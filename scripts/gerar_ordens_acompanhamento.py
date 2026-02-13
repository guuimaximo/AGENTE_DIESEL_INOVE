import os
import re
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

import vertexai
from vertexai.generative_models import GenerativeModel

from supabase import create_client
from playwright.sync_api import sync_playwright

# ==============================================================================
# 1. CONFIGURAÇÃO E ENV
# ==============================================================================
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID") or os.getenv("PROJECT_ID")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
VERTEX_MODEL = os.getenv("VERTEX_MODEL", "gemini-2.5-pro")

SUPABASE_A_URL = os.getenv("SUPABASE_A_URL")
SUPABASE_A_SERVICE_ROLE_KEY = os.getenv("SUPABASE_A_SERVICE_ROLE_KEY")

SUPABASE_B_URL = os.getenv("SUPABASE_B_URL")
SUPABASE_B_SERVICE_ROLE_KEY = os.getenv("SUPABASE_B_SERVICE_ROLE_KEY")

ORDEM_BATCH_ID = os.getenv("ORDEM_BATCH_ID")

# Tabelas
TABELA_ORIGEM = os.getenv("DIESEL_SOURCE_TABLE", "premiacao_diaria")
TABELA_LOTE = "acompanhamento_lotes"
TABELA_ITENS = "acompanhamento_lote_itens"
TABELA_DESTINO = "diesel_acompanhamentos"

# Storage
BUCKET = "relatorios"
REMOTE_PREFIX = "acompanhamento"
PASTA_SAIDA = Path("Ordens_Geradas")

# Configurações
KML_MIN = 1.0
KML_MAX = 6.0
FETCH_DAYS = 90   # Histórico para gráfico
JANELA_DIAS = 7   # Paginação
PAGE_SIZE = 2000

# Status operacionais (fluxo)
STATUS_ORDEM_INICIAL = "AGUARDANDO INSTRUTOR"   # ✅ novo (ordem criada, aguardando ação prática)
STATUS_LOTE_PROCESSANDO = "PROCESSANDO"
STATUS_LOTE_CONCLUIDO = "CONCLUIDO"
STATUS_LOTE_ERRO = "ERRO"

# ==============================================================================
# 2. CLIENTES E HELPERS
# ==============================================================================
def safe_num(val):
    if pd.isna(val) or val is None or val == "":
        return 0.0
    try:
        return float(val)
    except Exception:
        return 0.0

def _sb_a():
    return create_client(SUPABASE_A_URL, SUPABASE_A_SERVICE_ROLE_KEY)

def _sb_b():
    return create_client(SUPABASE_B_URL, SUPABASE_B_SERVICE_ROLE_KEY)

def _safe_filename(name: str) -> str:
    name = str(name or "").strip()
    name = re.sub(r"[^\w\-.() ]+", "_", name, flags=re.UNICODE)
    return name[:100] or "sem_nome"

def atualizar_status_lote(status: str, msg: str = None):
    """Atualiza o status do lote (processamento do batch)."""
    if not ORDEM_BATCH_ID:
        return
    print(f"🔄 [Lote {ORDEM_BATCH_ID}] Status: {status}")
    sb = _sb_b()
    payload = {"status": status}
    if msg:
        payload["erro_msg"] = msg
    sb.table(TABELA_LOTE).update(payload).eq("id", ORDEM_BATCH_ID).execute()

def to_num(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    s = series.astype(str).str.strip()
    s = s.str.replace(",", ".", regex=False)
    s = s.str.replace(r"[^0-9\.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

def get_cluster(v):
    v = str(v).strip()
    if v.startswith("2216"): return "C8"
    if v.startswith("2222"): return "C9"
    if v.startswith("2224"): return "C10"
    if v.startswith("2425"): return "C11"
    if v.startswith("W"):    return "C6"
    return "OUTROS"

def upload_storage(local_path: Path, remote_name: str, content_type: str):
    """
    Faz upload e retorna:
      - remote_path (relativo ao bucket)
      - public_url (se bucket for público)
    """
    if not ORDEM_BATCH_ID:
        return {"remote_path": None, "public_url": None}

    sb = _sb_b()
    remote_path = f"{REMOTE_PREFIX}/{ORDEM_BATCH_ID}/{remote_name}"

    if not local_path.exists():
        return {"remote_path": None, "public_url": None}

    with open(local_path, "rb") as f:
        sb.storage.from_(BUCKET).upload(
            path=remote_path,
            file=f,
            file_options={"content-type": content_type, "upsert": "true"},
        )

    public_url = f"{SUPABASE_B_URL}/storage/v1/object/public/{BUCKET}/{remote_path}"
    return {"remote_path": remote_path, "public_url": public_url}

def safe_update_lote_item(sb, lote_id, motorista_chapa, payload):
    """
    Atualiza item do lote SEM quebrar se a tabela não tiver as colunas.
    """
    try:
        sb.table(TABELA_ITENS).update(payload).eq("lote_id", lote_id).eq("motorista_chapa", motorista_chapa).execute()
    except Exception as e:
        print(f"⚠️ Não consegui atualizar item do lote ({motorista_chapa}): {e}")

def safe_insert_or_update_ordem(sb, payload_variants, match_where=None):
    """
    Tenta inserir (ou atualizar se existir) em diesel_acompanhamentos com robustez.
    - payload_variants: lista de dicts, do mais completo ao mais simples
    - match_where: dict com colunas para tentar achar registro existente e atualizar
    """
    # 1) tenta update se achar registro (opcional)
    if match_where:
        try:
            q = sb.table(TABELA_DESTINO).select("*")
            for k, v in match_where.items():
                q = q.eq(k, v)
            existing = q.limit(1).execute().data
            if existing:
                # tenta update com o primeiro payload que funcionar
                for p in payload_variants:
                    try:
                        uq = sb.table(TABELA_DESTINO).update(p)
                        for k, v in match_where.items():
                            uq = uq.eq(k, v)
                        uq.execute()
                        return "UPDATED"
                    except Exception:
                        continue
        except Exception:
            pass

    # 2) tenta insert com variações
    for p in payload_variants:
        try:
            sb.table(TABELA_DESTINO).insert(p).execute()
            return "INSERTED"
        except Exception as e:
            last = e

    raise last  # se nada funcionou, estoura

# ==============================================================================
# 3. LEITURA DE NOMES (CSV)
# ==============================================================================
def carregar_mapa_nomes(caminho_csv="motoristas_rows.csv"):
    if not os.path.exists(caminho_csv):
        print("⚠️ CSV de nomes não encontrado. Usando nomes do sistema.")
        return {}
    try:
        df = pd.read_csv(caminho_csv, dtype=str)
        df["chapa"] = df["chapa"].str.strip()

        mapa = {}
        for _, row in df.iterrows():
            mapa[row["chapa"]] = {
                "nome": str(row.get("nome", "")).strip().upper(),
                "cargo": str(row.get("cargo", "MOTORISTA")).strip().upper(),
            }
        print(f"📋 Mapa de nomes carregado: {len(mapa)} registros.")
        return mapa
    except Exception as e:
        print(f"❌ Erro ao ler CSV de nomes: {e}")
        return {}

# ==============================================================================
# 4. CARREGAMENTO DE DADOS
# ==============================================================================
def obter_motoristas_do_lote():
    """Busca a lista de chapas que o React gravou no Supabase B"""
    if not ORDEM_BATCH_ID:
        print("❌ ORDEM_BATCH_ID não fornecido.")
        return []

    print(f"📥 Buscando itens do lote {ORDEM_BATCH_ID}...")
    sb = _sb_b()
    try:
        res = sb.table(TABELA_ITENS).select("*").eq("lote_id", ORDEM_BATCH_ID).execute()
        itens = res.data or []
        print(f"📋 {len(itens)} motoristas para processar.")
        return itens
    except Exception as e:
        print(f"❌ Erro ao buscar itens: {e}")
        return []

def carregar_historico_motorista(chapa):
    """Busca dados brutos no Supabase A (90 dias para gráfico, 30 para análise)"""
    print(f"   ↳ Buscando dados brutos: {chapa}...")
    sb = _sb_a()

    agora = datetime.utcnow()
    limite = agora - timedelta(days=FETCH_DAYS)
    cursor = agora
    all_rows = []

    # Paginação reversa por data
    while cursor > limite:
        inicio_janela = cursor - timedelta(days=JANELA_DIAS)
        s_fim = cursor.strftime("%Y-%m-%d")
        s_ini = inicio_janela.strftime("%Y-%m-%d")

        start = 0
        while True:
            try:
                resp = (
                    sb.table(TABELA_ORIGEM)
                    .select('dia, motorista, veiculo, linha, "km/l", km_rodado, combustivel_consumido')
                    .eq("motorista", chapa)
                    .gte("dia", s_ini)
                    .lte("dia", s_fim)
                    .order("dia", desc=True)
                    .range(start, start + PAGE_SIZE - 1)
                    .execute()
                )

                rows = resp.data or []
                all_rows.extend(rows)
                if len(rows) < PAGE_SIZE:
                    break
                start += PAGE_SIZE
            except Exception:
                break

        cursor = inicio_janela - timedelta(days=1)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df.rename(
        columns={
            "dia": "Date",
            "motorista": "Motorista",
            "km_rodado": "Km",
            "combustivel_consumido": "Comb.",
            "km/l": "kml_db",
        },
        inplace=True,
    )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Km"] = to_num(df["Km"])
    df["Comb."] = to_num(df["Comb."])

    df = df.dropna(subset=["Date", "Km", "Comb."])
    df = df[(df["Comb."] > 0) & (df["Km"] > 0)].copy()
    df["Cluster"] = df["veiculo"].apply(get_cluster)

    df = df[(df["Km"] / df["Comb."] >= KML_MIN) & (df["Km"] / df["Comb."] <= KML_MAX)]
    return df

# ==============================================================================
# 5. CÁLCULOS E PROCESSAMENTO
# ==============================================================================
def processar_dados_prontuario(df, chapa, info_nome):
    if df.empty:
        return None

    max_date = df["Date"].max()
    min_date_30 = max_date - timedelta(days=30)
    df_30d = df[df["Date"] >= min_date_30].copy()
    if df_30d.empty:
        return None

    raio_x = (
        df_30d.groupby(["linha", "Cluster"])
        .agg({"Km": "sum", "Comb.": "sum", "veiculo": lambda x: list(x.unique())[0]})
        .reset_index()
    )

    METAS = {"C6": 2.5, "C8": 2.6, "C9": 2.73, "C10": 2.8, "C11": 2.9, "OUTROS": 2.5}
    raio_x["KML_Meta"] = raio_x["Cluster"].map(METAS).fillna(2.5)
    raio_x["KML_Real"] = raio_x["Km"] / raio_x["Comb."]

    def calc_metrics(r):
        litros_meta = r["Km"] / r["KML_Meta"]
        desp = (r["Comb."] - litros_meta) if r["KML_Real"] < r["KML_Meta"] else 0
        return pd.Series([litros_meta, desp])

    raio_x[["Litros_Meta", "Desperdicio"]] = raio_x.apply(calc_metrics, axis=1)
    raio_x = raio_x.sort_values("Desperdicio", ascending=False)

    total_km = float(raio_x["Km"].sum())
    total_litros = float(raio_x["Comb."].sum())
    total_desperdicio = float(raio_x["Desperdicio"].sum())

    kml_geral_real = (total_km / total_litros) if total_litros > 0 else 0.0
    soma_litros_teoricos = float(raio_x["Litros_Meta"].sum())
    kml_geral_meta = (total_km / soma_litros_teoricos) if soma_litros_teoricos > 0 else 0.0

    # Gráfico últimas 4-5 semanas
    min_date_chart = max_date - timedelta(days=35)
    df_chart = df[df["Date"] >= min_date_chart].copy()
    df_chart["Semana_Dt"] = df_chart["Date"].dt.to_period("W").apply(lambda r: r.start_time)
    grp_chart = df_chart.groupby("Semana_Dt").agg({"Km": "sum", "Comb.": "sum"}).sort_index()

    dates, real_vals, meta_vals = [], [], []
    for dt, row in grp_chart.iterrows():
        kml = (row["Km"] / row["Comb."]) if row["Comb."] > 0 else 0.0
        dates.append(dt.strftime("%d/%m"))
        real_vals.append(float(kml))
        meta_vals.append(float(kml_geral_meta))

    top_ofensor = raio_x.iloc[0] if not raio_x.empty else None

    return {
        "chapa": chapa,
        "nome": info_nome.get("nome", chapa),
        "cargo": info_nome.get("cargo", "MOTORISTA"),
        "periodo_txt": f"{min_date_30.strftime('%d/%m')} a {max_date.strftime('%d/%m')}",
        "raio_x": raio_x,
        "totais": {
            "km": total_km,
            "litros": total_litros,
            "desp": total_desperdicio,
            "kml_real": float(kml_geral_real),
            "kml_meta": float(kml_geral_meta),
        },
        "grafico": {"dates": dates, "real": real_vals, "meta": meta_vals},
        "foco_cluster": top_ofensor["Cluster"] if top_ofensor is not None else "-",
        # foco linha (a mais crítica)
        "foco_linha": str(top_ofensor["linha"]) if top_ofensor is not None else "-",
    }

# ==============================================================================
# 6. GERAÇÃO DE ASSETS (IA, GRÁFICO, HTML)
# ==============================================================================
def chamar_ia_coach(dados):
    if not VERTEX_PROJECT_ID:
        return "ANÁLISE: IA Indisponível."
    try:
        vertexai.init(project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION)
        model = GenerativeModel(VERTEX_MODEL)

        top_linhas = dados["raio_x"].head(3)[["linha", "Cluster", "KML_Real", "KML_Meta"]].to_string(index=False)

        prompt = f"""
Você é um Instrutor Técnico Master de Condução Econômica.

ALVO:
Motorista: {dados['nome']}
Performance: {dados['totais']['kml_real']:.2f} km/l (Meta: {dados['totais']['kml_meta']:.2f})
Desperdício Mensal: {dados['totais']['desp']:.0f} Litros de Diesel

ONDE ELE ESTÁ ERRANDO (Top Linhas):
{top_linhas}

Gere um PRONTUÁRIO curto e direto.
Responda ESTRITAMENTE com estes 3 tópicos (use as tags exatas):

DIAGNÓSTICO COMPORTAMENTAL: [Explique tecnicamente o provável erro: ex: giro alto, troca tardia, falta de antecipação]
FOCO DA MONITORIA: [3 itens bullet points para o instrutor treinar na prática]
FEEDBACK EDUCATIVO: [Um parágrafo de roteiro para o instrutor falar para o motorista, tom respeitoso mas firme sobre a meta]
"""
        return model.generate_content(prompt).text
    except Exception as e:
        print(f"Erro IA: {e}")
        return "DIAGNÓSTICO COMPORTAMENTAL: Não foi possível gerar."

def gerar_grafico_visual(dados_g, path):
    if not dados_g.get("dates"):
        return

    plt.figure(figsize=(10, 3.5))
    plt.plot(dados_g["dates"], dados_g["meta"], color="#c0392b", linestyle="--", linewidth=2, label="Meta (Ref)", alpha=0.6)
    plt.plot(dados_g["dates"], dados_g["real"], color="#2c3e50", marker="o", linewidth=2.5, markersize=6, label="Realizado")

    for x, y in zip(dados_g["dates"], dados_g["real"]):
        plt.text(x, y + 0.03, f"{y:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#2c3e50")

    plt.title("Evolução Semanal (Tendência)", fontsize=10, fontweight="bold", loc="left")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()

def extrair_secao(texto, chaves):
    if isinstance(chaves, str):
        chaves = [chaves]
    pattern = "|".join(chaves)
    m = re.search(
        rf"(?:{pattern})[:\s\-]*(.*?)(?=\n(?:DIAGN|FOCO|FEEDBACK|ROTEIRO)|$)",
        texto,
        re.DOTALL | re.IGNORECASE,
    )
    return m.group(1).strip() if m else "..."

def gerar_html_final(d, txt_ia, img_path):
    analise = extrair_secao(txt_ia, ["DIAGNÓSTICO COMPORTAMENTAL", "DIAGNOSTICO"])
    foco = extrair_secao(txt_ia, ["FOCO DA MONITORIA", "ROTEIRO", "AÇÃO"])
    feedback = extrair_secao(txt_ia, ["FEEDBACK EDUCATIVO", "FEEDBACK"])

    rows = ""
    for _, r in d["raio_x"].head(10).iterrows():
        style_loss = "color:#c0392b; font-weight:bold" if r["Desperdicio"] > 5 else ""
        rows += f"""<tr>
<td>{r['linha']}</td>
<td>{r['veiculo']}</td>
<td>{r['Cluster']}</td>
<td align="right">{r['Km']:.0f}</td>
<td align="right">{r['Comb.']:.0f}</td>
<td align="right"><b>{r['KML_Real']:.2f}</b></td>
<td align="right" style="color:#7f8c8d">{r['KML_Meta']:.2f}</td>
<td align="right" style="{style_loss}">{r['Desperdicio']:.1f} L</td>
</tr>"""

    t = d["totais"]

    return f"""
<!DOCTYPE html><html lang="pt-BR"><head><meta charset="UTF-8">
<style>
body {{ font-family: 'Segoe UI', sans-serif; padding: 25px; color: #333; background: #fff; }}
.header {{ border-bottom: 3px solid #2c3e50; padding-bottom: 15px; margin-bottom: 25px; display: flex; justify-content: space-between; }}
.title h1 {{ margin: 0; color: #2c3e50; font-size: 24px; text-transform: uppercase; }}
.tag {{ background:#c0392b; color:white; padding:5px 10px; border-radius:4px; font-size:12px; font-weight:bold; }}
.cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 30px; }}
.card {{ border: 1px solid #ddd; border-radius: 8px; padding: 15px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
.card b {{ display: block; font-size: 22px; color: #2c3e50; margin-bottom: 5px; }}
.card span {{ font-size: 11px; text-transform: uppercase; color: #7f8c8d; font-weight: bold; }}
table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin-bottom: 5px; }}
th {{ background: #34495e; color: white; padding: 8px; text-align: left; }}
td {{ border-bottom: 1px solid #eee; padding: 8px; }}
tfoot {{ background: #ecf0f1; font-weight: bold; border-top: 2px solid #bdc3c7; }}
h2 {{ font-size: 15px; color: #2980b9; margin-top: 30px; border-left: 5px solid #2980b9; padding-left: 10px; margin-bottom: 15px; }}
.ia-box {{ padding: 10px 15px; border-radius: 6px; margin-bottom: 10px; font-size: 13px; line-height: 1.5; }}
.footer {{ margin-top: 50px; font-size: 10px; color: #aaa; text-align: center; border-top: 1px solid #eee; padding-top: 10px; }}
</style></head>
<body>
<div class="header">
  <div>
    <h1>ORDEM DE ACOMPANHAMENTO</h1>
    <div style="font-size:16px; font-weight:bold; margin-top:5px">{d['nome']}</div>
    <div style="font-size:12px; color:#666">{d['chapa']} | {d['cargo']}</div>
  </div>
  <div style="text-align:right">
    <div class="tag">PRIORIDADE ALTA</div>
    <div style="font-size:12px; margin-top:5px; color:#666">Período: {d['periodo_txt']}</div>
  </div>
</div>

<div class="cards">
  <div class="card"><b>{d['foco_cluster']}</b><span>Foco Tecnologia</span></div>
  <div class="card"><b style="color:#c0392b">{t['desp']:.0f} L</b><span>Desperdício Total</span></div>
  <div class="card"><b>{t['kml_real']:.2f}</b><span>KM/L Real</span></div>
  <div class="card"><b style="color:#7f8c8d">{t['kml_meta']:.2f}</b><span>Meta Ajustada</span></div>
</div>

<h2>1. RAIO-X DA OPERAÇÃO</h2>
<table>
  <thead><tr><th>Linha</th><th>Veículo</th><th>Cluster</th><th align="right">KM</th><th align="right">Litros</th><th align="right">Real</th><th align="right">Meta</th><th align="right">Desp.</th></tr></thead>
  <tbody>{rows}</tbody>
  <tfoot><tr>
    <td colspan="3">TOTAIS / MÉDIAS</td>
    <td align="right">{t['km']:.0f}</td>
    <td align="right">{t['litros']:.0f}</td>
    <td align="right" style="color:#e67e22">{t['kml_real']:.2f}</td>
    <td align="right">{t['kml_meta']:.2f}</td>
    <td align="right" style="color:#c0392b">{t['desp']:.0f} L</td>
  </tr></tfoot>
</table>

<h2>2. EVOLUÇÃO SEMANAL</h2>
<div style="border:1px solid #eee; padding:5px; border-radius:6px; text-align:center">
  <img src="{os.path.basename(img_path)}" style="width:100%; height:auto; max-height:200px; object-fit:contain;">
</div>

<h2>3. DIAGNÓSTICO E AÇÃO (IA)</h2>
<div class="ia-box" style="background:#fff3e0; border:1px solid #ffe0b2;">
  <b>DIAGNÓSTICO COMPORTAMENTAL:</b><br>{analise}
</div>
<div class="ia-box" style="background:#e3f2fd; border:1px solid #bbdefb;">
  <b>FOCO DA MONITORIA:</b><br>{foco}
</div>
<div class="ia-box" style="background:#e8f5e9; border:1px solid #c8e6c9;">
  <b>FEEDBACK EDUCATIVO:</b><br>{feedback}
</div>

<div style="margin-top:40px; display:flex; justify-content:space-between; padding:0 20px;">
  <div style="border-top:1px solid #000; width:40%; text-align:center; font-size:11px; padding-top:5px;">Instrutor Responsável</div>
  <div style="border-top:1px solid #000; width:40%; text-align:center; font-size:11px; padding-top:5px;">Motorista</div>
</div>

<div class="footer">
  Gerado automaticamente pelo Agente Diesel AI em {datetime.now().strftime('%d/%m/%Y %H:%M')}
</div>
</body></html>
"""

# ==============================================================================
# 7. MAIN FLOW (AJUSTADO PARA STATUS OPERACIONAL)
# ==============================================================================
def main():
    if not ORDEM_BATCH_ID:
        print("❌ ORDEM_BATCH_ID não definido no ambiente.")
        return

    try:
        atualizar_status_lote(STATUS_LOTE_PROCESSANDO)
        PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

        mapa_nomes = carregar_mapa_nomes()
        itens = obter_motoristas_do_lote()
        if not itens:
            atualizar_status_lote(STATUS_LOTE_ERRO, "Lote sem itens")
            return

        print(f"🚀 Iniciando geração de {len(itens)} ordens...")
        sb_dest = _sb_b()

        ok_count = 0
        erro_count = 0

        for item in itens:
            mot_chapa = item.get("motorista_chapa")
            if not mot_chapa:
                continue

            # Marca item como "PROCESSANDO" (se existir coluna)
            safe_update_lote_item(
                sb_dest,
                ORDEM_BATCH_ID,
                mot_chapa,
                {"status": "PROCESSANDO", "erro_msg": None},
            )

            try:
                df_full = carregar_historico_motorista(mot_chapa)
                if df_full.empty:
                    print(f"⚠️ Sem dados para {mot_chapa}")
                    safe_update_lote_item(sb_dest, ORDEM_BATCH_ID, mot_chapa, {"status": "SEM_DADOS"})
                    continue

                info_nome = mapa_nomes.get(mot_chapa, {"nome": mot_chapa, "cargo": "MOTORISTA"})
                dados_pront = processar_dados_prontuario(df_full, mot_chapa, info_nome)
                if not dados_pront:
                    safe_update_lote_item(sb_dest, ORDEM_BATCH_ID, mot_chapa, {"status": "SEM_DADOS"})
                    continue

                print(f"   ⚙️ Gerando assets para {mot_chapa}...")

                safe = _safe_filename(mot_chapa)
                p_img = PASTA_SAIDA / f"{safe}.png"
                p_html = PASTA_SAIDA / f"{safe}.html"
                p_pdf = PASTA_SAIDA / f"{safe}.pdf"

                gerar_grafico_visual(dados_pront["grafico"], p_img)
                txt_ia = chamar_ia_coach(dados_pront)
                html_content = gerar_html_final(dados_pront, txt_ia, p_img)

                with open(p_html, "w", encoding="utf-8") as f:
                    f.write(html_content)

                # PDF Playwright
                with sync_playwright() as p:
                    browser = p.chromium.launch(args=["--no-sandbox"])
                    page = browser.new_page()
                    page.goto(p_html.resolve().as_uri())
                    page.pdf(
                        path=str(p_pdf),
                        format="A4",
                        print_background=True,
                        margin={"top": "10mm", "bottom": "10mm", "left": "10mm", "right": "10mm"},
                    )
                    browser.close()

                # Upload (retorna remote_path e public_url)
                up_pdf = upload_storage(p_pdf, f"{safe}.pdf", "application/pdf")
                up_html = upload_storage(p_html, f"{safe}.html", "text/html")

                # ✅ AJUSTE PRINCIPAL:
                # Criar a ORDEM no status operacional correto: AGUARDANDO INSTRUTOR
                # E guardar "status de geração" dentro do metadata, para não misturar conceitos.
                foco_txt = f"{dados_pront.get('foco_cluster','-')} - Linha {dados_pront.get('foco_linha','-')}"

                metadata_base = {
                    "versao": "V9_Status_Operacional",
                    "geracao": {
                        "status_geracao": "CONCLUIDO",
                        "lote_id": ORDEM_BATCH_ID,
                        "pdf": up_pdf,
                        "html": up_html,
                    },
                    "kpis": {
                        "kml_real": dados_pront["totais"]["kml_real"],
                        "kml_meta": dados_pront["totais"]["kml_meta"],
                        "perda_litros": dados_pront["totais"]["desp"],
                        "foco": foco_txt,
                    },
                }

                # Variante 1 (se seu schema já for o novo/operacional)
                payload_v1 = {
                    "motorista_chapa": mot_chapa,
                    "motorista_nome": dados_pront["nome"],
                    "status": STATUS_ORDEM_INICIAL,
                    "motivo": f"Média {dados_pront['totais']['kml_real']:.2f} vs Meta {dados_pront['totais']['kml_meta']:.2f}",
                    "observacao_inicial": foco_txt,
                    "kml_inicial": dados_pront["totais"]["kml_real"],
                    "kml_meta": dados_pront["totais"]["kml_meta"],
                    "metadata": metadata_base,
                }

                # Variante 2 (se seu schema for o “legado” do seu script anterior)
                payload_v2 = {
                    "lote_id": ORDEM_BATCH_ID,
                    "motorista_chapa": mot_chapa,
                    "motorista_nome": dados_pront["nome"],
                    "status": STATUS_ORDEM_INICIAL,  # ✅ mudou de CONCLUIDO para AGUARDANDO INSTRUTOR
                    # manter campos antigos, mas sem depender disso para o fluxo
                    "arquivo_pdf_path": up_pdf["remote_path"] or up_pdf["public_url"],
                    "arquivo_html_path": up_html["remote_path"] or up_html["public_url"],
                    "kml_real": dados_pront["totais"]["kml_real"],
                    "perda_litros": dados_pront["totais"]["desp"],
                    "created_at": datetime.utcnow().isoformat(),
                    "metadata": metadata_base,
                }

                # Variante 3 (mínimo viável, se houver colunas bem restritas)
                payload_v3 = {
                    "motorista_chapa": mot_chapa,
                    "status": STATUS_ORDEM_INICIAL,
                    "metadata": metadata_base,
                }

                # tenta update/insert robusto
                match_where = {"lote_id": ORDEM_BATCH_ID, "motorista_chapa": mot_chapa}
                try:
                    result = safe_insert_or_update_ordem(
                        sb_dest,
                        [payload_v1, payload_v2, payload_v3],
                        match_where=match_where,
                    )
                except Exception:
                    # se schema não tiver lote_id, tenta sem ele
                    result = safe_insert_or_update_ordem(
                        sb_dest,
                        [payload_v1, payload_v2, payload_v3],
                        match_where={"motorista_chapa": mot_chapa},
                    )

                # marca item do lote como "GERADO" (se existir coluna)
                safe_update_lote_item(
                    sb_dest,
                    ORDEM_BATCH_ID,
                    mot_chapa,
                    {
                        "status": "GERADO",
                        "pdf_path": up_pdf["remote_path"] or up_pdf["public_url"],
                        "html_path": up_html["remote_path"] or up_html["public_url"],
                    },
                )

                ok_count += 1
                print(f"✅ {mot_chapa}: Sucesso ({result}). Ordem criada em '{STATUS_ORDEM_INICIAL}'.")

            except Exception as e:
                erro_count += 1
                print(f"❌ {mot_chapa}: erro ao gerar ordem: {e}")
                safe_update_lote_item(sb_dest, ORDEM_BATCH_ID, mot_chapa, {"status": "ERRO", "erro_msg": str(e)})

        atualizar_status_lote(STATUS_LOTE_CONCLUIDO)
        print(f"🏁 Processo finalizado. OK={ok_count} | ERROS={erro_count}")

    except Exception as e:
        print(f"❌ Erro fatal no script: {e}")
        import traceback
        traceback.print_exc()
        atualizar_status_lote(STATUS_LOTE_ERRO, str(e))

if __name__ == "__main__":
    main()
