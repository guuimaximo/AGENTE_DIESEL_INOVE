import os
import re
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from supabase import create_client
from playwright.sync_api import sync_playwright

# ==============================================================================
# ENV / CONFIG
# ==============================================================================
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID") or os.getenv("PROJECT_ID")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
VERTEX_MODEL = os.getenv("VERTEX_MODEL", "gemini-2.5-pro")
VERTEX_ENABLED = (os.getenv("VERTEX_ENABLED", "0").strip() == "1")

GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON", "").strip()

SUPABASE_A_URL = os.getenv("SUPABASE_A_URL")
SUPABASE_A_SERVICE_ROLE_KEY = os.getenv("SUPABASE_A_SERVICE_ROLE_KEY")

SUPABASE_B_URL = os.getenv("SUPABASE_B_URL")
SUPABASE_B_SERVICE_ROLE_KEY = os.getenv("SUPABASE_B_SERVICE_ROLE_KEY")

ORDEM_BATCH_ID = os.getenv("ORDEM_BATCH_ID")  # ID do lote no Supabase B
TABELA_ORIGEM = os.getenv("DIESEL_SOURCE_TABLE", "premiacao_diaria")

# Tabelas (Supabase B)
TABELA_LOTE = "acompanhamento_lotes"
TABELA_ITENS = "acompanhamento_lote_itens"
TABELA_ORDEM = "diesel_acompanhamentos"
TABELA_EVENTOS = "diesel_acompanhamento_eventos"

# Storage (Supabase B)
BUCKET = "relatorios"
REMOTE_PREFIX = "acompanhamento"
PASTA_SAIDA = Path("Ordens_Geradas")

# Regras
KML_MIN = 1.5
KML_MAX = 5.0
FETCH_DAYS = 90
JANELA_DIAS = 7
PAGE_SIZE = 2000

# padrão: ordem nasce aguardando instrutor, monitoramento padrão 7d (o instrutor redefine no LANÇAR)
DEFAULT_DIAS_MONITORAMENTO = int(os.getenv("DEFAULT_DIAS_MONITORAMENTO", "7"))


# ==============================================================================
# HELPERS
# ==============================================================================
def _sb_a():
    if not SUPABASE_A_URL or not SUPABASE_A_SERVICE_ROLE_KEY:
        raise RuntimeError("ENV Supabase A ausente (SUPABASE_A_URL / SUPABASE_A_SERVICE_ROLE_KEY)")
    return create_client(SUPABASE_A_URL, SUPABASE_A_SERVICE_ROLE_KEY)


def _sb_b():
    if not SUPABASE_B_URL or not SUPABASE_B_SERVICE_ROLE_KEY:
        raise RuntimeError("ENV Supabase B ausente (SUPABASE_B_URL / SUPABASE_B_SERVICE_ROLE_KEY)")
    return create_client(SUPABASE_B_URL, SUPABASE_B_SERVICE_ROLE_KEY)


def _safe_filename(name: str) -> str:
    name = str(name or "").strip()
    name = re.sub(r"[^\w\-.() ]+", "_", name, flags=re.UNICODE)
    return name[:100] or "sem_nome"


def to_num(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    s = series.astype(str).str.strip()
    s = s.str.replace(",", ".", regex=False)
    s = s.str.replace(r"[^0-9\.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def n(v):
    try:
        x = float(v)
        return x if pd.notna(x) else 0.0
    except Exception:
        return 0.0


def get_cluster(v):
    v = str(v).strip()
    if v.startswith("2216"):
        return "C8"
    if v.startswith("2222"):
        return "C9"
    if v.startswith("2224"):
        return "C10"
    if v.startswith("2425"):
        return "C11"
    if v.startswith("W"):
        return "C6"
    return "OUTROS"


def atualizar_status_lote(status: str, msg: str = None, extra: dict = None):
    if not ORDEM_BATCH_ID:
        return
    sb = _sb_b()
    payload = {"status": status}
    if msg:
        payload["erro_msg"] = str(msg)[:1000]
    if extra is not None:
        payload["metadata"] = extra
    sb.table(TABELA_LOTE).update(payload).eq("id", ORDEM_BATCH_ID).execute()
    print(f"🔄 [Lote {ORDEM_BATCH_ID}] Status: {status}")


def upload_storage(local_path: Path, remote_name: str, content_type: str):
    """
    Retorna (remote_path, public_url)
    """
    if not ORDEM_BATCH_ID:
        return (None, None)
    if not local_path.exists():
        return (None, None)

    sb = _sb_b()
    remote_path = f"{REMOTE_PREFIX}/{ORDEM_BATCH_ID}/{remote_name}"

    with open(local_path, "rb") as f:
        sb.storage.from_(BUCKET).upload(
            path=remote_path,
            file=f,
            file_options={"content-type": content_type, "upsert": "true"},
        )

    public_url = f"{SUPABASE_B_URL}/storage/v1/object/public/{BUCKET}/{remote_path}"
    return (remote_path, public_url)


# ==============================================================================
# NOMES (CSV)
# ==============================================================================
def carregar_mapa_nomes(caminho_csv="motoristas_rows.csv"):
    if not os.path.exists(caminho_csv):
        print("⚠️ CSV de nomes não encontrado. Usando chapa como nome.")
        return {}
    try:
        df = pd.read_csv(caminho_csv, dtype=str)
        df["chapa"] = df["chapa"].astype(str).str.strip()

        mapa = {}
        for _, row in df.iterrows():
            mapa[row["chapa"]] = {
                "nome": str(row.get("nome", "")).strip().upper() or row["chapa"],
                "cargo": str(row.get("cargo", "MOTORISTA")).strip().upper() or "MOTORISTA",
            }
        print(f"📋 Mapa de nomes carregado: {len(mapa)} registros.")
        return mapa
    except Exception as e:
        print(f"❌ Erro ao ler CSV de nomes: {e}")
        return {}


# ==============================================================================
# LEITURA LOTE (Supabase B)
# ==============================================================================
def obter_motoristas_do_lote():
    if not ORDEM_BATCH_ID:
        print("❌ ORDEM_BATCH_ID não fornecido.")
        return []

    print(f"📥 Buscando itens do lote {ORDEM_BATCH_ID}...")
    sb = _sb_b()
    res = sb.table(TABELA_ITENS).select("*").eq("lote_id", ORDEM_BATCH_ID).execute()
    itens = res.data or []
    print(f"📋 {len(itens)} motoristas para processar.")
    return itens


# ==============================================================================
# DADOS BRUTOS (Supabase A)
# ==============================================================================
def carregar_historico_motorista(chapa):
    print(f"   ↳ Buscando dados brutos: {chapa}...")
    sb = _sb_a()

    agora = datetime.utcnow()
    limite = agora - timedelta(days=FETCH_DAYS)
    cursor = agora
    all_rows = []

    while cursor > limite:
        inicio_janela = cursor - timedelta(days=JANELA_DIAS)
        s_fim = cursor.strftime("%Y-%m-%d")
        s_ini = inicio_janela.strftime("%Y-%m-%d")

        start = 0
        while True:
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

    # outliers
    kml_calc = df["Km"] / df["Comb."]
    df = df[(kml_calc >= KML_MIN) & (kml_calc <= KML_MAX)].copy()

    return df


# ==============================================================================
# PROCESSAMENTO (gera raio_x + periodo)
# ==============================================================================
def processar_dados_prontuario(df, chapa, info_nome):
    if df.empty:
        return None

    max_date = df["Date"].max()
    if pd.isna(max_date):
        return None

    min_date_30 = max_date - timedelta(days=30)
    df_30d = df[df["Date"] >= min_date_30].copy()
    if df_30d.empty:
        return None

    # Raio-X por linha + cluster
    raio_x = (
        df_30d.groupby(["linha", "Cluster"])
        .agg(
            Km=("Km", "sum"),
            Comb=("Comb.", "sum"),
            veiculo=("veiculo", lambda x: list(pd.Series(x).dropna().unique())[0] if len(pd.Series(x).dropna().unique()) else None),
        )
        .reset_index()
    )

    METAS = {"C6": 2.5, "C8": 2.6, "C9": 2.73, "C10": 2.8, "C11": 2.9, "OUTROS": 2.5}
    raio_x["kml_meta"] = raio_x["Cluster"].map(METAS).fillna(2.5)
    raio_x["kml_real"] = raio_x["Km"] / raio_x["Comb"]

    def calc_desp(r):
        litros_meta = r["Km"] / r["kml_meta"] if r["kml_meta"] > 0 else 0
        desp = (r["Comb"] - litros_meta) if (r["kml_real"] < r["kml_meta"]) else 0
        return litros_meta, desp

    raio_x[["litros_meta", "desperdicio"]] = raio_x.apply(lambda r: pd.Series(calc_desp(r)), axis=1)
    raio_x = raio_x.sort_values("desperdicio", ascending=False)

    total_km = float(raio_x["Km"].sum())
    total_litros = float(raio_x["Comb"].sum())
    total_desperdicio = float(raio_x["desperdicio"].sum())

    kml_geral_real = (total_km / total_litros) if total_litros > 0 else 0.0
    soma_litros_teoricos = float(raio_x["litros_meta"].sum())
    kml_geral_meta = (total_km / soma_litros_teoricos) if soma_litros_teoricos > 0 else 0.0

    top = raio_x.iloc[0] if not raio_x.empty else None
    foco = f"{top['Cluster']} - Linha {top['linha']}" if top is not None else "Geral"

    return {
        "chapa": chapa,
        "nome": info_nome.get("nome", chapa),
        "cargo": info_nome.get("cargo", "MOTORISTA"),
        "periodo_inicio": min_date_30.date().isoformat(),
        "periodo_fim": max_date.date().isoformat(),
        "periodo_txt": f"{min_date_30.strftime('%d/%m')} a {max_date.strftime('%d/%m')}",
        "raio_x": raio_x,
        "totais": {
            "km": total_km,
            "litros": total_litros,
            "desp": total_desperdicio,
            "kml_real": kml_geral_real,
            "kml_meta": kml_geral_meta,
        },
        "foco": foco,
        "foco_cluster": top["Cluster"] if top is not None else "OUTROS",
        "linha_foco": top["linha"] if top is not None else None,
        "veiculo_foco": top["veiculo"] if top is not None else None,
    }


# ==============================================================================
# IA (Opcional)
# ==============================================================================
def chamar_ia_coach(dados):
    if not VERTEX_ENABLED or not VERTEX_PROJECT_ID:
        return "DIAGNÓSTICO COMPORTAMENTAL: IA desativada.\nFOCO DA MONITORIA: -\nFEEDBACK EDUCATIVO: -"

    try:
        credentials = None
        if GOOGLE_CREDENTIALS_JSON:
            from google.oauth2 import service_account
            info = json.loads(GOOGLE_CREDENTIALS_JSON)
            credentials = service_account.Credentials.from_service_account_info(info)

        import vertexai
        from vertexai.generative_models import GenerativeModel

        vertexai.init(project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION, credentials=credentials)
        model = GenerativeModel(VERTEX_MODEL)

        top_linhas = dados["raio_x"].head(3)[["linha", "Cluster", "kml_real", "kml_meta"]].to_string(index=False)

        prompt = f"""
Você é um Instrutor Técnico Master de Condução Econômica.

ALVO:
Motorista: {dados['nome']} ({dados['chapa']})
Performance 30d: {dados['totais']['kml_real']:.2f} km/l (Meta ref: {dados['totais']['kml_meta']:.2f})
Desperdício 30d: {dados['totais']['desp']:.0f} Litros

ONDE ESTÁ ERRANDO (Top Linhas):
{top_linhas}

Responda ESTRITAMENTE com 3 tópicos (tags exatas):

DIAGNÓSTICO COMPORTAMENTAL: ...
FOCO DA MONITORIA: ...
FEEDBACK EDUCATIVO: ...
"""
        return model.generate_content(prompt).text

    except Exception as e:
        print(f"Erro IA (ignorado): {e}")
        return "DIAGNÓSTICO COMPORTAMENTAL: IA indisponível.\nFOCO DA MONITORIA: -\nFEEDBACK EDUCATIVO: -"


# ==============================================================================
# HTML/PDF
# ==============================================================================
def gerar_html_final(d, txt_ia):
    return f"""
<!DOCTYPE html><html lang="pt-BR"><head><meta charset="UTF-8">
<style>
body {{ font-family: Arial, sans-serif; padding: 24px; }}
h1 {{ margin: 0 0 6px 0; }}
small {{ color:#666; }}
.box {{ border:1px solid #eee; border-radius:10px; padding:12px; margin-top:12px; }}
</style>
</head><body>
<h1>ORDEM DE ACOMPANHAMENTO</h1>
<small>{d['nome']} • {d['chapa']} • Período {d['periodo_txt']}</small>

<div class="box"><b>Foco:</b> {d['foco']}</div>
<div class="box">
  <b>KM/L Real:</b> {d['totais']['kml_real']:.2f}
  • <b>Meta ref:</b> {d['totais']['kml_meta']:.2f}
  • <b>Desperdício:</b> {d['totais']['desp']:.0f} L
</div>

<div class="box" style="white-space:pre-wrap"><b>IA:</b><br>{txt_ia}</div>
</body></html>
"""


def html_to_pdf(p_html: Path, p_pdf: Path):
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


# ==============================================================================
# SUPABASE B: CRIAR ORDEM + EVENTO
# ==============================================================================
def criar_ordem_e_evento(sb_b, dados, lote_id, pdf_path, pdf_url, html_path, html_url, txt_ia):
    # dt padrão (instrutor redefine no LANÇAR)
    dt_inicio = datetime.utcnow().date().isoformat()
    dias = DEFAULT_DIAS_MONITORAMENTO
    dt_fim_planejado = (datetime.utcnow().date() + timedelta(days=dias - 1)).isoformat()

    raio_top = dados["raio_x"].head(10).to_dict(orient="records")

    payload = {
        # vínculo lote (se existir coluna; se não existir, remove)
        "lote_id": lote_id,

        "motorista_chapa": dados["chapa"],
        "motorista_nome": dados["nome"],
        "motivo": dados["foco"],

        # ✅ fluxo certo
        "status": "AGUARDANDO_INSTRUTOR",
        "dias_monitoramento": dias,
        "dt_inicio": dt_inicio,
        "dt_fim_planejado": dt_fim_planejado,

        # baseline
        "kml_inicial": dados["totais"]["kml_real"],
        "kml_meta": dados["totais"]["kml_meta"],
        "observacao_inicial": txt_ia[:5000],

        # arquivos / evidências
        "arquivo_pdf_url": pdf_url,
        "arquivo_html_url": html_url,

        # opcional: salvar paths também (se você quiser)
        "metadata": {
            "versao": "V12_ordem_aguardando_instrutor",
            "lote_id": lote_id,
            "periodo_inicio_30d": dados["periodo_inicio"],
            "periodo_fim_30d": dados["periodo_fim"],
            "foco": dados["foco"],
            "cluster_foco": dados["foco_cluster"],
            "linha_foco": dados["linha_foco"],
            "veiculo_foco": dados["veiculo_foco"],
            "kpis_30d": dados["totais"],
            "raio_x_top10": raio_top,
            "pdf_path": pdf_path,
            "html_path": html_path,
        },

        # se sua coluna for jsonb array
        "evidencias_urls": [u for u in [pdf_url, html_url] if u],
    }

    # 1) cria ORDEM (diesel_acompanhamentos)
    ordem = sb_b.table(TABELA_ORDEM).insert(payload).execute().data
    if not ordem:
        raise RuntimeError("Falha ao inserir diesel_acompanhamentos (ordem vazia).")
    ordem_id = ordem[0].get("id")
    if not ordem_id:
        raise RuntimeError("Falha: diesel_acompanhamentos retornou sem id.")

    # 2) cria EVENTO LANCAMENTO
    evento = {
        "acompanhamento_id": ordem_id,
        "tipo": "LANCAMENTO",
        "observacoes": f"Ordem gerada automaticamente (lote {lote_id}). Foco: {dados['foco']}",
        "evidencias_urls": [u for u in [pdf_url, html_url] if u],
        "periodo_inicio": dados["periodo_inicio"],
        "periodo_fim": dados["periodo_fim"],
        "kml": dados["totais"]["kml_real"],
        "extra": {
            "kml_meta_ref": dados["totais"]["kml_meta"],
            "desperdicio_litros_30d": dados["totais"]["desp"],
            "cluster_foco": dados["foco_cluster"],
            "linha_foco": dados["linha_foco"],
            "veiculo_foco": dados["veiculo_foco"],
        },
    }
    sb_b.table(TABELA_EVENTOS).insert(evento).execute()

    return ordem_id


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    if not ORDEM_BATCH_ID:
        print("❌ ORDEM_BATCH_ID não definido no ambiente.")
        return

    ok = 0
    erros = 0
    erros_list = []

    atualizar_status_lote("PROCESSANDO", extra={"started_at": datetime.utcnow().isoformat()})

    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    mapa_nomes = carregar_mapa_nomes()
    itens = obter_motoristas_do_lote()
    if not itens:
        atualizar_status_lote("ERRO", "Lote sem itens")
        return

    sb_b = _sb_b()

    print(f"🚀 Iniciando geração de {len(itens)} ordens (diesel_acompanhamentos)...")

    for item in itens:
        mot_chapa = str(item.get("motorista_chapa") or "").strip()
        if not mot_chapa:
            continue

        try:
            df_full = carregar_historico_motorista(mot_chapa)
            if df_full.empty:
                print(f"⚠️ Sem dados para {mot_chapa}")
                continue

            info_nome = mapa_nomes.get(mot_chapa, {"nome": mot_chapa, "cargo": "MOTORISTA"})
            dados = processar_dados_prontuario(df_full, mot_chapa, info_nome)
            if not dados:
                print(f"⚠️ Sem janela válida para {mot_chapa}")
                continue

            print(f"   ⚙️ Gerando PDF/HTML para {mot_chapa}...")
            safe = _safe_filename(mot_chapa)
            p_html = PASTA_SAIDA / f"{safe}.html"
            p_pdf = PASTA_SAIDA / f"{safe}.pdf"

            txt_ia = chamar_ia_coach(dados)
            html = gerar_html_final(dados, txt_ia)
            p_html.write_text(html, encoding="utf-8")

            html_to_pdf(p_html, p_pdf)

            # upload Storage (Supabase B)
            pdf_path, pdf_url = upload_storage(p_pdf, f"{safe}.pdf", "application/pdf")
            html_path, html_url = upload_storage(p_html, f"{safe}.html", "text/html")

            # cria ordem+evento
            ordem_id = criar_ordem_e_evento(
                sb_b=sb_b,
                dados=dados,
                lote_id=ORDEM_BATCH_ID,
                pdf_path=pdf_path,
                pdf_url=pdf_url,
                html_path=html_path,
                html_url=html_url,
                txt_ia=txt_ia,
            )

            ok += 1
            print(f"✅ {mot_chapa}: Ordem criada (id={ordem_id}).")

        except Exception as e:
            erros += 1
            msg = str(e)
            erros_list.append({"motorista": mot_chapa, "erro": msg[:500]})
            print(f"❌ {mot_chapa}: erro: {msg}")

    # status final do lote
    finished = {"ok": ok, "erros": erros, "erros_list": erros_list, "finished_at": datetime.utcnow().isoformat()}
    if erros == 0 and ok > 0:
        atualizar_status_lote("CONCLUIDO", extra=finished)
    elif ok == 0 and erros > 0:
        atualizar_status_lote("ERRO", msg=f"OK={ok} | ERROS={erros}", extra=finished)
    else:
        atualizar_status_lote("CONCLUIDO_COM_ERROS", msg=f"OK={ok} | ERROS={erros}", extra=finished)

    print(f"🏁 Finalizado. OK={ok} | ERROS={erros}")


if __name__ == "__main__":
    main()
