import os
import re
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
# --- CONFIGURAÇÕES DE API ---
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
VERTEX_MODEL = os.getenv("VERTEX_MODEL", "gemini-2.5-pro")

SUPABASE_A_URL = os.getenv("SUPABASE_A_URL")
SUPABASE_A_SERVICE_ROLE_KEY = os.getenv("SUPABASE_A_SERVICE_ROLE_KEY")

SUPABASE_B_URL = os.getenv("SUPABASE_B_URL")
SUPABASE_B_SERVICE_ROLE_KEY = os.getenv("SUPABASE_B_SERVICE_ROLE_KEY")

ORDEM_BATCH_ID = os.getenv("ORDEM_BATCH_ID")

# --- CONFIGURAÇÕES DE NEGÓCIO ---
QTD_ACOMPANHAMENTOS = int(os.getenv("QTD", "10"))
MOTORISTA_FOCO = os.getenv("MOTORISTA_FOCO") # Se preenchido, foca só nele
NO_FILTERS = os.getenv("NO_FILTERS", "0") in ("1", "true", "TRUE", "yes", "YES")

TABELA_ORIGEM = os.getenv("DIESEL_SOURCE_TABLE", "premiacao_diaria")
TABELA_LOTE = "acompanhamento_lotes"
TABELA_DESTINO = "diesel_acompanhamentos"

BUCKET = "relatorios"
REMOTE_PREFIX = "acompanhamento"
PASTA_SAIDA = Path("Ordens_Acompanhamento")

# Limites de KML (Só aplica se NO_FILTERS=False)
KML_MIN = float(os.getenv("KML_MIN", "1.5"))
KML_MAX = float(os.getenv("KML_MAX", "5.0"))

# --- CONFIGURAÇÕES TÉCNICAS (O SEGREDO DA CORREÇÃO) ---
PAGE_SIZE = int(os.getenv("PAGE_SIZE", "2000"))
FETCH_DAYS = int(os.getenv("FETCH_DAYS", "120")) # Busca 4 meses para garantir histórico
JANELA_DIAS = 20 # IMPORTANTE: Baixa de 20 em 20 dias para não travar o banco

RANKING_DIAS = int(os.getenv("RANKING_DIAS", "30"))
DETALHE_DIAS = int(os.getenv("DETALHE_DIAS", "30"))

# ==============================================================================
# 2. CLIENTES E HELPERS
# ==============================================================================
def _sb_a():
    if not SUPABASE_A_URL or not SUPABASE_A_SERVICE_ROLE_KEY:
        raise Exception("⚠️ ENV faltando: SUPABASE_A_URL / SUPABASE_A_SERVICE_ROLE_KEY")
    return create_client(SUPABASE_A_URL, SUPABASE_A_SERVICE_ROLE_KEY)

def _sb_b():
    if not SUPABASE_B_URL or not SUPABASE_B_SERVICE_ROLE_KEY:
        raise Exception("⚠️ ENV faltando: SUPABASE_B_URL / SUPABASE_B_SERVICE_ROLE_KEY")
    return create_client(SUPABASE_B_URL, SUPABASE_B_SERVICE_ROLE_KEY)

def _safe_filename(name: str) -> str:
    name = str(name or "").strip()
    name = re.sub(r"[^\w\-.() ]+", "_", name, flags=re.UNICODE)
    return name[:100] or "sem_nome"

def atualizar_status_lote(status: str, msg: str = None):
    if not ORDEM_BATCH_ID: return
    print(f"🔄 [Lote {ORDEM_BATCH_ID}] Status: {status}")
    sb = _sb_b()
    payload = {"status": status}
    if msg: payload["erro_msg"] = msg
    sb.table(TABELA_LOTE).update(payload).eq("id", ORDEM_BATCH_ID).execute()

def upload_storage(local_path: Path, remote_name: str, content_type: str) -> str:
    if not ORDEM_BATCH_ID: return None
    sb = _sb_b()
    remote_path = f"{REMOTE_PREFIX}/{ORDEM_BATCH_ID}/{remote_name}"
    if not local_path.exists(): return None
    with open(local_path, "rb") as f:
        sb.storage.from_(BUCKET).upload(
            path=remote_path, file=f,
            file_options={"content-type": content_type, "upsert": "true"},
        )
    return remote_path

def extrair_bloco(texto, tag_chave):
    if not texto: return "..."
    mapa = {
        "ANALISE": [r"AN[ÁA]LISE", r"DIAGN[ÓO]STICO", r"PROBLEMA"],
        "ROTEIRO": [r"ROTEIRO", r"PLANO", r"A[ÇC][ÕO]ES", r"O QUE FAZER"],
        "FEEDBACK": [r"FEEDBACK", r"MENSAGEM", r"GESTOR", r"CONCLUS[ÃA]O"],
    }
    chaves_possiveis = mapa.get(tag_chave, [tag_chave])
    pattern_chave = "|".join(chaves_possiveis)
    regex = rf"(?:^|\n|#|\*|[\d]+\.)\s*(?:{pattern_chave})[:\s\-]*(.*?)(?=\n(?:AN[ÁA]LISE|ROTEIRO|PLANO|FEEDBACK|RESUMO)[:#\*]|$)"
    match = re.search(regex, texto, re.IGNORECASE | re.DOTALL)
    if match: return re.sub(r"^[\*\-\s]+", "", match.group(1).strip())
    return "..."

def to_num(series: pd.Series) -> pd.Series:
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
    if v.startswith("W"): return "C6"
    return None

# ==============================================================================
# 2.1 MÉTRICAS
# ==============================================================================
def resumo_60d(df_hist_mot: pd.DataFrame, df_hist_linha: pd.DataFrame):
    df_hist_mot = df_hist_mot.copy()
    df_hist_linha = df_hist_linha.copy()
    df_hist_mot["Date"] = pd.to_datetime(df_hist_mot["Date"], errors="coerce")
    df_hist_linha["Date"] = pd.to_datetime(df_hist_linha["Date"], errors="coerce")

    dmax_m = df_hist_mot["Date"].max()
    dmax_l = df_hist_linha["Date"].max()
    data_max = max(dmax_m, dmax_l) if pd.notna(dmax_l) else dmax_m

    if pd.isna(data_max):
        return {}

    ini = (data_max.normalize() - timedelta(days=59))
    fim = data_max.normalize()

    mot = df_hist_mot[(df_hist_mot["Date"] >= ini) & (df_hist_mot["Date"] <= fim)].copy()
    lin = df_hist_linha[(df_hist_linha["Date"] >= ini) & (df_hist_linha["Date"] <= fim)].copy()

    km_60 = float(mot["Km"].sum())
    litros_60 = float(mot["Comb."].sum())
    kml_60 = (km_60 / litros_60) if litros_60 > 0 else None
    
    km_lin_60 = float(lin["Km"].sum())
    litros_lin_60 = float(lin["Comb."].sum())
    kml_linha_60 = (km_lin_60 / litros_lin_60) if litros_lin_60 > 0 else None

    gap_60 = (kml_60 - kml_linha_60) if (kml_60 is not None and kml_linha_60 is not None) else None

    return {
        "inicio": ini.strftime("%Y-%m-%d"),
        "fim": fim.strftime("%Y-%m-%d"),
        "dias_com_dados_60": int(mot["Date"].dt.date.nunique()) if not mot.empty else 0,
        "km_total_60": km_60,
        "litros_total_60": litros_60,
        "kml_medio_60": kml_60,
        "kml_linha_medio_60": kml_linha_60,
        "gap_60": gap_60,
    }

def top_dias_criticos(df_hist_mot: pd.DataFrame, kml_ref_linha_60: float, topn: int = 5):
    # CORREÇÃO: Filtra data para não mostrar Outubro
    if not kml_ref_linha_60 or kml_ref_linha_60 <= 0: return []
    
    d = df_hist_mot.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d.dropna(subset=["Date", "Km", "Comb."])
    
    # Filtro de 60 dias
    data_max = d["Date"].max()
    if pd.isna(data_max): return []
    ini_60 = data_max.normalize() - timedelta(days=59)
    d = d[d["Date"] >= ini_60].copy()

    d["Dia"] = d["Date"].dt.date
    dia = d.groupby(["Dia", "linha", "veiculo"]).agg({"Km": "sum", "Comb.": "sum"}).reset_index()
    dia["KML"] = dia["Km"] / dia["Comb."]

    def perda(row):
        if row["Comb."] > 0 and row["KML"] < kml_ref_linha_60:
            return float(row["Comb."] - (row["Km"] / kml_ref_linha_60))
        return 0.0

    dia["litros_perdidos_estim"] = dia.apply(perda, axis=1)
    dia = dia.sort_values("litros_perdidos_estim", ascending=False).head(topn)

    out = []
    for _, r in dia.iterrows():
        out.append({
            "dia": str(r["Dia"]), "linha": str(r["linha"]), "veiculo": str(r["veiculo"]),
            "km": float(r["Km"]), "kml": float(r["KML"]),
            "litros_perdidos_estim": float(r["litros_perdidos_estim"]),
        })
    return out

def detalhamento_dias_dia_a_dia(df_hist_mot, df_hist_linha, linha_foco, cluster_foco, dias=30):
    mot = df_hist_mot.copy()
    lin = df_hist_linha.copy()
    mot["Date"] = pd.to_datetime(mot["Date"], errors="coerce")
    lin["Date"] = pd.to_datetime(lin["Date"], errors="coerce")
    
    mot = mot.dropna(subset=["Date", "Km", "Comb."])
    if mot.empty: return []

    data_max = mot["Date"].max().normalize()
    ini = (data_max - timedelta(days=dias - 1)).normalize()
    fim = data_max.normalize()

    motw = mot[(mot["Date"] >= ini) & (mot["Date"] <= fim)].copy()
    linw = lin[(lin["Date"] >= ini) & (lin["Date"] <= fim)].copy()

    if linha_foco and cluster_foco and not linw.empty:
        linw = linw[(linw["linha"].astype(str) == str(linha_foco)) & (linw["Cluster"].astype(str) == str(cluster_foco))]

    motw["Dia"] = motw["Date"].dt.date
    linw["Dia"] = linw["Date"].dt.date

    m = motw.groupby("Dia", dropna=False).agg(
        km=("Km", "sum"), litros=("Comb.", "sum"),
        veiculos=("veiculo", lambda s: ", ".join(sorted(set(map(str, s))))[:160]),
        linhas=("linha", lambda s: ", ".join(sorted(set(map(str, s))))[:120]),
    ).reset_index()
    m["kml_motorista"] = m.apply(lambda r: (r["km"]/r["litros"]) if r["litros"]>0 else None, axis=1)

    l = linw.groupby("Dia", dropna=False).agg(km=("Km", "sum"), litros=("Comb.", "sum")).reset_index()
    l["kml_linha"] = l.apply(lambda r: (r["km"]/r["litros"]) if r["litros"]>0 else None, axis=1)

    cal = pd.DataFrame({"Dia": pd.date_range(ini, fim, freq="D").date})
    out = cal.merge(m, on="Dia", how="left").merge(l[["Dia", "kml_linha"]], on="Dia", how="left")
    out["gap_dia"] = out.apply(lambda r: (r["kml_motorista"]-r["kml_linha"]) if pd.notna(r["kml_motorista"]) and pd.notna(r["kml_linha"]) else None, axis=1)
    
    detalhes = []
    for _, r in out.sort_values("Dia", ascending=False).iterrows():
        detalhes.append({
            "dia": str(r["Dia"]), "veiculos": str(r.get("veiculos") or ""), "linhas": str(r.get("linhas") or ""),
            "km": float(r["km"]) if pd.notna(r.get("km")) else None,
            "litros": float(r["litros"]) if pd.notna(r.get("litros")) else None,
            "kml_motorista": float(r["kml_motorista"]) if pd.notna(r.get("kml_motorista")) else None,
            "kml_linha": float(r["kml_linha"]) if pd.notna(r.get("kml_linha")) else None,
            "gap_dia": float(r["gap_dia"]) if pd.notna(r.get("gap_dia")) else None,
        })
    return detalhes

# ==============================================================================
# 3. CARREGAMENTO (A CORREÇÃO DO BURACO)
# ==============================================================================
def carregar_dados():
    """
    Usa Janelas de Tempo (Loop) para evitar travamento em paginação profunda.
    """
    print("📦 [Supabase A] Buscando histórico (Modo Janelas)...")
    sb = _sb_a()
    
    data_final_global = datetime.utcnow()
    data_limite_global = data_final_global - timedelta(days=FETCH_DAYS)
    all_rows = []
    cursor_data = data_final_global
    
    # Loop Principal: Volta no tempo em fatias de JANELA_DIAS
    while cursor_data > data_limite_global:
        inicio_janela = cursor_data - timedelta(days=JANELA_DIAS)
        s_fim = cursor_data.strftime("%Y-%m-%d")
        s_ini = inicio_janela.strftime("%Y-%m-%d")
        
        print(f"   📅 Buscando janela: {s_ini} até {s_fim}...")
        start = 0
        sel = 'dia, motorista, veiculo, linha, "km/l", km_rodado, combustivel_consumido'
        
        while True:
            try:
                # Se tiver motorista foco, filtra no banco (mais rápido)
                query = sb.table(TABELA_ORIGEM).select(sel).gte("dia", s_ini).lte("dia", s_fim)
                if MOTORISTA_FOCO:
                    query = query.eq("motorista", MOTORISTA_FOCO)
                
                resp = query.order("dia", desc=True).range(start, start + PAGE_SIZE - 1).execute()
                rows = resp.data or []
                all_rows.extend(rows)
                
                if len(rows) < PAGE_SIZE: break
                start += PAGE_SIZE
                print(f"      -> Baixados +{len(rows)}...")
            except Exception as e:
                print(f"⚠️ Erro janela {s_ini}: {e}")
                break
        
        cursor_data = inicio_janela - timedelta(days=1)

    print(f"📦 Total de registros: {len(all_rows)}")
    if not all_rows: return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df.rename(columns={
        "dia": "Date", "motorista": "Motorista", "veiculo": "veiculo",
        "linha": "linha", "km/l": "kml_db", "km_rodado": "Km", "combustivel_consumido": "Comb."
    }, inplace=True)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Km"] = to_num(df["Km"])
    df["Comb."] = to_num(df["Comb."])
    return df

# ==============================================================================
# 4. PROCESSAMENTO
# ==============================================================================
def preparar_base(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["Date", "Motorista", "veiculo", "linha", "Km", "Comb."])
    df = df[(df["Comb."] > 0) & (df["Km"] > 0)].copy()
    df["kml"] = df["Km"] / df["Comb."]
    df["Cluster"] = df["veiculo"].astype(str).apply(get_cluster)

    if NO_FILTERS: return df

    # Filtros de Qualidade
    df = df.dropna(subset=["Cluster"]).copy()
    df = df[(df["kml"] >= KML_MIN) & (df["kml"] <= KML_MAX)].copy()
    return df

def processar_item(df_full, mot):
    # Pega data mais recente DO MOTORISTA para ancorar
    df_mot = df_full[df_full["Motorista"] == mot]
    if df_mot.empty: return None
    
    fim_rank = df_mot["Date"].max().normalize()
    ini_rank = (fim_rank - timedelta(days=RANKING_DIAS - 1)).normalize()
    
    # Recorte do Ranking
    df_rank = df_mot[(df_mot["Date"] >= ini_rank) & (df_mot["Date"] <= fim_rank)].copy()
    if df_rank.empty: return None

    # Meta e Linha Foco
    if ("Cluster" in df_full.columns) and (not df_rank["Cluster"].isna().all()):
        ref = df_rank.groupby(["linha", "Cluster"]).agg({"Km": "sum", "Comb.": "sum"}).reset_index()
        ref["Meta"] = ref["Km"] / ref["Comb."]
        df_rank = df_rank.merge(ref[["linha", "Cluster", "Meta"]], on=["linha", "Cluster"], how="left")
        df_rank["Perda"] = df_rank.apply(lambda r: (r["Comb."] - (r["Km"]/r["Meta"])) if (r["Meta"]>0 and r["kml"]<r["Meta"]) else 0.0, axis=1)
    else:
        df_rank["Meta"] = None; df_rank["Perda"] = 0.0

    # Acha pior linha
    pior = df_rank.groupby(["linha", "Cluster"], dropna=False).agg({"Perda":"sum", "Meta":"mean"}).reset_index().sort_values("Perda", ascending=False)
    if pior.empty: return None
    top = pior.iloc[0]
    
    # Históricos
    df_hist_lin = df_full[(df_full["linha"] == top["linha"])].copy()
    if top.get("Cluster"): df_hist_lin = df_hist_lin[df_hist_lin["Cluster"]==top["Cluster"]]

    # Métricas Finais
    res60 = resumo_60d(df_mot, df_hist_lin)
    top5 = top_dias_criticos(df_mot, res60.get("kml_linha_medio_60"))
    det = detalhamento_dias_dia_a_dia(df_mot, df_hist_lin, top["linha"], top.get("Cluster"), dias=DETALHE_DIAS)

    return {
        "Motorista": mot,
        "Litros_Total": float(df_rank["Perda"].sum()),
        "Linha_Foco": top["linha"],
        "Cluster_Foco": str(top.get("Cluster") or ""),
        "KML_Real": float(df_rank["Km"].sum() / df_rank["Comb."].sum()) if df_rank["Comb."].sum() > 0 else 0,
        "KML_Meta": float(top["Meta"]) if pd.notna(top["Meta"]) else 0,
        "Dados_Hist_Mot": df_mot,
        "Dados_Hist_Linha": df_hist_lin,
        "Dados_RaioX": df_rank,
        "Resumo_60D": res60,
        "Top_Dias_Criticos": top5,
        "Detalhamento_30D_DiaADia": det,
        "Periodo_Txt": f"{ini_rank.strftime('%Y-%m-%d')} a {fim_rank.strftime('%Y-%m-%d')}",
        "Veiculos_Recentes": ", ".join(sorted(df_rank["veiculo"].unique().astype(str)))
    }

# ==============================================================================
# 5. GERAÇÃO DE RELATÓRIO (VERTEX + HTML + PDF)
# ==============================================================================
def chamar_ia(dados):
    if not VERTEX_PROJECT_ID: return "ANÁLISE: IA Desativada."
    try:
        vertexai.init(project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION)
        model = GenerativeModel(VERTEX_MODEL)
        prompt = f"""
Atue como Instrutor de Motoristas. Analise este motorista:
Motorista: {dados['Motorista']} | Veículo: {dados['Cluster_Foco']} | Linha: {dados['Linha_Foco']}
Meta: {dados['KML_Meta']:.2f} | Real: {dados['KML_Real']:.2f} | Perda: {dados['Litros_Total']:.0f} L

Responda neste formato estrito:
ANÁLISE: [Causa provável]
ROTEIRO: [3 ações]
FEEDBACK: [Frase de impacto]
"""
        return (model.generate_content(prompt).text or "").replace("**", "")
    except: return "ANÁLISE: Erro API."

def gerar_html(d, txt_ia, img_path, tbl_html):
    analise = extrair_bloco(txt_ia, "ANALISE")
    roteiro = extrair_bloco(txt_ia, "ROTEIRO")
    feedback = extrair_bloco(txt_ia, "FEEDBACK")
    
    # Monta tabelas HTML
    rows_det = ""
    for x in d['Detalhamento_30D_DiaADia']:
        cor_gap = "color:#c0392b;font-weight:bold;" if (x['gap_dia'] or 0) < 0 else ""
        rows_det += f"<tr><td>{x['dia']}</td><td>{x['veiculos']}</td><td>{x['linhas']}</td><td>{x['km']:.0f}</td><td>{x['litros']:.0f}</td><td>{x['kml_motorista']:.2f}</td><td>{x['kml_linha']:.2f}</td><td style='{cor_gap}'>{x['gap_dia']:.2f}</td></tr>"

    rows_top = ""
    for x in d['Top_Dias_Criticos']:
        rows_top += f"<tr><td>{x['dia']}</td><td>{x['veiculo']}</td><td>{x['linha']}</td><td>{x['km']:.0f}</td><td>{x['kml']:.2f}</td><td style='color:#c0392b'>{x['litros_perdidos_estim']:.1f} L</td></tr>"

    return f"""
    <!DOCTYPE html><html lang="pt-BR"><head><meta charset="UTF-8">
    <style>
        body {{ font-family: sans-serif; padding: 20px; background: #f4f7f6; }}
        .box {{ background: white; padding: 20px; border-radius: 8px; border-top: 5px solid #2c3e50; }}
        h1 {{ font-size: 18px; margin: 0 0 10px 0; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 11px; margin-bottom: 15px; }}
        th {{ background: #34495e; color: white; padding: 5px; text-align: left; }}
        td {{ padding: 5px; border-bottom: 1px solid #ddd; }}
        .kpi-box {{ display: flex; gap: 10px; margin-bottom: 15px; }}
        .kpi {{ background: #ecf0f1; padding: 10px; flex: 1; text-align: center; border-radius: 5px; }}
        .kpi b {{ display: block; font-size: 16px; color: #2c3e50; }}
        .ia {{ border-left: 4px solid #ddd; padding: 10px; margin-top: 5px; background: #fff; font-size: 12px; }}
    </style></head><body>
    <div class="box">
        <h1>RELATÓRIO DE PERFORMANCE: {d['Motorista']}</h1>
        <div style="font-size:11px; color:#666; margin-bottom:10px;">Período Ranking: {d['Periodo_Txt']} | Linha Foco: {d['Linha_Foco']}</div>
        
        <div class="kpi-box">
            <div class="kpi"><b>{d['KML_Real']:.2f}</b><span>Real</span></div>
            <div class="kpi"><b>{d['KML_Meta']:.2f}</b><span>Meta</span></div>
            <div class="kpi"><b style="color:#c0392b">{d['Litros_Total']:.0f} L</b><span>Perda</span></div>
        </div>

        <h3>1. Detalhamento (30 Dias)</h3>
        <table><thead><tr><th>Dia</th><th>Veículo</th><th>Linha</th><th>Km</th><th>L</th><th>KML</th><th>Ref</th><th>Gap</th></tr></thead><tbody>{rows_det}</tbody></table>

        <h3>2. Top Dias Críticos (60D)</h3>
        <table><thead><tr><th>Dia</th><th>Veículo</th><th>Linha</th><th>Km</th><th>KML</th><th>Excesso</th></tr></thead><tbody>{rows_top}</tbody></table>

        <h3>3. Evolução</h3>
        <img src="{os.path.basename(img_path)}" style="width:100%; height:150px; object-fit:cover; border:1px solid #ccc;">

        <h3>4. Diagnóstico IA</h3>
        <div class="ia" style="border-color:#f39c12"><b>Análise:</b> {analise}</div>
        <div class="ia" style="border-color:#3498db"><b>Ação:</b> {roteiro}</div>
        <div class="ia" style="border-color:#27ae60"><b>Feedback:</b> {feedback}</div>
    </div></body></html>
    """

def gerar_grafico(df_mot, df_lin, path):
    plt.figure(figsize=(10, 3))
    if not df_mot.empty:
        g = df_mot.groupby("Date")["kml"].mean()
        plt.plot(g.index, g.values, marker="o", label="Mot")
    if not df_lin.empty:
        l = df_lin.groupby("Date")["kml"].mean()
        plt.plot(l.index, l.values, linestyle="--", alpha=0.5, label="Linha")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def gerar_tabela_raiox(df):
    # Gera HTML simples para a tabela de Raio-X se precisar (opcional no layout final)
    return "" 

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    if not ORDEM_BATCH_ID: return print("❌ Sem Batch ID")
    try:
        atualizar_status_lote("PROCESSANDO")
        PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

        df_full = carregar_dados()
        df_clean = preparar_base(df_full)

        # Seleção de Motoristas
        if MOTORISTA_FOCO:
            lista_mots = [MOTORISTA_FOCO]
        else:
            # Lógica simples de ranking se não tiver foco
            ranking = df_clean.groupby("Motorista")["Km"].sum().sort_values(ascending=False).head(QTD_ACOMPANHAMENTOS)
            lista_mots = ranking.index.tolist()

        print(f"🎯 Processando {len(lista_mots)} motoristas...")
        
        sb = _sb_b()
        
        for mot in lista_mots:
            print(f"   > {mot}...")
            item = processar_item(df_clean, mot)
            if not item: continue
            
            safe = _safe_filename(mot)
            p_img = PASTA_SAIDA / f"{safe}.png"
            p_html = PASTA_SAIDA / f"{safe}.html"
            p_pdf = PASTA_SAIDA / f"{safe}.pdf"

            gerar_grafico(item["Dados_Hist_Mot"], item["Dados_Hist_Linha"], p_img)
            txt_ia = chamar_ia(item)
            html = gerar_html(item, txt_ia, p_img, "")
            
            with open(p_html, "w", encoding="utf-8") as f: f.write(html)
            
            with sync_playwright() as p:
                browser = p.chromium.launch(args=["--no-sandbox"])
                page = browser.new_page()
                page.goto(p_html.resolve().as_uri())
                page.pdf(path=str(p_pdf), format="A4")
                browser.close()

            url_pdf = upload_storage(p_pdf, f"{safe}.pdf", "application/pdf")
            url_html = upload_storage(p_html, f"{safe}.html", "text/html")
            
            if url_pdf:
                sb.table(TABELA_DESTINO).insert({
                    "lote_id": ORDEM_BATCH_ID,
                    "motorista_nome": mot,
                    "motorista_chapa": mot,
                    "veiculo_foco": item["Cluster_Foco"],
                    "linha_foco": item["Linha_Foco"],
                    "kml_real": item["KML_Real"],
                    "kml_meta": item["KML_Meta"],
                    "perda_litros": item["Litros_Total"],
                    "arquivo_pdf_path": url_pdf,
                    "arquivo_html_path": url_html,
                    "status": "CONCLUIDO",
                    "metadata": {"debug_info": "Versão com Deep Pagination Fix"}
                }).execute()

        atualizar_status_lote("CONCLUIDO")
        print("✅ Sucesso Absoluto!")

    except Exception as e:
        print(f"❌ Erro: {e}")
        atualizar_status_lote("ERRO", str(e))

if __name__ == "__main__":
    main()
