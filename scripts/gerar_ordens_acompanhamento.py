import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import vertexai
from vertexai.generative_models import GenerativeModel

from supabase import create_client
from playwright.sync_api import sync_playwright

# ==============================================================================
# 1. CONFIGURAÇÃO E ENV
# ==============================================================================
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
VERTEX_MODEL = os.getenv("VERTEX_MODEL", "gemini-2.5-pro")

SUPABASE_A_URL = os.getenv("SUPABASE_A_URL")
SUPABASE_A_SERVICE_ROLE_KEY = os.getenv("SUPABASE_A_SERVICE_ROLE_KEY")

SUPABASE_B_URL = os.getenv("SUPABASE_B_URL")
SUPABASE_B_SERVICE_ROLE_KEY = os.getenv("SUPABASE_B_SERVICE_ROLE_KEY")

ORDEM_BATCH_ID = os.getenv("ORDEM_BATCH_ID")

QTD_ACOMPANHAMENTOS = int(os.getenv("QTD", "10"))
MOTORISTA_FOCO = os.getenv("MOTORISTA_FOCO")
NO_FILTERS = os.getenv("NO_FILTERS", "0") in ("1", "true", "TRUE", "yes", "YES")

TABELA_ORIGEM = os.getenv("DIESEL_SOURCE_TABLE", "premiacao_diaria")
TABELA_LOTE = "acompanhamento_lotes"
TABELA_DESTINO = "diesel_acompanhamentos"

BUCKET = "relatorios"
REMOTE_PREFIX = "acompanhamento"
PASTA_SAIDA = Path("Ordens_Acompanhamento")

KML_MIN = float(os.getenv("KML_MIN", "1.5"))
KML_MAX = float(os.getenv("KML_MAX", "5.0"))

# --- PERFORMANCE ---
PAGE_SIZE = int(os.getenv("PAGE_SIZE", "2000"))
FETCH_DAYS = 90  # Histórico geral de 3 meses
JANELA_DIAS = 5  # Janelas pequenas de busca

RANKING_DIAS = int(os.getenv("RANKING_DIAS", "30"))
DETALHE_DIAS = int(os.getenv("DETALHE_DIAS", "30"))

# ==============================================================================
# 2. CLIENTES E HELPERS
# ==============================================================================
def safe_num(val):
    if pd.isna(val) or val is None or val == "": return 0.0
    try: return float(val)
    except: return 0.0

def _sb_a():
    if not SUPABASE_A_URL or not SUPABASE_A_SERVICE_ROLE_KEY:
        raise Exception("⚠️ Falta SUPABASE_A_URL ou KEY")
    return create_client(SUPABASE_A_URL, SUPABASE_A_SERVICE_ROLE_KEY)

def _sb_b():
    if not SUPABASE_B_URL or not SUPABASE_B_SERVICE_ROLE_KEY:
        raise Exception("⚠️ Falta SUPABASE_B_URL ou KEY")
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
        "ANALISE": [r"AN[ÁA]LISE", r"DIAGN[ÓO]STICO"],
        "ROTEIRO": [r"ROTEIRO", r"A[ÇC][ÕO]ES"],
        "FEEDBACK": [r"FEEDBACK", r"CONCLUS[ÃA]O"],
    }
    chaves = mapa.get(tag_chave, [tag_chave])
    pattern = "|".join(chaves)
    regex = rf"(?:^|\n|#|\*|[\d]+\.)\s*(?:{pattern})[:\s\-]*(.*?)(?=\n(?:AN[ÁA]LISE|ROTEIRO|PLANO|FEEDBACK)[:#\*]|$)"
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
# 3. CÁLCULOS MATEMÁTICOS
# ==============================================================================
def resumo_60d(df_hist_mot: pd.DataFrame, df_hist_linha: pd.DataFrame):
    df_hist_mot = df_hist_mot.copy()
    df_hist_linha = df_hist_linha.copy()
    dmax = df_hist_mot["Date"].max()
    if pd.isna(dmax): return {}
    
    ini = dmax - timedelta(days=59)
    mot = df_hist_mot[df_hist_mot["Date"] >= ini]
    lin = df_hist_linha[df_hist_linha["Date"] >= ini]

    km_60 = safe_num(mot["Km"].sum())
    litros_60 = safe_num(mot["Comb."].sum())
    kml_60 = (km_60 / litros_60) if litros_60 > 0 else 0.0
    
    km_lin_60 = safe_num(lin["Km"].sum())
    litros_lin_60 = safe_num(lin["Comb."].sum())
    kml_linha_60 = (km_lin_60 / litros_lin_60) if litros_lin_60 > 0 else 0.0

    return {
        "inicio": ini.strftime("%Y-%m-%d"),
        "fim": dmax.strftime("%Y-%m-%d"),
        "dias_com_dados_60": int(mot["Date"].nunique()),
        "km_total_60": km_60,
        "litros_total_60": litros_60,
        "kml_medio_60": kml_60,
        "kml_linha_medio_60": kml_linha_60,
        "gap_60": (kml_60 - kml_linha_60),
    }

def top_dias_criticos(df_hist_mot: pd.DataFrame, kml_ref_linha_60: float, topn: int = 5):
    kml_ref = safe_num(kml_ref_linha_60)
    if kml_ref <= 0: return []
    
    d = df_hist_mot.copy()
    data_max = d["Date"].max()
    if pd.isna(data_max): return []
    ini_60 = data_max - timedelta(days=59)
    d = d[d["Date"] >= ini_60]

    dia = d.groupby(["Date", "linha", "veiculo"]).agg({"Km": "sum", "Comb.": "sum"}).reset_index()
    dia["KML"] = dia["Km"] / dia["Comb."]

    def perda(row):
        comb = safe_num(row["Comb."])
        km = safe_num(row["Km"])
        if comb > 0 and row["KML"] < kml_ref:
            return comb - (km / kml_ref)
        return 0.0

    dia["litros_perdidos_estim"] = dia.apply(perda, axis=1)
    
    out = []
    for _, r in dia.sort_values("litros_perdidos_estim", ascending=False).head(topn).iterrows():
        out.append({
            "dia": r["Date"].strftime("%Y-%m-%d"),
            "linha": str(r["linha"]),
            "veiculo": str(r["veiculo"]),
            "km": safe_num(r["Km"]),
            "kml": safe_num(r["KML"]),
            "litros_perdidos_estim": safe_num(r["litros_perdidos_estim"]),
        })
    return out

def detalhamento_dias_dia_a_dia(df_hist_mot, df_hist_linha, linha_foco, cluster_foco, dias=30):
    mot = df_hist_mot.copy()
    lin = df_hist_linha.copy()
    
    data_max = mot["Date"].max()
    ini = data_max - timedelta(days=dias - 1)
    
    linw = lin[(lin["Date"] >= ini) & (lin["Date"] <= data_max)]
    if linha_foco and cluster_foco and not linw.empty:
        linw = linw[(linw["linha"].astype(str) == str(linha_foco)) & (linw["Cluster"].astype(str) == str(cluster_foco))]
    
    motw = mot[(mot["Date"] >= ini) & (mot["Date"] <= data_max)]

    m = motw.groupby(motw["Date"].dt.date).agg(
        km=("Km", "sum"), litros=("Comb.", "sum"),
        veiculos=("veiculo", lambda s: ", ".join(sorted(set(map(str, s))))[:20]),
        linhas=("linha", lambda s: ", ".join(sorted(set(map(str, s))))[:20]),
    ).reset_index().rename(columns={"Date": "Dia"})
    m["kml_mot"] = m["km"] / m["litros"]

    l = linw.groupby(linw["Date"].dt.date).agg(km=("Km", "sum"), litros=("Comb.", "sum")).reset_index().rename(columns={"Date": "Dia"})
    l["kml_lin"] = l["km"] / l["litros"]

    cal = pd.DataFrame({"Dia": pd.date_range(ini, data_max, freq="D").date})
    out = cal.merge(m, on="Dia", how="left").merge(l[["Dia", "kml_lin"]], on="Dia", how="left")
    
    out["gap"] = out.apply(lambda r: (safe_num(r["kml_mot"]) - safe_num(r["kml_lin"])) if (pd.notna(r["kml_mot"]) and pd.notna(r["kml_lin"])) else 0.0, axis=1)
    
    return out.sort_values("Dia", ascending=False).to_dict("records")

# ==============================================================================
# 4. CARREGAMENTO INTELIGENTE
# ==============================================================================
def obter_lista_motoristas():
    if MOTORISTA_FOCO:
        print(f"🎯 Modo Foco: {MOTORISTA_FOCO}")
        return [str(MOTORISTA_FOCO).strip()]

    print("📊 Ranking: Consultando Top Consumidores...")
    sb = _sb_a()
    
    # 30 dias para ranking
    data_corte = (datetime.utcnow() - timedelta(days=RANKING_DIAS)).strftime("%Y-%m-%d")

    res = sb.table(TABELA_ORIGEM)\
            .select("motorista, km_rodado, combustivel_consumido")\
            .gte("dia", data_corte)\
            .execute()
            
    df = pd.DataFrame(res.data)
    if df.empty: return []

    df["km"] = to_num(df["km_rodado"])
    df["litros"] = to_num(df["combustivel_consumido"])
    
    rank = df.groupby("motorista")[["km", "litros"]].sum()
    rank = rank.sort_values("litros", ascending=False).head(QTD_ACOMPANHAMENTOS)
    
    lista = rank.index.tolist()
    print(f"📋 Top {len(lista)} identificados: {lista}")
    return lista

def carregar_dados_do_motorista(motorista_id):
    print(f"📦 [Supabase] Detalhes: {motorista_id} (Janelas 5d)...")
    sb = _sb_a()
    
    agora = datetime.utcnow()
    limite = agora - timedelta(days=FETCH_DAYS) # 90 dias
    cursor = agora
    
    all_rows = []
    
    while cursor > limite:
        inicio_janela = cursor - timedelta(days=JANELA_DIAS) # 5 dias
        s_fim = cursor.strftime("%Y-%m-%d")
        s_ini = inicio_janela.strftime("%Y-%m-%d")
        
        start = 0
        while True:
            try:
                resp = sb.table(TABELA_ORIGEM)\
                         .select('dia, motorista, veiculo, linha, "km/l", km_rodado, combustivel_consumido')\
                         .eq("motorista", motorista_id)\
                         .gte("dia", s_ini)\
                         .lte("dia", s_fim)\
                         .order("dia", desc=True)\
                         .range(start, start + PAGE_SIZE - 1)\
                         .execute()
                         
                rows = resp.data or []
                all_rows.extend(rows)
                
                if len(rows) < PAGE_SIZE: break
                start += PAGE_SIZE
                
            except Exception as e:
                print(f"⚠️ Erro parcial: {e}")
                break
        
        cursor = inicio_janela - timedelta(days=1)

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
# 5. PROCESSAMENTO
# ==============================================================================
def preparar_base(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["Date", "Km", "Comb."])
    df = df[(df["Comb."] > 0) & (df["Km"] > 0)].copy()
    df["kml"] = df["Km"] / df["Comb."]
    df["Cluster"] = df["veiculo"].apply(get_cluster)

    if NO_FILTERS: return df

    df = df.dropna(subset=["Cluster"])
    df = df[(df["kml"] >= KML_MIN) & (df["kml"] <= KML_MAX)]
    return df

def processar_motorista(df_full, mot):
    df_mot = df_full[df_full["Motorista"] == mot]
    if df_mot.empty: return None
    
    fim_rank = df_mot["Date"].max()
    ini_rank = fim_rank - timedelta(days=RANKING_DIAS - 1)
    
    df_rank = df_mot[(df_mot["Date"] >= ini_rank) & (df_mot["Date"] <= fim_rank)].copy()
    if df_rank.empty: return None

    ref = df_rank.groupby(["linha", "Cluster"]).agg({"Km": "sum", "Comb.": "sum"}).reset_index()
    ref["Meta"] = ref["Km"] / ref["Comb."]
    
    df_rank = df_rank.merge(ref[["linha", "Cluster", "Meta"]], on=["linha", "Cluster"], how="left")
    
    def calc_perda(r):
        meta = safe_num(r["Meta"])
        real = safe_num(r["kml"])
        comb = safe_num(r["Comb."])
        if meta > 0 and real < meta:
            return comb - (safe_num(r["Km"]) / meta)
        return 0.0

    df_rank["Perda"] = df_rank.apply(calc_perda, axis=1)

    pior = df_rank.groupby(["linha", "Cluster"]).agg({"Perda":"sum", "Meta":"mean"}).reset_index().sort_values("Perda", ascending=False)
    if pior.empty: return None
    top = pior.iloc[0]
    
    df_hist_lin = df_full[(df_full["linha"] == top["linha"]) & (df_full["Cluster"] == top["Cluster"])]

    return {
        "Motorista": mot,
        "Litros_Total": safe_num(df_rank["Perda"].sum()),
        "Linha_Foco": top["linha"],
        "Cluster_Foco": str(top["Cluster"]),
        "KML_Real": safe_num(df_rank["Km"].sum() / df_rank["Comb."].sum()),
        "KML_Meta": safe_num(top["Meta"]),
        "Dados_Hist_Mot": df_mot,
        "Dados_Hist_Linha": df_hist_lin,
        "Resumo_60D": resumo_60d(df_mot, df_hist_lin),
        "Top_Dias_Criticos": top_dias_criticos(df_mot, safe_num(top["Meta"])),
        "Detalhamento_30D_DiaADia": detalhamento_dias_dia_a_dia(df_mot, df_hist_lin, top["linha"], top["Cluster"], dias=DETALHE_DIAS),
        "Periodo_Txt": f"{ini_rank.strftime('%d/%m')} a {fim_rank.strftime('%d/%m')}",
        "Veiculos_Recentes": ", ".join(sorted(df_rank["veiculo"].unique().astype(str)))
    }

# ==============================================================================
# 6. GERAÇÃO DE ASSETS (GRÁFICO NOVO 4 SEMANAS)
# ==============================================================================
def chamar_ia(dados):
    if not VERTEX_PROJECT_ID: return "ANÁLISE: IA Desativada."
    try:
        vertexai.init(project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION)
        model = GenerativeModel(VERTEX_MODEL)
        prompt = f"""
Atue como Instrutor. Analise:
Motorista: {dados['Motorista']} | Veículo: {dados['Cluster_Foco']} | Linha: {dados['Linha_Foco']}
Meta: {dados['KML_Meta']:.2f} | Real: {dados['KML_Real']:.2f} | Perda: {dados['Litros_Total']:.0f} L
Responda: ANÁLISE: [Causa] ROTEIRO: [3 ações] FEEDBACK: [Frase]
"""
        return (model.generate_content(prompt).text or "").replace("**", "")
    except: return "ANÁLISE: Erro API."

def gerar_grafico(df_mot, df_lin, path):
    # 1. Filtra últimas 4 semanas (28 dias)
    if df_mot.empty: return
    
    max_date = df_mot['Date'].max()
    min_date = max_date - timedelta(days=28)
    
    df_m = df_mot[df_mot['Date'] >= min_date].copy()
    df_l = df_lin[df_lin['Date'] >= min_date].copy()
    
    if df_m.empty: return

    # 2. Agrupa por Semana (Início da semana)
    def calc_kml_grupo(df):
        k = df['Km'].sum()
        l = df['Comb.'].sum()
        return k/l if l > 0 else 0

    # Cria coluna de semana
    df_m['Semana'] = df_m['Date'].dt.to_period('W').apply(lambda r: r.start_time)
    df_l['Semana'] = df_l['Date'].dt.to_period('W').apply(lambda r: r.start_time)

    # Calcula médias semanais
    grp_m = df_m.groupby('Semana').apply(calc_kml_grupo).reset_index(name='Realizado')
    grp_l = df_l.groupby('Semana').apply(calc_kml_grupo).reset_index(name='Referencia')

    # Junta
    df_chart = pd.merge(grp_m, grp_l, on='Semana', how='outer').fillna(0).sort_values('Semana')
    
    # 3. Plota
    dates = df_chart['Semana'].dt.strftime('%d/%m')
    
    plt.figure(figsize=(10, 4))
    plt.plot(dates, df_chart['Realizado'], marker='o', lw=2, label="Realizado", color='#2980b9')
    plt.plot(dates, df_chart['Referencia'], marker='x', ls='--', lw=2, label="Referência Linha", color='#7f8c8d')
    
    plt.title("Evolução - Últimas 4 Semanas")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def gerar_html(d, txt_ia, img_path):
    analise = extrair_bloco(txt_ia, "ANALISE")
    roteiro = extrair_bloco(txt_ia, "ROTEIRO")
    feedback = extrair_bloco(txt_ia, "FEEDBACK")
    
    rows_det = ""
    for x in d['Detalhamento_30D_DiaADia']:
        km = safe_num(x.get('km')); litros = safe_num(x.get('litros'))
        kml_m = safe_num(x.get('kml_mot')); kml_l = safe_num(x.get('kml_lin'))
        gap = safe_num(x.get('gap'))
        cor = "color:#c0392b;font-weight:bold;" if gap < -0.1 else ""
        rows_det += f"<tr><td>{x['Dia']}</td><td>{x['veiculos']}</td><td>{x['linhas']}</td><td>{km:.0f}</td><td>{litros:.0f}</td><td>{kml_m:.2f}</td><td>{kml_l:.2f}</td><td style='{cor}'>{gap:.2f}</td></tr>"

    rows_top = ""
    for x in d['Top_Dias_Criticos']:
        rows_top += f"<tr><td>{x['dia']}</td><td>{x['veiculo']}</td><td>{x['linha']}</td><td>{x['km']:.0f}</td><td>{x['kml']:.2f}</td><td style='color:#c0392b'>{x['litros_perdidos_estim']:.1f} L</td></tr>"

    return f"""
    <!DOCTYPE html><html lang="pt-BR"><head><meta charset="UTF-8">
    <style>
        body {{ font-family: sans-serif; padding: 20px; background: #f4f7f6; }}
        .box {{ background: white; padding: 20px; border-radius: 8px; border-top: 5px solid #2c3e50; }}
        h1 {{ font-size: 18px; margin: 0 0 10px 0; color: #2c3e50; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 11px; margin-bottom: 15px; }}
        th {{ background: #34495e; color: white; padding: 5px; text-align: left; }}
        td {{ padding: 5px; border-bottom: 1px solid #ddd; }}
        .kpi-box {{ display: flex; gap: 10px; margin-bottom: 15px; }}
        .kpi {{ background: #ecf0f1; padding: 10px; flex: 1; text-align: center; border-radius: 5px; }}
        .kpi b {{ display: block; font-size: 16px; color: #2c3e50; }}
        .ia {{ border-left: 4px solid #ddd; padding: 10px; margin-top: 5px; background: #fff; font-size: 12px; }}
    </style></head><body>
    <div class="box">
        <h1>RELATÓRIO: {d['Motorista']}</h1>
        <div style="font-size:11px; color:#666; margin-bottom:10px;">
            Período: {d['Periodo_Txt']} | Linha Foco: {d['Linha_Foco']} ({d['Cluster_Foco']})
        </div>
        
        <div class="kpi-box">
            <div class="kpi"><b>{d['KML_Real']:.2f}</b><span>Real</span></div>
            <div class="kpi"><b>{d['KML_Meta']:.2f}</b><span>Meta</span></div>
            <div class="kpi"><b style="color:#c0392b">{d['Litros_Total']:.0f} L</b><span>Perda</span></div>
        </div>

        <h3>1. Dia a Dia</h3>
        <table><thead><tr><th>Dia</th><th>Carros</th><th>Linhas</th><th>Km</th><th>L</th><th>KML</th><th>Ref</th><th>Gap</th></tr></thead><tbody>{rows_det}</tbody></table>

        <h3>2. Top 5 Piores Dias</h3>
        <table><thead><tr><th>Dia</th><th>Carro</th><th>Linha</th><th>Km</th><th>KML</th><th>Excesso</th></tr></thead><tbody>{rows_top}</tbody></table>

        <h3>3. Tendência (4 Semanas)</h3>
        <img src="{os.path.basename(img_path)}" style="width:100%; height:150px; object-fit:cover; border:1px solid #ccc;">

        <h3>4. IA Coach</h3>
        <div class="ia" style="border-color:#f39c12"><b>Análise:</b> {analise}</div>
        <div class="ia" style="border-color:#3498db"><b>Ação:</b> {roteiro}</div>
        <div class="ia" style="border-color:#27ae60"><b>Feedback:</b> {feedback}</div>
    </div></body></html>
    """

# ==============================================================================
# 7. MAIN
# ==============================================================================
def main():
    if not ORDEM_BATCH_ID: return
    try:
        atualizar_status_lote("PROCESSANDO")
        PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
        sb_dest = _sb_b()

        lista_mots = obter_lista_motoristas()
        if not lista_mots: return

        print(f"🚀 Iniciando {len(lista_mots)} motoristas...")

        for mot in lista_mots:
            print(f"   Processando {mot}...")
            df_full = carregar_dados_do_motorista(mot)
            if df_full.empty: continue
                
            df_clean = preparar_base(df_full)
            item = processar_motorista(df_clean, mot)
            if not item: continue
            
            safe = _safe_filename(mot)
            p_img = PASTA_SAIDA / f"{safe}.png"
            p_html = PASTA_SAIDA / f"{safe}.html"
            p_pdf = PASTA_SAIDA / f"{safe}.pdf"

            gerar_grafico(item["Dados_Hist_Mot"], item["Dados_Hist_Linha"], p_img)
            txt_ia = chamar_ia(item)
            html = gerar_html(item, txt_ia, p_img)
            
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
                sb_dest.table(TABELA_DESTINO).insert({
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
                    "metadata": {"versao": "V6_Grafico4W"}
                }).execute()

        atualizar_status_lote("CONCLUIDO")
        print("🏁 Fim!")

    except Exception as e:
        print(f"❌ Erro: {e}")
        atualizar_status_lote("ERRO", str(e))

if __name__ == "__main__":
    main()
