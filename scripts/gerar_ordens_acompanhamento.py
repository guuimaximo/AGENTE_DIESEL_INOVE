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
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID")  # opcional (pode ficar vazio)
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
VERTEX_MODEL = os.getenv("VERTEX_MODEL", "gemini-2.5-pro")

SUPABASE_A_URL = os.getenv("SUPABASE_A_URL")
SUPABASE_A_SERVICE_ROLE_KEY = os.getenv("SUPABASE_A_SERVICE_ROLE_KEY")

SUPABASE_B_URL = os.getenv("SUPABASE_B_URL")
SUPABASE_B_SERVICE_ROLE_KEY = os.getenv("SUPABASE_B_SERVICE_ROLE_KEY")

ORDEM_BATCH_ID = os.getenv("ORDEM_BATCH_ID")
QTD_ACOMPANHAMENTOS = int(os.getenv("QTD", "10"))

TABELA_ORIGEM = os.getenv("DIESEL_SOURCE_TABLE", "premiacao_diaria")
TABELA_LOTE = "acompanhamento_lotes"
TABELA_DESTINO = "diesel_acompanhamentos"

BUCKET = "relatorios"
REMOTE_PREFIX = "acompanhamento"

PASTA_SAIDA = Path("Ordens_Acompanhamento")

# Regra oficial de limpeza (pedido seu)
KML_MIN = float(os.getenv("KML_MIN", "1.5"))
KML_MAX = float(os.getenv("KML_MAX", "5.0"))

PAGE_SIZE = int(os.getenv("PAGE_SIZE", "2000"))

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
    if not ORDEM_BATCH_ID:
        return
    print(f"🔄 [Lote {ORDEM_BATCH_ID}] Status: {status}")
    sb = _sb_b()
    payload = {"status": status}
    if msg:
        payload["erro_msg"] = msg
    sb.table(TABELA_LOTE).update(payload).eq("id", ORDEM_BATCH_ID).execute()

def upload_storage(local_path: Path, remote_name: str, content_type: str) -> str:
    if not ORDEM_BATCH_ID:
        return None
    sb = _sb_b()
    remote_path = f"{REMOTE_PREFIX}/{ORDEM_BATCH_ID}/{remote_name}"
    if not local_path.exists():
        return None
    with open(local_path, "rb") as f:
        sb.storage.from_(BUCKET).upload(
            path=remote_path,
            file=f,
            file_options={"content-type": content_type, "upsert": "true"},
        )
    return remote_path

def extrair_bloco(texto, tag_chave):
    """
    Extrai texto entre tags de forma robusta (aceita markdown, maiúsculas, etc).
    """
    if not texto:
        return "..."

    mapa = {
        "ANALISE": [r"AN[ÁA]LISE", r"DIAGN[ÓO]STICO", r"PROBLEMA"],
        "ROTEIRO": [r"ROTEIRO", r"PLANO", r"A[ÇC][ÕO]ES", r"O QUE FAZER"],
        "FEEDBACK": [r"FEEDBACK", r"MENSAGEM", r"GESTOR", r"CONCLUS[ÃA]O"],
    }
    chaves_possiveis = mapa.get(tag_chave, [tag_chave])
    pattern_chave = "|".join(chaves_possiveis)

    regex = rf"(?:^|\n|#|\*|[\d]+\.)\s*(?:{pattern_chave})[:\s\-]*(.*?)(?=\n(?:AN[ÁA]LISE|ROTEIRO|PLANO|FEEDBACK|RESUMO)[:#\*]|$)"
    match = re.search(regex, texto, re.IGNORECASE | re.DOTALL)
    if match:
        conteudo = match.group(1).strip()
        return re.sub(r"^[\*\-\s]+", "", conteudo)

    return "..."

# ==============================================================================
# 2.1 MÉTRICAS (60D + DETALHE 30D DIA A DIA)  — ENRIQUECIMENTO DA ORDEM
# ==============================================================================
def _kml_from(df_):
    km = float(df_["Km"].sum())
    comb = float(df_["Comb."].sum())
    if comb <= 0:
        return None
    return km / comb

def resumo_60d(df_hist_mot: pd.DataFrame, df_hist_linha: pd.DataFrame):
    """
    Retorna métricas de 60 dias (motorista e benchmark da linha) + tendência 14d.
    Janela é ANCORADA NO ÚLTIMO DIA REAL DO MOTORISTA/linha (max Date), não em utcnow().
    """
    df_hist_mot = df_hist_mot.copy()
    df_hist_linha = df_hist_linha.copy()

    df_hist_mot["Date"] = pd.to_datetime(df_hist_mot["Date"], errors="coerce")
    df_hist_linha["Date"] = pd.to_datetime(df_hist_linha["Date"], errors="coerce")

    dmax_m = df_hist_mot["Date"].max()
    dmax_l = df_hist_linha["Date"].max()
    data_max = max(dmax_m, dmax_l) if pd.notna(dmax_l) else dmax_m

    if pd.isna(data_max):
        return {
            "inicio": None, "fim": None,
            "dias_com_dados_60": 0,
            "km_total_60": 0.0, "litros_total_60": 0.0,
            "kml_medio_60": None,
            "kml_linha_medio_60": None,
            "gap_60": None,
            "kml_ult_14d": None,
            "kml_14d_ant": None,
            "delta_14d": None,
        }

    ini = (data_max.normalize() - timedelta(days=59))
    fim = data_max.normalize()

    mot = df_hist_mot[(df_hist_mot["Date"] >= ini) & (df_hist_mot["Date"] <= fim)].copy()
    lin = df_hist_linha[(df_hist_linha["Date"] >= ini) & (df_hist_linha["Date"] <= fim)].copy()

    km_60 = float(mot["Km"].sum())
    litros_60 = float(mot["Comb."].sum())
    kml_60 = (km_60 / litros_60) if litros_60 > 0 else None
    dias_op_60 = int(mot["Date"].dt.date.nunique()) if not mot.empty else 0

    km_lin_60 = float(lin["Km"].sum())
    litros_lin_60 = float(lin["Comb."].sum())
    kml_linha_60 = (km_lin_60 / litros_lin_60) if litros_lin_60 > 0 else None

    gap_60 = (kml_60 - kml_linha_60) if (kml_60 is not None and kml_linha_60 is not None) else None

    # tendência 14d vs 14d anteriores (motorista)
    d0 = fim
    ini14 = d0 - timedelta(days=13)
    ini14_prev = ini14 - timedelta(days=14)
    fim14_prev = ini14 - timedelta(days=1)

    mot_14 = mot[(mot["Date"] >= ini14) & (mot["Date"] <= d0)]
    mot_14_prev = mot[(mot["Date"] >= ini14_prev) & (mot["Date"] <= fim14_prev)]

    kml_14 = _kml_from(mot_14) if len(mot_14) else None
    kml_14_prev = _kml_from(mot_14_prev) if len(mot_14_prev) else None
    delta_14 = (kml_14 - kml_14_prev) if (kml_14 is not None and kml_14_prev is not None) else None

    return {
        "inicio": ini.strftime("%Y-%m-%d"),
        "fim": fim.strftime("%Y-%m-%d"),
        "dias_com_dados_60": dias_op_60,
        "km_total_60": km_60,
        "litros_total_60": litros_60,
        "kml_medio_60": kml_60,
        "kml_linha_medio_60": kml_linha_60,
        "gap_60": gap_60,
        "kml_ult_14d": kml_14,
        "kml_14d_ant": kml_14_prev,
        "delta_14d": delta_14,
    }

def top_dias_criticos(df_hist_mot: pd.DataFrame, kml_ref_linha_60: float, topn: int = 5):
    """
    Top dias com maior litros excedentes estimados vs benchmark 60D da linha.
    """
    if not kml_ref_linha_60 or kml_ref_linha_60 <= 0:
        return []

    d = df_hist_mot.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d.dropna(subset=["Date", "Km", "Comb."])

    d["Dia"] = d["Date"].dt.date

    dia = (
        d.groupby(["Dia", "linha", "veiculo"])
        .agg({"Km": "sum", "Comb.": "sum"})
        .reset_index()
    )
    dia["KML"] = dia["Km"] / dia["Comb."]

    def perda(row):
        if row["Comb."] and row["Comb."] > 0 and row["KML"] < kml_ref_linha_60:
            comb_ideal = row["Km"] / kml_ref_linha_60
            return float(row["Comb."] - comb_ideal)
        return 0.0

    dia["litros_perdidos_estim"] = dia.apply(perda, axis=1)
    dia = dia.sort_values("litros_perdidos_estim", ascending=False).head(topn)

    out = []
    for _, r in dia.iterrows():
        out.append(
            {
                "dia": str(r["Dia"]),
                "linha": str(r["linha"]),
                "veiculo": str(r["veiculo"]),
                "km": float(r["Km"]),
                "kml": float(r["KML"]) if pd.notna(r["KML"]) else None,
                "litros_perdidos_estim": float(r["litros_perdidos_estim"]),
            }
        )
    return out

def detalhamento_30d_dia_a_dia(df_hist_mot: pd.DataFrame, df_hist_linha: pd.DataFrame, linha_foco: str, cluster_foco: str):
    """
    Retorna lista (últimos 30 dias) com dia a dia do motorista e benchmark (linha+cluster foco).
    Janela é ancorada no último dia real do MOTORISTA.
    """
    mot = df_hist_mot.copy()
    lin = df_hist_linha.copy()

    mot["Date"] = pd.to_datetime(mot["Date"], errors="coerce")
    lin["Date"] = pd.to_datetime(lin["Date"], errors="coerce")

    mot = mot.dropna(subset=["Date", "Km", "Comb.", "kml"])
    lin = lin.dropna(subset=["Date", "Km", "Comb.", "kml"])

    if mot.empty:
        return []

    data_max = mot["Date"].max().normalize()
    ini30 = (data_max - timedelta(days=29)).normalize()
    fim30 = data_max.normalize()

    mot30 = mot[(mot["Date"] >= ini30) & (mot["Date"] <= fim30)].copy()
    lin30 = lin[(lin["Date"] >= ini30) & (lin["Date"] <= fim30)].copy()

    # garante benchmark é da linha+cluster foco
    lin30 = lin30[(lin30["linha"].astype(str) == str(linha_foco)) & (lin30["Cluster"].astype(str) == str(cluster_foco))].copy()

    mot30["Dia"] = mot30["Date"].dt.date
    lin30["Dia"] = lin30["Date"].dt.date

    # motorista por dia
    m = (
        mot30.groupby("Dia", dropna=False)
        .agg(
            km=("Km", "sum"),
            litros=("Comb.", "sum"),
            veiculos=("veiculo", lambda s: ", ".join(sorted(set(map(str, s))))[:160]),
            linhas=("linha", lambda s: ", ".join(sorted(set(map(str, s))))[:120]),
        )
        .reset_index()
    )
    m["kml_motorista"] = m.apply(lambda r: (r["km"] / r["litros"]) if r["litros"] and r["litros"] > 0 else None, axis=1)

    # benchmark linha+cluster por dia
    l = (
        lin30.groupby("Dia", dropna=False)
        .agg(km=("Km", "sum"), litros=("Comb.", "sum"))
        .reset_index()
    )
    l["kml_linha"] = l.apply(lambda r: (r["km"] / r["litros"]) if r["litros"] and r["litros"] > 0 else None, axis=1)

    out = m.merge(l[["Dia", "kml_linha"]], on="Dia", how="left")
    out["gap_dia"] = out.apply(
        lambda r: (r["kml_motorista"] - r["kml_linha"]) if (r["kml_motorista"] is not None and r["kml_linha"] is not None) else None,
        axis=1
    )

    out = out.sort_values("Dia", ascending=False)

    detalhes = []
    for _, r in out.iterrows():
        detalhes.append(
            {
                "dia": str(r["Dia"]),
                "veiculos": str(r.get("veiculos") or ""),
                "linhas": str(r.get("linhas") or ""),
                "km": float(r["km"]) if pd.notna(r["km"]) else 0.0,
                "litros": float(r["litros"]) if pd.notna(r["litros"]) else 0.0,
                "kml_motorista": float(r["kml_motorista"]) if pd.notna(r["kml_motorista"]) else None,
                "kml_linha": float(r["kml_linha"]) if pd.notna(r["kml_linha"]) else None,
                "gap_dia": float(r["gap_dia"]) if pd.notna(r["gap_dia"]) else None,
            }
        )
    return detalhes

# ==============================================================================
# 3. CARREGAMENTO DE DADOS (60 dias)
# ==============================================================================
def carregar_dados():
    """
    Busca 60 dias a partir de hoje (filtro no banco) — mas a análise/relatórios são
    ANCORA DOS NO ÚLTIMO DIA REAL encontrado (data_max), pra não gerar janela errada.
    """
    print("📦 [Supabase A] Buscando histórico (>= 60 dias do filtro)...")
    sb = _sb_a()

    data_corte = (datetime.utcnow() - timedelta(days=60)).strftime("%Y-%m-%d")

    all_rows = []
    start = 0

    sel = 'dia, motorista, veiculo, linha, "km/l", km_rodado, combustivel_consumido'

    while True:
        resp = (
            sb.table(TABELA_ORIGEM)
            .select(sel)
            .gte("dia", data_corte)
            .range(start, start + PAGE_SIZE - 1)
            .execute()
        )
        rows = resp.data or []
        all_rows.extend(rows)
        if len(rows) < PAGE_SIZE:
            break
        start += PAGE_SIZE
        print(f"   -> Lendo registros: {len(all_rows)}...")

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df.rename(
        columns={
            "dia": "Date",
            "motorista": "Motorista",
            "veiculo": "veiculo",
            "linha": "linha",
            "km/l": "kml",
            "km_rodado": "Km",
            "combustivel_consumido": "Comb.",
        },
        inplace=True,
    )
    return df

# ==============================================================================
# 4. PROCESSAMENTO (METODOLOGIA V4 + ENRIQUECIMENTO 60D + DETALHE 30D)
# ==============================================================================
def processar_dados(df: pd.DataFrame):
    print("⚙️ [Core] Calculando eficiência e desperdício...")

    # Tipagem
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Mes_Ano"] = df["Date"].dt.to_period("M")

    for c in ["kml", "Km", "Comb."]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df.dropna(subset=["Date", "Motorista", "veiculo", "Km", "Comb.", "kml"], inplace=True)

    # Cluster por prefixo do veículo
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
        return None

    df["Cluster"] = df["veiculo"].apply(get_cluster)
    df = df.dropna(subset=["Cluster"])

    # ✅ REGRA FINAL: remove KML < 1,5 e > 5 (pedido)
    before = len(df)
    df = df[(df["kml"] >= KML_MIN) & (df["kml"] <= KML_MAX)].copy()
    after = len(df)
    print(f"   -> Filtro KML [{KML_MIN}, {KML_MAX}] removeu {before - after} linhas (restou {after}).")

    # Âncora de data real
    data_max = df["Date"].max()
    if pd.isna(data_max):
        return []

    mes_atual = data_max.to_period("M")

    # Período de foco (ranking)
    df_foco = df[df["Mes_Ano"] == mes_atual].copy()
    if len(df_foco) < 100:
        mes_anterior = (data_max - timedelta(days=30)).to_period("M")
        print(f"   -> Poucos dados em {mes_atual}, expandindo para {mes_anterior}...")
        df_foco = df[(df["Mes_Ano"] == mes_atual) | (df["Mes_Ano"] == mes_anterior)].copy()
        periodo_txt = f"{mes_anterior} e {mes_atual}"
    else:
        periodo_txt = str(mes_atual)

    # Meta dinâmica (média linha+cluster no período de foco)
    ref = df_foco.groupby(["linha", "Cluster"]).agg({"Km": "sum", "Comb.": "sum"}).reset_index()
    ref["KML_Meta_Linha"] = ref["Km"] / ref["Comb."]

    df_foco = df_foco.merge(ref[["linha", "Cluster", "KML_Meta_Linha"]], on=["linha", "Cluster"], how="left")

    def calc_perda(row):
        meta = row["KML_Meta_Linha"]
        real = row["kml"]
        if pd.notna(meta) and meta > 0 and real < meta:
            comb_ideal = row["Km"] / meta
            return row["Comb."] - comb_ideal
        return 0.0

    df_foco["Litros_Perdidos"] = df_foco.apply(calc_perda, axis=1)

    # Ranking
    ranking = (
        df_foco.groupby("Motorista")
        .agg({"Litros_Perdidos": "sum", "Km": "sum"})
        .reset_index()
        .sort_values("Litros_Perdidos", ascending=False)
        .head(QTD_ACOMPANHAMENTOS)
    )

    lista_final = []

    for _, r in ranking.iterrows():
        mot = r["Motorista"]
        perda_total = float(r["Litros_Perdidos"])

        df_mot = df_foco[df_foco["Motorista"] == mot]
        if df_mot.empty:
            continue

        # Pior linha (onde mais perdeu)
        pior = (
            df_mot.groupby(["linha", "Cluster"])
            .agg({"Litros_Perdidos": "sum", "Km": "sum", "Comb.": "sum", "KML_Meta_Linha": "mean"})
            .reset_index()
            .sort_values("Litros_Perdidos", ascending=False)
        )
        if pior.empty:
            continue

        top = pior.iloc[0]
        linha_foco = top["linha"]
        cluster_foco = top["Cluster"]

        # Histórico bruto (já filtrado KML) para 60D e 30D
        df_hist_mot = df[df["Motorista"] == mot].copy()
        df_hist_linha = df[(df["linha"] == linha_foco) & (df["Cluster"] == cluster_foco)].copy()

        # Veículos recentes (15d, com âncora data_max)
        d15 = data_max - timedelta(days=15)
        vecs = df_hist_mot[df_hist_mot["Date"] >= d15]["veiculo"].unique()
        vecs_str = ", ".join(sorted(map(str, vecs))) if len(vecs) > 0 else "Nenhum"

        kml_real = float(top["Km"] / top["Comb."]) if float(top["Comb."]) > 0 else 0.0
        kml_meta = float(top["KML_Meta_Linha"]) if pd.notna(top["KML_Meta_Linha"]) else 0.0

        # Enriquecimento 60D / top dias críticos / detalhamento 30D
        res60 = resumo_60d(df_hist_mot, df_hist_linha)
        top5 = top_dias_criticos(df_hist_mot, res60.get("kml_linha_medio_60"), topn=5)
        det30 = detalhamento_30d_dia_a_dia(df_hist_mot, df_hist_linha, linha_foco, cluster_foco)

        lista_final.append(
            {
                "Motorista": mot,
                "Litros_Total": perda_total,
                "Linha_Foco": linha_foco,
                "Cluster_Foco": cluster_foco,
                "KML_Real": kml_real,
                "KML_Meta": kml_meta,
                "Gap": kml_real - kml_meta,
                "Dados_RaioX": df_mot,
                "Dados_Hist_Mot": df_hist_mot,
                "Dados_Hist_Linha": df_hist_linha,
                "Veiculos_Recentes": vecs_str,
                "Periodo_Txt": periodo_txt,
                "Resumo_60D": res60,
                "Top_Dias_Criticos": top5,
                "Detalhamento_30D_DiaADia": det30,
            }
        )

    return lista_final

# ==============================================================================
# 5. IA (OPCIONAL) + ASSETS
# ==============================================================================
def chamar_vertex_ai(dados):
    # Você pediu para não focar na IA agora: se não tiver VERTEX_PROJECT_ID, fica off.
    if not VERTEX_PROJECT_ID:
        return "ANÁLISE: IA Desativada."

    try:
        vertexai.init(project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION)
        model = GenerativeModel(VERTEX_MODEL)

        tec = {
            "C11": "VW 17.230 Automático. DICA: Evitar kickdown (pé no fundo).",
            "C10": "MB 1721 Euro 6. DICA: Trocar marcha no verde.",
            "C6": "MB 1721 Manual. DICA: Não esticar marcha.",
            "C8": "Micro MB.",
        }.get(dados["Cluster_Foco"], "Ônibus Urbano.")

        prompt = f"""
Atue como Instrutor de Motoristas Sênior.
Analise este motorista com ALTO DESPERDÍCIO DE COMBUSTÍVEL.

DADOS:
- Motorista: {dados['Motorista']}
- Veículo: {dados['Cluster_Foco']} ({tec})
- Linha: {dados['Linha_Foco']}
- Carros recentes: {dados['Veiculos_Recentes']}

PERFORMANCE (período de ranking):
- Meta: {dados['KML_Meta']:.2f} km/l
- Realizado: {dados['KML_Real']:.2f} km/l
- Perda: {dados['Litros_Total']:.0f} Litros

Responda ESTRITAMENTE neste formato (sem asteriscos, sem introdução):

ANÁLISE:
[Escreva aqui a provável causa técnica]

ROTEIRO:
[Liste 3 ações práticas]

FEEDBACK:
[Uma frase de impacto profissional]
"""
        resp = model.generate_content(prompt)
        text = resp.text.replace("**", "").replace("##", "")
        return text
    except Exception as e:
        print(f"⚠️ Erro IA: {e}")
        return "ANÁLISE: Indisponível (Erro API)."

def gerar_grafico(df_mot, df_linha, caminho):
    # Agrupa por Semana (começa segunda)
    mot_w = df_mot.groupby(pd.Grouper(key="Date", freq="W-MON")).agg({"Km": "sum", "Comb.": "sum"}).reset_index()
    mot_w["KML"] = mot_w["Km"] / mot_w["Comb."]

    lin_w = df_linha.groupby(pd.Grouper(key="Date", freq="W-MON")).agg({"Km": "sum", "Comb.": "sum"}).reset_index()
    lin_w["KML"] = lin_w["Km"] / lin_w["Comb."]

    dados = pd.merge(
        mot_w,
        lin_w[["Date", "KML"]],
        on="Date",
        how="outer",
        suffixes=("_Mot", "_Linha"),
    ).sort_values("Date")

    dados = dados.dropna(subset=["KML_Mot", "KML_Linha"], how="all")
    if len(dados) == 0:
        return

    dates = dados["Date"].dt.strftime("%d/%m")

    plt.figure(figsize=(10, 4))
    plt.plot(dates, dados["KML_Mot"], marker="o", lw=3, color="#2980b9", label="Motorista")
    plt.plot(dates, dados["KML_Linha"], ls="--", lw=2, color="#c0392b", label="Média Linha (semana)")

    for x, y in zip(dates, dados["KML_Mot"]):
        if pd.notna(y):
            plt.text(x, y + 0.05, f"{y:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.title("Evolução Semanal (Últimos 60 dias)", fontsize=11, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho, dpi=110)
    plt.close()

def gerar_tabela_html(df_mot):
    df_mot = df_mot.copy()
    df_mot["Mes"] = df_mot["Mes_Ano"].astype(str)

    res = (
        df_mot.groupby(["Mes", "veiculo", "linha", "Cluster", "KML_Meta_Linha"])
        .agg(Km=("Km", "sum"), Comb=("Comb.", "sum"), Loss=("Litros_Perdidos", "sum"))
        .reset_index()
        .sort_values("Loss", ascending=False)
    )
    res["KML"] = res["Km"] / res["Comb"]

    rows = ""
    for _, r in res.iterrows():
        style = "background:#ffebee; color:#c0392b; font-weight:bold;" if r["Loss"] > 5 else ""
        rows += f"""<tr style="{style}">
            <td>{r['veiculo']}</td><td>{r['linha']}</td>
            <td style="text-align:right;">{r['Km']:.0f}</td>
            <td style="text-align:right;">{r['KML']:.2f}</td>
            <td style="text-align:right;">{r['KML_Meta_Linha']:.2f}</td>
            <td style="text-align:right;">{r['Loss']:.1f} L</td></tr>"""
    return rows

def gerar_html_final(dados, texto_ia, img, tabela):
    analise = extrair_bloco(texto_ia, "ANALISE")
    roteiro = extrair_bloco(texto_ia, "ROTEIRO")
    feedback = extrair_bloco(texto_ia, "FEEDBACK")
    cor = "#c0392b" if dados["Gap"] < 0 else "#27ae60"

    r60 = dados.get("Resumo_60D") or {}
    top5 = dados.get("Top_Dias_Criticos") or []
    det30 = dados.get("Detalhamento_30D_DiaADia") or []

    # Bloco 60D
    b60 = f"""
      <div class="obs">
        <b>Janela 60D:</b> {r60.get('inicio','?')} → {r60.get('fim','?')}
        &nbsp;|&nbsp; <b>Dias c/ dados:</b> {r60.get('dias_com_dados_60','?')}
        &nbsp;|&nbsp; <b>KM:</b> {float(r60.get('km_total_60') or 0):.0f}
        &nbsp;|&nbsp; <b>L:</b> {float(r60.get('litros_total_60') or 0):.0f}
        &nbsp;|&nbsp; <b>KM/L 60D:</b> {float(r60.get('kml_medio_60') or 0):.2f}
        &nbsp;|&nbsp; <b>KM/L Linha 60D:</b> {float(r60.get('kml_linha_medio_60') or 0):.2f}
        &nbsp;|&nbsp; <b>Gap 60D:</b> {float(r60.get('gap_60') or 0):.2f}
      </div>
      <div class="obs">
        <b>Tendência (14D):</b>
        Últimos 14D {float(r60.get('kml_ult_14d') or 0):.2f} |
        14D anteriores {float(r60.get('kml_14d_ant') or 0):.2f} |
        Δ {float(r60.get('delta_14d') or 0):.2f}
        &nbsp;|&nbsp; <b>Regra KML:</b> [{KML_MIN}..{KML_MAX}]
      </div>
    """

    # Top dias críticos (60D)
    rows_top = ""
    for t in top5:
        rows_top += f"""
          <tr>
            <td>{t.get('dia','')}</td>
            <td>{t.get('veiculo','')}</td>
            <td>{t.get('linha','')}</td>
            <td style="text-align:right;">{float(t.get('km') or 0):.0f}</td>
            <td style="text-align:right;">{float(t.get('kml') or 0):.2f}</td>
            <td style="text-align:right; font-weight:bold; color:#c0392b;">{float(t.get('litros_perdidos_estim') or 0):.1f} L</td>
          </tr>
        """

    top_tbl = ""
    if rows_top:
        top_tbl = f"""
          <h3>1. Top dias críticos (60D)</h3>
          <table>
            <thead>
              <tr>
                <th>Dia</th><th>Carro</th><th>Linha</th>
                <th style="text-align:right;">Km</th>
                <th style="text-align:right;">KM/L</th>
                <th style="text-align:right;">Litros exced.</th>
              </tr>
            </thead>
            <tbody>{rows_top}</tbody>
          </table>
        """

    # Detalhamento 30D (dia a dia)
    rows_30 = ""
    for d in det30:
        gap = d.get("gap_dia")
        gap_txt = f"{gap:.2f}" if gap is not None else ""
        gap_style = "color:#c0392b;font-weight:bold;" if (gap is not None and gap < 0) else "color:#2c3e50;"
        rows_30 += f"""
          <tr>
            <td>{d.get('dia','')}</td>
            <td>{d.get('veiculos','')}</td>
            <td>{d.get('linhas','')}</td>
            <td style="text-align:right;">{float(d.get('km') or 0):.0f}</td>
            <td style="text-align:right;">{float(d.get('litros') or 0):.0f}</td>
            <td style="text-align:right;">{float(d.get('kml_motorista') or 0):.2f}</td>
            <td style="text-align:right;">{float(d.get('kml_linha') or 0):.2f}</td>
            <td style="text-align:right;{gap_style}">{gap_txt}</td>
          </tr>
        """

    tbl_30d = ""
    if rows_30:
        tbl_30d = f"""
          <h3>0. Detalhamento 30D (dia a dia)</h3>
          <table>
            <thead>
              <tr>
                <th>Dia</th>
                <th>Veículos</th>
                <th>Linhas (do motorista)</th>
                <th style="text-align:right;">Km</th>
                <th style="text-align:right;">Litros</th>
                <th style="text-align:right;">KM/L Mot</th>
                <th style="text-align:right;">KM/L Linha</th>
                <th style="text-align:right;">Gap</th>
              </tr>
            </thead>
            <tbody>{rows_30}</tbody>
          </table>
        """

    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head><meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; background: #f4f7f6; }}
        .box {{ background: white; padding: 22px; border-radius: 10px; border-top: 6px solid #2c3e50; }}
        h1 {{ margin: 0; color: #2c3e50; font-size: 20px; }}
        .kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 16px 0; }}
        .kpi {{ background: #ecf0f1; padding: 10px; text-align: center; border-radius: 7px; }}
        .kpi b {{ display: block; font-size: 18px; color: #2c3e50; }}
        .kpi span {{ font-size: 10px; color: #7f8c8d; text-transform: uppercase; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 11px; margin-top: 10px; }}
        th {{ background: #34495e; color: white; padding: 7px; text-align: left; }}
        td {{ padding: 7px; border-bottom: 1px solid #eee; vertical-align: top; }}
        .ia {{ background: #fff; border-left: 4px solid #ddd; padding: 10px; margin-top: 8px; font-size: 13px; }}
        .obs {{ font-size: 11px; background: #eee; padding: 8px; margin: 8px 0; border-radius: 6px; }}
        .tag {{ background:#c0392b; color:white; padding:6px 8px; border-radius:6px; font-size:11px; height:fit-content; }}
        img {{ border-radius: 6px; }}
    </style>
    </head>
    <body>
    <div class="box">
        <div style="display:flex; justify-content:space-between; gap: 10px;">
            <div>
              <h1>ORDEM DE ACOMPANHAMENTO</h1>
              <div style="font-size:12px;color:#666"><b>Motorista/Chapa:</b> {dados['Motorista']}</div>
              <div style="font-size:12px;color:#666"><b>Período Ranking:</b> {dados['Periodo_Txt']}</div>
            </div>
            <div class="tag">ALTO CUSTO</div>
        </div>

        {b60}

        <div class="kpis">
            <div class="kpi"><b>{dados['Cluster_Foco']}</b><span>Veículo Foco</span></div>
            <div class="kpi"><b>{dados['Linha_Foco']}</b><span>Linha Foco</span></div>
            <div class="kpi"><b style="color:{cor}">{dados['KML_Real']:.2f}</b><span>Real (Meta {dados['KML_Meta']:.2f})</span></div>
            <div class="kpi"><b style="color:#c0392b">{dados['Litros_Total']:.0f} L</b><span>Perda Estimada</span></div>
        </div>

        <div class="obs"><b>Veículos (últ. 15d):</b> {dados['Veiculos_Recentes']}</div>

        {tbl_30d}
        {top_tbl}

        <h3>2. Evolução (60D)</h3>
        <img src="{os.path.basename(img)}" style="width:100%; border:1px solid #ddd; margin-bottom:10px;">

        <h3>3. Raio-X da Perda (período de ranking)</h3>
        <table>
          <thead>
            <tr>
              <th>Carro</th><th>Linha</th>
              <th style="text-align:right;">Km</th>
              <th style="text-align:right;">Real</th>
              <th style="text-align:right;">Meta</th>
              <th style="text-align:right;">Perda</th>
            </tr>
          </thead>
          <tbody>{tabela}</tbody>
        </table>

        <h3>4. Diagnóstico Técnico (opcional)</h3>
        <div class="ia" style="border-color:#f39c12"><b>Análise:</b> {analise}</div>
        <div class="ia" style="border-color:#3498db"><b>Ação:</b> {roteiro}</div>
        <div class="ia" style="border-color:#27ae60"><b>Feedback:</b> {feedback}</div>
    </div>
    </body></html>
    """

def gerar_pdf(html_path, pdf_path):
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page()
        page.goto(html_path.resolve().as_uri())
        page.pdf(
            path=str(pdf_path),
            format="A4",
            margin={"top": "1cm", "right": "1cm", "bottom": "1cm", "left": "1cm"},
            print_background=True,
        )
        browser.close()

# ==============================================================================
# 6. MAIN
# ==============================================================================
def main():
    if not ORDEM_BATCH_ID:
        print("❌ Sem Batch ID (ORDEM_BATCH_ID)")
        return

    try:
        atualizar_status_lote("PROCESSANDO")
        PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

        df = carregar_dados()
        if df.empty:
            raise Exception("Base vazia (Supabase A)")

        lista = processar_dados(df)
        print(f"🎯 Gerando {len(lista)} ordens...")

        sb = _sb_b()

        for item in lista:
            mot = item["Motorista"]
            print(f"   > {mot}...")
            safe = _safe_filename(mot)

            p_img = PASTA_SAIDA / f"{safe}.png"
            p_html = PASTA_SAIDA / f"{safe}.html"
            p_pdf = PASTA_SAIDA / f"{safe}.pdf"

            gerar_grafico(item["Dados_Hist_Mot"], item["Dados_Hist_Linha"], p_img)
            tbl = gerar_tabela_html(item["Dados_RaioX"])

            # IA opcional (se quiser desligar, basta deixar VERTEX_PROJECT_ID vazio no GitHub)
            txt_ia = chamar_vertex_ai(item)

            html = gerar_html_final(item, txt_ia, p_img, tbl)
            with open(p_html, "w", encoding="utf-8") as f:
                f.write(html)

            gerar_pdf(p_html, p_pdf)

            url_pdf = upload_storage(p_pdf, f"{safe}.pdf", "application/pdf")
            url_html = upload_storage(p_html, f"{safe}.html", "text/html")

            r60 = item.get("Resumo_60D") or {}
            top5 = item.get("Top_Dias_Criticos") or []
            det30 = item.get("Detalhamento_30D_DiaADia") or []

            if url_pdf:
                sb.table(TABELA_DESTINO).insert(
                    {
                        "lote_id": ORDEM_BATCH_ID,

                        # você pediu: não colocar nome agora
                        "motorista_nome": mot,
                        "motorista_chapa": mot,

                        "motivo": "BAIXO_DESEMPENHO",
                        "veiculo_foco": item["Cluster_Foco"],
                        "linha_foco": item["Linha_Foco"],

                        "kml_real": float(item["KML_Real"]),
                        "kml_meta": float(item["KML_Meta"]),
                        "gap": float(item["Gap"]),
                        "perda_litros": float(item["Litros_Total"]),

                        "arquivo_pdf_path": url_pdf,
                        "arquivo_html_path": url_html,
                        "status": "CONCLUIDO",

                        # enriquecimento 60D (colunas que você já tinha no código)
                        "analise_inicio": r60.get("inicio"),
                        "analise_fim": r60.get("fim"),
                        "dias_com_dados_60": r60.get("dias_com_dados_60"),
                        "km_total_60": r60.get("km_total_60"),
                        "litros_total_60": r60.get("litros_total_60"),
                        "kml_medio_60": r60.get("kml_medio_60"),
                        "kml_linha_medio_60": r60.get("kml_linha_medio_60"),
                        "gap_60": r60.get("gap_60"),
                        "kml_ult_14d": r60.get("kml_ult_14d"),
                        "kml_14d_ant": r60.get("kml_14d_ant"),
                        "delta_14d": r60.get("delta_14d"),

                        # jsons (sem mexer em schema)
                        "top_dias_criticos": top5,
                        "metadata": {
                            "periodo_ranking": item.get("Periodo_Txt"),
                            "regra_kml": {"min": KML_MIN, "max": KML_MAX},
                            "analise_60d": r60,
                            "top_dias_criticos": top5,
                            "detalhamento_30d_dia_a_dia": det30,
                        },
                    }
                ).execute()

        atualizar_status_lote("CONCLUIDO")
        print("✅ Sucesso!")

    except Exception as e:
        print(f"❌ Erro: {e}")
        atualizar_status_lote("ERRO", str(e))
        raise

if __name__ == "__main__":
    main()
