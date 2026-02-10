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
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID")  # opcional
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
VERTEX_MODEL = os.getenv("VERTEX_MODEL", "gemini-2.5-pro")

SUPABASE_A_URL = os.getenv("SUPABASE_A_URL")
SUPABASE_A_SERVICE_ROLE_KEY = os.getenv("SUPABASE_A_SERVICE_ROLE_KEY")

SUPABASE_B_URL = os.getenv("SUPABASE_B_URL")
SUPABASE_B_SERVICE_ROLE_KEY = os.getenv("SUPABASE_B_SERVICE_ROLE_KEY")

ORDEM_BATCH_ID = os.getenv("ORDEM_BATCH_ID")

# Quantidade de motoristas no ranking (modo normal)
QTD_ACOMPANHAMENTOS = int(os.getenv("QTD", "10"))

# ✅ MODO FOCO (se preencher, gera SOMENTE esse motorista)
MOTORISTA_FOCO = os.getenv("MOTORISTA_FOCO")

# ✅ Desliga filtros (KML/Cluster etc) no modo foco, se quiser
NO_FILTERS = os.getenv("NO_FILTERS", "0") in ("1", "true", "TRUE", "yes", "YES")

TABELA_ORIGEM = os.getenv("DIESEL_SOURCE_TABLE", "premiacao_diaria")
TABELA_LOTE = "acompanhamento_lotes"
TABELA_DESTINO = "diesel_acompanhamentos"

BUCKET = "relatorios"
REMOTE_PREFIX = "acompanhamento"

PASTA_SAIDA = Path("Ordens_Acompanhamento")

# Regra oficial de limpeza (padrão). Se NO_FILTERS=1, a gente ignora.
KML_MIN = float(os.getenv("KML_MIN", "1.5"))
KML_MAX = float(os.getenv("KML_MAX", "5.0"))

PAGE_SIZE = int(os.getenv("PAGE_SIZE", "2000"))

# BUSCA NO BANCO (puxa com folga para não “sumir” dados após filtros)
FETCH_DAYS = int(os.getenv("FETCH_DAYS", "120"))

# JANELA DO RANKING e DETALHE (calendário)
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
# 2.1 MÉTRICAS (60D + DETALHE 30D)
# ==============================================================================
def _kml_from(df_):
    km = float(df_["Km"].sum())
    comb = float(df_["Comb."].sum())
    if comb <= 0:
        return None
    return km / comb

def resumo_60d(df_hist_mot: pd.DataFrame, df_hist_linha: pd.DataFrame):
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
            "kml_medio_60": None, "kml_linha_medio_60": None, "gap_60": None,
            "kml_ult_14d": None, "kml_14d_ant": None, "delta_14d": None,
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
        out.append({
            "dia": str(r["Dia"]),
            "linha": str(r["linha"]),
            "veiculo": str(r["veiculo"]),
            "km": float(r["Km"]),
            "kml": float(r["KML"]) if pd.notna(r["KML"]) else None,
            "litros_perdidos_estim": float(r["litros_perdidos_estim"]),
        })
    return out

def detalhamento_dias_dia_a_dia(df_hist_mot: pd.DataFrame, df_hist_linha: pd.DataFrame, linha_foco: str, cluster_foco: str, dias: int = 30):
    mot = df_hist_mot.copy()
    lin = df_hist_linha.copy()

    mot["Date"] = pd.to_datetime(mot["Date"], errors="coerce")
    lin["Date"] = pd.to_datetime(lin["Date"], errors="coerce")

    mot = mot.dropna(subset=["Date", "Km", "Comb."])
    lin = lin.dropna(subset=["Date", "Km", "Comb."])

    if mot.empty:
        return []

    data_max = mot["Date"].max().normalize()
    ini = (data_max - timedelta(days=dias - 1)).normalize()
    fim = data_max.normalize()

    motw = mot[(mot["Date"] >= ini) & (mot["Date"] <= fim)].copy()
    linw = lin[(lin["Date"] >= ini) & (lin["Date"] <= fim)].copy()

    if (linha_foco is not None) and (cluster_foco is not None) and (not linw.empty):
        linw = linw[
            (linw["linha"].astype(str) == str(linha_foco)) &
            (linw["Cluster"].astype(str) == str(cluster_foco))
        ].copy()

    motw["Dia"] = motw["Date"].dt.date
    linw["Dia"] = linw["Date"].dt.date

    m = (
        motw.groupby("Dia", dropna=False)
        .agg(
            km=("Km", "sum"),
            litros=("Comb.", "sum"),
            veiculos=("veiculo", lambda s: ", ".join(sorted(set(map(str, s))))[:160]),
            linhas=("linha", lambda s: ", ".join(sorted(set(map(str, s))))[:120]),
        )
        .reset_index()
    )
    m["kml_motorista"] = m.apply(lambda r: (r["km"] / r["litros"]) if r["litros"] and r["litros"] > 0 else None, axis=1)

    l = (
        linw.groupby("Dia", dropna=False)
        .agg(km=("Km", "sum"), litros=("Comb.", "sum"))
        .reset_index()
    )
    l["kml_linha"] = l.apply(lambda r: (r["km"] / r["litros"]) if r["litros"] and r["litros"] > 0 else None, axis=1)

    cal = pd.DataFrame({"Dia": pd.date_range(ini, fim, freq="D").date})
    out = cal.merge(m, on="Dia", how="left").merge(l[["Dia", "kml_linha"]], on="Dia", how="left")

    out["gap_dia"] = out.apply(
        lambda r: (r["kml_motorista"] - r["kml_linha"])
        if (pd.notna(r["kml_motorista"]) and pd.notna(r["kml_linha"]))
        else None,
        axis=1,
    )
    out = out.sort_values("Dia", ascending=False)

    detalhes = []
    for _, r in out.iterrows():
        detalhes.append({
            "dia": str(r["Dia"]),
            "veiculos": str(r.get("veiculos") or ""),
            "linhas": str(r.get("linhas") or ""),
            "km": float(r["km"]) if pd.notna(r.get("km")) else None,
            "litros": float(r["litros"]) if pd.notna(r.get("litros")) else None,
            "kml_motorista": float(r["kml_motorista"]) if pd.notna(r.get("kml_motorista")) else None,
            "kml_linha": float(r["kml_linha"]) if pd.notna(r.get("kml_linha")) else None,
            "gap_dia": float(r["gap_dia"]) if pd.notna(r.get("gap_dia")) else None,
        })
    return detalhes

# ==============================================================================
# 3. CARREGAMENTO DE DADOS (FIX: Janelas Temporais para evitar travamento)
# ==============================================================================
def carregar_dados():
    """
    ✅ FIX APLICADO: Busca dividida em janelas de 20 dias.
    Isso força o banco a buscar dados passados (Janeiro/Dezembro) sem travar
    no 'Deep Pagination' (offset alto) em Fevereiro.
    """
    print("📦 [Supabase A] Buscando histórico (Iteração por Janelas de Tempo)...")
    sb = _sb_a()
    
    # Configuração da janela (fatias de 20 dias evitam timeout)
    JANELA_DIAS = 20
    data_final_global = datetime.utcnow()
    data_limite_global = data_final_global - timedelta(days=FETCH_DAYS)
    
    all_rows = []
    
    # Loop Externo: Janelas de Tempo (Do presente para o passado)
    cursor_data = data_final_global
    while cursor_data > data_limite_global:
        inicio_janela = cursor_data - timedelta(days=JANELA_DIAS)
        
        s_fim = cursor_data.strftime("%Y-%m-%d")
        s_ini = inicio_janela.strftime("%Y-%m-%d")
        
        print(f"   📅 Buscando janela: {s_ini} até {s_fim}...")
        
        start = 0
        sel = 'dia, motorista, veiculo, linha, "km/l", km_rodado, combustivel_consumido'
        
        # Loop Interno: Paginação dentro da janela (Offset controlado)
        while True:
            try:
                resp = (
                    sb.table(TABELA_ORIGEM)
                    .select(sel)
                    .gte("dia", s_ini)
                    .lte("dia", s_fim) # Trava o limite superior da janela
                    .order("dia", desc=True)
                    .range(start, start + PAGE_SIZE - 1)
                    .execute()
                )
                rows = resp.data or []
                all_rows.extend(rows)
                
                # Se trouxe menos que o limite, acabou esta janela
                if len(rows) < PAGE_SIZE:
                    break
                
                start += PAGE_SIZE
                print(f"      -> Baixados +{len(rows)} registros (Offset: {start})...")
                
            except Exception as e:
                print(f"⚠️ Erro ao baixar lote {start} na janela {s_ini}: {e}")
                break

        # Move o cursor para trás (para a próxima janela antiga)
        cursor_data = inicio_janela - timedelta(days=1)

    print(f"📦 Total de registros carregados: {len(all_rows)}")

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df.rename(
        columns={
            "dia": "Date",
            "motorista": "Motorista",
            "veiculo": "veiculo",
            "linha": "linha",
            "km/l": "kml_db",
            "km_rodado": "Km",
            "combustivel_consumido": "Comb.",
        },
        inplace=True,
    )

    # Normalização
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Km"] = to_num(df["Km"])
    df["Comb."] = to_num(df["Comb."])
    df["kml_db"] = to_num(df.get("kml_db", pd.Series([None] * len(df))))

    return df

# ==============================================================================
# 4. PROCESSAMENTO
# ==============================================================================
def preparar_base(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza base e (opcionalmente) aplica filtros padrão."""
    df = df.copy()
    df = df.dropna(subset=["Date", "Motorista", "veiculo", "linha", "Km", "Comb."])
    df = df[(df["Comb."] > 0) & (df["Km"] > 0)].copy()
    df["kml"] = df["Km"] / df["Comb."]

    df["Cluster"] = df["veiculo"].astype(str).apply(get_cluster)

    if NO_FILTERS:
        return df

    df = df.dropna(subset=["Cluster"]).copy()

    before = len(df)
    df = df[(df["kml"] >= KML_MIN) & (df["kml"] <= KML_MAX)].copy()
    after = len(df)
    print(f"   -> Filtro KML calc [{KML_MIN}, {KML_MAX}] removeu {before - after} linhas (restou {after}).")

    return df

def montar_item_motorista(df: pd.DataFrame, mot: str, fim_rank: pd.Timestamp):
    fim_rank = fim_rank.normalize()
    ini_rank = (fim_rank - timedelta(days=RANKING_DIAS - 1)).normalize()
    periodo_txt = f"{ini_rank.strftime('%Y-%m-%d')} → {fim_rank.strftime('%Y-%m-%d')} (últ. {RANKING_DIAS} dias)"

    df_mot_rank = df[(df["Motorista"] == mot) & (df["Date"] >= ini_rank) & (df["Date"] <= fim_rank)].copy()
    if df_mot_rank.empty:
        return None

    if ("Cluster" in df.columns) and (not df_mot_rank["Cluster"].isna().all()):
        ref = df_mot_rank.groupby(["linha", "Cluster"]).agg({"Km": "sum", "Comb.": "sum"}).reset_index()
        ref["KML_Meta_Linha"] = ref["Km"] / ref["Comb."]
        df_mot_rank = df_mot_rank.merge(ref[["linha", "Cluster", "KML_Meta_Linha"]], on=["linha", "Cluster"], how="left")
        
        def calc_perda(row):
            meta = row["KML_Meta_Linha"]
            real = row["kml"]
            if pd.notna(meta) and meta > 0 and real < meta:
                comb_ideal = row["Km"] / meta
                return row["Comb."] - comb_ideal
            return 0.0
        df_mot_rank["Litros_Perdidos"] = df_mot_rank.apply(calc_perda, axis=1)
    else:
        df_mot_rank["KML_Meta_Linha"] = None
        df_mot_rank["Litros_Perdidos"] = 0.0

    pior = (
        df_mot_rank.groupby(["linha", "Cluster"], dropna=False)
        .agg({"Litros_Perdidos": "sum", "Km": "sum", "Comb.": "sum", "KML_Meta_Linha": "mean"})
        .reset_index()
        .sort_values("Litros_Perdidos", ascending=False)
    )

    top = pior.iloc[0]
    linha_foco = top["linha"]
    cluster_foco = top.get("Cluster")

    df_hist_mot = df[df["Motorista"] == mot].copy()
    df_hist_linha = df[(df["linha"] == linha_foco)].copy()
    if ("Cluster" in df.columns) and (cluster_foco is not None) and pd.notna(cluster_foco):
        df_hist_linha = df_hist_linha[df_hist_linha["Cluster"] == cluster_foco].copy()

    d15 = fim_rank - timedelta(days=15)
    vecs = df_hist_mot[df_hist_mot["Date"] >= d15]["veiculo"].unique()
    vecs_str = ", ".join(sorted(map(str, vecs))) if len(vecs) > 0 else "Nenhum"

    km_rank = float(df_mot_rank["Km"].sum())
    comb_rank = float(df_mot_rank["Comb."].sum())
    kml_real = (km_rank / comb_rank) if comb_rank > 0 else 0.0

    kml_meta = float(top["KML_Meta_Linha"]) if pd.notna(top.get("KML_Meta_Linha")) else 0.0
    perda_total = float(df_mot_rank["Litros_Perdidos"].sum())

    res60 = resumo_60d(df_hist_mot, df_hist_linha)
    top5 = top_dias_criticos(df_hist_mot, res60.get("kml_linha_medio_60"), topn=5) if not NO_FILTERS else []
    det = detalhamento_dias_dia_a_dia(df_hist_mot, df_hist_linha, linha_foco, cluster_foco, dias=DETALHE_DIAS)

    return {
        "Motorista": mot,
        "Litros_Total": perda_total,
        "Linha_Foco": linha_foco,
        "Cluster_Foco": str(cluster_foco) if cluster_foco is not None else "",
        "KML_Real": kml_real,
        "KML_Meta": kml_meta,
        "Gap": (kml_real - kml_meta) if kml_meta else 0.0,
        "Dados_RaioX": df_mot_rank,
        "Dados_Hist_Mot": df_hist_mot,
        "Dados_Hist_Linha": df_hist_linha,
        "Veiculos_Recentes": vecs_str,
        "Periodo_Txt": periodo_txt,
        "Resumo_60D": res60,
        "Top_Dias_Criticos": top5,
        "Detalhamento_30D_DiaADia": det,
    }

def processar_dados(df_raw: pd.DataFrame):
    print("⚙️ [Core] Preparando base...")
    df = preparar_base(df_raw)

    if df.empty:
        return []

    if MOTORISTA_FOCO:
        mot = str(MOTORISTA_FOCO).strip()
        df_mot = df[df["Motorista"].astype(str) == mot].copy()
        if df_mot.empty:
            print(f"⚠️ Motorista foco {mot} não tem dados dentro do FETCH_DAYS={FETCH_DAYS}.")
            return []
        fim_rank = df_mot["Date"].max()
        item = montar_item_motorista(df, mot, fim_rank)
        return [item] if item else []

    data_max = df["Date"].max()
    if pd.isna(data_max):
        return []

    fim_rank = data_max.normalize()
    ini_rank = (fim_rank - timedelta(days=RANKING_DIAS - 1)).normalize()
    periodo_txt = f"{ini_rank.strftime('%Y-%m-%d')} → {fim_rank.strftime('%Y-%m-%d')} (últ. {RANKING_DIAS} dias)"
    print(f"🗓️ Período ranking: {periodo_txt}")

    df_foco = df[(df["Date"] >= ini_rank) & (df["Date"] <= fim_rank)].copy()

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

    ranking = (
        df_foco.groupby("Motorista")
        .agg({"Litros_Perdidos": "sum", "Km": "sum"})
        .reset_index()
        .sort_values("Litros_Perdidos", ascending=False)
        .head(QTD_ACOMPANHAMENTOS)
    )

    lista_final = []
    for _, rr in ranking.iterrows():
        mot = rr["Motorista"]
        fim_mot = df_foco[df_foco["Motorista"] == mot]["Date"].max()
        item = montar_item_motorista(df, mot, fim_mot if pd.notna(fim_mot) else fim_rank)
        if item:
            lista_final.append(item)

    return lista_final

# ==============================================================================
# 5. IA (OPCIONAL) + ASSETS
# ==============================================================================
def chamar_vertex_ai(dados):
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
        }.get(dados.get("Cluster_Foco"), "Ônibus Urbano.")

        prompt = f"""
Atue como Instrutor de Motoristas Sênior.
Analise este motorista com ALTO DESPERDÍCIO DE COMBUSTÍVEL.

DADOS:
- Motorista: {dados['Motorista']}
- Veículo: {dados.get('Cluster_Foco','')} ({tec})
- Linha: {dados.get('Linha_Foco','')}
- Carros recentes: {dados.get('Veiculos_Recentes','')}

PERFORMANCE (período ranking):
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
        return (resp.text or "").replace("**", "").replace("##", "")
    except Exception as e:
        print(f"⚠️ Erro IA: {e}")
        return "ANÁLISE: Indisponível (Erro API)."

def gerar_grafico(df_mot, df_linha, caminho):
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
    plt.plot(dates, dados["KML_Mot"], marker="o", lw=3, label="Motorista")
    plt.plot(dates, dados["KML_Linha"], ls="--", lw=2, label="Média Linha (semana)")
    plt.title("Evolução Semanal (Últimos 60 dias)", fontsize=11, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho, dpi=110)
    plt.close()

def gerar_tabela_html(df_mot):
    df_mot = df_mot.copy()
    df_mot["Mes"] = df_mot["Date"].dt.to_period("M").astype(str)

    if "KML_Meta_Linha" not in df_mot.columns:
        df_mot["KML_Meta_Linha"] = None
    if "Litros_Perdidos" not in df_mot.columns:
        df_mot["Litros_Perdidos"] = 0.0

    res = (
        df_mot.groupby(["Mes", "veiculo", "linha", "Cluster", "KML_Meta_Linha"], dropna=False)
        .agg(Km=("Km", "sum"), Comb=("Comb.", "sum"), Loss=("Litros_Perdidos", "sum"))
        .reset_index()
        .sort_values("Loss", ascending=False)
    )
    res["KML"] = res["Km"] / res["Comb"]

    rows = ""
    for _, r in res.iterrows():
        style = "background:#ffebee; color:#c0392b; font-weight:bold;" if (pd.notna(r["Loss"]) and r["Loss"] > 5) else ""
        meta = r["KML_Meta_Linha"]
        meta_txt = f"{float(meta):.2f}" if pd.notna(meta) else ""
        rows += f"""<tr style="{style}">
            <td>{r['veiculo']}</td><td>{r['linha']}</td>
            <td style="text-align:right;">{r['Km']:.0f}</td>
            <td style="text-align:right;">{r['KML']:.2f}</td>
            <td style="text-align:right;">{meta_txt}</td>
            <td style="text-align:right;">{float(r['Loss'] or 0):.1f} L</td></tr>"""
    return rows

def gerar_html_final(dados, texto_ia, img, tabela):
    analise = extrair_bloco(texto_ia, "ANALISE")
    roteiro = extrair_bloco(texto_ia, "ROTEIRO")
    feedback = extrair_bloco(texto_ia, "FEEDBACK")
    cor = "#c0392b" if (dados.get("Gap", 0) < 0) else "#27ae60"

    r60 = dados.get("Resumo_60D") or {}
    top5 = dados.get("Top_Dias_Criticos") or []
    det = dados.get("Detalhamento_30D_DiaADia") or []

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
        <b>Período Ranking:</b> {dados.get('Periodo_Txt','')}
        &nbsp;|&nbsp; <b>NO_FILTERS:</b> {"SIM" if NO_FILTERS else "NÃO"}
        &nbsp;|&nbsp; <b>FETCH_DAYS:</b> {FETCH_DAYS}
      </div>
    """

    rows_det = ""
    for d in det:
        km = d.get("km"); litros = d.get("litros")
        kml_m = d.get("kml_motorista"); kml_l = d.get("kml_linha")
        gap = d.get("gap_dia")

        km_txt = f"{km:.0f}" if km is not None else ""
        litros_txt = f"{litros:.0f}" if litros is not None else ""
        kml_m_txt = f"{kml_m:.3f}" if kml_m is not None else ""
        kml_l_txt = f"{kml_l:.3f}" if kml_l is not None else ""
        gap_txt = f"{gap:.3f}" if gap is not None else ""

        gap_style = "color:#c0392b;font-weight:bold;" if (gap is not None and gap < 0) else "color:#2c3e50;"

        rows_det += f"""
          <tr>
            <td>{d.get('dia','')}</td>
            <td>{d.get('veiculos','')}</td>
            <td>{d.get('linhas','')}</td>
            <td style="text-align:right;">{km_txt}</td>
            <td style="text-align:right;">{litros_txt}</td>
            <td style="text-align:right;">{kml_m_txt}</td>
            <td style="text-align:right;">{kml_l_txt}</td>
            <td style="text-align:right;{gap_style}">{gap_txt}</td>
          </tr>
        """

    tbl_det = f"""
      <h3>0. Detalhamento {DETALHE_DIAS}D (dia a dia)</h3>
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
        <tbody>{rows_det}</tbody>
      </table>
    """

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
            </div>
            <div class="tag">ALTO CUSTO</div>
        </div>

        {b60}

        <div class="kpis">
            <div class="kpi"><b>{dados.get('Cluster_Foco','')}</b><span>Veículo Foco</span></div>
            <div class="kpi"><b>{dados.get('Linha_Foco','')}</b><span>Linha Foco</span></div>
            <div class="kpi"><b style="color:{cor}">{dados.get('KML_Real',0):.2f}</b><span>Real (Meta {dados.get('KML_Meta',0):.2f})</span></div>
            <div class="kpi"><b style="color:#c0392b">{dados.get('Litros_Total',0):.0f} L</b><span>Perda Estimada</span></div>
        </div>

        <div class="obs"><b>Veículos (últ. 15d):</b> {dados.get('Veiculos_Recentes','')}</div>

        {tbl_det}
        {top_tbl}

        <h3>2. Evolução (60D)</h3>
        <img src="{os.path.basename(img)}" style="width:100%; border:1px solid #ddd; margin-bottom:10px;">

        <h3>3. Raio-X da Perda (período ranking)</h3>
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

        df_raw = carregar_dados()
        if df_raw.empty:
            raise Exception("Base vazia (Supabase A)")

        lista = processar_dados(df_raw)
        lista = [x for x in lista if x]

        print(f"🎯 Gerando {len(lista)} ordens... (MOTORISTA_FOCO={MOTORISTA_FOCO or 'NÃO'})")

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

            txt_ia = chamar_vertex_ai(item)
            html = gerar_html_final(item, txt_ia, p_img, tbl)

            with open(p_html, "w", encoding="utf-8") as f:
                f.write(html)

            gerar_pdf(p_html, p_pdf)

            url_pdf = upload_storage(p_pdf, f"{safe}.pdf", "application/pdf")
            url_html = upload_storage(p_html, f"{safe}.html", "text/html")

            r60 = item.get("Resumo_60D") or {}
            top5 = item.get("Top_Dias_Criticos") or []
            det = item.get("Detalhamento_30D_DiaADia") or []

            if url_pdf:
                sb.table(TABELA_DESTINO).insert(
                    {
                        "lote_id": ORDEM_BATCH_ID,
                        "motorista_nome": mot,
                        "motorista_chapa": mot,
                        "motivo": "BAIXO_DESEMPENHO",
                        "veiculo_foco": item.get("Cluster_Foco") or "",
                        "linha_foco": item.get("Linha_Foco") or "",
                        "kml_real": float(item.get("KML_Real") or 0),
                        "kml_meta": float(item.get("KML_Meta") or 0),
                        "gap": float(item.get("Gap") or 0),
                        "perda_litros": float(item.get("Litros_Total") or 0),
                        "arquivo_pdf_path": url_pdf,
                        "arquivo_html_path": url_html,
                        "status": "CONCLUIDO",
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
                        "top_dias_criticos": top5,
                        "metadata": {
                            "periodo_ranking": item.get("Periodo_Txt"),
                            "fetch_days": FETCH_DAYS,
                            "ranking_dias": RANKING_DIAS,
                            "detalhe_dias": DETALHE_DIAS,
                            "no_filters": NO_FILTERS,
                            "motorista_foco": MOTORISTA_FOCO,
                            "regra_kml": {"min": KML_MIN, "max": KML_MAX},
                            "analise_60d": r60,
                            "top_dias_criticos": top5,
                            "detalhamento_dia_a_dia": det,
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
