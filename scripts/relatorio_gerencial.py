# scripts/relatorio_gerencial.py
import os
import re
import json  # ✅ Adicionado para manipular o JSON
from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

import vertexai
from vertexai.generative_models import GenerativeModel

from supabase import create_client
from playwright.sync_api import sync_playwright

# ✅ NOVO: tratar ausência de credenciais ADC sem estourar stacktrace
try:
    from google.auth.exceptions import DefaultCredentialsError
except Exception:  # fallback (caso lib não exista no ambiente)
    class DefaultCredentialsError(Exception):
        pass


# ==============================================================================
# CONFIG (ENV FIRST)
# ==============================================================================
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID") or os.getenv("PROJECT_ID")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
VERTEX_MODEL = os.getenv("VERTEX_MODEL", "gemini-2.5-pro")

SUPABASE_A_URL = os.getenv("SUPABASE_A_URL")
SUPABASE_A_SERVICE_ROLE_KEY = os.getenv("SUPABASE_A_SERVICE_ROLE_KEY")

SUPABASE_B_URL = os.getenv("SUPABASE_B_URL")
SUPABASE_B_SERVICE_ROLE_KEY = os.getenv("SUPABASE_B_SERVICE_ROLE_KEY")

REPORT_ID = os.getenv("REPORT_ID")
REPORT_TIPO = os.getenv("REPORT_TIPO", "diesel_gerencial")
REPORT_PERIODO_INICIO = os.getenv("REPORT_PERIODO_INICIO")  # YYYY-MM-DD
REPORT_PERIODO_FIM = os.getenv("REPORT_PERIODO_FIM")        # YYYY-MM-DD

PASTA_SAIDA = os.getenv("REPORT_OUTPUT_DIR", "Relatorios_Diesel_Final")
TABELA_ORIGEM = os.getenv("DIESEL_SOURCE_TABLE", "premiacao_diaria")
BUCKET_RELATORIOS = os.getenv("REPORT_BUCKET", "relatorios")

# (opcional) override do path folder remoto
REMOTE_BASE_PREFIX = os.getenv("REPORT_REMOTE_PREFIX", "diesel")

# PERFORMANCE / RESILIÊNCIA
REPORT_PAGE_SIZE = int(os.getenv("REPORT_PAGE_SIZE", "1000"))
REPORT_MAX_ROWS = int(os.getenv("REPORT_MAX_ROWS", "250000"))
REPORT_FETCH_WINDOW_DAYS = int(os.getenv("REPORT_FETCH_WINDOW_DAYS", "7"))  # ✅ janelas menores

# ✅ TABELA DE SUGESTÕES (Supabase B)
SUGESTOES_TABLE = os.getenv("SUGESTOES_TABLE", "diesel_sugestoes_acompanhamento")

# ✅ NOVO (opcional): você pode injetar o JSON do service account via ENV e o script monta ADC
# Ex.: VERTEX_SA_JSON='{"type":"service_account", ... }'
VERTEX_SA_JSON = os.getenv("VERTEX_SA_JSON")  # opcional


# ==============================================================================
# Helpers
# ==============================================================================
def _assert_env():
    missing = []
    for k in [
        "SUPABASE_A_URL",
        "SUPABASE_A_SERVICE_ROLE_KEY",
        "SUPABASE_B_URL",
        "SUPABASE_B_SERVICE_ROLE_KEY",
        "REPORT_ID",
    ]:
        if not os.getenv(k):
            missing.append(k)

    if missing:
        raise RuntimeError(f"Variáveis obrigatórias ausentes: {missing}")


def _parse_iso(d: str | None) -> date | None:
    if not d:
        return None
    return datetime.strptime(d, "%Y-%m-%d").date()


def _safe_filename(name: str) -> str:
    name = str(name or "").strip()
    name = re.sub(r"[^\w\-.() ]+", "_", name, flags=re.UNICODE)
    name = re.sub(r"\s+", "_", name)
    return name[:180] if name else "arquivo"


def _sb_a():
    return create_client(SUPABASE_A_URL, SUPABASE_A_SERVICE_ROLE_KEY)


def _sb_b():
    return create_client(SUPABASE_B_URL, SUPABASE_B_SERVICE_ROLE_KEY)


def atualizar_status_relatorio(status: str, **fields):
    sb = _sb_b()
    payload = {"status": status, **fields}
    sb.table("relatorios_gerados").update(payload).eq("id", REPORT_ID).execute()


def upload_storage_b(local_path: Path, remote_path: str, content_type: str) -> int:
    sb = _sb_b()
    storage = sb.storage.from_(BUCKET_RELATORIOS)
    data = local_path.read_bytes()

    storage.upload(
        path=remote_path,
        file=data,
        file_options={"content-type": content_type, "upsert": "true"},
    )
    return len(data)


def _fmt_br_date(d: date | None) -> str:
    if not d:
        return ""
    return d.strftime("%d/%m/%Y")


def _to_num(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    s = series.astype(str).str.strip()
    s = s.str.replace(",", ".", regex=False)
    s = s.str.replace(r"[^0-9\.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def _extract_chapa(motorista_val) -> str:
    """
    Tenta extrair chapa do campo Motorista.
    Aceita padrões tipo:
      '12345 - JOAO' / '12345 JOAO' / 'Chapa 12345' / '12345'
    Se não achar número, usa o texto original "limpo".
    """
    s = str(motorista_val or "").strip()
    if not s:
        return "N/D"
    m = re.search(r"\b(\d{3,10})\b", s)  # chapa geralmente numérica
    if m:
        return m.group(1)
    return _safe_filename(s)[:40]


# ==============================================================================
# ✅ NOVO: garante ADC sem quebrar (opcional via ENV VERTEX_SA_JSON)
# ==============================================================================
def _ensure_vertex_adc_if_possible():
    """
    Se o ambiente não tiver ADC (GOOGLE_APPLICATION_CREDENTIALS),
    mas tiver VERTEX_SA_JSON, cria um arquivo temporário e aponta a env.
    Isso evita DefaultCredentialsError no GitHub Actions.
    """
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        return

    if not VERTEX_SA_JSON:
        return

    try:
        tmp = Path("/tmp/vertex_sa.json")
        tmp.write_text(VERTEX_SA_JSON, encoding="utf-8")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(tmp)
        print("✅ [Vertex] ADC configurado via VERTEX_SA_JSON (/tmp/vertex_sa.json).")
    except Exception as e:
        print("⚠️ [Vertex] Falha ao montar ADC via VERTEX_SA_JSON:", repr(e))


# ==============================================================================
# NOVO: CARREGAR NOMES DO CSV
# ==============================================================================
def carregar_mapa_nomes(caminho_csv="motoristas_rows.csv"):
    """
    Lê o CSV e retorna um dicionário { '30061012': 'NOME DO MOTORISTA' }
    """
    if not os.path.exists(caminho_csv):
        print("⚠️ CSV de nomes não encontrado. Usando nomes do sistema.")
        return {}

    try:
        # Lê como string para preservar zeros à esquerda na chapa
        df = pd.read_csv(caminho_csv, dtype=str)
        # Remove espaços e cria dicionário
        df["chapa"] = df["chapa"].str.strip()
        df["nome"] = df["nome"].str.strip().str.upper()
        return dict(zip(df["chapa"], df["nome"]))
    except Exception as e:
        print(f"❌ Erro ao ler CSV de nomes: {e}")
        return {}


# ==============================================================================
# CÁLCULO DE DETALHES (AJUSTADO: Gráfico Ordenado)
# ==============================================================================
def calcular_detalhes_json(df_motorista):
    """
    Gera o JSON com Raio-X e Gráfico Semanal para o Frontend
    """
    if df_motorista.empty:
        return {}

    # --- 1. RAIO-X (Agrupado por Linha + Cluster) ---
    grp = df_motorista.groupby(["linha", "Cluster"]).agg({
        "Km": "sum",
        "Comb.": "sum",
        "veiculo": lambda x: list(x.unique())[0],
        "KML_Ref": "mean"
    }).reset_index()

    grp["kml_real"] = grp["Km"] / grp["Comb."]

    def calc_waste(row):
        meta = row["KML_Ref"]
        try:
            if meta > 0 and row["kml_real"] < meta:
                return row["Comb."] - (row["Km"] / meta)
        except Exception:
            pass
        return 0.0

    grp["desperdicio"] = grp.apply(calc_waste, axis=1)

    raio_x = []
    for _, row in grp.sort_values("desperdicio", ascending=False).iterrows():
        raio_x.append({
            "linha": str(row["linha"]),
            "cluster": str(row["Cluster"]),
            "km": float(row["Km"]),
            "litros": float(row["Comb."]),
            "kml_real": float(row["kml_real"]),
            "kml_meta": float(row["KML_Ref"]),
            "desperdicio": float(row["desperdicio"])
        })

    # --- 2. GRÁFICO SEMANAL (CORRIGIDO ORDENAÇÃO) ---
    df_chart = df_motorista.copy()
    df_chart["Semana_Dt"] = df_chart["Date"].dt.to_period("W").apply(lambda r: r.start_time)

    grp_sem = df_chart.groupby("Semana_Dt").agg({
        "Km": "sum",
        "Comb.": "sum",
        "KML_Ref": "mean"
    }).sort_index()

    grafico = []
    for dt, row in grp_sem.iterrows():
        kml_real = row["Km"] / row["Comb."] if row["Comb."] > 0 else 0
        grafico.append({
            "label": dt.strftime("%d/%m"),
            "real": float(kml_real),
            "meta": float(row["KML_Ref"])
        })

    return {"raio_x": raio_x, "grafico_semanal": grafico}


def gerar_sugestoes_acompanhamento(dados_proc: dict) -> pd.DataFrame:
    """
    Gera a lista para a tela (Sugestões de Acompanhamento) a partir do df_atual.
    E popula o campo 'detalhes_json' com o Raio-X e Gráficos.
    """
    df_atual = dados_proc.get("df_atual")
    if df_atual is None or df_atual.empty:
        return pd.DataFrame(columns=[
            "chapa", "linha_mais_rodada", "km_percorrido", "consumo_realizado",
            "kml_realizado", "kml_meta", "combustivel_desperdicado",
            "motorista_nome", "cluster", "detalhes_json"
        ])

    df = df_atual.copy()

    # Normaliza tipos
    df["Km"] = pd.to_numeric(df["Km"], errors="coerce").fillna(0)
    df["Comb."] = pd.to_numeric(df["Comb."], errors="coerce").fillna(0)
    df["kml"] = pd.to_numeric(df["kml"], errors="coerce")
    df["KML_Ref"] = pd.to_numeric(df["KML_Ref"], errors="coerce")
    df["Litros_Desperdicio"] = pd.to_numeric(df["Litros_Desperdicio"], errors="coerce").fillna(0)

    # Chapa
    df["chapa"] = df["Motorista"].apply(_extract_chapa)

    # CARREGA O DE/PARA DE NOMES
    mapa_nomes = carregar_mapa_nomes("motoristas_rows.csv")

    def resolver_nome(row):
        nome_csv = mapa_nomes.get(row["chapa"])
        if nome_csv:
            return nome_csv
        return row["Motorista"]

    df["Motorista_Final"] = df.apply(resolver_nome, axis=1)

    # 1. Agregação Geral (Para a lista principal)
    agg = (
        df.groupby(["chapa"], as_index=False)
        .agg(
            km_percorrido=("Km", "sum"),
            consumo_realizado=("Comb.", "sum"),
            combustivel_desperdicado=("Litros_Desperdicio", "sum"),
            motorista_nome=("Motorista_Final", "first"),
            cluster=("Cluster", "first"),
            kml_meta=("KML_Ref", "mean"),
        )
    )

    agg["kml_realizado"] = agg["km_percorrido"] / agg["consumo_realizado"]

    # 2. Gera os Detalhes JSON para cada motorista
    print("⚙️  [Sugestões] Calculando Raio-X e Gráficos detalhados...")
    detalhes_map = {}
    for chapa in agg["chapa"].unique():
        df_mot = df[df["chapa"] == chapa]
        detalhes_map[chapa] = json.dumps(calcular_detalhes_json(df_mot))

    agg["detalhes_json"] = agg["chapa"].map(detalhes_map)

    # Linha mais rodada por chapa
    linha_top = (
        df.groupby(["chapa", "linha"], as_index=False)["Km"].sum()
        .sort_values(["chapa", "Km"], ascending=[True, False])
    )
    linha_top = (
        linha_top.drop_duplicates("chapa", keep="first")[["chapa", "linha"]]
        .rename(columns={"linha": "linha_mais_rodada"})
    )

    agg = agg.merge(linha_top, on="chapa", how="left")
    agg = agg.sort_values(["combustivel_desperdicado", "km_percorrido"], ascending=[False, False])

    agg = agg[[
        "chapa",
        "linha_mais_rodada",
        "km_percorrido",
        "consumo_realizado",
        "kml_realizado",
        "kml_meta",
        "combustivel_desperdicado",
        "motorista_nome",
        "cluster",
        "detalhes_json",
    ]].reset_index(drop=True)

    agg = agg.dropna(subset=["chapa"])
    return agg


def salvar_sugestoes_supabase_b(df_sug: pd.DataFrame, mes_ref: str, periodo_inicio: date | None, periodo_fim: date | None):
    """
    Upsert das sugestões no Supabase B em diesel_sugestoes_acompanhamento.
    Incluindo o campo detalhes_json.
    """
    if df_sug is None or df_sug.empty:
        print("ℹ️  [Sugestões] Nenhuma sugestão para salvar.")
        return

    sb = _sb_b()

    rows = []
    for _, r in df_sug.iterrows():
        try:
            detalhes = json.loads(r.get("detalhes_json", "{}"))
        except Exception:
            detalhes = {}

        rows.append({
            "mes_ref": mes_ref,
            "periodo_inicio": str(periodo_inicio) if periodo_inicio else None,
            "periodo_fim": str(periodo_fim) if periodo_fim else None,

            "chapa": str(r.get("chapa") or "N/D"),
            "linha_mais_rodada": r.get("linha_mais_rodada"),
            "km_percorrido": float(r.get("km_percorrido") or 0),
            "consumo_realizado": float(r.get("consumo_realizado") or 0),
            "kml_realizado": float(r.get("kml_realizado") or 0),
            "kml_meta": float(r.get("kml_meta") or 0),
            "combustivel_desperdicado": float(r.get("combustivel_desperdicado") or 0),

            "motorista_nome": str(r.get("motorista_nome") or ""),
            "cluster": str(r.get("cluster") or ""),

            "detalhes_json": detalhes,
            "extra": {},
        })

    sb.table(SUGESTOES_TABLE).upsert(rows, on_conflict="mes_ref,chapa").execute()
    print(f"✅ [Sugestões] Salvas/atualizadas: {len(rows)} linhas (mes_ref={mes_ref}).")


# ==============================================================================
# 0) BUSCA DADOS SUPABASE A -> DF padrão interno (fetch por janelas)
# ==============================================================================
def carregar_dados_supabase_a(periodo_inicio: date | None, periodo_fim: date | None) -> pd.DataFrame:
    sb = _sb_a()

    if not periodo_inicio or not periodo_fim:
        hoje = datetime.utcnow().date()
        periodo_inicio = periodo_inicio or hoje.replace(day=1)
        periodo_fim = periodo_fim or hoje

    if periodo_inicio > periodo_fim:
        periodo_inicio, periodo_fim = periodo_fim, periodo_inicio

    SELECT_FIELDS = (
        'id_premiacao_diaria, dia, motorista, veiculo, linha, '
        'km_rodado, combustivel_consumido, minutos_em_viagem, "km/l"'
    )

    all_rows = []
    pages = 0

    cursor_fim = periodo_fim
    while cursor_fim >= periodo_inicio:
        cursor_ini = max(periodo_inicio, cursor_fim - timedelta(days=REPORT_FETCH_WINDOW_DAYS - 1))
        s_ini = str(cursor_ini)
        s_fim = str(cursor_fim)

        start = 0
        while True:
            end = start + REPORT_PAGE_SIZE - 1
            q = (
                sb.table(TABELA_ORIGEM)
                .select(SELECT_FIELDS)
                .gte("dia", s_ini)
                .lte("dia", s_fim)
                .order("dia", desc=False)
                .order("id_premiacao_diaria", desc=False)
                .range(start, end)
            )

            resp = q.execute()
            rows = resp.data or []
            pages += 1
            all_rows.extend(rows)

            print(
                f"📦 [SupabaseA] janela={s_ini}..{s_fim} "
                f"page={pages} range={start}-{end} fetched={len(rows)} total={len(all_rows)}"
            )

            if len(rows) < REPORT_PAGE_SIZE:
                break

            if len(all_rows) >= REPORT_MAX_ROWS:
                all_rows = all_rows[:REPORT_MAX_ROWS]
                print(f"⚠️ [SupabaseA] REPORT_MAX_ROWS atingido: {REPORT_MAX_ROWS}")
                break

            start += REPORT_PAGE_SIZE

        if len(all_rows) >= REPORT_MAX_ROWS:
            break

        cursor_fim = cursor_ini - timedelta(days=1)

    if not all_rows:
        return pd.DataFrame(columns=["Date", "Motorista", "veiculo", "linha", "kml", "Km", "Comb."])

    df = pd.DataFrame(all_rows)

    if "km/l" in df.columns and "kml" not in df.columns:
        df["kml"] = df["km/l"]

    out = pd.DataFrame()
    out["Date"] = df.get("dia")
    out["Motorista"] = df.get("motorista")
    out["veiculo"] = df.get("veiculo")
    out["linha"] = df.get("linha")
    out["kml"] = df.get("kml")
    out["Km"] = df.get("km_rodado")
    out["Comb."] = df.get("combustivel_consumido")
    return out


# ==============================================================================
# 1) PROCESSAMENTO
# ==============================================================================
def processar_dados_gerenciais_df(df: pd.DataFrame, periodo_inicio: date | None, periodo_fim: date | None):
    print("⚙️  [Sistema] Processando dados para visão gerencial...")

    obrigatorias = ["Date", "Motorista", "veiculo", "linha", "kml", "Km", "Comb."]
    faltando = [c for c in obrigatorias if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas obrigatórias ausentes: {faltando}. Colunas atuais: {df.columns.tolist()}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    df["kml"] = _to_num(df["kml"])
    df["Km"] = _to_num(df["Km"])
    df["Comb."] = _to_num(df["Comb."])

    bruto_min = df["Date"].min()
    bruto_max = df["Date"].max()
    bruto_min_txt = bruto_min.strftime("%d/%m/%Y") if pd.notna(bruto_min) else "N/D"
    bruto_max_txt = bruto_max.strftime("%d/%m/%Y") if pd.notna(bruto_max) else "N/D"
    qtd_bruto = len(df)

    def definir_cluster(v):
        v = str(v).strip()
        if v in ["W511", "W513", "W515"]:
            return None
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

    df["Cluster"] = df["veiculo"].apply(definir_cluster)
    qtd_cluster_invalido = int(df["Cluster"].isna().sum())
    df = df.dropna(subset=["Cluster"])

    df = df.dropna(subset=["Km", "Comb."])
    df = df[(df["Km"] > 0) & (df["Comb."] > 0)].copy()
    df["kml"] = df["Km"] / df["Comb."]

    df_clean = df[(df["kml"] >= 1.5) & (df["kml"] <= 5)].copy()
    if df_clean.empty:
        raise ValueError("Sem dados válidos após filtros (kml entre 1.5 e 5 e cluster válido).")

    if periodo_inicio and periodo_fim:
        periodo_txt = f"{_fmt_br_date(periodo_inicio)} a {_fmt_br_date(periodo_fim)}"
    else:
        data_ini = df_clean["Date"].min().strftime("%d/%m/%Y")
        data_fim = df_clean["Date"].max().strftime("%d/%m/%Y")
        periodo_txt = f"{data_ini} a {data_fim}"

    ultimo_mes_dt = df_clean["Date"].max()
    mes_en = ultimo_mes_dt.strftime("%B").lower()
    mapa_meses = {
        "january": "JANEIRO",
        "february": "FEVEREIRO",
        "march": "MARÇO",
        "april": "ABRIL",
        "may": "MAIO",
        "june": "JUNHO",
        "july": "JULHO",
        "august": "AGOSTO",
        "september": "SETEMBRO",
        "october": "OUTUBRO",
        "november": "NOVEMBRO",
        "december": "DEZEMBRO",
    }
    mes_pt = mapa_meses.get(mes_en, mes_en.upper())
    mes_atual_txt = f"{mes_pt}/{ultimo_mes_dt.year}"

    outliers = df[(df["kml"] > 5) | (df["kml"] < 1.5)].copy()
    qtd_excluidos = len(outliers)

    if not outliers.empty:
        top_veiculos_contaminados = (
            outliers.groupby(["veiculo", "Cluster", "linha"])
            .agg(
                Qtd_Contaminacoes=("kml", "count"),
                KML_Min=("kml", "min"),
                KML_Max=("kml", "max"),
            )
            .reset_index()
            .sort_values("Qtd_Contaminacoes", ascending=False)
            .head(10)
        )
    else:
        top_veiculos_contaminados = pd.DataFrame(
            columns=["veiculo", "Cluster", "linha", "Qtd_Contaminacoes", "KML_Min", "KML_Max"]
        )

    df_clean["Mes_Ano"] = df_clean["Date"].dt.to_period("M")

    tabela_cluster = (
        df_clean.groupby(["Cluster", "Mes_Ano"])
        .agg({"Km": "sum", "Comb.": "sum"})
        .reset_index()
    )
    tabela_cluster["KML"] = tabela_cluster["Km"] / tabela_cluster["Comb."]
    tabela_pivot = tabela_cluster.pivot(index="Cluster", columns="Mes_Ano", values="KML")

    linha_trend = (
        df_clean.groupby(["linha", "Mes_Ano"])
        .agg({"Km": "sum", "Comb.": "sum"})
        .reset_index()
    )
    linha_trend["KML"] = linha_trend["Km"] / linha_trend["Comb."]
    linha_pivot = linha_trend.pivot(index="linha", columns="Mes_Ano", values="KML")

    cols_meses = linha_pivot.columns.sort_values()
    if len(cols_meses) >= 2:
        mes_atual = cols_meses[-1]
        mes_anterior = cols_meses[-2]

        linha_pivot["KML_Atual"] = linha_pivot[mes_atual]
        linha_pivot["KML_Anterior"] = linha_pivot[mes_anterior]
        linha_pivot["Variacao_Pct"] = ((linha_pivot["KML_Atual"] - linha_pivot["KML_Anterior"]) / linha_pivot["KML_Anterior"]) * 100

        top_linhas_queda = (
            linha_pivot.dropna(subset=["Variacao_Pct"])
            .sort_values("Variacao_Pct", ascending=True)
            .head(5)
        )
        top_linhas_queda = top_linhas_queda[["KML_Anterior", "KML_Atual", "Variacao_Pct"]].reset_index()
    else:
        linha_pivot["KML_Atual"] = linha_pivot[cols_meses[-1]]
        linha_pivot["KML_Anterior"] = 0
        linha_pivot["Variacao_Pct"] = 0
        top_linhas_queda = (
            linha_pivot.sort_values("KML_Atual", ascending=True)
            .head(5)
            .reset_index()
        )

    ultimo_mes = df_clean["Mes_Ano"].max()
    df_atual = df_clean[df_clean["Mes_Ano"] == ultimo_mes].copy()

    ref_grupo = (
        df_atual.groupby(["linha", "Cluster"])
        .agg({"Km": "sum", "Comb.": "sum"})
        .reset_index()
    )
    ref_grupo["KML_Ref"] = ref_grupo["Km"] / ref_grupo["Comb."]
    ref_grupo.rename(columns={"Km": "KM_Total_Linha"}, inplace=True)

    df_atual = pd.merge(
        df_atual,
        ref_grupo[["linha", "Cluster", "KML_Ref", "KM_Total_Linha"]],
        on=["linha", "Cluster"],
        how="left",
    )

    def calc_desperdicio(r):
        try:
            if r["KML_Ref"] > 0 and r["kml"] < r["KML_Ref"]:
                return r["Comb."] - (r["Km"] / r["KML_Ref"])
            return 0
        except Exception:
            return 0

    df_atual["Litros_Desperdicio"] = df_atual.apply(calc_desperdicio, axis=1)
    total_desperdicio = float(df_atual["Litros_Desperdicio"].sum() or 0)

    top_veiculos = (
        df_atual.groupby(["veiculo", "Cluster", "linha"])
        .agg({"Litros_Desperdicio": "sum", "Km": "sum", "Comb.": "sum", "KML_Ref": "mean"})
        .reset_index()
    )
    top_veiculos["KML_Real"] = top_veiculos["Km"] / top_veiculos["Comb."]
    top_veiculos["KML_Meta"] = top_veiculos["KML_Ref"]
    top_veiculos = top_veiculos.sort_values("Litros_Desperdicio", ascending=False).head(5)

    top_motoristas = (
        df_atual.groupby(["Motorista", "Cluster", "linha", "KM_Total_Linha"])
        .agg({"Litros_Desperdicio": "sum", "Km": "sum", "Comb.": "sum", "KML_Ref": "mean"})
        .reset_index()
    )
    top_motoristas["KML_Real"] = top_motoristas["Km"] / top_motoristas["Comb."]
    top_motoristas["KML_Meta"] = top_motoristas["KML_Ref"]
    top_motoristas["Impacto_Pct"] = (top_motoristas["Km"] / top_motoristas["KM_Total_Linha"]) * 100
    top_motoristas = top_motoristas.sort_values("Litros_Desperdicio", ascending=False).head(5)

    clean_min = df_clean["Date"].min()
    clean_max = df_clean["Date"].max()
    clean_min_txt = clean_min.strftime("%d/%m/%Y") if pd.notna(clean_min) else "N/D"
    clean_max_txt = clean_max.strftime("%d/%m/%Y") if pd.notna(clean_max) else "N/D"
    qtd_clean = len(df_clean)

    return {
        "df_clean": df_clean,
        "df_atual": df_atual,
        "qtd_excluidos": int(qtd_excluidos),
        "total_desperdicio": total_desperdicio,
        "top_veiculos": top_veiculos,
        "top_linhas_queda": top_linhas_queda,
        "top_motoristas": top_motoristas,
        "top_veiculos_contaminados": top_veiculos_contaminados,
        "periodo": periodo_txt,
        "mes_atual_nome": mes_atual_txt,
        "tabela_pivot": tabela_pivot,
        "cobertura": {
            "bruto_min": bruto_min_txt,
            "bruto_max": bruto_max_txt,
            "qtd_bruto": int(qtd_bruto),
            "qtd_cluster_invalido": int(qtd_cluster_invalido),
            "qtd_contaminacao": int(qtd_excluidos),
            "clean_min": clean_min_txt,
            "clean_max": clean_max_txt,
            "qtd_clean": int(qtd_clean),
        },
    }


# ==============================================================================
# 2) GRÁFICO
# ==============================================================================
def gerar_grafico_geral(df_clean: pd.DataFrame, caminho_img: Path):
    df_clean = df_clean.copy()
    df_clean["Semana"] = df_clean["Date"].dt.to_period("W").apply(lambda r: r.start_time)

    evolucao_cluster = (
        df_clean.groupby(["Semana", "Cluster"])
        .agg({"Km": "sum", "Comb.": "sum"})
        .reset_index()
    )
    evolucao_cluster["KML"] = evolucao_cluster["Km"] / evolucao_cluster["Comb."]
    pivot_chart = evolucao_cluster.pivot(index="Semana", columns="Cluster", values="KML")

    evolucao_geral = (
        df_clean.groupby(["Semana"])
        .agg({"Km": "sum", "Comb.": "sum"})
        .reset_index()
    )
    evolucao_geral["KML"] = evolucao_geral["Km"] / evolucao_geral["Comb."]

    plt.figure(figsize=(10, 5))

    cores = {
        "C11": "#e67e22",
        "C10": "#2ecc71",
        "C9": "#3498db",
        "C8": "#9b59b6",
        "C6": "#95a5a6",
    }

    for cluster in pivot_chart.columns:
        plt.plot(
            pivot_chart.index,
            pivot_chart[cluster],
            marker=".",
            linewidth=1.5,
            label=cluster,
            color=cores.get(cluster, "gray"),
            alpha=0.7,
        )

    plt.plot(
        evolucao_geral["Semana"],
        evolucao_geral["KML"],
        marker="o",
        linewidth=3,
        label="MÉDIA FROTA",
        color="black",
    )

    for x, y in zip(evolucao_geral["Semana"], evolucao_geral["KML"]):
        plt.text(
            x, y + 0.05, f"{y:.2f}",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold",
            color="black"
        )

    plt.title("Evolução de Eficiência: Clusters vs Média Frota", fontsize=12, fontweight="bold")
    plt.xlabel("Semana")
    plt.ylabel("KM/L")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(caminho_img, dpi=110)
    plt.close()


# ==============================================================================
# 3) IA (fallback se Vertex não configurado OU sem credenciais ADC)
# ==============================================================================
def consultar_ia_gerencial(dados_proc: dict) -> str:
    print("🧠 [Gerente] Solicitando análise estratégica à IA...")

    if not VERTEX_PROJECT_ID:
        return (
            "<p><b>Visão Geral da Eficiência no Período</b><br>"
            "IA desativada (VERTEX_PROJECT_ID não configurado). Relatório gerado apenas com dados.</p>"
        )

    # ✅ NOVO: tenta montar ADC se vier por ENV (sem quebrar)
    _ensure_vertex_adc_if_possible()

    try:
        df_clean = dados_proc["df_clean"].copy()

        km_total_periodo = float(df_clean["Km"].sum() or 0)
        comb_total_periodo = float(df_clean["Comb."].sum() or 0)
        kml_periodo = km_total_periodo / comb_total_periodo if comb_total_periodo > 0 else 0.0

        if "Mes_Ano" not in df_clean.columns:
            df_clean["Mes_Ano"] = df_clean["Date"].dt.to_period("M")

        mensal = (
            df_clean.groupby("Mes_Ano")
            .agg({"Km": "sum", "Comb.": "sum"})
            .reset_index()
        )
        mensal["KML"] = mensal["Km"] / mensal["Comb."]
        mensal = mensal.sort_values("Mes_Ano")
        tabela_mensal_md = mensal[["Mes_Ano", "Km", "Comb.", "KML"]].to_markdown(index=False)

        if len(mensal) >= 1:
            mes_atual_row = mensal.iloc[-1]
            kml_mes_atual = float(mes_atual_row["KML"])
            mes_atual_label = str(mes_atual_row["Mes_Ano"])
        else:
            kml_mes_atual = 0.0
            mes_atual_label = "N/D"

        if len(mensal) >= 2:
            mes_ant_row = mensal.iloc[-2]
            kml_mes_anterior = float(mes_ant_row["KML"])
            mes_ant_label = str(mes_ant_row["Mes_Ano"])
            delta_kml_mes = ((kml_mes_atual - kml_mes_anterior) / kml_mes_anterior * 100) if kml_mes_anterior > 0 else 0.0
        else:
            kml_mes_anterior = 0.0
            mes_ant_label = "N/D"
            delta_kml_mes = 0.0

        df_clean["Semana"] = df_clean["Date"].dt.to_period("W").apply(lambda r: r.start_time)
        semanal = (
            df_clean.groupby("Semana")
            .agg({"Km": "sum", "Comb.": "sum"})
            .reset_index()
        )
        semanal["KML"] = semanal["Km"] / semanal["Comb."]
        semanal = semanal.sort_values("Semana")

        if len(semanal) >= 1:
            semana_atual_row = semanal.iloc[-1]
            kml_semana_atual = float(semana_atual_row["KML"])
            semana_atual_inicio_txt = semana_atual_row["Semana"].strftime("%d/%m/%Y")
        else:
            kml_semana_atual = 0.0
            semana_atual_inicio_txt = "N/D"

        if len(semanal) >= 2:
            semana_ant_row = semanal.iloc[-2]
            kml_semana_anterior = float(semana_ant_row["KML"])
            semana_ant_inicio_txt = semana_ant_row["Semana"].strftime("%d/%m/%Y")
            delta_kml_semana = ((kml_semana_atual - kml_semana_anterior) / kml_semana_anterior * 100) if kml_semana_anterior > 0 else 0.0
        else:
            kml_semana_anterior = 0.0
            semana_ant_inicio_txt = "N/D"
            delta_kml_semana = 0.0

        vertexai.init(project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION)
        model = GenerativeModel(VERTEX_MODEL)

        cob = dados_proc.get("cobertura", {}) or {}

        prompt = f"""
Você é Diretor de Operações de uma empresa de transporte urbano, especialista em eficiência energética (KM/L).

Analise a performance de KM/L no período: {dados_proc['periodo']}
Foco:
- MÊS DE REFERÊNCIA: {dados_proc['mes_atual_nome']}
- ÚLTIMA SEMANA do período

TRANSPARÊNCIA DA BASE:
- Base bruta no range: {cob.get('bruto_min','N/D')} a {cob.get('bruto_max','N/D')} | registros: {cob.get('qtd_bruto','?')}
- Removidos por cluster inválido: {cob.get('qtd_cluster_invalido','?')}
- Removidos por contaminação (kml < 1.5 ou > 5): {cob.get('qtd_contaminacao','?')}
- Base limpa analisada: {cob.get('clean_min','N/D')} a {cob.get('clean_max','N/D')} | registros: {cob.get('qtd_clean','?')}

VISÃO GERAL (FROTA):
- KM/L médio período: {kml_periodo:.2f}
- KM/L mês atual ({mes_atual_label}): {kml_mes_atual:.2f}
- KM/L mês anterior ({mes_ant_label}): {kml_mes_anterior:.2f}
- Variação mês atual vs anterior: {delta_kml_mes:+.1f}%
- KM/L última semana (início {semana_atual_inicio_txt}): {kml_semana_atual:.2f}
- KM/L semana anterior (início {semana_ant_inicio_txt}): {kml_semana_anterior:.2f}
- Variação última semana vs anterior: {delta_kml_semana:+.1f}%
- Desperdício total estimado no mês atual: {dados_proc['total_desperdicio']:.0f} litros

TABELA MENSAL (base limpa):
{tabela_mensal_md}

CLUSTER (KML por mês):
{dados_proc['tabela_pivot'].to_markdown()}

TOP ALVOS DO MÊS:
VEÍCULOS:
{dados_proc['top_veiculos'].to_markdown()}

LINHAS EM QUEDA:
{dados_proc['top_linhas_queda'].to_markdown()}

MOTORISTAS:
{dados_proc['top_motoristas'].to_markdown()}

FORMATO DE RESPOSTA:
Gere um resumo executivo em HTML (sem markdown, sem ```), usando apenas: <p>, <b>, <br>, <ul>, <li>.

Estrutura obrigatória:
1) <b>Visão Geral da Eficiência no Período</b>
2) <b>Zoom na Última Semana</b>
3) <b>Recomendações Práticas para o Próximo Ciclo</b>
   - mínimo 8 itens, agrupados por:
     (a) Operação (b) Manutenção (c) Dados/Sistema
   - Cada item com objetivo + alvo + métrica de sucesso.

Regras:
- Não invente fatos. Baseie em dados.
- Direto e acionável (linguagem de diretoria).
""".strip()

        resp = model.generate_content(prompt)
        texto = getattr(resp, "text", None) or "Análise indisponível."
        return texto.replace("```html", "").replace("```", "")

    # ✅ NOVO: quando não há ADC, NÃO imprime stacktrace gigante; apenas fallback
    except DefaultCredentialsError as e:
        print("⚠️ [IA] Credenciais ADC não encontradas. IA desativada neste run.")
        return (
            "<p><b>Visão Geral da Eficiência no Período</b><br>"
            "IA desativada (credenciais Google/ADC não configuradas no runner). "
            "Relatório gerado apenas com dados.</p>"
        )

    except Exception as e:
        import traceback
        print("❌ Erro ao chamar IA:", repr(e))
        print(traceback.format_exc())
        return "<p>Análise indisponível (erro na IA).</p>"


# ==============================================================================
# 4) HTML + PDF
# ==============================================================================
def gerar_html_gerencial(dados: dict, texto_ia: str, img_path: Path, html_path: Path):
    img_src = img_path.name

    def make_rows(df, cols, fmt_map):
        rows = ""
        if df is None or df.empty:
            return rows
        for _, row in df.iterrows():
            rows += "<tr>"
            for col in cols:
                val = row.get(col, "")
                fmt = fmt_map.get(col, "{}")
                if isinstance(val, (int, float)):
                    val_str = fmt.format(val)
                else:
                    val_str = str(val)

                style = ""
                if col == "Variacao_Pct":
                    try:
                        v = float(val)
                    except Exception:
                        v = 0
                    if v < -5:
                        style = "color: #c0392b; font-weight: bold;"
                    elif v < 0:
                        style = "color: #e67e22;"
                    else:
                        style = "color: #27ae60;"
                    val_str = f"{v:.1f}%"

                rows += f"<td style='{style}'>{val_str}</td>"
            rows += "</tr>"
        return rows

    df_clean = dados["df_clean"].copy()
    if "Mes_Ano" not in df_clean.columns:
        df_clean["Mes_Ano"] = df_clean["Date"].dt.to_period("M")

    mensal = (
        df_clean.groupby("Mes_Ano")
        .agg({"Km": "sum", "Comb.": "sum"})
        .reset_index()
    )
    mensal["KML"] = mensal["Km"] / mensal["Comb."]
    mensal = mensal.sort_values("Mes_Ano")

    if len(mensal) >= 1:
        kml_mes_atual = float(mensal.iloc[-1]["KML"])
    else:
        kml_mes_atual = 0.0

    if len(mensal) >= 2:
        kml_mes_anterior = float(mensal.iloc[-2]["KML"])
        delta_kml_mes = ((kml_mes_atual - kml_mes_anterior) / kml_mes_anterior * 100) if kml_mes_anterior > 0 else 0
    else:
        delta_kml_mes = 0

    if delta_kml_mes < 0:
        texto_var = f"{abs(delta_kml_mes):.1f}% de QUEDA"
        cor_var = "#c0392b"
    elif delta_kml_mes > 0:
        texto_var = f"{delta_kml_mes:.1f}% de MELHORA"
        cor_var = "#27ae60"
    else:
        texto_var = "0,0% (Estável)"
        cor_var = "#7f8c8d"

    rows_veic = make_rows(
        dados["top_veiculos"],
        ["veiculo", "Cluster", "linha", "KML_Real", "KML_Meta", "Litros_Desperdicio"],
        {"KML_Real": "{:.2f}", "KML_Meta": "{:.2f}", "Litros_Desperdicio": "{:.0f}"},
    )

    rows_lin = make_rows(
        dados["top_linhas_queda"],
        ["linha", "KML_Anterior", "KML_Atual", "Variacao_Pct"],
        {"KML_Anterior": "{:.2f}", "KML_Atual": "{:.2f}", "Variacao_Pct": "{:.1f}"},
    )

    rows_mot = ""
    if dados["top_motoristas"] is not None and not dados["top_motoristas"].empty:
        for _, row in dados["top_motoristas"].iterrows():
            rows_mot += f"""
            <tr>
                <td>{row['Motorista']}</td>
                <td>{row['Cluster']}</td>
                <td>{row['linha']}</td>
                <td><b>{row['KML_Real']:.2f}</b></td>
                <td style="color:#777">{row['KML_Meta']:.2f}</td>
                <td><b>{row['Litros_Desperdicio']:.0f}</b></td>
                <td><span class="badge">{row['Impacto_Pct']:.1f}%</span></td>
            </tr>"""

    rows_cont = make_rows(
        dados["top_veiculos_contaminados"],
        ["veiculo", "Cluster", "linha", "Qtd_Contaminacoes", "KML_Min", "KML_Max"],
        {"Qtd_Contaminacoes": "{:.0f}", "KML_Min": "{:.2f}", "KML_Max": "{:.2f}"},
    )

    cob = dados.get("cobertura", {}) or {}

    html = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <title>Relatório Gerencial</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background-color: #f0f2f5; margin: 0; padding: 20px; }}
            .container {{ max-width: 1000px; margin: auto; background: white; padding: 40px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); border-radius: 8px; }}
            .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 3px solid #2c3e50; padding-bottom: 20px; margin-bottom: 18px; }}
            .title h1 {{ margin: 0; color: #2c3e50; text-transform: uppercase; letter-spacing: 1px; }}
            .month-card {{ background: #2c3e50; color: white; padding: 10px 20px; border-radius: 6px; text-align: center; }}
            .month-label {{ font-size: 10px; text-transform: uppercase; opacity: 0.8; }}
            .month-val {{ font-size: 18px; font-weight: bold; }}
            .kpi-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 18px; }}
            .kpi-card {{ background: #f8f9fa; padding: 18px; border-radius: 8px; text-align: center; border: 1px solid #e0e0e0; }}
            .kpi-val {{ display: block; font-size: 26px; font-weight: bold; }}
            .kpi-lbl {{ font-size: 12px; text-transform: uppercase; color: #666; letter-spacing: 1px; }}
            h2 {{ color: #2980b9; font-size: 18px; border-left: 5px solid #2980b9; padding-left: 10px; margin-top: 22px; margin-bottom: 14px; }}
            .ai-box {{ background-color: #fffde7; border: 1px solid #fbc02d; padding: 18px; border-radius: 6px; line-height: 1.6; color: #333; margin-bottom: 18px; }}
            .chart-box {{ text-align: center; margin-bottom: 18px; border: 1px solid #eee; padding: 10px; border-radius: 8px; }}
            .chart-box img {{ max-width: 100%; height: auto; }}
            .row-split {{ display: flex; gap: 22px; }}
            .col {{ flex: 1; }}
            table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
            th {{ background-color: #34495e; color: white; padding: 10px; text-align: left; }}
            td {{ border-bottom: 1px solid #eee; padding: 8px; vertical-align: top; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .badge {{ background: #e67e22; color: white; padding: 3px 8px; border-radius: 10px; font-size: 11px; font-weight: bold; }}
            .muted {{ font-size: 12px; color: #666; }}
            .footer {{ margin-top: 26px; text-align: center; font-size: 11px; color: #aaa; border-top: 1px solid #eee; padding-top: 14px; }}

            @page {{ size: A4; margin: 10mm; }}
            @media print {{
              html, body {{ background: #fff !important; padding: 0 !important; margin: 0 !important; }}
              .container {{ max-width: none !important; margin: 0 !important; padding: 0 !important; box-shadow: none !important; border-radius: 0 !important; }}
              .header, .kpi-grid {{ break-inside: avoid; page-break-inside: avoid; }}
              .row-split {{ display: block !important; }}
              .col {{ width: 100% !important; }}
              .kpi-card, .ai-box, .chart-box, table {{ break-inside: avoid; page-break-inside: avoid; }}
              h2 {{ break-after: avoid; page-break-after: avoid; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="title">
                    <h1>Relatório Gerencial</h1>
                    <div class="muted" style="margin-top:5px;">Eficiência Energética de Frota</div>
                    <div class="muted" style="margin-top:6px;">
                      <b>Período de Análise:</b> {dados['periodo']}
                    </div>
                </div>
                <div class="month-card">
                    <div class="month-label">MÊS DE REFERÊNCIA</div>
                    <div class="month-val">{dados['mes_atual_nome']}</div>
                </div>
            </div>

            <div class="kpi-grid">
                <div class="kpi-card">
                    <span class="kpi-val" style="color:#2c3e50">{kml_mes_atual:.2f}</span>
                    <span class="kpi-lbl">KM/L MÊS BASE (SEM CONTAMINAÇÕES)</span>
                </div>
                <div class="kpi-card">
                    <span class="kpi-val" style="color:{cor_var}">{texto_var}</span>
                    <span class="kpi-lbl">VARIAÇÃO VS MÊS ANTERIOR</span>
                </div>
                <div class="kpi-card">
                    <span class="kpi-val" style="color:#27ae60">AGENTE AI</span>
                    <span class="kpi-lbl">MONITORAMENTO ATIVO</span>
                </div>
            </div>

            <div class="muted" style="margin-bottom:16px;">
              <b>Cobertura (transparência):</b><br>
              Base bruta no range: {cob.get('bruto_min','N/D')} → {cob.get('bruto_max','N/D')} ({cob.get('qtd_bruto','?')} registros)<br>
              Removidos por cluster inválido: {cob.get('qtd_cluster_invalido','?')}<br>
              Removidos por contaminação (kml &lt; 1,5 ou &gt; 5): {cob.get('qtd_contaminacao','?')}<br>
              Base limpa analisada: {cob.get('clean_min','N/D')} → {cob.get('clean_max','N/D')} ({cob.get('qtd_clean','?')} registros)
            </div>

            <h2>1. Inteligência Executiva</h2>
            <div class="ai-box">{texto_ia}</div>

            <h2>2. Evolução de Eficiência (Clusters vs Média Frota)</h2>
            <div class="chart-box"><img src="{img_src}"></div>

            <div class="row-split">
                <div class="col">
                    <h2>3. Top 5 Veículos (Máquinas)</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Veículo</th><th>Cluster</th><th>Linha</th>
                                <th>Real</th><th>Ref</th><th>Perda (L)</th>
                            </tr>
                        </thead>
                        <tbody>{rows_veic}</tbody>
                    </table>
                </div>
                <div class="col">
                    <h2>4. Top 5 Linhas em Queda (Piora de KM/L)</h2>
                    <p class="muted">Comparativo Mês Atual vs Anterior.</p>
                    <table>
                        <thead>
                            <tr>
                                <th>Linha</th><th>Mês Ant.</th><th>Mês Atual</th><th>Variação</th>
                            </tr>
                        </thead>
                        <tbody>{rows_lin}</tbody>
                    </table>
                </div>
            </div>

            <h2>5. Fator Humano (Top 5 Motoristas Críticos)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Motorista</th><th>Cluster</th><th>Linha</th>
                        <th>Real</th><th>Ref</th><th>Perda (L)</th><th>Impacto (%)</th>
                    </tr>
                </thead>
                <tbody>{rows_mot}</tbody>
            </table>

            <h2>6. Top 10 Veículos com Leituras Contaminadas (Auditoria de Dados)</h2>
            <p class="muted">
                Veículos abaixo apresentam leituras de KM/L fora da faixa aceitável (kml &lt; 1,5 ou kml &gt; 5).
            </p>
            <table>
                <thead>
                    <tr>
                        <th>Veículo</th><th>Cluster</th><th>Linha</th>
                        <th>Qtd. Contaminações</th><th>KML Mínimo</th><th>KML Máximo</th>
                    </tr>
                </thead>
                <tbody>{rows_cont}</tbody>
            </table>

            <div class="footer">
                Relatório Gerado Automaticamente pelo Agente Diesel AI.<br>
                Período de Análise: {dados['periodo']}
            </div>
        </div>
    </body>
    </html>
    """
    html_path.write_text(html, encoding="utf-8")
    print(f"✅ HTML salvo: {html_path}")


def gerar_pdf_do_html(html_path: Path, pdf_path: Path):
    html_path = html_path.resolve()
    pdf_path = pdf_path.resolve()

    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1280, "height": 720})
        page.goto(html_path.as_uri(), wait_until="networkidle")
        page.wait_for_timeout(300)

        page.pdf(
            path=str(pdf_path),
            format="A4",
            print_background=True,
            margin={"top": "0mm", "right": "0mm", "bottom": "0mm", "left": "0mm"},
            prefer_css_page_size=True,
        )

        browser.close()

    print(f"✅ PDF (Chromium) salvo: {pdf_path}")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    _assert_env()
    Path(PASTA_SAIDA).mkdir(parents=True, exist_ok=True)

    periodo_inicio = _parse_iso(REPORT_PERIODO_INICIO)
    periodo_fim = _parse_iso(REPORT_PERIODO_FIM)

    if not periodo_inicio and not periodo_fim:
        hoje = datetime.utcnow().date()
        periodo_inicio = hoje.replace(day=1)
        periodo_fim = hoje

    atualizar_status_relatorio(
        "PROCESSANDO",
        tipo=REPORT_TIPO,
        periodo_inicio=str(periodo_inicio) if periodo_inicio else None,
        periodo_fim=str(periodo_fim) if periodo_fim else None,
    )

    try:
        df_base = carregar_dados_supabase_a(periodo_inicio, periodo_fim)
        dados = processar_dados_gerenciais_df(df_base, periodo_inicio, periodo_fim)

        mes_ref = str(dados["df_clean"]["Date"].max().to_period("M"))  # ex: 2026-02
        df_sug = gerar_sugestoes_acompanhamento(dados)
        salvar_sugestoes_supabase_b(df_sug, mes_ref, periodo_inicio, periodo_fim)

        out_dir = Path(PASTA_SAIDA)
        img_path = out_dir / "cluster_evolution_unificado.png"
        html_path = out_dir / "Relatorio_Gerencial.html"
        pdf_path = out_dir / "Relatorio_Gerencial.pdf"

        gerar_grafico_geral(dados["df_clean"], img_path)
        texto_ia = consultar_ia_gerencial(dados)  # ✅ agora não estoura stacktrace quando não há ADC
        gerar_html_gerencial(dados, texto_ia, img_path, html_path)
        gerar_pdf_do_html(html_path, pdf_path)

        base_folder = f"{REMOTE_BASE_PREFIX}/{mes_ref}/report_{REPORT_ID}"

        remote_img = f"{base_folder}/{img_path.name}"
        remote_html = f"{base_folder}/{html_path.name}"
        remote_pdf = f"{base_folder}/{pdf_path.name}"

        upload_storage_b(img_path, remote_img, "image/png")
        upload_storage_b(html_path, remote_html, "text/html; charset=utf-8")
        upload_storage_b(pdf_path, remote_pdf, "application/pdf")

        atualizar_status_relatorio(
            "CONCLUIDO",
            arquivo_pdf_path=remote_pdf,
            arquivo_html_path=remote_html,
            arquivo_png_path=remote_img,
            erro_msg=None,
            mes_ref=mes_ref,  # opcional (se a coluna existir)
        )

        print("✅ [OK] Relatório concluído e enviado para Supabase B (PDF + HTML + PNG).")

    except Exception as e:
        err = repr(e)
        print("❌ ERRO:", err)
        try:
            atualizar_status_relatorio("ERRO", erro_msg=err)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
