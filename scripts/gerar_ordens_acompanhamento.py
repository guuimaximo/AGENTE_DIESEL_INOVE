import os
from datetime import datetime, timedelta
import pandas as pd
from supabase import create_client

# =========================
# ENV
# =========================
SUPABASE_A_URL = os.getenv("SUPABASE_A_URL")
SUPABASE_A_SERVICE_ROLE_KEY = os.getenv("SUPABASE_A_SERVICE_ROLE_KEY")
TABELA_ORIGEM = os.getenv("DIESEL_SOURCE_TABLE", "premiacao_diaria")

MOTORISTA = "30061012"
DIAS = 30
PAGE_SIZE = int(os.getenv("PAGE_SIZE", "2000"))
EXPORT_CSV = os.getenv("EXPORT_CSV", "1") == "1"

# =========================
# Helpers
# =========================
def sb_a():
    if not SUPABASE_A_URL or not SUPABASE_A_SERVICE_ROLE_KEY:
        raise Exception("⚠️ ENV faltando: SUPABASE_A_URL / SUPABASE_A_SERVICE_ROLE_KEY")
    return create_client(SUPABASE_A_URL, SUPABASE_A_SERVICE_ROLE_KEY)

def to_num(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(",", ".", regex=False)
    s = s.str.replace(r"[^0-9\.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

# =========================
# Carregar dados (últimos 30 dias)
# =========================
def carregar_ultimos_30_dias_motorista(motorista: str, dias: int = 30) -> pd.DataFrame:
    sb = sb_a()
    data_corte = (datetime.utcnow() - timedelta(days=dias + 5)).strftime("%Y-%m-%d")  # folga

    # pega colunas base (não confia no km/l do banco)
    sel = 'dia, motorista, veiculo, linha, "km/l", km_rodado, combustivel_consumido'

    all_rows = []
    start = 0

    while True:
        resp = (
            sb.table(TABELA_ORIGEM)
            .select(sel)
            .eq("motorista", motorista)
            .gte("dia", data_corte)
            .order("dia", desc=False)
            .range(start, start + PAGE_SIZE - 1)
            .execute()
        )

        rows = resp.data or []
        all_rows.extend(rows)

        if len(rows) < PAGE_SIZE:
            break

        start += PAGE_SIZE

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
            "combustivel_consumido": "Litros",
        },
        inplace=True,
    )

    # tipagem
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df["Km"] = to_num(df["Km"])
    df["Litros"] = to_num(df["Litros"])
    df["kml_db"] = to_num(df["kml_db"])  # só para referência (não usamos no cálculo)

    # remove só o que é impossível (data inválida)
    df = df.dropna(subset=["Date"])

    # ancora no último dia real do motorista
    data_max = df["Date"].max()
    ini = data_max - timedelta(days=dias - 1)
    df = df[(df["Date"] >= ini) & (df["Date"] <= data_max)].copy()

    # kml calculado (sem filtro)
    df["kml_calc"] = df.apply(
        lambda r: (r["Km"] / r["Litros"]) if pd.notna(r["Km"]) and pd.notna(r["Litros"]) and r["Litros"] and r["Litros"] > 0 else None,
        axis=1,
    )

    return df.sort_values("Date", ascending=True)

# =========================
# Agregar dia a dia (com linhas/veículos do dia)
# =========================
def consolidar_dia_a_dia(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = (
        df.groupby("Date", dropna=False)
        .agg(
            km=("Km", "sum"),
            litros=("Litros", "sum"),
            veiculos=("veiculo", lambda s: ", ".join(sorted(set(map(str, s.dropna()))))[:200]),
            linhas=("linha", lambda s: ", ".join
