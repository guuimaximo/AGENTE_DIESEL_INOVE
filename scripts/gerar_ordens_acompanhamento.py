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

MOTORISTA = os.getenv("MOTORISTA", "30061012")
DIAS = int(os.getenv("DIAS", "30"))
PAGE_SIZE = int(os.getenv("PAGE_SIZE", "2000"))
EXPORT_CSV = os.getenv("EXPORT_CSV", "1") == "1"
SAIDA_DIR = os.getenv("SAIDA_DIR", "saida_30d")

# =========================
# Helpers
# =========================
def sb_a():
    if not SUPABASE_A_URL or not SUPABASE_A_SERVICE_ROLE_KEY:
        raise Exception("⚠️ ENV faltando: SUPABASE_A_URL / SUPABASE_A_SERVICE_ROLE_KEY")
    return create_client(SUPABASE_A_URL, SUPABASE_A_SERVICE_ROLE_KEY)

def to_num(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    # aceita decimal com vírgula
    s = s.str.replace(",", ".", regex=False)
    # remove lixo (mantém dígitos, ponto e sinal)
    s = s.str.replace(r"[^0-9\.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

# =========================
# Carregar dados (últimos N dias do motorista)
# =========================
def carregar_motorista(motorista: str, dias: int) -> pd.DataFrame:
    sb = sb_a()

    # folga pra garantir que fecha 30 dias mesmo se tiver buraco
    data_corte = (datetime.utcnow() - timedelta(days=dias + 10)).strftime("%Y-%m-%d")

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

    # Tipagem
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df["Km"] = to_num(df["Km"])
    df["Litros"] = to_num(df["Litros"])
    df["kml_db"] = to_num(df["kml_db"])

    # remove só datas inválidas
    df = df.dropna(subset=["Date"])

    # ancora no último dia real do motorista
    data_max = df["Date"].max()
    if pd.isna(data_max):
        return pd.DataFrame()

    ini = (data_max - timedelta(days=dias - 1)).normalize()
    df = df[(df["Date"] >= ini) & (df["Date"] <= data_max)].copy()

    # KML calculado (não usa filtro; não depende do "km/l" do banco)
    df["kml_calc"] = df.apply(
        lambda r: (r["Km"] / r["Litros"])
        if pd.notna(r["Km"]) and pd.notna(r["Litros"]) and r["Litros"] and r["Litros"] > 0
        else None,
        axis=1,
    )

    return df.sort_values("Date", ascending=True)

# =========================
# Consolida dia a dia (somatório do dia)
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
            linhas=("linha", lambda s: ", ".join(sorted(set(map(str, s.dropna()))))[:200]),
        )
        .reset_index()
    )

    out["kml_dia"] = out.apply(
        lambda r: (r["km"] / r["litros"]) if pd.notna(r["km"]) and pd.notna(r["litros"]) and r["litros"] and r["litros"] > 0 else None,
        axis=1,
    )

    return out.sort_values("Date", ascending=True)

# =========================
# Detalhe por dia + linha + veículo (se quiser ver “onde” no dia)
# =========================
def detalhar_por_dia_linha_veiculo(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce").dt.normalize()

    out = (
        d.groupby(["Date", "linha", "veiculo"], dropna=False)
        .agg(
            km=("Km", "sum"),
            litros=("Litros", "sum"),
        )
        .reset_index()
    )
    out["kml"] = out.apply(
        lambda r: (r["km"] / r["litros"]) if r["litros"] and r["litros"] > 0 else None,
        axis=1,
    )
    return out.sort_values(["Date", "linha", "veiculo"], ascending=True)

# =========================
# MAIN
# =========================
def main():
    print(f"🔎 Buscando últimos {DIAS} dias do motorista {MOTORISTA} na tabela {TABELA_ORIGEM}...")

    df_raw = carregar_motorista(MOTORISTA, DIAS)
    if df_raw.empty:
        print("⚠️ Nenhum dado encontrado para o período.")
        return

    data_min = df_raw["Date"].min().strftime("%Y-%m-%d")
    data_max = df_raw["Date"].max().strftime("%Y-%m-%d")

    # Consolidados
    df_dia = consolidar_dia_a_dia(df_raw)
    df_det = detalhar_por_dia_linha_veiculo(df_raw)

    # Resumo geral do período
    km_total = float(df_raw["Km"].sum(skipna=True))
    litros_total = float(df_raw["Litros"].sum(skipna=True))
    kml_periodo = (km_total / litros_total) if litros_total > 0 else None

    print("✅ Período ancorado no último dia real do motorista")
    print(f"📅 Janela: {data_min} → {data_max}")
    print(f"📌 Registros brutos: {len(df_raw)} | Dias: {df_dia['Date'].nunique()}")
    print(f"Σ KM: {km_total:.2f} | Σ Litros: {litros_total:.2f} | KM/L período: {kml_periodo:.3f}" if kml_periodo is not None else "KM/L período: N/A")

    if EXPORT_CSV:
        os.makedirs(SAIDA_DIR, exist_ok=True)

        p_raw = os.path.join(SAIDA_DIR, f"{MOTORISTA}_raw_{data_min}_a_{data_max}.csv")
        p_dia = os.path.join(SAIDA_DIR, f"{MOTORISTA}_dia_a_dia_{data_min}_a_{data_max}.csv")
        p_det = os.path.join(SAIDA_DIR, f"{MOTORISTA}_detalhe_dia_linha_veiculo_{data_min}_a_{data_max}.csv")

        df_raw.to_csv(p_raw, index=False, encoding="utf-8")
        df_dia.to_csv(p_dia, index=False, encoding="utf-8")
        df_det.to_csv(p_det, index=False, encoding="utf-8")

        print("📄 CSVs gerados:")
        print(f" - {p_raw}")
        print(f" - {p_dia}")
        print(f" - {p_det}")

    # imprime os 30 dias dia a dia no log (topo)
    print("\n📊 Dia a dia (últimos 30 dias):")
    preview = df_dia.copy()
    preview["Date"] = preview["Date"].dt.strftime("%Y-%m-%d")
    # mostra tudo (30 linhas), mas sem estourar log gigante
    for _, r in preview.iterrows():
        kml_txt = f"{r['kml_dia']:.3f}" if pd.notna(r["kml_dia"]) else "N/A"
        print(f"{r['Date']} | km={r['km']:.2f} | L={r['litros']:.2f} | kml={kml_txt} | linhas={r['linhas']} | veiculos={r['veiculos']}")

if __name__ == "__main__":
    main()
