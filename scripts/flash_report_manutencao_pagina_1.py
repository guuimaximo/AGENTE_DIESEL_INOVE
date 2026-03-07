# scripts/flash_report_manutencao.py
# ------------------------------------------------------------------------------
# FLASH REPORT MANUTENÇÃO - PÁGINAS 1, 2 E 3
# ------------------------------------------------------------------------------

import os
import re
from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from supabase import create_client
from playwright.sync_api import sync_playwright

# IA opcional
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel
    from google.auth.exceptions import DefaultCredentialsError
except Exception:
    vertexai = None
    GenerativeModel = None

    class DefaultCredentialsError(Exception):
        pass


# ==============================================================================
# CONFIG
# ==============================================================================
SUPABASE_B_URL = os.getenv("SUPABASE_B_URL")
SUPABASE_B_SERVICE_ROLE_KEY = os.getenv("SUPABASE_B_SERVICE_ROLE_KEY")

REPORT_ID = os.getenv("REPORT_ID", "")
REPORT_TIPO = os.getenv("REPORT_TIPO", "flash_report_manutencao")
REPORT_PERIODO_INICIO = os.getenv("REPORT_PERIODO_INICIO")
REPORT_PERIODO_FIM = os.getenv("REPORT_PERIODO_FIM")

PASTA_SAIDA = os.getenv("REPORT_OUTPUT_DIR", "Relatorios_Manutencao")
BUCKET_RELATORIOS = os.getenv("REPORT_BUCKET", "relatorios")
REMOTE_BASE_PREFIX = os.getenv("REPORT_REMOTE_PREFIX", "manutencao")

REPORT_PAGE_SIZE = int(os.getenv("REPORT_PAGE_SIZE", "1000"))
REPORT_MAX_ROWS = int(os.getenv("REPORT_MAX_ROWS", "200000"))

MKBF_META = float(os.getenv("MKBF_META", "7000"))

# IA opcional
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID") or os.getenv("PROJECT_ID")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
VERTEX_MODEL = os.getenv("VERTEX_MODEL", "gemini-2.5-pro")
VERTEX_SA_JSON = os.getenv("VERTEX_SA_JSON")


# ==============================================================================
# HELPERS BASE
# ==============================================================================
def _assert_env():
    missing = []
    for k in ["SUPABASE_B_URL", "SUPABASE_B_SERVICE_ROLE_KEY"]:
        if not os.getenv(k):
            missing.append(k)
    if missing:
        raise RuntimeError(f"Variáveis obrigatórias ausentes: {missing}")


def _parse_iso(d: str | None) -> date | None:
    if not d:
        return None
    return datetime.strptime(d, "%Y-%m-%d").date()


def _sb_b():
    return create_client(SUPABASE_B_URL, SUPABASE_B_SERVICE_ROLE_KEY)


def atualizar_status_relatorio(status: str, **fields):
    if not REPORT_ID:
        return
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


def _fmt_int(v):
    try:
        return f"{int(round(float(v))):,}".replace(",", ".")
    except Exception:
        return "0"


def _fmt_pct(v):
    try:
        return f"{float(v):+.1f}%"
    except Exception:
        return "0,0%"


def _parse_number(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None

    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
        try:
            return float(s)
        except Exception:
            return None

    if "," in s and "." not in s:
        s = s.replace(",", ".")
        try:
            return float(s)
        except Exception:
            return None

    try:
        return float(s)
    except Exception:
        pass

    s = re.sub(r"[^0-9\.\-]", "", s)
    try:
        return float(s)
    except Exception:
        return None


def _to_num(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    return series.apply(_parse_number).astype("float64")


def month_start(d: date) -> date:
    return d.replace(day=1)


def month_end(d: date) -> date:
    if d.month == 12:
        return d.replace(day=31)
    return d.replace(month=d.month + 1, day=1) - timedelta(days=1)


def add_months(d: date, months: int) -> date:
    y = d.year + ((d.month - 1 + months) // 12)
    m = ((d.month - 1 + months) % 12) + 1
    day = min(d.day, month_end(date(y, m, 1)).day)
    return date(y, m, day)


def pt_month_name(d: date) -> str:
    meses = {
        1: "JANEIRO", 2: "FEVEREIRO", 3: "MARÇO", 4: "ABRIL",
        5: "MAIO", 6: "JUNHO", 7: "JULHO", 8: "AGOSTO",
        9: "SETEMBRO", 10: "OUTUBRO", 11: "NOVEMBRO", 12: "DEZEMBRO",
    }
    return meses[d.month]


def periodo_label(d_ini: date, d_fim: date) -> str:
    return f"{_fmt_br_date(d_ini)} a {_fmt_br_date(d_fim)}"


def cls_var(v):
    try:
        v = float(v)
    except Exception:
        v = 0
    if v > 0:
        return "#16a34a"
    if v < 0:
        return "#dc2626"
    return "#64748b"


def _ensure_vertex_adc_if_possible():
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        return
    if not VERTEX_SA_JSON:
        return
    try:
        tmp = Path("/tmp/vertex_sa.json")
        tmp.write_text(VERTEX_SA_JSON, encoding="utf-8")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(tmp)
    except Exception as e:
        print("⚠️ Falha ao montar ADC:", repr(e))


def definir_cluster_manutencao(veiculo):
    v = str(veiculo or "").strip().upper()
    if not v:
        return "OUTROS"
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


# ==============================================================================
# REGRA DE OCORRÊNCIA
# ==============================================================================
TIPOS_GRAFICO = ["RECOLHEU", "SOS", "AVARIA", "TROCA", "IMPROCEDENTE"]

def normalize_tipo(oc):
    o = str(oc or "").upper().strip()
    if not o:
        return ""
    if o in ("RA", "R.A", "R.A."):
        return "RECOLHEU"
    if "RECOLH" in o:
        return "RECOLHEU"
    if "IMPRO" in o:
        return "IMPROCEDENTE"
    if "TROC" in o:
        return "TROCA"
    if o == "S.O.S":
        return "SOS"
    if "AVARI" in o:
        return "AVARIA"
    if "SEGUIU" in o:
        return "SEGUIU VIAGEM"
    return o


def is_ocorrencia_valida_para_mkbf(oc):
    tipo = normalize_tipo(oc)
    if not tipo:
        return False
    if tipo == "SEGUIU VIAGEM":
        return False
    return True


# ==============================================================================
# BUSCA DADOS
# ==============================================================================
def fetch_all_table_period(
    table_name: str, select_fields: str, date_field: str, start_date: date, end_date: date
) -> list[dict]:
    sb = _sb_b()
    all_rows = []
    offset = 0

    while True:
        end = offset + REPORT_PAGE_SIZE - 1
        resp = (
            sb.table(table_name)
            .select(select_fields)
            .gte(date_field, str(start_date))
            .lte(date_field, str(end_date))
            .order(date_field, desc=False)
            .range(offset, end)
            .execute()
        )
        rows = resp.data or []
        all_rows.extend(rows)

        print(f"📦 [{table_name}] range={offset}-{end} fetched={len(rows)} total={len(all_rows)}")

        if len(rows) < REPORT_PAGE_SIZE:
            break
        if len(all_rows) >= REPORT_MAX_ROWS:
            all_rows = all_rows[:REPORT_MAX_ROWS]
            print(f"⚠️ [{table_name}] REPORT_MAX_ROWS atingido: {REPORT_MAX_ROWS}")
            break

        offset += REPORT_PAGE_SIZE
    return all_rows


def carregar_km_rodado(periodo_inicio: date, periodo_fim: date) -> pd.DataFrame:
    rows = fetch_all_table_period(
        table_name="km_rodado_diario", select_fields="data, km_total", date_field="data",
        start_date=periodo_inicio, end_date=periodo_fim,
    )
    if not rows:
        return pd.DataFrame(columns=["data", "km_total"])
    return pd.DataFrame(rows)


def carregar_sos(periodo_inicio: date, periodo_fim: date) -> pd.DataFrame:
    rows = fetch_all_table_period(
        table_name="sos_acionamentos", select_fields="id, numero_sos, data_sos, ocorrencia, status",
        date_field="data_sos", start_date=periodo_inicio, end_date=periodo_fim,
    )
    if not rows:
        return pd.DataFrame(columns=["id", "numero_sos", "data_sos", "ocorrencia", "status"])
    return pd.DataFrame(rows)


def carregar_sos_pagina_3(periodo_inicio: date, periodo_fim: date) -> pd.DataFrame:
    rows = fetch_all_table_period(
        table_name="sos_acionamentos",
        select_fields="id, numero_sos, data_sos, hora_sos, veiculo, linha, ocorrencia, status",
        date_field="data_sos",
        start_date=periodo_inicio,
        end_date=periodo_fim,
    )
    if not rows:
        return pd.DataFrame(
            columns=["id", "numero_sos", "data_sos", "hora_sos", "veiculo", "linha", "ocorrencia", "status"]
        )
    return pd.DataFrame(rows)


# ==============================================================================
# PROCESSAMENTO - PÁGINAS 1 E 2
# ==============================================================================
def processar_km_df(df_km: pd.DataFrame) -> pd.DataFrame:
    df = df_km.copy()
    if df.empty:
        return pd.DataFrame(columns=["data", "km_total"])
    df["data"] = pd.to_datetime(df["data"], errors="coerce").dt.date
    df["km_total"] = _to_num(df["km_total"]).fillna(0)
    df = (df.dropna(subset=["data"]).groupby("data", as_index=False)
          .agg(km_total=("km_total", "sum")).sort_values("data"))
    return df


def processar_sos_df(df_sos: pd.DataFrame) -> pd.DataFrame:
    df = df_sos.copy()
    if df.empty:
        return pd.DataFrame(columns=["id", "numero_sos", "data_sos", "ocorrencia", "status", "tipo_norm", "valida_mkbf"])
    df["data_sos"] = pd.to_datetime(df["data_sos"], errors="coerce").dt.date
    df["tipo_norm"] = df["ocorrencia"].apply(normalize_tipo)
    df["valida_mkbf"] = df["ocorrencia"].apply(is_ocorrencia_valida_para_mkbf)
    return df.dropna(subset=["data_sos"]).copy()


def montar_diario_mkbf(df_km_proc: pd.DataFrame, df_sos_proc: pd.DataFrame) -> pd.DataFrame:
    km_dia = df_km_proc.copy()
    if df_sos_proc.empty:
        ocorr_dia = pd.DataFrame(columns=["data", "intervencoes"])
    else:
        ocorr_dia = (df_sos_proc[df_sos_proc["valida_mkbf"]].groupby("data_sos", as_index=False)
                     .size().rename(columns={"data_sos": "data", "size": "intervencoes"}))
    diario = km_dia.merge(ocorr_dia, on="data", how="left")
    diario["intervencoes"] = diario["intervencoes"].fillna(0).astype(int)
    diario["mkbf"] = diario.apply(
        lambda r: (r["km_total"] / r["intervencoes"]) if r["intervencoes"] > 0 else 0, axis=1,
    )
    return diario.sort_values("data").reset_index(drop=True)


def resumo_periodo(df_diario: pd.DataFrame, df_sos_proc: pd.DataFrame) -> dict:
    km_total = float(df_diario["km_total"].sum()) if not df_diario.empty else 0.0
    interv = int(df_diario["intervencoes"].sum()) if not df_diario.empty else 0
    mkbf = (km_total / interv) if interv > 0 else 0.0
    por_tipo = (df_sos_proc[df_sos_proc["tipo_norm"].isin(TIPOS_GRAFICO)]
                .groupby("tipo_norm", as_index=False).size()
                .rename(columns={"size": "total"}).sort_values("total", ascending=False))
    por_tipo_map = {r["tipo_norm"]: int(r["total"]) for _, r in por_tipo.iterrows()}
    for t in TIPOS_GRAFICO:
        por_tipo_map.setdefault(t, 0)
    return {
        "km_total": km_total, "intervencoes": interv, "mkbf": mkbf,
        "por_tipo_map": por_tipo_map, "por_tipo_df": por_tipo,
    }


def processar_pagina_1(periodo_inicio: date, periodo_fim: date) -> dict:
    df_km_atual = carregar_km_rodado(periodo_inicio, periodo_fim)
    df_sos_atual = carregar_sos(periodo_inicio, periodo_fim)

    km_atual = processar_km_df(df_km_atual)
    sos_atual = processar_sos_df(df_sos_atual)
    diario_atual = montar_diario_mkbf(km_atual, sos_atual)
    resumo_atual = resumo_periodo(diario_atual, sos_atual)

    ini_ant = month_start(add_months(periodo_inicio, -1))
    fim_ant = month_end(ini_ant)

    df_km_ant = carregar_km_rodado(ini_ant, fim_ant)
    df_sos_ant = carregar_sos(ini_ant, fim_ant)

    km_ant = processar_km_df(df_km_ant)
    sos_ant = processar_sos_df(df_sos_ant)
    diario_ant = montar_diario_mkbf(km_ant, sos_ant)
    resumo_ant = resumo_periodo(diario_ant, sos_ant)

    hist_rows = []
    mes_base = month_start(periodo_fim)
    meses_hist = [add_months(mes_base, -i) for i in range(11, -1, -1)]

    for mes_dt in meses_hist:
        ini_m = month_start(mes_dt)
        fim_m = month_end(mes_dt)
        df_km_m = processar_km_df(carregar_km_rodado(ini_m, fim_m))
        df_sos_m = processar_sos_df(carregar_sos(ini_m, fim_m))
        diario_m = montar_diario_mkbf(df_km_m, df_sos_m)
        resumo_m = resumo_periodo(diario_m, df_sos_m)
        hist_rows.append({
            "mes_dt": ini_m,
            "mes_label": f"{pt_month_name(ini_m)[:3]}/{str(ini_m.year)[2:]}",
            "mkbf": float(resumo_m["mkbf"]),
            "meta": float(MKBF_META),
            "km_total": float(resumo_m["km_total"]),
            "intervencoes": int(resumo_m["intervencoes"]),
        })

    df_hist = pd.DataFrame(hist_rows)

    tipos_all = sorted(set(TIPOS_GRAFICO) | set(resumo_atual["por_tipo_map"]) | set(resumo_ant["por_tipo_map"]))
    rows_motivos = []
    for t in tipos_all:
        rows_motivos.append({
            "tipo": t,
            "mes_anterior": int(resumo_ant["por_tipo_map"].get(t, 0)),
            "mes_atual": int(resumo_atual["por_tipo_map"].get(t, 0)),
        })

    df_motivos = pd.DataFrame(rows_motivos).sort_values(["mes_atual", "mes_anterior", "tipo"], ascending=[False, False, True])

    def pct_var(atual, anterior):
        if anterior in (0, None): return 0.0
        return ((atual - anterior) / anterior) * 100.0

    variacoes = {
        "intervencoes_pct": pct_var(resumo_atual["intervencoes"], resumo_ant["intervencoes"]),
        "km_pct": pct_var(resumo_atual["km_total"], resumo_ant["km_total"]),
        "mkbf_pct": pct_var(resumo_atual["mkbf"], resumo_ant["mkbf"]),
    }

    aderencia_meta_pct = (resumo_atual["mkbf"] / MKBF_META * 100.0) if MKBF_META > 0 else 0.0

    return {
        "periodo_inicio": periodo_inicio, "periodo_fim": periodo_fim,
        "periodo_label": periodo_label(periodo_inicio, periodo_fim),
        "mes_atual_label": f"{pt_month_name(periodo_fim)}/{periodo_fim.year}",
        "mes_anterior_label": f"{pt_month_name(ini_ant)}/{ini_ant.year}",
        "resumo_atual": resumo_atual, "resumo_ant": resumo_ant,
        "df_diario_atual": diario_atual, "df_hist": df_hist,
        "df_motivos": df_motivos, "variacoes": variacoes,
        "aderencia_meta_pct": aderencia_meta_pct,
    }


def processar_pagina_2(periodo_inicio: date, periodo_fim: date, diario: pd.DataFrame, resumo_atual: dict) -> dict:
    df = diario.copy()
    if df.empty:
        df = pd.DataFrame(columns=["data", "km_total", "intervencoes", "mkbf"])
        df["dia"] = []
    else:
        df["dia"] = pd.to_datetime(df["data"]).dt.strftime("%d/%m")

    total_interv = int(resumo_atual.get("intervencoes", 0))
    total_km = float(resumo_atual.get("km_total", 0.0))
    mkbf_mes = float(resumo_atual.get("mkbf", 0.0))

    dias_periodo = (periodo_fim - periodo_inicio).days + 1
    dias_decorridos = len(df)
    media_interv_dia = (total_interv / dias_decorridos) if dias_decorridos > 0 else 0.0

    meta_interv_mes = (total_km / MKBF_META) if MKBF_META > 0 else 0.0
    meta_interv_dia = (meta_interv_mes / dias_periodo) if dias_periodo > 0 else 0.0
    delta_interv = total_interv - meta_interv_mes

    ultimo_dia_mes = month_end(periodo_fim)
    dias_mes = ultimo_dia_mes.day
    dia_atual_mes = min(periodo_fim.day, dias_mes)

    proj_km_mes = (total_km / dia_atual_mes) * dias_mes if dia_atual_mes > 0 else total_km
    proj_interv_mes = (total_interv / dia_atual_mes) * dias_mes if dia_atual_mes > 0 else total_interv

    return {
        "periodo_inicio": periodo_inicio,
        "periodo_fim": periodo_fim,
        "periodo_label": periodo_label(periodo_inicio, periodo_fim),
        "mes_atual_label": f"{pt_month_name(periodo_fim)}/{periodo_fim.year}",
        "diario": df,
        "total_interv": total_interv,
        "total_km": total_km,
        "mkbf_mes": mkbf_mes,
        "media_interv_dia": media_interv_dia,
        "meta_interv_mes": meta_interv_mes,
        "meta_interv_dia": meta_interv_dia,
        "delta_interv": delta_interv,
        "proj_km_mes": proj_km_mes,
        "proj_interv_mes": proj_interv_mes,
        "dias_periodo": dias_periodo,
        "dias_decorridos": dias_decorridos,
    }


# ==============================================================================
# PROCESSAMENTO - PÁGINA 3
# ==============================================================================
def processar_sos_pagina_3(df_sos: pd.DataFrame) -> pd.DataFrame:
    df = df_sos.copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "id", "numero_sos", "data_sos", "hora_sos", "veiculo",
                "linha", "ocorrencia", "status", "tipo_norm",
                "valida_mkbf", "hora_int", "cluster",
            ]
        )

    df["data_sos"] = pd.to_datetime(df["data_sos"], errors="coerce").dt.date
    df["tipo_norm"] = df["ocorrencia"].apply(normalize_tipo)
    df["valida_mkbf"] = df["ocorrencia"].apply(is_ocorrencia_valida_para_mkbf)

    def extrair_hora(v):
        s = str(v or "").strip()
        if not s: return None
        try: return int(s[:2])
        except: return None

    df["hora_int"] = df["hora_sos"].apply(extrair_hora)
    df["linha"] = df["linha"].astype(str).str.strip().replace({"": "N/D"})
    df["veiculo"] = df["veiculo"].astype(str).str.strip().replace({"": "N/D"})
    df["cluster"] = df["veiculo"].apply(definir_cluster_manutencao)

    df = df.dropna(subset=["data_sos"]).copy()
    return df


def processar_pagina_3(periodo_fim: date) -> dict:
    # Restringe a análise para os últimos 3 meses
    periodo_fim = month_end(periodo_fim)
    periodo_inicio = month_start(add_months(periodo_fim, -2))

    df_sos = carregar_sos_pagina_3(periodo_inicio, periodo_fim)
    sos_proc = processar_sos_pagina_3(df_sos)
    base = sos_proc[sos_proc["valida_mkbf"]].copy()

    meses_lista = [month_start(add_months(periodo_fim, -i)) for i in range(2, -1, -1)]
    meses_labels = [f"{pt_month_name(m)[:3]}/{str(m.year)[2:]}" for m in meses_lista]
    mes_ref_label = meses_labels[-1]

    if base.empty:
        vazio_linha = pd.DataFrame(columns=["linha", "int_total", "int_ref"])
        vazio_hora = pd.DataFrame(columns=["hora_int", "int_total", "int_ref"])
        vazio_carro = pd.DataFrame(columns=["veiculo", "int_total", "int_ref"])
        vazio_cluster = pd.DataFrame(columns=["cluster", "frota_ref", "int_veiculo_ref"])
        for m in meses_labels: vazio_cluster[m] = 0

        return {
            "periodo_inicio": periodo_inicio, "periodo_fim": periodo_fim,
            "periodo_label": periodo_label(periodo_inicio, periodo_fim),
            "mes_ref_label": mes_ref_label,
            "meses_labels": meses_labels,
            "total_interv": 0,
            "total_interv_ref": 0,
            "df_linha": vazio_linha,
            "df_horario": vazio_hora,
            "df_top_carro": vazio_carro,
            "df_cluster": vazio_cluster,
        }

    base['is_ref'] = base['data_sos'].apply(lambda d: d.year == periodo_fim.year and d.month == periodo_fim.month)
    base['mes_label'] = base['data_sos'].apply(lambda d: f"{pt_month_name(d)[:3]}/{str(d.year)[2:]}")

    # Sempre focar/ordenar pelo mês de referência
    df_linha = base.groupby("linha").agg(
        int_total=('id', 'count'),
        int_ref=('is_ref', 'sum')
    ).reset_index().sort_values(["int_ref", "int_total"], ascending=[False, False]).head(14)

    df_horario = base.dropna(subset=["hora_int"]).groupby("hora_int").agg(
        int_total=('id', 'count'),
        int_ref=('is_ref', 'sum')
    ).reset_index().sort_values(["hora_int"], ascending=[True])

    df_top_carro = base.groupby("veiculo").agg(
        int_total=('id', 'count'),
        int_ref=('is_ref', 'sum')
    ).reset_index().sort_values(["int_ref", "int_total"], ascending=[False, False]).head(10)

    # Cross tab (pivot) para a tabela de Clusters
    pivot = base.pivot_table(index="cluster", columns="mes_label", values="id", aggfunc="count", fill_value=0)
    for m_lbl in meses_labels:
        if m_lbl not in pivot.columns:
            pivot[m_lbl] = 0
    pivot = pivot.reset_index()

    # Frota considera os veículos no mês de referência
    frota_ref = base[base['is_ref']].groupby('cluster')['veiculo'].nunique().reset_index().rename(columns={'veiculo': 'frota_ref'})
    df_cluster = pd.merge(pivot, frota_ref, on='cluster', how='left').fillna(0)
    df_cluster['frota_ref'] = df_cluster['frota_ref'].astype(int)

    df_cluster['int_veiculo_ref'] = df_cluster.apply(
        lambda r: (r[mes_ref_label] / r['frota_ref']) if r['frota_ref'] > 0 else 0.0, axis=1
    )
    df_cluster = df_cluster.sort_values(by=mes_ref_label, ascending=False).reset_index(drop=True)

    return {
        "periodo_inicio": periodo_inicio,
        "periodo_fim": periodo_fim,
        "periodo_label": periodo_label(periodo_inicio, periodo_fim),
        "mes_ref_label": mes_ref_label,
        "meses_labels": meses_labels,
        "total_interv": len(base),
        "total_interv_ref": int(base['is_ref'].sum()),
        "df_linha": df_linha,
        "df_horario": df_horario,
        "df_top_carro": df_top_carro,
        "df_cluster": df_cluster,
    }


# ==============================================================================
# GRÁFICOS
# ==============================================================================
def gerar_grafico_mkbf_historico(df_hist: pd.DataFrame, caminho_img: Path):
    df = df_hist.copy()
    if df.empty:
        plt.figure(figsize=(10.5, 4.8))
        plt.title("Evolução Histórica do MKBF", fontsize=12, fontweight="bold")
        plt.text(0.5, 0.5, "Sem dados para exibição", ha="center", va="center", transform=plt.gca().transAxes)
        plt.tight_layout()
        plt.savefig(caminho_img, dpi=120)
        plt.close()
        return

    x = range(len(df))
    y_mkbf = df["mkbf"].tolist()
    y_meta = df["meta"].tolist()

    plt.figure(figsize=(11.5, 5.0))
    
    plt.plot(x, y_mkbf, marker="o", markersize=6, linewidth=2.5, label="MKBF", color="#2563eb")
    plt.fill_between(x, y_mkbf, alpha=0.1, color="#3b82f6")
    plt.plot(x, y_meta, linewidth=2, linestyle="--", label="Meta", color="#ef4444")

    offset_val = max(max(y_mkbf + y_meta) * 0.05, 100) if (y_mkbf + y_meta) else 100
    
    for i, v in enumerate(y_mkbf):
        plt.text(i, v + offset_val * 0.2, f"{v:,.0f}".replace(",", "."), ha="center", va="bottom", fontsize=8, fontweight="bold", color="#1e293b")
        if i > 0:
            prev = y_mkbf[i-1]
            if prev > 0:
                pct = ((v - prev) / prev) * 100
                color = "#16a34a" if pct > 0 else ("#dc2626" if pct < 0 else "#64748b")
                sinal = "+" if pct > 0 else ""
                plt.text(i, v - offset_val * 0.4, f"{sinal}{pct:.1f}%", ha="center", va="top", fontsize=7, fontweight="bold", color=color)

    plt.xticks(list(x), df["mes_label"].tolist(), fontsize=8, rotation=0)
    plt.yticks(fontsize=8)
    plt.ylabel("MKBF", fontsize=9)
    plt.title("Evolução Histórica do MKBF", fontsize=12, fontweight="bold", color="#0f172a")
    
    plt.grid(True, linestyle=":", alpha=0.6, axis="y")
    
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")

    plt.legend(loc="upper left", fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(caminho_img, dpi=140, bbox_inches='tight')
    plt.close()


def gerar_grafico_pagina_2(dados: dict, caminho_img: Path):
    df = dados["diario"].copy()

    if df.empty:
        plt.figure(figsize=(11.5, 4.8))
        plt.title("Evolução Diária: Intervenções e MKBF", fontsize=12, fontweight="bold")
        plt.text(0.5, 0.5, "Sem dados para exibição", ha="center", va="center", transform=plt.gca().transAxes)
        plt.tight_layout()
        plt.savefig(caminho_img, dpi=140)
        plt.close()
        return

    fig, ax1 = plt.subplots(figsize=(11.5, 4.8))

    x = range(len(df))
    labels = df["dia"].tolist()
    interv = df["intervencoes"].tolist()
    mkbf = df["mkbf"].tolist()

    bars = ax1.bar(x, interv, label="Intervenções", color="#3b82f6", alpha=0.85, width=0.6)
    ax1.set_ylabel("Intervenções", fontsize=9)
    ax1.set_xticks(list(x))
    
    ax1.set_xticklabels(labels, rotation=45, fontsize=7, ha="right")
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)

    max_interv = max(interv) if interv else 1
    ax1.set_ylim(0, max_interv * 1.5)

    for i, v in enumerate(interv):
        if v > 0:
            ax1.text(i, v + (max_interv * 0.02), str(v), ha="center", va="bottom", fontsize=7, fontweight="bold", color="#0f172a")

    ax2 = ax1.twinx()
    ax2.plot(x, mkbf, marker="o", markersize=4, linewidth=2.0, label="MKBF diário", color="#ef4444")
    ax2.set_ylabel("MKBF", fontsize=9)

    max_mkbf = max(mkbf) if mkbf else 1
    min_mkbf = min(mkbf) if mkbf else 0
    ax2.set_ylim(min_mkbf * 0.5, max_mkbf * 1.15)

    for i, v in enumerate(mkbf):
        if v > 0:
            ax2.text(i, v + (max_mkbf * 0.02), f"{v:,.0f}".replace(",", "."), ha="center", va="bottom", fontsize=6.5, color="#dc2626")

    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    plt.title("Evolução Diária: Intervenções e MKBF", fontsize=12, fontweight="bold", color="#0f172a")
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", fontsize=8, frameon=False)

    fig.tight_layout()
    plt.savefig(caminho_img, dpi=140, bbox_inches='tight')
    plt.close()


def gerar_grafico_pagina_3_linha(df_linha: pd.DataFrame, caminho_img: Path):
    df = df_linha.copy()
    plt.figure(figsize=(11.5, 3.2))
    
    if df.empty:
        plt.title("Ocorrências por Linha", fontsize=12, fontweight="bold")
        plt.text(0.5, 0.5, "Sem dados", ha="center", va="center", transform=plt.gca().transAxes)
        plt.tight_layout()
        plt.savefig(caminho_img, dpi=140, transparent=True)
        plt.close()
        return

    x = range(len(df))
    x_total = [i - 0.2 for i in x]
    x_ref = [i + 0.2 for i in x]
    
    y_total = df["int_total"].tolist()
    y_ref = df["int_ref"].tolist()
    labels = df["linha"].tolist()

    plt.bar(x_total, y_total, width=0.4, color="#cbd5e1", label="Acumulado (3 Meses)")
    plt.bar(x_ref, y_ref, width=0.4, color="#1e3a8a", label="Mês Atual")

    max_y = max(y_total) if y_total else 1
    plt.ylim(0, max_y * 1.3)

    for i, v in enumerate(y_total):
        if v > 0:
            plt.text(x_total[i], v + (max_y * 0.02), str(int(v)), ha="center", va="bottom", fontsize=7, color="#64748b")
    for i, v in enumerate(y_ref):
        if v > 0:
            plt.text(x_ref[i], v + (max_y * 0.02), str(int(v)), ha="center", va="bottom", fontsize=8, fontweight="bold", color="#0f172a")

    plt.xticks(list(x), labels, rotation=0, fontsize=8)
    plt.yticks([])
    plt.title("Top 14 - Ocorrências por Linha", fontsize=11, fontweight="bold", color="#0f172a")
    plt.legend(loc="upper right", fontsize=8, frameon=False)
    
    for spine in ["top", "right", "left", "bottom"]:
        plt.gca().spines[spine].set_visible(False)
        
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(caminho_img, dpi=140, transparent=True)
    plt.close()


def gerar_grafico_pagina_3_horario(df_horario: pd.DataFrame, caminho_img: Path):
    df = df_horario.copy()
    plt.figure(figsize=(11.5, 3.2))

    if df.empty:
        plt.title("Ocorrências por Horário", fontsize=12, fontweight="bold")
        plt.text(0.5, 0.5, "Sem dados", ha="center", va="center", transform=plt.gca().transAxes)
        plt.tight_layout()
        plt.savefig(caminho_img, dpi=140, transparent=True)
        plt.close()
        return

    x = range(len(df))
    x_total = [i - 0.2 for i in x]
    x_ref = [i + 0.2 for i in x]
    
    y_total = df["int_total"].tolist()
    y_ref = df["int_ref"].tolist()
    labels = [f"{int(h):02d}h" for h in df["hora_int"].tolist()]

    plt.bar(x_total, y_total, width=0.4, color="#cbd5e1", label="Acumulado (3 Meses)")
    plt.bar(x_ref, y_ref, width=0.4, color="#3b82f6", label="Mês Atual")

    max_y = max(y_total) if y_total else 1
    plt.ylim(0, max_y * 1.3)

    for i, v in enumerate(y_total):
        if v > 0:
            plt.text(x_total[i], v + (max_y * 0.02), str(int(v)), ha="center", va="bottom", fontsize=7, color="#64748b")
    for i, v in enumerate(y_ref):
        if v > 0:
            plt.text(x_ref[i], v + (max_y * 0.02), str(int(v)), ha="center", va="bottom", fontsize=8, fontweight="bold", color="#0f172a")

    plt.xticks(list(x), labels, fontsize=8)
    plt.yticks([])
    plt.title("Volume de Ocorrências por Faixa de Horário", fontsize=11, fontweight="bold", color="#0f172a")
    plt.legend(loc="upper right", fontsize=8, frameon=False)
    
    for spine in ["top", "right", "left", "bottom"]:
        plt.gca().spines[spine].set_visible(False)
        
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(caminho_img, dpi=140, transparent=True)
    plt.close()


def gerar_grafico_pagina_3_top_carro(df_top_carro: pd.DataFrame, caminho_img: Path):
    df = df_top_carro.copy()
    plt.figure(figsize=(5.5, 4.0))
    
    if df.empty:
        plt.title("Top 10 Carros", fontsize=11, fontweight="bold")
        plt.text(0.5, 0.5, "Sem dados", ha="center", va="center", transform=plt.gca().transAxes)
        plt.tight_layout()
        plt.savefig(caminho_img, dpi=140, transparent=True)
        plt.close()
        return

    x = range(len(df))
    x_total = [i - 0.2 for i in x]
    x_ref = [i + 0.2 for i in x]
    
    y_total = df["int_total"].tolist()
    y_ref = df["int_ref"].tolist()
    labels = df["veiculo"].tolist()

    plt.bar(x_total, y_total, width=0.4, color="#cbd5e1", label="Acumulado (3 Meses)")
    plt.bar(x_ref, y_ref, width=0.4, color="#ef4444", label="Mês Atual")

    max_y = max(y_total) if y_total else 1
    plt.ylim(0, max_y * 1.3)

    for i, v in enumerate(y_total):
        if v > 0:
            plt.text(x_total[i], v + (max_y * 0.02), str(int(v)), ha="center", va="bottom", fontsize=7, color="#64748b")
    for i, v in enumerate(y_ref):
        if v > 0:
            plt.text(x_ref[i], v + (max_y * 0.02), str(int(v)), ha="center", va="bottom", fontsize=8, fontweight="bold", color="#0f172a")

    plt.xticks(list(x), labels, rotation=45, ha="right", fontsize=8)
    plt.yticks([])
    plt.legend(loc="upper right", fontsize=8, frameon=False)
    
    for spine in ["top", "right", "left", "bottom"]:
        plt.gca().spines[spine].set_visible(False)
        
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(caminho_img, dpi=140, transparent=True)
    plt.close()


# ==============================================================================
# CONSIDERAÇÕES IA E FALLBACK
# ==============================================================================
def gerar_consideracoes_fallback_p1(dados: dict) -> str:
    atual = dados["resumo_atual"]
    var = dados["variacoes"]
    ader = dados["aderencia_meta_pct"]

    top_motivos = dados["df_motivos"].head(3).copy()
    motivos_txt = ", ".join(
        [f"{r['tipo']} ({int(r['mes_atual'])})" for _, r in top_motivos.iterrows() if int(r["mes_atual"]) > 0]
    )
    if not motivos_txt:
        motivos_txt = "sem concentração relevante de ofensores"

    tendencia_mkbf = "melhora" if var["mkbf_pct"] > 0 else ("queda" if var["mkbf_pct"] < 0 else "estabilidade")
    meta_txt = "acima" if atual["mkbf"] >= MKBF_META else "abaixo"

    return (
        f"No período de {dados['periodo_label']}, a operação registrou "
        f"{atual['intervencoes']} intervenções válidas para MKBF, com "
        f"{_fmt_int(atual['km_total'])} km rodados e MKBF de {_fmt_int(atual['mkbf'])}. "
        f"Na comparação com {dados['mes_anterior_label']}, houve {tendencia_mkbf} no indicador, "
        f"com variação de {_fmt_pct(var['mkbf_pct'])}. "
        f"A aderência frente à meta de {_fmt_int(MKBF_META)} ficou em {ader:.1f}%, "
        f"mantendo o resultado {meta_txt} do patamar esperado. "
        f"Os principais ofensores do mês foram {motivos_txt}."
    )

def gerar_consideracoes_ia_p1(dados: dict) -> str:
    if not VERTEX_PROJECT_ID or vertexai is None or GenerativeModel is None:
        return gerar_consideracoes_fallback_p1(dados)

    try:
        _ensure_vertex_adc_if_possible()
        vertexai.init(project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION)
        model = GenerativeModel(VERTEX_MODEL)

        atual = dados["resumo_atual"]
        ant = dados["resumo_ant"]
        var = dados["variacoes"]
        top_motivos = dados["df_motivos"].head(5).to_dict(orient="records")

        prompt = f"""
Você é um gerente executivo de manutenção de frota.
Escreva uma consideração executiva curta, objetiva e profissional, em português do Brasil,
com foco em confiabilidade operacional.

DADOS:
- Período: {dados['periodo_label']}
- Mês atual: {dados['mes_atual_label']}
- Mês anterior: {dados['mes_anterior_label']}
- Intervenções mês atual: {atual['intervencoes']}
- Intervenções mês anterior: {ant['intervencoes']}
- KM rodado mês atual: {atual['km_total']}
- KM rodado mês anterior: {ant['km_total']}
- MKBF mês atual: {atual['mkbf']}
- MKBF mês anterior: {ant['mkbf']}
- Variação MKBF: {var['mkbf_pct']:.2f}%
- Meta MKBF: {MKBF_META}
- Aderência à meta: {dados['aderencia_meta_pct']:.2f}%
- Principais ofensores: {top_motivos}

REGRAS:
- Máximo de 110 palavras.
- Não invente fatos.
- Não use markdown.
- Linguagem executiva e natural.
- Cite se o resultado ficou acima ou abaixo da meta.
"""
        resp = model.generate_content(prompt)
        texto = getattr(resp, "text", None) or ""
        texto = texto.strip().replace("```", "")
        return texto if texto else gerar_consideracoes_fallback_p1(dados)

    except Exception as e:
        print("⚠️ Erro na IA P1:", repr(e))
        return gerar_consideracoes_fallback_p1(dados)


def gerar_consideracoes_fallback_p2(dados: dict) -> str:
    df = dados["diario"].copy()
    if not df.empty:
        df["dt"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")
        wk = df["dt"].dt.weekday
        int_bd = int(df[wk < 5]["intervencoes"].sum())
        int_fds = int(df[wk >= 5]["intervencoes"].sum())
    else:
        int_bd = int_fds = 0

    media_dia = dados["media_interv_dia"]
    meta_mes = dados["meta_interv_mes"]
    delta = dados["delta_interv"]
    status_meta = "acima da meta de intervenções" if delta > 0 else "dentro da meta de intervenções"
    direcao = "pressiona" if delta > 0 else "favorece"

    return (
        f"No período de {dados['periodo_label']}, a média diária observada foi de {media_dia:.2f} "
        f"intervenções/dia, com projeção {status_meta}. A distribuição aponta {int_bd} quebras em dias úteis "
        f"e {int_fds} aos finais de semana. O comportamento diário {direcao} diretamente a "
        f"confiabilidade operacional. O acompanhamento contínuo é essencial para antecipar ações corretivas."
    )

def gerar_consideracoes_ia_p2(dados: dict) -> str:
    if not VERTEX_PROJECT_ID or vertexai is None or GenerativeModel is None:
        return gerar_consideracoes_fallback_p2(dados)

    try:
        _ensure_vertex_adc_if_possible()
        vertexai.init(project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION)
        model = GenerativeModel(VERTEX_MODEL)

        df = dados["diario"].copy()
        if not df.empty:
            df["dt"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")
            wk = df["dt"].dt.weekday
            int_bd = int(df[wk < 5]["intervencoes"].sum())
            int_sat = int(df[wk == 5]["intervencoes"].sum())
            int_sun = int(df[wk == 6]["intervencoes"].sum())
        else:
            int_bd = int_sat = int_sun = 0

        status_proj = "ALERTA (projeção supera a meta)" if dados["proj_interv_mes"] > dados["meta_interv_mes"] else "DENTRO DO ESPERADO"

        prompt = f"""
Você é um gerente executivo de manutenção de frota.
Escreva uma consideração executiva curta, objetiva e profissional, em português do Brasil.

DADOS GERAIS:
- Período: {dados['periodo_label']}
- Total de intervenções no período: {dados['total_interv']}
- Meta de intervenções para o mês: {dados['meta_interv_mes']:.1f}
- Projeção de intervenções até o fim do mês: {dados['proj_interv_mes']:.1f}
- Status da Projeção vs Meta: {status_proj}

DISTRIBUIÇÃO POR DIA DA SEMANA (Foco da Análise):
- Dias Úteis (Segunda a Sexta): {int_bd} ocorrências
- Sábados: {int_sat} ocorrências
- Domingos: {int_sun} ocorrências

REGRAS OBRIGATÓRIAS:
- Fale OBRIGATORIAMENTE sobre o volume de quebras nos dias úteis comparado aos finais de semana (sábado e domingo).
- Analise se essa tendência diária está pressionando ou favorecendo a meta e a confiabilidade.
- Máximo de 110 palavras.
- Não invente fatos, use apenas os dados fornecidos.
- Linguagem técnica e direta. Sem usar sintaxe markdown.
"""
        resp = model.generate_content(prompt)
        texto = getattr(resp, "text", None) or ""
        texto = texto.strip().replace("```", "")
        return texto if texto else gerar_consideracoes_fallback_p2(dados)

    except Exception as e:
        print("⚠️ Erro na IA P2:", repr(e))
        return gerar_consideracoes_fallback_p2(dados)


def gerar_consideracoes_fallback_p3(dados: dict) -> str:
    return (
        f"No mês atual ({dados['mes_ref_label']}), a operação registrou "
        f"{_fmt_int(dados['total_interv_ref'])} intervenções. A análise aponta a variação e concentração "
        f"das ocorrências de forma direta frente aos últimos meses, direcionando o foco das ações corretivas."
    )

def gerar_consideracoes_ia_p3(dados: dict) -> str:
    if not VERTEX_PROJECT_ID or vertexai is None or GenerativeModel is None:
        return gerar_consideracoes_fallback_p3(dados)
    
    try:
        _ensure_vertex_adc_if_possible()
        vertexai.init(project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION)
        model = GenerativeModel(VERTEX_MODEL)

        top_linha = dados["df_linha"].iloc[0] if not dados["df_linha"].empty else None
        top_carro = dados["df_top_carro"].iloc[0] if not dados["df_top_carro"].empty else None
        
        info_linha = f"A linha {top_linha['linha']} foi o maior ofensor do mês com {top_linha['int_ref']} quebras." if top_linha is not None else "Sem dados de linha."
        info_carro = f"O carro {top_carro['veiculo']} foi o mais ofensor no mês com {top_carro['int_ref']} falhas." if top_carro is not None else "Sem dados de carro."

        prompt = f"""
Você é um gerente executivo de manutenção de frota.
Escreva uma consideração executiva focada ESTRITAMENTE na Análise Estratégica do mês atual.

DADOS:
- Mês de Referência: {dados['mes_ref_label']}
- Total Intervenções Mês Ref: {dados['total_interv_ref']}
- Total Intervenções Acumulado (3 Meses): {dados['total_interv']}
- Detalhe Ofensores Mês Ref: {info_linha} | {info_carro}

REGRAS:
- A Análise SEMPRE precisa ser do Mês de Referência. Cite os dados do Acumulado (3 Meses) apenas como fator de alerta.
- Destaque objetivamente o ofensor por linha e por veículo.
- Máximo de 110 palavras.
- Não invente fatos. Linguagem técnica e direta. Sem markdown.
"""
        resp = model.generate_content(prompt)
        texto = getattr(resp, "text", None) or ""
        texto = texto.strip().replace("```", "")
        return texto if texto else gerar_consideracoes_fallback_p3(dados)

    except Exception as e:
        print("⚠️ Erro na IA P3:", repr(e))
        return gerar_consideracoes_fallback_p3(dados)


# ==============================================================================
# HTML E PDF - RELATÓRIO COMPLETO
# ==============================================================================
def gerar_html_relatorio_completo(
    dados_p1: dict, dados_p2: dict, dados_p3: dict,
    img_path_1: Path, img_path_2: Path,
    img_p3_linha: Path, img_p3_horario: Path, img_p3_top: Path,
    html_path: Path
):
    # ---- DADOS PÁGINA 1 ----
    atual = dados_p1["resumo_atual"]
    ant = dados_p1["resumo_ant"]
    var = dados_p1["variacoes"]
    ader = dados_p1["aderencia_meta_pct"]
    motivos = dados_p1["df_motivos"].copy()
    cons_p1 = gerar_consideracoes_ia_p1(dados_p1)
    
    status_text_p1 = "ATINGIU" if atual["mkbf"] >= MKBF_META else "ABAIXO"
    status_bg_p1 = "#dcfce7" if atual["mkbf"] >= MKBF_META else "#fee2e2"
    status_fg_p1 = "#166534" if atual["mkbf"] >= MKBF_META else "#991b1b"

    rows_motivos = ""
    for _, r in motivos.iterrows():
        rows_motivos += f"""
        <tr>
            <td style="text-align: left; padding-left: 8px;">{r['tipo']}</td>
            <td>{_fmt_int(r['mes_anterior'])}</td>
            <td style="font-weight:700;">{_fmt_int(r['mes_atual'])}</td>
        </tr>
        """

    # ---- DADOS PÁGINA 2 ----
    cons_p2 = gerar_consideracoes_ia_p2(dados_p2)
    status_proj = "ALERTA" if dados_p2["proj_interv_mes"] > dados_p2["meta_interv_mes"] else "DENTRO DO ESPERADO"
    status_bg_proj = "#fee2e2" if dados_p2["proj_interv_mes"] > dados_p2["meta_interv_mes"] else "#dcfce7"
    status_fg_proj = "#991b1b" if dados_p2["proj_interv_mes"] > dados_p2["meta_interv_mes"] else "#166534"
    cor_delta = "#dc2626" if dados_p2["delta_interv"] > 0 else "#16a34a"
    sinal_delta = "+" if dados_p2["delta_interv"] > 0 else ""

    # ---- DADOS PÁGINA 3 ----
    df_cluster = dados_p3["df_cluster"].copy()
    cons_p3 = gerar_consideracoes_ia_p3(dados_p3)
    
    lbl_m1 = dados_p3['meses_labels'][0]
    lbl_m2 = dados_p3['meses_labels'][1]
    lbl_m3 = dados_p3['meses_labels'][2]

    rows_cluster = ""
    for _, r in df_cluster.iterrows():
        rows_cluster += f"""
        <tr>
            <td style="font-weight:bold; text-align: left; padding-left:8px;">{r['cluster']}</td>
            <td>{int(r[lbl_m1])}</td>
            <td>{int(r[lbl_m2])}</td>
            <td style="font-weight:700; color:#1e3a8a;">{int(r[lbl_m3])}</td>
            <td>{r['int_veiculo_ref']:.2f}</td>
            <td>{int(r['frota_ref'])}</td>
        </tr>
        """
    total_interv_cluster_ref = _fmt_int(df_cluster[lbl_m3].sum()) if not df_cluster.empty else "0"

    footer_right = f"Relatório ID: {REPORT_ID}" if REPORT_ID else "Flash Report Manutenção"

    html = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
      <meta charset="UTF-8" />
      <title>Flash Report Manutenção</title>
      <style>
        * {{ box-sizing: border-box; }}
        body {{ margin: 0; font-family: Arial, Helvetica, sans-serif; background: #eef2f7; color: #111827; }}
        .page {{
          width: 210mm; min-height: 297mm; margin: 0 auto;
          background: radial-gradient(circle at top right, rgba(37,99,235,0.06), transparent 22%), linear-gradient(180deg, #ffffff 0%, #fbfcfe 100%);
          padding: 8mm 10mm 8mm 10mm; position: relative;
        }}
        .page-break {{ page-break-before: always; }}
        .header {{ display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 4px solid #0f172a; padding-bottom: 8px; margin-bottom: 12px; }}
        .title h1 {{ margin: 0; font-size: 24px; line-height: 1; letter-spacing: 0.3px; color: #0f172a; }}
        .title .sub {{ margin-top: 6px; font-size: 11px; color: #475569; }}
        .period-box {{ min-width: 220px; text-align: right; background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%); color: white; padding: 10px 12px; border-radius: 14px; box-shadow: 0 10px 24px rgba(15,23,42,0.16); }}
        .period-box .ref {{ font-size: 10px; text-transform: uppercase; font-weight: 700; opacity: 0.8; }}
        .period-box .val {{ font-size: 18px; font-weight: 800; margin-top: 3px; }}
        
        .grid-top {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }}
        .grid-top-p3 {{ display: grid; grid-template-columns: 1fr 1.35fr; gap: 12px; margin-bottom: 12px; align-items: start; }}
        
        .card {{ border: 1px solid #dbe3ee; border-radius: 14px; overflow: hidden; background: #fff; box-shadow: 0 6px 20px rgba(15,23,42,0.06); }}
        .card-title {{ padding: 10px 12px; background: linear-gradient(90deg, #0f172a 0%, #172554 100%); color: white; font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.8px; }}
        .card-body {{ padding: 12px; }}
        
        table {{ width: 100%; border-collapse: collapse; font-size: 10px; }}
        th {{ background: #eef2f7; color: #0f172a; text-transform: uppercase; font-size: 9px; padding: 6px 4px; border: 1px solid #dbe3ee; text-align: center; }}
        td {{ padding: 6px 4px; border: 1px solid #dbe3ee; vertical-align: middle; text-align: center; word-wrap: break-word; }}
        
        .metric-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }}
        .metric {{ border: 1px solid #dbe3ee; border-radius: 12px; padding: 10px 8px; background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%); box-shadow: inset 0 1px 0 rgba(255,255,255,0.8); }}
        .metric .lbl {{ font-size: 9px; color: #64748b; text-transform: uppercase; font-weight: 800; white-space: nowrap; }}
        .metric .val {{ margin-top: 4px; font-size: 18px; font-weight: 800; color: #0f172a; }}
        .metric .aux {{ margin-top: 3px; font-size: 9px; color: #64748b; line-height: 1.1; }}
        
        .metric-wide {{ margin-top: 10px; display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
        .badge {{ display: inline-block; padding: 6px 10px; border-radius: 999px; font-size: 11px; font-weight: 800; color: white; background: linear-gradient(90deg, #0f172a 0%, #1e3a8a 100%); box-shadow: 0 6px 14px rgba(30,58,138,0.18); }}
        .muted {{ color: #64748b; }}
        
        .chart-wrap {{ padding: 10px; border: 1px solid #dbe3ee; border-radius: 14px; background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%); }}
        .chart-wrap img {{ width: 100%; height: auto; display: block; }}
        
        .cons-box {{ margin-top: 12px; border: 1px solid #dbe3ee; border-radius: 14px; background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%); padding: 12px; box-shadow: 0 6px 20px rgba(15,23,42,0.05); }}
        .cons-title {{ font-size: 11px; font-weight: 800; text-transform: uppercase; margin-bottom: 8px; color: #0f172a; letter-spacing: 0.7px; }}
        .cons-text {{ font-size: 12px; line-height: 1.62; color: #1f2937; text-align: justify; }}
        .footer {{ position: absolute; left: 10mm; right: 10mm; bottom: 6mm; font-size: 10px; color: #64748b; display: flex; justify-content: space-between; border-top: 1px solid #dbe3ee; padding-top: 4px; }}
        @page {{ size: A4; margin: 0; }}
      </style>
    </head>
    <body>
      
      <div class="page">
        <div class="header">
          <div class="title">
            <h1>FLASH REPORT MANUTENÇÃO</h1>
            <div class="sub">Página 1 · Intervenções do mês / MKBF</div>
            <div class="sub">Período analisado: <b>{dados_p1['periodo_label']}</b></div>
          </div>
          <div class="period-box">
            <div class="ref">Mês de referência</div>
            <div class="val">{dados_p1['mes_atual_label']}</div>
          </div>
        </div>

        <div class="grid-top">
          <div class="card">
            <div class="card-title">Comparativo mensal</div>
            <div class="card-body">
              <table>
                <thead>
                  <tr>
                    <th style="text-align: left; padding-left: 8px;">Indicador</th>
                    <th>{dados_p1['mes_anterior_label']}</th>
                    <th>{dados_p1['mes_atual_label']}</th>
                    <th>Var</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td style="text-align: left; padding-left: 8px;"><b>Intervenções</b></td>
                    <td>{_fmt_int(ant['intervencoes'])}</td>
                    <td><b>{_fmt_int(atual['intervencoes'])}</b></td>
                    <td style="color:{cls_var(-var['intervencoes_pct'])}; font-weight:bold;">{_fmt_pct(var['intervencoes_pct'])}</td>
                  </tr>
                  <tr>
                    <td style="text-align: left; padding-left: 8px;"><b>KM rodado</b></td>
                    <td>{_fmt_int(ant['km_total'])}</td>
                    <td><b>{_fmt_int(atual['km_total'])}</b></td>
                    <td style="color:{cls_var(var['km_pct'])}; font-weight:bold;">{_fmt_pct(var['km_pct'])}</td>
                  </tr>
                  <tr>
                    <td style="text-align: left; padding-left: 8px;"><b>MKBF</b></td>
                    <td>{_fmt_int(ant['mkbf'])}</td>
                    <td><b>{_fmt_int(atual['mkbf'])}</b></td>
                    <td style="color:{cls_var(var['mkbf_pct'])}; font-weight:bold;">{_fmt_pct(var['mkbf_pct'])}</td>
                  </tr>
                  <tr>
                    <td style="text-align: left; padding-left: 8px;"><b>Meta MKBF</b></td>
                    <td>{_fmt_int(MKBF_META)}</td>
                    <td>{_fmt_int(MKBF_META)}</td>
                    <td class="muted">-</td>
                  </tr>
                </tbody>
              </table>

              <div class="metric-wide">
                <div class="metric">
                  <div class="lbl">Aderência à meta</div>
                  <div class="val">{ader:.1f}%</div>
                  <div class="aux">Meta de referência: {_fmt_int(MKBF_META)}</div>
                </div>
                <div class="metric" style="text-align: center;">
                  <div class="lbl">Status do mês</div>
                  <div style="margin-top:10px;">
                    <span style="display:inline-block; padding:6px 10px; border-radius:10px; font-size:12px; font-weight:800; background:{status_bg_p1}; color:{status_fg_p1};">{status_text_p1}</span>
                  </div>
                  <div class="aux" style="margin-top:10px;">Base consolidada</div>
                </div>
              </div>
            </div>
          </div>

          <div class="card">
            <div class="card-title">Resumo executivo do período</div>
            <div class="card-body">
              <div class="metric-grid">
                <div class="metric">
                  <div class="lbl">Intervenções</div>
                  <div class="val">{_fmt_int(atual['intervencoes'])}</div>
                  <div class="aux">Ocorrências válidas</div>
                </div>
                <div class="metric">
                  <div class="lbl">KM rodado</div>
                  <div class="val">{_fmt_int(atual['km_total'])}</div>
                  <div class="aux">Acumulado total</div>
                </div>
                <div class="metric">
                  <div class="lbl">MKBF</div>
                  <div class="val">{_fmt_int(atual['mkbf'])}</div>
                  <div class="aux">KM ÷ intervenções</div>
                </div>
              </div>

              <div style="margin-top:12px;">
                <div class="badge">Meta MKBF: {_fmt_int(MKBF_META)}</div>
              </div>

              <div style="margin-top:14px;">
                <table>
                  <thead>
                    <tr>
                      <th style="text-align: left; padding-left: 8px;">Tipo de ocorrência</th>
                      <th>{dados_p1['mes_anterior_label']}</th>
                      <th>{dados_p1['mes_atual_label']}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows_motivos}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>

        <div class="card" style="margin-bottom:12px;">
          <div class="card-title">Evolução histórica do MKBF</div>
          <div class="card-body">
            <div class="chart-wrap">
              <img src="{img_path_1.name}" alt="Gráfico histórico do MKBF" />
            </div>
          </div>
        </div>

        <div class="cons-box">
          <div class="cons-title">Considerações executivas</div>
          <div class="cons-text">{cons_p1}</div>
        </div>

        <div class="footer">
          <div>Gerado automaticamente · Página 1/3</div>
          <div>{footer_right}</div>
        </div>
      </div>

      <div class="page-break"></div>

      <div class="page">
        <div class="header">
          <div class="title">
            <h1>FLASH REPORT MANUTENÇÃO</h1>
            <div class="sub">Página 2 · Intervenções por Dia e Projeções</div>
            <div class="sub">Período analisado: <b>{dados_p2['periodo_label']}</b></div>
          </div>
          <div class="period-box">
            <div class="ref">Mês de referência</div>
            <div class="val">{dados_p2['mes_atual_label']}</div>
          </div>
        </div>

        <div class="grid-top">
          <div class="card">
            <div class="card-title">Projeções e Metas</div>
            <div class="card-body">
              <table>
                <thead>
                  <tr>
                    <th style="text-align: left; padding-left: 8px;">Indicador</th>
                    <th>Valor Acumulado / Projeção</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td style="text-align: left; padding-left: 8px;"><b>Média de Intervenções / Dia</b></td>
                    <td style="font-weight: bold; font-size: 13px;">{dados_p2['media_interv_dia']:.1f}</td>
                  </tr>
                  <tr>
                    <td style="text-align: left; padding-left: 8px;"><b>Meta de Intervenções / Mês</b></td>
                    <td>{_fmt_int(dados_p2['meta_interv_mes'])}</td>
                  </tr>
                  <tr>
                    <td style="text-align: left; padding-left: 8px;"><b>Projeção Fim do Mês</b></td>
                    <td><b>{_fmt_int(dados_p2['proj_interv_mes'])}</b></td>
                  </tr>
                  <tr>
                    <td style="text-align: left; padding-left: 8px;"><b>Desvio (Delta vs Meta)</b></td>
                    <td style="color:{cor_delta}; font-weight:bold;">
                        {sinal_delta}{_fmt_int(dados_p2['delta_interv'])}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <div class="card">
            <div class="card-title">Status e Base Analítica</div>
            <div class="card-body" style="display: flex; flex-direction: column; gap: 12px;">
              <div style="padding: 10px; background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 8px;">
                <div style="font-size: 11px; font-weight: bold; color: #334155; margin-bottom: 4px;">Informação de Base</div>
                <div style="font-size: 10px; color: #64748b; line-height: 1.4;">
                    Dias decorridos na análise: <b>{dados_p2['dias_decorridos']} dias</b>.<br/>
                    A meta de intervenções é calculada através do KM rodado dividido pela meta do MKBF ({_fmt_int(MKBF_META)}).
                </div>
              </div>
              
              <div style="padding: 10px; border: 1px solid #dbe3ee; border-radius: 8px; background: #fff; text-align: center;">
                 <div style="font-size:10px; color:#64748b; font-weight:bold; text-transform:uppercase; margin-bottom:8px;">Status da Projeção</div>
                 <div><span style="display:inline-block; padding:6px 10px; border-radius:10px; font-size:12px; font-weight:800; background:{status_bg_proj}; color:{status_fg_proj};">{status_proj}</span></div>
              </div>
            </div>
          </div>
        </div>

        <div class="card" style="margin-bottom:12px;">
          <div class="card-title">Evolução Diária - Intervenções e MKBF</div>
          <div class="card-body">
            <div class="chart-wrap">
              <img src="{img_path_2.name}" alt="Gráfico de Intervenções por Dia" />
            </div>
          </div>
        </div>

        <div class="cons-box">
          <div class="cons-title">Considerações executivas · Análise Diária</div>
          <div class="cons-text">{cons_p2}</div>
        </div>

        <div class="footer">
          <div>Gerado automaticamente · Página 2/3</div>
          <div>{footer_right}</div>
        </div>
      </div>

      <div class="page-break"></div>

      <div class="page">
        <div class="header">
          <div class="title">
            <h1>FLASH REPORT MANUTENÇÃO</h1>
            <div class="sub">Página 3 · Análise Estratégica (Linha / Horário / Carro / Cluster)</div>
            <div class="sub">Período consolidado: <b>{dados_p3['periodo_label']} (3 Meses)</b></div>
          </div>
          <div class="period-box">
            <div class="ref">Mês Referência Principal</div>
            <div class="val">{dados_p3['mes_ref_label']}</div>
          </div>
        </div>

        <div class="card" style="margin-bottom:12px;">
          <div class="card-title">Ocorrências por Linha e Horário</div>
          <div class="card-body" style="padding-bottom: 4px;">
            <img src="{img_p3_linha.name}" alt="Intervenções por linha" style="width: 100%; margin-bottom: 8px;">
            <hr style="border: 0; height: 1px; background: #dbe3ee; margin: 10px 0;">
            <img src="{img_p3_horario.name}" alt="Intervenções por horário" style="width: 100%;">
          </div>
        </div>

        <div class="grid-top-p3">
          <div class="card" style="height: 100%;">
            <div class="card-title">Top 10 - Veículos Ofensores</div>
            <div class="card-body" style="height: calc(100% - 35px); display: flex; align-items: center; justify-content: center;">
              <img src="{img_p3_top.name}" alt="Top 10 carro" style="width: 100%;">
            </div>
          </div>

          <div class="card" style="height: 100%;">
            <div class="card-title">Volume Histórico por Cluster</div>
            <div class="card-body">
              <table>
                <thead>
                  <tr>
                    <th style="text-align: left; padding-left: 8px;">Cluster</th>
                    <th>{lbl_m1}</th>
                    <th>{lbl_m2}</th>
                    <th style="color: #1e3a8a;">{lbl_m3} (Ref)</th>
                    <th>Int/Veículo<br/>(Mês Atual)</th>
                    <th>Frota<br/>(Mês Atual)</th>
                  </tr>
                </thead>
                <tbody>
                  {rows_cluster}
                </tbody>
              </table>
              <div style="margin-top: 12px; padding: 10px; text-align: right; background: #f8fafc; border-radius: 8px; font-size: 13px; font-weight: 800; color: #111827; border: 1px solid #dbe3ee;">
                TOTAL MÊS ATUAL <span style="color:#dc2626; margin-left: 15px; font-size: 16px;">{total_interv_cluster_ref}</span>
              </div>
            </div>
          </div>
        </div>

        <div class="cons-box">
          <div class="cons-title">Considerações executivas · Análise Estratégica</div>
          <div class="cons-text">{cons_p3}</div>
        </div>

        <div class="footer">
          <div>Gerado automaticamente · Página 3/3</div>
          <div>{footer_right}</div>
        </div>
      </div>
    </body>
    </html>
    """
    html_path.write_text(html, encoding="utf-8")
    print(f"✅ HTML completo salvo: {html_path}")


def gerar_pdf_do_html(html_path: Path, pdf_path: Path):
    html_path = html_path.resolve()
    pdf_path = pdf_path.resolve()

    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        page.goto(html_path.as_uri(), wait_until="networkidle")
        page.wait_for_timeout(500)
        page.pdf(
            path=str(pdf_path),
            format="A4",
            print_background=True,
            margin={"top": "0mm", "right": "0mm", "bottom": "0mm", "left": "0mm"},
            prefer_css_page_size=True,
        )
        browser.close()

    print(f"✅ PDF salvo: {pdf_path}")


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
        periodo_inicio = month_start(hoje)
        periodo_fim = hoje
    elif periodo_inicio and not periodo_fim:
        periodo_fim = month_end(periodo_inicio)
    elif not periodo_inicio and periodo_fim:
        periodo_inicio = month_start(periodo_fim)

    atualizar_status_relatorio(
        "PROCESSANDO",
        tipo=REPORT_TIPO,
        periodo_inicio=str(periodo_inicio),
        periodo_fim=str(periodo_fim),
    )

    try:
        # Processamento das três páginas
        dados_p1 = processar_pagina_1(periodo_inicio, periodo_fim)
        dados_p2 = processar_pagina_2(periodo_inicio, periodo_fim, dados_p1["df_diario_atual"], dados_p1["resumo_atual"])
        dados_p3 = processar_pagina_3(periodo_fim)

        # Caminhos dos arquivos
        out_dir = Path(PASTA_SAIDA)
        img_path_p1 = out_dir / "flash_manutencao_mkbf_hist.png"
        img_path_p2 = out_dir / "flash_manutencao_diario.png"
        img_p3_linha = out_dir / "flash_manutencao_p3_linha.png"
        img_p3_horario = out_dir / "flash_manutencao_p3_horario.png"
        img_p3_top = out_dir / "flash_manutencao_p3_top_carro.png"
        
        html_path = out_dir / "Flash_Report_Manutencao.html"
        pdf_path = out_dir / "Flash_Report_Manutencao.pdf"

        # Geração dos recursos visuais e arquivo final
        gerar_grafico_mkbf_historico(dados_p1["df_hist"], img_path_p1)
        gerar_grafico_pagina_2(dados_p2, img_path_p2)
        gerar_grafico_pagina_3_linha(dados_p3["df_linha"], img_p3_linha)
        gerar_grafico_pagina_3_horario(dados_p3["df_horario"], img_p3_horario)
        gerar_grafico_pagina_3_top_carro(dados_p3["df_top_carro"], img_p3_top)

        gerar_html_relatorio_completo(
            dados_p1, dados_p2, dados_p3, 
            img_path_p1, img_path_p2, 
            img_p3_linha, img_p3_horario, img_p3_top, 
            html_path
        )
        
        gerar_pdf_do_html(html_path, pdf_path)

        # Configuração para upload (diretório unificado do relatório)
        mes_ref = f"{periodo_fim.year}-{str(periodo_fim.month).zfill(2)}"
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        base_folder = f"{REMOTE_BASE_PREFIX}/{mes_ref}/flash_report/{stamp}"

        # Enviando arquivos para o Supabase
        upload_storage_b(img_path_p1, f"{base_folder}/{img_path_p1.name}", "image/png")
        upload_storage_b(img_path_p2, f"{base_folder}/{img_path_p2.name}", "image/png")
        upload_storage_b(img_p3_linha, f"{base_folder}/{img_p3_linha.name}", "image/png")
        upload_storage_b(img_p3_horario, f"{base_folder}/{img_p3_horario.name}", "image/png")
        upload_storage_b(img_p3_top, f"{base_folder}/{img_p3_top.name}", "image/png")
        
        upload_storage_b(html_path, f"{base_folder}/{html_path.name}", "text/html; charset=utf-8")
        upload_storage_b(pdf_path, f"{base_folder}/{pdf_path.name}", "application/pdf")

        # Status Final
        atualizar_status_relatorio(
            "CONCLUIDO",
            arquivo_pdf_path=f"{base_folder}/{pdf_path.name}",
            arquivo_html_path=f"{base_folder}/{html_path.name}",
            arquivo_png_path=f"{base_folder}/{img_path_p1.name}",
            erro_msg=None,
            mes_ref=mes_ref,
        )

        print("✅ [OK] Flash Report Manutenção (Páginas 1, 2 e 3) concluído.")
        print(f"📄 PDF Unificado: {pdf_path}")
        print(f"🌐 HTML: {html_path}")
        print(f"☁️ Storage base: {base_folder}")

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
