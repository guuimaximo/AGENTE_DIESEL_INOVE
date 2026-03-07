# scripts/flash_report_manutencao.py
# ------------------------------------------------------------------------------
# FLASH REPORT MANUTENÇÃO - PÁGINAS 1 E 2
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


# ==============================================================================
# PROCESSAMENTO - PÁGINA 1
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


# ==============================================================================
# PROCESSAMENTO - PÁGINA 2
# ==============================================================================
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
        plt.figure(figsize=(11.5, 5.0))
        plt.title("Evolução Diária: Intervenções e MKBF", fontsize=12, fontweight="bold")
        plt.text(0.5, 0.5, "Sem dados para exibição", ha="center", va="center", transform=plt.gca().transAxes)
        plt.tight_layout()
        plt.savefig(caminho_img, dpi=140)
        plt.close()
        return

    fig, ax1 = plt.subplots(figsize=(11.5, 5.0))

    x = range(len(df))
    labels = df["dia"].tolist()
    interv = df["intervencoes"].tolist()
    mkbf = df["mkbf"].tolist()

    bars = ax1.bar(x, interv, label="Intervenções", color="#1e3a8a", alpha=0.85, width=0.6)
    ax1.set_ylabel("Intervenções", fontsize=9)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=45, fontsize=8, ha="right")
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)

    for i, v in enumerate(interv):
        if v > 0:
            ax1.text(i, v + (max(interv)*0.02), str(v), ha="center", va="bottom", fontsize=8, fontweight="bold", color="#0f172a")

    ax2 = ax1.twinx()
    ax2.plot(x, mkbf, marker="o", markersize=5, linewidth=2.2, label="MKBF diário", color="#ef4444")
    ax2.set_ylabel("MKBF", fontsize=9)

    for i, v in enumerate(mkbf):
        if v > 0:
            ax2.text(i, v + max(max(mkbf) * 0.05, 100), f"{v:,.0f}".replace(",", "."), ha="center", fontsize=7, color="#dc2626")

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

    except DefaultCredentialsError:
        return gerar_consideracoes_fallback_p1(dados)
    except Exception as e:
        print("⚠️ Erro na IA:", repr(e))
        return gerar_consideracoes_fallback_p1(dados)


def gerar_consideracoes_pagina_2(dados: dict) -> str:
    total_interv = dados["total_interv"]
    total_km = dados["total_km"]
    mkbf_mes = dados["mkbf_mes"]
    media_dia = dados["media_interv_dia"]
    meta_mes = dados["meta_interv_mes"]
    delta = dados["delta_interv"]

    status_meta = "acima da meta de intervenções" if delta > 0 else "dentro da meta de intervenções"
    direcao = "pressiona" if delta > 0 else "favorece"

    return (
        f"No período de {dados['periodo_label']}, a média diária observada foi de {media_dia:.2f} "
        f"intervenções por dia, frente a uma meta estimada de {meta_mes:.1f} intervenções no mês. "
        f"O comportamento diário indica um nível de ocorrência que {direcao} diretamente a "
        f"confiabilidade operacional, mantendo a projeção {status_meta}. O acompanhamento dessa "
        f"distribuição ao longo dos dias é essencial para identificar concentração de falhas e antecipar ações corretivas."
    )


# ==============================================================================
# HTML E PDF - RELATÓRIO COMPLETO
# ==============================================================================
def gerar_html_relatorio_completo(dados_p1: dict, dados_p2: dict, img_path_1: Path, img_path_2: Path, html_path: Path):
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
            <td>{r['tipo']}</td>
            <td style="text-align:center;">{_fmt_int(r['mes_anterior'])}</td>
            <td style="text-align:center; font-weight:700;">{_fmt_int(r['mes_atual'])}</td>
        </tr>
        """

    # ---- DADOS PÁGINA 2 ----
    cons_p2 = gerar_consideracoes_pagina_2(dados_p2)
    
    status_proj = "ALERTA" if dados_p2["proj_interv_mes"] > dados_p2["meta_interv_mes"] else "DENTRO DO ESPERADO"
    status_bg_proj = "#fee2e2" if dados_p2["proj_interv_mes"] > dados_p2["meta_interv_mes"] else "#dcfce7"
    status_fg_proj = "#991b1b" if dados_p2["proj_interv_mes"] > dados_p2["meta_interv_mes"] else "#166534"
    
    cor_delta = "#dc2626" if dados_p2["delta_interv"] > 0 else "#16a34a"
    sinal_delta = "+" if dados_p2["delta_interv"] > 0 else ""

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
        .grid-top {{ display: grid; grid-template-columns: 1.04fr 0.96fr; gap: 12px; margin-bottom: 12px; }}
        .grid-top-p2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }}
        .card {{ border: 1px solid #dbe3ee; border-radius: 14px; overflow: hidden; background: #fff; box-shadow: 0 6px 20px rgba(15,23,42,0.06); }}
        .card-title {{ padding: 10px 12px; background: linear-gradient(90deg, #0f172a 0%, #172554 100%); color: white; font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.8px; }}
        .card-body {{ padding: 12px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 11px; table-layout: fixed; }}
        th {{ background: #eef2f7; color: #0f172a; text-transform: uppercase; font-size: 9px; padding: 6px 4px; border: 1px solid #dbe3ee; }}
        td {{ padding: 6px 4px; border: 1px solid #dbe3ee; vertical-align: middle; word-wrap: break-word; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }}
        .metric {{ border: 1px solid #dbe3ee; border-radius: 12px; padding: 8px 6px; background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%); box-shadow: inset 0 1px 0 rgba(255,255,255,0.8); }}
        .metric .lbl {{ font-size: 9px; color: #64748b; text-transform: uppercase; font-weight: 800; }}
        .metric .val {{ margin-top: 4px; font-size: 21px; font-weight: 800; color: #0f172a; }}
        .metric .aux {{ margin-top: 3px; font-size: 9px; color: #64748b; }}
        .metric-wide {{ margin-top: 10px; display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
        .badge {{ display: inline-block; padding: 6px 10px; border-radius: 999px; font-size: 11px; font-weight: 800; color: white; background: linear-gradient(90deg, #0f172a 0%, #1e3a8a 100%); box-shadow: 0 6px 14px rgba(30,58,138,0.18); }}
        .center {{ text-align: center; }}
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
                    <th>Indicador</th>
                    <th>{dados_p1['mes_anterior_label']}</th>
                    <th>{dados_p1['mes_atual_label']}</th>
                    <th>Variação</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><b>Intervenções do mês</b></td>
                    <td class="center">{_fmt_int(ant['intervencoes'])}</td>
                    <td class="center"><b>{_fmt_int(atual['intervencoes'])}</b></td>
                    <td class="center" style="color:{cls_var(-var['intervencoes_pct'])}; font-weight:bold;">{_fmt_pct(var['intervencoes_pct'])}</td>
                  </tr>
                  <tr>
                    <td><b>KM rodado</b></td>
                    <td class="center">{_fmt_int(ant['km_total'])}</td>
                    <td class="center"><b>{_fmt_int(atual['km_total'])}</b></td>
                    <td class="center" style="color:{cls_var(var['km_pct'])}; font-weight:bold;">{_fmt_pct(var['km_pct'])}</td>
                  </tr>
                  <tr>
                    <td><b>MKBF</b></td>
                    <td class="center">{_fmt_int(ant['mkbf'])}</td>
                    <td class="center"><b>{_fmt_int(atual['mkbf'])}</b></td>
                    <td class="center" style="color:{cls_var(var['mkbf_pct'])}; font-weight:bold;">{_fmt_pct(var['mkbf_pct'])}</td>
                  </tr>
                  <tr>
                    <td><b>Meta MKBF</b></td>
                    <td class="center">{_fmt_int(MKBF_META)}</td>
                    <td class="center">{_fmt_int(MKBF_META)}</td>
                    <td class="center muted">-</td>
                  </tr>
                </tbody>
              </table>

              <div class="metric-wide">
                <div class="metric">
                  <div class="lbl">Aderência à meta</div>
                  <div class="val">{ader:.1f}%</div>
                  <div class="aux">Meta de referência: {_fmt_int(MKBF_META)}</div>
                </div>
                <div class="metric">
                  <div class="lbl">Status do mês</div>
                  <div style="margin-top:10px;">
                    <span style="display:inline-block; padding:6px 10px; border-radius:10px; font-size:12px; font-weight:800; background:{status_bg_p1}; color:{status_fg_p1};">{status_text_p1}</span>
                  </div>
                  <div class="aux" style="margin-top:10px;">Com base no MKBF consolidado</div>
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
                  <div class="aux">KM total consolidado</div>
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
                      <th style="width: 46%; text-align: left; padding-left: 8px;">Tipo de ocorrência</th>
                      <th style="width: 27%;">{dados_p1['mes_anterior_label']}</th>
                      <th style="width: 27%;">{dados_p1['mes_atual_label']}</th>
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
          <div>Gerado automaticamente · Página 1/2</div>
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

        <div class="grid-top-p2">
          <div class="card">
            <div class="card-title">Projeções e Metas</div>
            <div class="card-body">
              <table>
                <thead>
                  <tr>
                    <th style="width: 55%; text-align: left; padding-left: 8px;">Indicador</th>
                    <th style="width: 45%;">Valor Acumulado / Projeção</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><b>Média de Intervenções / Dia</b></td>
                    <td class="center" style="font-weight: bold; font-size: 13px;">{dados_p2['media_interv_dia']:.1f}</td>
                  </tr>
                  <tr>
                    <td><b>Meta de Intervenções / Mês</b></td>
                    <td class="center">{_fmt_int(dados_p2['meta_interv_mes'])}</td>
                  </tr>
                  <tr>
                    <td><b>Projeção Fim do Mês</b></td>
                    <td class="center"><b>{_fmt_int(dados_p2['proj_interv_mes'])}</b></td>
                  </tr>
                  <tr>
                    <td><b>Desvio (Delta vs Meta)</b></td>
                    <td class="center" style="color:{cor_delta}; font-weight:bold;">
                        {sinal_delta}{_fmt_int(dados_p2['delta_interv'])}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <div class="card">
            <div class="card-title">Status e Base Analítica</div>
            <div class="card-body" style="display: flex; flex-direction: column; justify-content: space-between; height: 100%;">
              <div style="padding: 10px; background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 8px;">
                <div style="font-size: 11px; font-weight: bold; color: #334155; margin-bottom: 4px;">Informação de Base</div>
                <div style="font-size: 10px; color: #64748b; line-height: 1.4;">
                    Dias decorridos na análise: <b>{dados_p2['dias_decorridos']} dias</b>.<br/>
                    A meta mensal de intervenções é calculada proporcionalmente através do total de KM rodado no período dividido pela meta global do MKBF ({_fmt_int(MKBF_META)}).
                </div>
              </div>
              
              <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; padding: 10px; border: 1px solid #dbe3ee; border-radius: 8px; background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);">
                 <div style="font-size:10px; color:#64748b; font-weight:bold; text-transform:uppercase; margin-bottom:6px;">Status da Projeção</div>
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
          <div>Gerado automaticamente · Página 2/2</div>
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
        # Processamento das duas páginas
        dados_p1 = processar_pagina_1(periodo_inicio, periodo_fim)
        dados_p2 = processar_pagina_2(periodo_inicio, periodo_fim, dados_p1["df_diario_atual"], dados_p1["resumo_atual"])

        # Caminhos dos arquivos
        out_dir = Path(PASTA_SAIDA)
        img_path_p1 = out_dir / "flash_manutencao_mkbf_hist.png"
        img_path_p2 = out_dir / "flash_manutencao_diario.png"
        html_path = out_dir / "Flash_Report_Manutencao.html"
        pdf_path = out_dir / "Flash_Report_Manutencao.pdf"

        # Geração dos recursos visuais e arquivo final
        gerar_grafico_mkbf_historico(dados_p1["df_hist"], img_path_p1)
        gerar_grafico_pagina_2(dados_p2, img_path_p2)
        gerar_html_relatorio_completo(dados_p1, dados_p2, img_path_p1, img_path_p2, html_path)
        gerar_pdf_do_html(html_path, pdf_path)

        # Configuração para upload (diretório unificado do relatório)
        mes_ref = f"{periodo_fim.year}-{str(periodo_fim.month).zfill(2)}"
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        base_folder = f"{REMOTE_BASE_PREFIX}/{mes_ref}/flash_report/{stamp}"

        # Enviando arquivos para o Supabase
        upload_storage_b(img_path_p1, f"{base_folder}/{img_path_p1.name}", "image/png")
        upload_storage_b(img_path_p2, f"{base_folder}/{img_path_p2.name}", "image/png")
        upload_storage_b(html_path, f"{base_folder}/{html_path.name}", "text/html; charset=utf-8")
        upload_storage_b(pdf_path, f"{base_folder}/{pdf_path.name}", "application/pdf")

        # Status Final
        atualizar_status_relatorio(
            "CONCLUIDO",
            arquivo_pdf_path=f"{base_folder}/{pdf_path.name}",
            arquivo_html_path=f"{base_folder}/{html_path.name}",
            arquivo_png_path=f"{base_folder}/{img_path_p1.name}", # Referência base para capa
            erro_msg=None,
            mes_ref=mes_ref,
        )

        print("✅ [OK] Flash Report Manutenção (Páginas 1 e 2) concluído.")
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
