# scripts/flash_report_manutencao_pagina_1.py
# ------------------------------------------------------------------------------
# FLASH REPORT MANUTENÇÃO - PÁGINA 1 (MKBF)
#
# O que este script gera:
# - Apenas a PRIMEIRA PÁGINA do flash report
# - Busca dados SOMENTE do Supabase B
# - Calcula:
#     * Intervenções do mês
#     * KM rodado do mês
#     * MKBF do mês
#     * Comparativo mês atual x mês anterior
#     * Principais tipos de ocorrência (mês atual x mês anterior)
#     * Evolução histórica mensal do MKBF + meta
# - Gera:
#     * 1 PNG do gráfico
#     * 1 HTML
#     * 1 PDF com somente a página 1
# - Faz upload no Storage
#
# REGRA OFICIAL DO MKBF (copiada do seu dashboard):
# - Conta toda ocorrência válida, exceto "SEGUIU VIAGEM"
# - Normaliza ocorrências:
#     RA / R.A / R.A. / RECOLH* => RECOLHEU
#     IMPRO* => IMPROCEDENTE
#     TROC* => TROCA
#     S.O.S => SOS
#     AVARI* => AVARIA
#     SEGUIU* => SEGUIU VIAGEM
#
# ENV obrigatórias:
# - SUPABASE_B_URL
# - SUPABASE_B_SERVICE_ROLE_KEY
#
# ENV opcionais:
# - REPORT_TIPO=flash_report_manutencao
# - REPORT_PERIODO_INICIO=2026-03-01
# - REPORT_PERIODO_FIM=2026-03-31
# - REPORT_OUTPUT_DIR=Relatorios_Manutencao
# - REPORT_BUCKET=relatorios
# - REPORT_REMOTE_PREFIX=manutencao
# - MKBF_META=7000
# ------------------------------------------------------------------------------

import os
import re
from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from supabase import create_client
from playwright.sync_api import sync_playwright


# ==============================================================================
# CONFIG
# ==============================================================================
SUPABASE_B_URL = os.getenv("SUPABASE_B_URL")
SUPABASE_B_SERVICE_ROLE_KEY = os.getenv("SUPABASE_B_SERVICE_ROLE_KEY")

# opcional, não é mais obrigatório
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


def _safe_filename(name: str) -> str:
    name = str(name or "").strip()
    name = re.sub(r"[^\w\-.() ]+", "_", name, flags=re.UNICODE)
    name = re.sub(r"\s+", "_", name)
    return name[:180] if name else "arquivo"


def _sb_b():
    return create_client(SUPABASE_B_URL, SUPABASE_B_SERVICE_ROLE_KEY)


def atualizar_status_relatorio(status: str, **fields):
    # agora é opcional: só atualiza se vier REPORT_ID
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


def _to_num(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    s = series.astype(str).str.strip()
    s = s.str.replace(".", "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    s = s.str.replace(r"[^0-9\.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


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
        1: "JANEIRO",
        2: "FEVEREIRO",
        3: "MARÇO",
        4: "ABRIL",
        5: "MAIO",
        6: "JUNHO",
        7: "JULHO",
        8: "AGOSTO",
        9: "SETEMBRO",
        10: "OUTUBRO",
        11: "NOVEMBRO",
        12: "DEZEMBRO",
    }
    return meses[d.month]


def periodo_label(d_ini: date, d_fim: date) -> str:
    return f"{_fmt_br_date(d_ini)} a {_fmt_br_date(d_fim)}"


# ==============================================================================
# REGRA DE OCORRÊNCIA (IGUAL AO DASHBOARD)
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


def tipo_display(oc):
    t = normalize_tipo(oc)
    return t if t else "N/D"


# ==============================================================================
# BUSCA DADOS - SUPABASE B
# ==============================================================================
def fetch_all_table_period(
    table_name: str,
    select_fields: str,
    date_field: str,
    start_date: date,
    end_date: date,
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

        print(
            f"📦 [{table_name}] range={offset}-{end} fetched={len(rows)} total={len(all_rows)}"
        )

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
        table_name="km_rodado_diario",
        select_fields="data, km_total",
        date_field="data",
        start_date=periodo_inicio,
        end_date=periodo_fim,
    )
    if not rows:
        return pd.DataFrame(columns=["data", "km_total"])
    return pd.DataFrame(rows)


def carregar_sos(periodo_inicio: date, periodo_fim: date) -> pd.DataFrame:
    rows = fetch_all_table_period(
        table_name="sos_acionamentos",
        select_fields="id, numero_sos, data_sos, ocorrencia, status",
        date_field="data_sos",
        start_date=periodo_inicio,
        end_date=periodo_fim,
    )
    if not rows:
        return pd.DataFrame(columns=["id", "numero_sos", "data_sos", "ocorrencia", "status"])
    return pd.DataFrame(rows)


# ==============================================================================
# PROCESSAMENTO PÁGINA 1
# ==============================================================================
def processar_km_df(df_km: pd.DataFrame) -> pd.DataFrame:
    df = df_km.copy()
    if df.empty:
        return pd.DataFrame(columns=["data", "km_total"])

    df["data"] = pd.to_datetime(df["data"], errors="coerce").dt.date
    df["km_total"] = _to_num(df["km_total"]).fillna(0)

    df = (
        df.dropna(subset=["data"])
        .groupby("data", as_index=False)
        .agg(km_total=("km_total", "sum"))
        .sort_values("data")
    )
    return df


def processar_sos_df(df_sos: pd.DataFrame) -> pd.DataFrame:
    df = df_sos.copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "id",
                "numero_sos",
                "data_sos",
                "ocorrencia",
                "status",
                "tipo_norm",
                "valida_mkbf",
            ]
        )

    df["data_sos"] = pd.to_datetime(df["data_sos"], errors="coerce").dt.date
    df["tipo_norm"] = df["ocorrencia"].apply(normalize_tipo)
    df["valida_mkbf"] = df["ocorrencia"].apply(is_ocorrencia_valida_para_mkbf)

    df = df.dropna(subset=["data_sos"]).copy()
    return df


def montar_diario_mkbf(df_km_proc: pd.DataFrame, df_sos_proc: pd.DataFrame) -> pd.DataFrame:
    km_dia = df_km_proc.copy()

    if df_sos_proc.empty:
        ocorr_dia = pd.DataFrame(columns=["data", "intervencoes"])
    else:
        ocorr_dia = (
            df_sos_proc[df_sos_proc["valida_mkbf"]]
            .groupby("data_sos", as_index=False)
            .size()
            .rename(columns={"data_sos": "data", "size": "intervencoes"})
        )

    diario = km_dia.merge(ocorr_dia, on="data", how="left")
    diario["intervencoes"] = diario["intervencoes"].fillna(0).astype(int)
    diario["mkbf"] = diario.apply(
        lambda r: (r["km_total"] / r["intervencoes"]) if r["intervencoes"] > 0 else 0,
        axis=1,
    )

    return diario.sort_values("data").reset_index(drop=True)


def resumo_periodo(df_diario: pd.DataFrame, df_sos_proc: pd.DataFrame) -> dict:
    km_total = float(df_diario["km_total"].sum()) if not df_diario.empty else 0.0
    interv = int(df_diario["intervencoes"].sum()) if not df_diario.empty else 0
    mkbf = (km_total / interv) if interv > 0 else 0.0

    por_tipo = (
        df_sos_proc[df_sos_proc["tipo_norm"].isin(TIPOS_GRAFICO)]
        .groupby("tipo_norm", as_index=False)
        .size()
        .rename(columns={"size": "total"})
        .sort_values("total", ascending=False)
    )

    por_tipo_map = {r["tipo_norm"]: int(r["total"]) for _, r in por_tipo.iterrows()}
    for t in TIPOS_GRAFICO:
        por_tipo_map.setdefault(t, 0)

    return {
        "km_total": km_total,
        "intervencoes": interv,
        "mkbf": mkbf,
        "por_tipo_map": por_tipo_map,
        "por_tipo_df": por_tipo,
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
    meses_hist = [add_months(mes_base, -i) for i in range(5, -1, -1)]

    for mes_dt in meses_hist:
        ini_m = month_start(mes_dt)
        fim_m = month_end(mes_dt)

        df_km_m = processar_km_df(carregar_km_rodado(ini_m, fim_m))
        df_sos_m = processar_sos_df(carregar_sos(ini_m, fim_m))
        diario_m = montar_diario_mkbf(df_km_m, df_sos_m)
        resumo_m = resumo_periodo(diario_m, df_sos_m)

        hist_rows.append(
            {
                "mes_dt": ini_m,
                "mes_label": f"{pt_month_name(ini_m)[:3]}/{str(ini_m.year)[2:]}",
                "mkbf": float(resumo_m["mkbf"]),
                "meta": float(MKBF_META),
                "km_total": float(resumo_m["km_total"]),
                "intervencoes": int(resumo_m["intervencoes"]),
            }
        )

    df_hist = pd.DataFrame(hist_rows)

    tipos_all = sorted(set(TIPOS_GRAFICO) | set(resumo_atual["por_tipo_map"]) | set(resumo_ant["por_tipo_map"]))
    rows_motivos = []
    for t in tipos_all:
        rows_motivos.append(
            {
                "tipo": t,
                "mes_anterior": int(resumo_ant["por_tipo_map"].get(t, 0)),
                "mes_atual": int(resumo_atual["por_tipo_map"].get(t, 0)),
            }
        )

    df_motivos = pd.DataFrame(rows_motivos).sort_values(
        ["mes_atual", "mes_anterior", "tipo"], ascending=[False, False, True]
    )

    def pct_var(atual, anterior):
        if anterior in (0, None):
            return 0.0
        return ((atual - anterior) / anterior) * 100.0

    variacoes = {
        "intervencoes_pct": pct_var(resumo_atual["intervencoes"], resumo_ant["intervencoes"]),
        "km_pct": pct_var(resumo_atual["km_total"], resumo_ant["km_total"]),
        "mkbf_pct": pct_var(resumo_atual["mkbf"], resumo_ant["mkbf"]),
    }

    aderencia_meta_pct = (resumo_atual["mkbf"] / MKBF_META * 100.0) if MKBF_META > 0 else 0.0

    return {
        "periodo_inicio": periodo_inicio,
        "periodo_fim": periodo_fim,
        "periodo_label": periodo_label(periodo_inicio, periodo_fim),
        "mes_atual_label": f"{pt_month_name(periodo_fim)}/{periodo_fim.year}",
        "mes_anterior_label": f"{pt_month_name(ini_ant)}/{ini_ant.year}",
        "resumo_atual": resumo_atual,
        "resumo_ant": resumo_ant,
        "df_diario_atual": diario_atual,
        "df_hist": df_hist,
        "df_motivos": df_motivos,
        "variacoes": variacoes,
        "aderencia_meta_pct": aderencia_meta_pct,
    }


# ==============================================================================
# GRÁFICO
# ==============================================================================
def gerar_grafico_mkbf_historico(df_hist: pd.DataFrame, caminho_img: Path):
    df = df_hist.copy()
    if df.empty:
        plt.figure(figsize=(10, 4.8))
        plt.title("Evolução Histórica do MKBF", fontsize=12, fontweight="bold")
        plt.text(0.5, 0.5, "Sem dados para exibição", ha="center", va="center", transform=plt.gca().transAxes)
        plt.tight_layout()
        plt.savefig(caminho_img, dpi=120)
        plt.close()
        return

    x = range(len(df))
    y_mkbf = df["mkbf"].tolist()
    y_meta = df["meta"].tolist()

    plt.figure(figsize=(10.5, 4.8))
    plt.plot(x, y_mkbf, marker="o", linewidth=2.8, label="MKBF", color="black")
    plt.plot(x, y_meta, marker="", linewidth=2, linestyle="--", label="Meta", color="#c0392b")

    offset = max(max(y_mkbf + y_meta) * 0.015, 60) if (y_mkbf + y_meta) else 60
    for i, v in enumerate(y_mkbf):
        plt.text(i, v + offset, f"{v:,.0f}".replace(",", "."), ha="center", fontsize=9)

    plt.xticks(list(x), df["mes_label"].tolist(), fontsize=9)
    plt.ylabel("MKBF", fontsize=10)
    plt.title("Evolução Histórica do MKBF", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.25, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(caminho_img, dpi=130)
    plt.close()


# ==============================================================================
# TEXTO DE CONSIDERAÇÕES
# ==============================================================================
def gerar_consideracoes_pagina_1(dados: dict) -> str:
    atual = dados["resumo_atual"]
    ant = dados["resumo_ant"]
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
        f"{atual['km_total']:,.0f} km rodados e MKBF de {atual['mkbf']:,.0f}. "
        f"Na comparação com {dados['mes_anterior_label']}, houve "
        f"{tendencia_mkbf} no indicador, com variação de {var['mkbf_pct']:+.1f}%. "
        f"A aderência frente à meta de {MKBF_META:,.0f} ficou em {ader:.1f}%, "
        f"mantendo o resultado {meta_txt} do patamar esperado. "
        f"Os principais ofensores do mês foram: {motivos_txt}."
    ).replace(",", ".")


# ==============================================================================
# HTML - APENAS PÁGINA 1
# ==============================================================================
def gerar_html_pagina_1(dados: dict, img_path: Path, html_path: Path):
    atual = dados["resumo_atual"]
    ant = dados["resumo_ant"]
    var = dados["variacoes"]
    ader = dados["aderencia_meta_pct"]
    motivos = dados["df_motivos"].copy()
    consideracoes = gerar_consideracoes_pagina_1(dados)

    img_src = img_path.name

    def fmt_int(v):
        try:
            return f"{int(round(float(v))):,}".replace(",", ".")
        except Exception:
            return "0"

    def fmt_pct(v):
        try:
            return f"{float(v):+.1f}%"
        except Exception:
            return "0,0%"

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

    rows_motivos = ""
    for _, r in motivos.iterrows():
        rows_motivos += f"""
        <tr>
            <td>{r['tipo']}</td>
            <td style="text-align:center;">{fmt_int(r['mes_anterior'])}</td>
            <td style="text-align:center;">{fmt_int(r['mes_atual'])}</td>
        </tr>
        """

    footer_right = (
        f"Relatório ID: {REPORT_ID}" if REPORT_ID else "Flash Report Manutenção · Página 1"
    )

    html = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
      <meta charset="UTF-8" />
      <title>Flash Report Manutenção - Página 1</title>
      <style>
        * {{
          box-sizing: border-box;
        }}
        body {{
          margin: 0;
          font-family: Arial, Helvetica, sans-serif;
          background: #f4f6f8;
          color: #111827;
        }}
        .page {{
          width: 210mm;
          min-height: 297mm;
          margin: 0 auto;
          background: #ffffff;
          padding: 8mm 10mm 8mm 10mm;
          position: relative;
        }}
        .header {{
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          border-bottom: 3px solid #111827;
          padding-bottom: 6px;
          margin-bottom: 10px;
        }}
        .title h1 {{
          margin: 0;
          font-size: 22px;
          line-height: 1;
          letter-spacing: 0.5px;
        }}
        .title .sub {{
          margin-top: 5px;
          font-size: 11px;
          color: #4b5563;
        }}
        .period-box {{
          min-width: 190px;
          text-align: right;
        }}
        .period-box .ref {{
          font-size: 10px;
          color: #6b7280;
          text-transform: uppercase;
          font-weight: bold;
        }}
        .period-box .val {{
          font-size: 18px;
          font-weight: bold;
          margin-top: 2px;
        }}

        .grid-top {{
          display: grid;
          grid-template-columns: 1.05fr 0.95fr;
          gap: 10px;
          margin-bottom: 10px;
        }}

        .card {{
          border: 1px solid #d1d5db;
          border-radius: 10px;
          overflow: hidden;
          background: #fff;
        }}
        .card-title {{
          padding: 8px 10px;
          background: #111827;
          color: white;
          font-size: 11px;
          font-weight: bold;
          text-transform: uppercase;
          letter-spacing: 0.6px;
        }}
        .card-body {{
          padding: 10px;
        }}

        table {{
          width: 100%;
          border-collapse: collapse;
          font-size: 12px;
        }}
        th {{
          background: #f3f4f6;
          color: #111827;
          text-transform: uppercase;
          font-size: 10px;
          letter-spacing: 0.4px;
          padding: 6px 7px;
          border: 1px solid #e5e7eb;
        }}
        td {{
          padding: 7px 7px;
          border: 1px solid #e5e7eb;
          vertical-align: middle;
        }}

        .metric-grid {{
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 8px;
        }}
        .metric {{
          border: 1px solid #e5e7eb;
          border-radius: 10px;
          padding: 10px;
          background: #fafafa;
        }}
        .metric .lbl {{
          font-size: 10px;
          color: #6b7280;
          text-transform: uppercase;
          font-weight: bold;
          letter-spacing: 0.4px;
        }}
        .metric .val {{
          margin-top: 6px;
          font-size: 24px;
          font-weight: 800;
          color: #111827;
        }}
        .metric .aux {{
          margin-top: 3px;
          font-size: 11px;
          color: #6b7280;
        }}

        .metric-wide {{
          margin-top: 8px;
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 8px;
        }}

        .badge {{
          display: inline-block;
          padding: 4px 8px;
          border-radius: 999px;
          font-size: 11px;
          font-weight: bold;
          color: white;
          background: #111827;
        }}

        .center {{
          text-align: center;
        }}
        .right {{
          text-align: right;
        }}
        .muted {{
          color: #6b7280;
        }}

        .chart-wrap {{
          padding: 8px;
          border: 1px solid #e5e7eb;
          border-radius: 10px;
          background: #fff;
        }}
        .chart-wrap img {{
          width: 100%;
          height: auto;
          display: block;
        }}

        .cons-box {{
          margin-top: 10px;
          border: 1px solid #d1d5db;
          border-radius: 10px;
          background: #fafafa;
          padding: 10px;
        }}
        .cons-title {{
          font-size: 11px;
          font-weight: bold;
          text-transform: uppercase;
          margin-bottom: 6px;
          color: #111827;
        }}
        .cons-text {{
          font-size: 12px;
          line-height: 1.55;
          color: #1f2937;
          text-align: justify;
        }}

        .footer {{
          position: absolute;
          left: 10mm;
          right: 10mm;
          bottom: 6mm;
          font-size: 10px;
          color: #6b7280;
          display: flex;
          justify-content: space-between;
          border-top: 1px solid #e5e7eb;
          padding-top: 4px;
        }}

        @page {{
          size: A4;
          margin: 0;
        }}
      </style>
    </head>
    <body>
      <div class="page">
        <div class="header">
          <div class="title">
            <h1>FLASH REPORT MANUTENÇÃO</h1>
            <div class="sub">Página 1 · Intervenções do mês / MKBF</div>
            <div class="sub">Período analisado: <b>{dados['periodo_label']}</b></div>
          </div>
          <div class="period-box">
            <div class="ref">Mês de referência</div>
            <div class="val">{dados['mes_atual_label']}</div>
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
                    <th>{dados['mes_anterior_label']}</th>
                    <th>{dados['mes_atual_label']}</th>
                    <th>Variação</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><b>Intervenções do mês</b></td>
                    <td class="center">{fmt_int(ant['intervencoes'])}</td>
                    <td class="center"><b>{fmt_int(atual['intervencoes'])}</b></td>
                    <td class="center" style="color:{cls_var(-var['intervencoes_pct'])}; font-weight:bold;">{fmt_pct(var['intervencoes_pct'])}</td>
                  </tr>
                  <tr>
                    <td><b>KM rodado</b></td>
                    <td class="center">{fmt_int(ant['km_total'])}</td>
                    <td class="center"><b>{fmt_int(atual['km_total'])}</b></td>
                    <td class="center" style="color:{cls_var(var['km_pct'])}; font-weight:bold;">{fmt_pct(var['km_pct'])}</td>
                  </tr>
                  <tr>
                    <td><b>MKBF</b></td>
                    <td class="center">{fmt_int(ant['mkbf'])}</td>
                    <td class="center"><b>{fmt_int(atual['mkbf'])}</b></td>
                    <td class="center" style="color:{cls_var(var['mkbf_pct'])}; font-weight:bold;">{fmt_pct(var['mkbf_pct'])}</td>
                  </tr>
                  <tr>
                    <td><b>Meta MKBF</b></td>
                    <td class="center">{fmt_int(MKBF_META)}</td>
                    <td class="center">{fmt_int(MKBF_META)}</td>
                    <td class="center muted">-</td>
                  </tr>
                </tbody>
              </table>

              <div class="metric-wide">
                <div class="metric">
                  <div class="lbl">Aderência à meta</div>
                  <div class="val">{ader:.1f}%</div>
                  <div class="aux">Meta de referência: {fmt_int(MKBF_META)}</div>
                </div>
                <div class="metric">
                  <div class="lbl">Status do mês</div>
                  <div class="val" style="font-size:20px;">
                    {"ATINGIU" if atual['mkbf'] >= MKBF_META else "ABAIXO"}
                  </div>
                  <div class="aux">Com base no MKBF consolidado do período</div>
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
                  <div class="val">{fmt_int(atual['intervencoes'])}</div>
                  <div class="aux">Ocorrências válidas no MKBF</div>
                </div>

                <div class="metric">
                  <div class="lbl">KM rodado</div>
                  <div class="val">{fmt_int(atual['km_total'])}</div>
                  <div class="aux">KM total consolidado</div>
                </div>

                <div class="metric">
                  <div class="lbl">MKBF</div>
                  <div class="val">{fmt_int(atual['mkbf'])}</div>
                  <div class="aux">KM ÷ intervenções</div>
                </div>
              </div>

              <div style="margin-top:10px;">
                <div class="badge">Meta MKBF: {fmt_int(MKBF_META)}</div>
              </div>

              <div style="margin-top:12px;">
                <table>
                  <thead>
                    <tr>
                      <th>Tipo de ocorrência</th>
                      <th>{dados['mes_anterior_label']}</th>
                      <th>{dados['mes_atual_label']}</th>
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

        <div class="card" style="margin-bottom:10px;">
          <div class="card-title">Evolução histórica do MKBF</div>
          <div class="card-body">
            <div class="chart-wrap">
              <img src="{img_src}" alt="Gráfico histórico do MKBF" />
            </div>
          </div>
        </div>

        <div class="cons-box">
          <div class="cons-title">Considerações</div>
          <div class="cons-text">{consideracoes}</div>
        </div>

        <div class="footer">
          <div>Gerado automaticamente · Página 1</div>
          <div>{footer_right}</div>
        </div>
      </div>
    </body>
    </html>
    """
    html_path.write_text(html, encoding="utf-8")
    print(f"✅ HTML salvo: {html_path}")


# ==============================================================================
# PDF
# ==============================================================================
def gerar_pdf_do_html(html_path: Path, pdf_path: Path):
    html_path = html_path.resolve()
    pdf_path = pdf_path.resolve()

    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        page.goto(html_path.as_uri(), wait_until="networkidle")
        page.wait_for_timeout(400)
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
        dados = processar_pagina_1(periodo_inicio, periodo_fim)

        out_dir = Path(PASTA_SAIDA)
        img_path = out_dir / "flash_manutencao_pagina1_mkbf.png"
        html_path = out_dir / "Flash_Report_Manutencao_Pagina_1.html"
        pdf_path = out_dir / "Flash_Report_Manutencao_Pagina_1.pdf"

        gerar_grafico_mkbf_historico(dados["df_hist"], img_path)
        gerar_html_pagina_1(dados, img_path, html_path)
        gerar_pdf_do_html(html_path, pdf_path)

        mes_ref = f"{periodo_fim.year}-{str(periodo_fim.month).zfill(2)}"
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        base_folder = f"{REMOTE_BASE_PREFIX}/{mes_ref}/pagina_1/{stamp}"

        upload_storage_b(img_path, f"{base_folder}/{img_path.name}", "image/png")
        upload_storage_b(html_path, f"{base_folder}/{html_path.name}", "text/html; charset=utf-8")
        upload_storage_b(pdf_path, f"{base_folder}/{pdf_path.name}", "application/pdf")

        atualizar_status_relatorio(
            "CONCLUIDO",
            arquivo_pdf_path=f"{base_folder}/{pdf_path.name}",
            arquivo_html_path=f"{base_folder}/{html_path.name}",
            arquivo_png_path=f"{base_folder}/{img_path.name}",
            erro_msg=None,
            mes_ref=mes_ref,
        )

        print("✅ [OK] Página 1 do Flash Report concluída.")
        print(f"📄 PDF: {pdf_path}")
        print(f"🌐 HTML: {html_path}")
        print(f"🖼️ PNG: {img_path}")
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
