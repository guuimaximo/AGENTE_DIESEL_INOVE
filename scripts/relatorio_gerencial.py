# scripts/relatorio_gerencial.py
import os
import re
import json
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

import vertexai
from vertexai.generative_models import GenerativeModel

from supabase import create_client
from playwright.sync_api import sync_playwright

try:
    from google.auth.exceptions import DefaultCredentialsError
except Exception:
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
REPORT_PERIODO_INICIO = os.getenv("REPORT_PERIODO_INICIO")
REPORT_PERIODO_FIM = os.getenv("REPORT_PERIODO_FIM")

PASTA_SAIDA = os.getenv("REPORT_OUTPUT_DIR", "Relatorios_Diesel_Final")
TABELA_ORIGEM = os.getenv("DIESEL_SOURCE_TABLE", "fato_kml_meta_ponderada_dia")
BUCKET_RELATORIOS = os.getenv("REPORT_BUCKET", "relatorios")

REMOTE_BASE_PREFIX = os.getenv("REPORT_REMOTE_PREFIX", "diesel")

REPORT_PAGE_SIZE = int(os.getenv("REPORT_PAGE_SIZE", "1000"))
REPORT_MAX_ROWS = int(os.getenv("REPORT_MAX_ROWS", "250000"))
REPORT_FETCH_WINDOW_DAYS = int(os.getenv("REPORT_FETCH_WINDOW_DAYS", "7"))

SUGESTOES_TABLE = os.getenv("SUGESTOES_TABLE", "diesel_sugestoes_acompanhamento")
VERTEX_SA_JSON = os.getenv("VERTEX_SA_JSON")


# ==============================================================================
# Helpers
# ==============================================================================
def _assert_env():
    missing = []
    for k in [
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
    s = str(motorista_val or "").strip()
    if not s:
        return "N/D"
    m = re.search(r"\b(\d{3,10})\b", s)
    if m:
        return m.group(1)
    return _safe_filename(s)[:40]


def _ensure_vertex_adc_if_possible():
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


def _safe_json(obj):
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except Exception:
            return {}
    return {}


def _json_num(v, default=0.0):
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def _fmt_int(v):
    try:
        return f"{int(round(float(v))):,}".replace(",", ".")
    except Exception:
        return "0"


def _fmt_num(v, dec=2):
    try:
        return f"{float(v):,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return f"0,{''.join(['0'] * dec)}"


def carregar_mapa_nomes():
    mapa = {}
    if not SUPABASE_A_URL or not SUPABASE_A_SERVICE_ROLE_KEY:
        return mapa

    try:
        sb = _sb_a()
        all_rows = []
        start = 0
        while True:
            end = start + 1000 - 1
            resp = sb.table("funcionarios").select("nr_cracha, nm_funcionario").range(start, end).execute()
            rows = resp.data or []
            all_rows.extend(rows)
            if len(rows) < 1000:
                break
            start += 1000

        for row in all_rows:
            chapa = str(row.get("nr_cracha") or "").strip()
            nome = str(row.get("nm_funcionario") or "").strip().upper()
            if chapa:
                mapa[chapa] = nome
    except Exception as e:
        print(f"❌ Erro ao ler tabela funcionarios: {e}")

    return mapa


def carregar_metas_consumo():
    try:
        sb_b = _sb_b()
        resp_b = sb_b.table("metas_consumo").select("*").execute()
        if resp_b.data:
            return pd.DataFrame(resp_b.data)
    except Exception:
        pass

    if SUPABASE_A_URL and SUPABASE_A_SERVICE_ROLE_KEY:
        try:
            sb = _sb_a()
            resp = sb.table("metas_consumo").select("*").execute()
            if resp.data:
                return pd.DataFrame(resp.data)
        except Exception as e:
            print(f"⚠️ Erro ao carregar metas_consumo: {e}")

    return pd.DataFrame()


def carregar_acompanhamentos():
    try:
        sb = _sb_b()
        resp = sb.table("diesel_acompanhamentos").select("*").execute()
        if resp.data:
            return pd.DataFrame(resp.data)
    except Exception:
        pass

    if SUPABASE_A_URL and SUPABASE_A_SERVICE_ROLE_KEY:
        try:
            sb = _sb_a()
            resp = sb.table("diesel_acompanhamentos").select("*").execute()
            if resp.data:
                return pd.DataFrame(resp.data)
        except Exception as e:
            print(f"⚠️ Erro ao carregar acompanhamentos: {e}")

    return pd.DataFrame()


def carregar_prompt_ia(prompt_id: str) -> str:
    try:
        sb = _sb_b()
        resp = sb.table("ia_prompts").select("prompt_text").eq("id", prompt_id).execute()
        if resp.data and len(resp.data) > 0:
            return resp.data[0]["prompt_text"]
    except Exception as e:
        print(f"⚠️ Erro ao buscar prompt {prompt_id} no banco: {e}")
    return ""


# ==============================================================================
# CHECKPOINTS / EVENTOS PÓS-ACOMPANHAMENTO
# ==============================================================================
def carregar_eventos_checkpoints(periodo_inicio: date | None, periodo_fim: date | None):
    sb = _sb_b()
    tipos_validos = ["PRONTUARIO_10", "PRONTUARIO_20", "PRONTUARIO_30"]

    try:
        resp = (
            sb.table("diesel_acompanhamento_eventos")
            .select("id, acompanhamento_id, created_at, tipo, observacoes, extra")
            .in_("tipo", tipos_validos)
            .order("created_at", desc=False)
            .execute()
        )
        eventos = pd.DataFrame(resp.data or [])
    except Exception as e:
        print(f"⚠️ Erro ao carregar diesel_acompanhamento_eventos: {e}")
        eventos = pd.DataFrame()

    if eventos.empty:
        return {
            "kpis": {
                "cp10_total": 0,
                "cp10_melhorou": 0,
                "cp10_piorou": 0,
                "cp10_sem_evolucao": 0,
                "cp10_litros_recuperados": 0.0,
                "cp10_delta_kml_medio": 0.0,

                "cp20_total": 0,
                "cp20_melhorou": 0,
                "cp20_piorou": 0,
                "cp20_sem_evolucao": 0,
                "cp20_litros_recuperados": 0.0,
                "cp20_delta_kml_medio": 0.0,

                "cp30_total": 0,
                "cp30_melhorou": 0,
                "cp30_piorou": 0,
                "cp30_sem_evolucao": 0,
                "cp30_litros_recuperados": 0.0,
                "cp30_delta_kml_medio": 0.0,

                "fase_lt_10": 0,
                "fase_cp10": 0,
                "fase_cp20": 0,
                "fase_cp30": 0,
                "fase_analise_final": 0,
            },
            "tabela_eventos": pd.DataFrame(),
            "cards_motoristas": [],
            "resumo_por_linha": {
                "PRONTUARIO_10": pd.DataFrame(),
                "PRONTUARIO_20": pd.DataFrame(),
                "PRONTUARIO_30": pd.DataFrame(),
            },
        }

    eventos["created_at_dt"] = pd.to_datetime(eventos["created_at"], errors="coerce", utc=True).dt.tz_convert(None)

    if periodo_inicio and periodo_fim:
        pi = pd.Timestamp(periodo_inicio)
        pf = pd.Timestamp(periodo_fim) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        eventos = eventos[(eventos["created_at_dt"] >= pi) & (eventos["created_at_dt"] <= pf)].copy()

    if eventos.empty:
        return {
            "kpis": {
                "cp10_total": 0,
                "cp10_melhorou": 0,
                "cp10_piorou": 0,
                "cp10_sem_evolucao": 0,
                "cp10_litros_recuperados": 0.0,
                "cp10_delta_kml_medio": 0.0,

                "cp20_total": 0,
                "cp20_melhorou": 0,
                "cp20_piorou": 0,
                "cp20_sem_evolucao": 0,
                "cp20_litros_recuperados": 0.0,
                "cp20_delta_kml_medio": 0.0,

                "cp30_total": 0,
                "cp30_melhorou": 0,
                "cp30_piorou": 0,
                "cp30_sem_evolucao": 0,
                "cp30_litros_recuperados": 0.0,
                "cp30_delta_kml_medio": 0.0,

                "fase_lt_10": 0,
                "fase_cp10": 0,
                "fase_cp20": 0,
                "fase_cp30": 0,
                "fase_analise_final": 0,
            },
            "tabela_eventos": pd.DataFrame(),
            "cards_motoristas": [],
            "resumo_por_linha": {
                "PRONTUARIO_10": pd.DataFrame(),
                "PRONTUARIO_20": pd.DataFrame(),
                "PRONTUARIO_30": pd.DataFrame(),
            },
        }

    eventos["extra_json"] = eventos["extra"].apply(_safe_json)
    eventos["comparativo"] = eventos["extra_json"].apply(lambda x: x.get("comparativo", {}) if isinstance(x, dict) else {})

    eventos["delta_kml"] = eventos["comparativo"].apply(lambda x: _json_num(x.get("delta_kml")))
    eventos["delta_desperdicio"] = eventos["comparativo"].apply(lambda x: _json_num(x.get("delta_desperdicio")))
    eventos["delta_desperdicio_pct"] = eventos["comparativo"].apply(lambda x: _json_num(x.get("delta_desperdicio_pct")))
    eventos["conclusao"] = eventos["comparativo"].apply(lambda x: str(x.get("conclusao") or "SEM_EVOLUCAO").upper())

    eventos["antes_kml"] = eventos["comparativo"].apply(
        lambda x: _json_num((x.get("antes_periodo") or {}).get("kml_real"))
    )
    eventos["depois_kml"] = eventos["comparativo"].apply(
        lambda x: _json_num((x.get("depois_periodo") or {}).get("kml_real"))
    )
    eventos["antes_desp"] = eventos["comparativo"].apply(
        lambda x: _json_num((x.get("antes_periodo") or {}).get("desperdicio"))
    )
    eventos["depois_desp"] = eventos["comparativo"].apply(
        lambda x: _json_num((x.get("depois_periodo") or {}).get("desperdicio"))
    )

    df_acomp = carregar_acompanhamentos()
    if not df_acomp.empty:
        keep_cols = [
            "id",
            "motorista_nome",
            "motorista_chapa",
            "status",
            "motivo",
            "metadata",
            "dt_inicio_monitoramento",
            "prontuario_10_gerado_em",
            "prontuario_20_gerado_em",
            "prontuario_30_gerado_em",
        ]
        keep_cols = [c for c in keep_cols if c in df_acomp.columns]
        df_acomp = df_acomp[keep_cols].copy()
        df_acomp = df_acomp.rename(columns={"id": "acomp_id"})
        eventos = eventos.merge(
            df_acomp,
            left_on="acompanhamento_id",
            right_on="acomp_id",
            how="left"
        )
    else:
        eventos["motorista_nome"] = ""
        eventos["motorista_chapa"] = ""
        eventos["status"] = ""
        eventos["motivo"] = ""
        eventos["metadata"] = None
        eventos["dt_inicio_monitoramento"] = None
        eventos["prontuario_10_gerado_em"] = None
        eventos["prontuario_20_gerado_em"] = None
        eventos["prontuario_30_gerado_em"] = None

    def get_linha_foco(row):
        md = row.get("metadata")
        md = _safe_json(md)
        if isinstance(md, dict):
            linha = md.get("linha_foco")
            if linha:
                return str(linha)
        motivo = str(row.get("motivo") or "")
        m = re.search(r"Linha\s+([A-Z0-9]+)", motivo, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
        return "-"

    eventos["linha_foco"] = eventos.apply(get_linha_foco, axis=1)

    kpis = {}
    for tipo, prefix in [
        ("PRONTUARIO_10", "cp10"),
        ("PRONTUARIO_20", "cp20"),
        ("PRONTUARIO_30", "cp30"),
    ]:
        sub = eventos[eventos["tipo"] == tipo].copy()

        kpis[f"{prefix}_total"] = int(len(sub))
        kpis[f"{prefix}_melhorou"] = int((sub["conclusao"] == "MELHOROU").sum())
        kpis[f"{prefix}_piorou"] = int((sub["conclusao"] == "PIOROU").sum())
        kpis[f"{prefix}_sem_evolucao"] = int((sub["conclusao"] == "SEM_EVOLUCAO").sum())
        kpis[f"{prefix}_litros_recuperados"] = float((-sub["delta_desperdicio"].clip(upper=0)).sum()) if not sub.empty else 0.0
        kpis[f"{prefix}_delta_kml_medio"] = float(sub["delta_kml"].mean()) if not sub.empty else 0.0

    fase_lt_10 = 0
    fase_cp10 = 0
    fase_cp20 = 0
    fase_cp30 = 0
    fase_analise_final = 0

    if not df_acomp.empty:
        agora = pd.Timestamp(datetime.utcnow().date())
        if "dt_inicio_monitoramento" in df_acomp.columns:
            df_acomp["dt_inicio_monitoramento"] = pd.to_datetime(
                df_acomp["dt_inicio_monitoramento"], errors="coerce", utc=True
            ).dt.tz_convert(None)

        df_acomp["dias_decorridos"] = (
            agora - df_acomp["dt_inicio_monitoramento"].dt.normalize()
        ).dt.days + 1

        df_acomp["status_norm"] = df_acomp["status"].astype(str).str.upper().str.strip()

        fase_lt_10 = int(len(df_acomp[
            (df_acomp["status_norm"] == "EM_MONITORAMENTO") &
            (df_acomp["dias_decorridos"] < 10) &
            (df_acomp["prontuario_10_gerado_em"].isna())
        ]))

        fase_cp10 = int(len(df_acomp[
            df_acomp["prontuario_10_gerado_em"].notna() &
            df_acomp["prontuario_20_gerado_em"].isna()
        ]))

        fase_cp20 = int(len(df_acomp[
            df_acomp["prontuario_20_gerado_em"].notna() &
            df_acomp["prontuario_30_gerado_em"].isna()
        ]))

        fase_cp30 = int(len(df_acomp[
            df_acomp["prontuario_30_gerado_em"].notna()
        ]))

        fase_analise_final = int(len(df_acomp[
            df_acomp["status_norm"].isin(["EM_ANALISE", "OK", "ENCERRADO", "ATAS"])
        ]))

    kpis["fase_lt_10"] = fase_lt_10
    kpis["fase_cp10"] = fase_cp10
    kpis["fase_cp20"] = fase_cp20
    kpis["fase_cp30"] = fase_cp30
    kpis["fase_analise_final"] = fase_analise_final

    tabela_eventos = eventos[[
        "created_at_dt",
        "tipo",
        "motorista_nome",
        "motorista_chapa",
        "linha_foco",
        "antes_kml",
        "depois_kml",
        "antes_desp",
        "depois_desp",
        "delta_kml",
        "delta_desperdicio",
        "conclusao"
    ]].copy()

    tabela_eventos = tabela_eventos.sort_values("created_at_dt", ascending=False)

    def resumir_por_linha(df_eventos: pd.DataFrame, tipo_checkpoint: str) -> pd.DataFrame:
        sub = df_eventos[df_eventos["tipo"] == tipo_checkpoint].copy()
        if sub.empty:
            return pd.DataFrame(columns=[
                "linha_foco",
                "qtd_motoristas",
                "antes_kml",
                "depois_kml",
                "delta_kml",
                "antes_desp",
                "depois_desp",
                "delta_desperdicio",
                "melhorou",
                "piorou",
                "sem_evolucao",
            ])

        resumo = (
            sub.groupby("linha_foco", dropna=False)
            .agg(
                qtd_motoristas=("motorista_chapa", "count"),
                antes_kml=("antes_kml", "mean"),
                depois_kml=("depois_kml", "mean"),
                delta_kml=("delta_kml", "mean"),
                antes_desp=("antes_desp", "mean"),
                depois_desp=("depois_desp", "mean"),
                delta_desperdicio=("delta_desperdicio", "mean"),
                melhorou=("conclusao", lambda x: int((x == "MELHOROU").sum())),
                piorou=("conclusao", lambda x: int((x == "PIOROU").sum())),
                sem_evolucao=("conclusao", lambda x: int((x == "SEM_EVOLUCAO").sum())),
            )
            .reset_index()
        )

        resumo["linha_foco"] = resumo["linha_foco"].fillna("-").astype(str)
        resumo = resumo.sort_values(
            ["delta_desperdicio", "delta_kml", "qtd_motoristas"],
            ascending=[True, False, False]
        ).reset_index(drop=True)

        return resumo

    resumo_por_linha = {
        "PRONTUARIO_10": resumir_por_linha(eventos, "PRONTUARIO_10"),
        "PRONTUARIO_20": resumir_por_linha(eventos, "PRONTUARIO_20"),
        "PRONTUARIO_30": resumir_por_linha(eventos, "PRONTUARIO_30"),
    }

    cards_motoristas = []
    for _, r in tabela_eventos.head(12).iterrows():
        cards_motoristas.append({
            "motorista_nome": str(r.get("motorista_nome") or "-"),
            "motorista_chapa": str(r.get("motorista_chapa") or "-"),
            "linha_foco": str(r.get("linha_foco") or "-"),
            "tipo": str(r.get("tipo") or "-"),
            "delta_kml": float(r.get("delta_kml") or 0),
            "delta_desperdicio": float(r.get("delta_desperdicio") or 0),
            "conclusao": str(r.get("conclusao") or "SEM_EVOLUCAO"),
            "data": r.get("created_at_dt"),
        })

    return {
        "kpis": kpis,
        "tabela_eventos": tabela_eventos,
        "cards_motoristas": cards_motoristas,
        "resumo_por_linha": resumo_por_linha,
    }


# ==============================================================================
# CÁLCULO DE DETALHES
# ==============================================================================
def calcular_detalhes_json(df_motorista):
    if df_motorista.empty:
        return {}

    grp = (
        df_motorista.groupby(["linha", "Cluster"])
        .agg(
            {
                "Km": "sum",
                "Comb.": "sum",
                "veiculo": lambda x: list(x.unique())[0],
                "KML_Ref": "mean",
                "Meta_Linha": "mean",
            }
        )
        .reset_index()
    )

    grp["kml_real"] = grp["Km"] / grp["Comb."]

    def calc_waste_ref(row):
        meta = row["KML_Ref"]
        if meta > 0 and row["kml_real"] < meta:
            return row["Comb."] - (row["Km"] / meta)
        return 0.0

    def calc_waste_meta(row):
        m = row.get("Meta_Linha", 0)
        if m > 0 and row["kml_real"] < m:
            return row["Comb."] - (row["Km"] / m)
        return 0.0

    grp["desperdicio"] = grp.apply(calc_waste_ref, axis=1)
    grp["desp_meta"] = grp.apply(calc_waste_meta, axis=1)

    raio_x = []
    for _, row in grp.sort_values("desp_meta", ascending=False).iterrows():
        raio_x.append(
            {
                "linha": str(row["linha"]),
                "cluster": str(row["Cluster"]),
                "km": float(row["Km"]),
                "litros": float(row["Comb."]),
                "kml_real": float(row["kml_real"]),
                "kml_meta": float(row["KML_Ref"]),
                "desperdicio": float(row["desperdicio"]),
                "meta_linha_oficial": float(row.get("Meta_Linha", 0)),
                "desp_meta_oficial": float(row.get("desp_meta", 0)),
            }
        )

    df_chart = df_motorista.copy()
    df_chart["Semana_Dt"] = df_chart["Date"].dt.to_period("W").apply(lambda r: r.start_time)

    grp_sem = (
        df_chart.groupby("Semana_Dt")
        .agg({"Km": "sum", "Comb.": "sum", "KML_Ref": "mean", "Meta_Linha": "mean"})
        .sort_index()
    )

    grafico = []
    for dt, row in grp_sem.iterrows():
        kml_real = row["Km"] / row["Comb."] if row["Comb."] > 0 else 0
        grafico.append(
            {
                "label": dt.strftime("%d/%m"),
                "real": float(kml_real),
                "meta": float(row["KML_Ref"]),
                "meta_linha": float(row.get("Meta_Linha", 0)),
            }
        )

    return {"raio_x": raio_x, "grafico_semanal": grafico}


def gerar_sugestoes_acompanhamento(dados_proc: dict) -> pd.DataFrame:
    df_atual = dados_proc.get("df_atual")
    if df_atual is None or df_atual.empty:
        return pd.DataFrame(
            columns=[
                "chapa",
                "linha_mais_rodada",
                "km_percorrido",
                "consumo_realizado",
                "kml_realizado",
                "kml_meta",
                "combustivel_desperdicado",
                "meta_linha_oficial",
                "desp_meta_oficial",
                "motorista_nome",
                "cluster",
                "detalhes_json",
            ]
        )

    df = df_atual.copy()

    df["Km"] = pd.to_numeric(df["Km"], errors="coerce").fillna(0)
    df["Comb."] = pd.to_numeric(df["Comb."], errors="coerce").fillna(0)
    df["kml"] = pd.to_numeric(df["kml"], errors="coerce")
    df["KML_Ref"] = pd.to_numeric(df["KML_Ref"], errors="coerce")
    df["Litros_Desperdicio"] = pd.to_numeric(df["Litros_Desperdicio"], errors="coerce").fillna(0)
    df["Meta_Linha"] = pd.to_numeric(df.get("Meta_Linha", 0), errors="coerce").fillna(0)
    df["Litros_Desp_Meta"] = pd.to_numeric(df.get("Litros_Desp_Meta", 0), errors="coerce").fillna(0)

    df["chapa"] = df["Motorista"].apply(_extract_chapa)

    mapa_nomes = carregar_mapa_nomes()

    def resolver_nome(row):
        nome_db = mapa_nomes.get(row["chapa"])
        if nome_db:
            return nome_db
        return row["Motorista"]

    df["Motorista_Final"] = df.apply(resolver_nome, axis=1)

    agg = (
        df.groupby(["chapa"], as_index=False)
        .agg(
            km_percorrido=("Km", "sum"),
            consumo_realizado=("Comb.", "sum"),
            combustivel_desperdicado=("Litros_Desperdicio", "sum"),
            meta_linha_oficial=("Meta_Linha", "mean"),
            desp_meta_oficial=("Litros_Desp_Meta", "sum"),
            motorista_nome=("Motorista_Final", "first"),
            cluster=("Cluster", "first"),
            kml_meta=("KML_Ref", "mean"),
        )
    )

    agg["kml_realizado"] = agg["km_percorrido"] / agg["consumo_realizado"]

    print("⚙️  [Sugestões] Calculando Raio-X e Gráficos detalhados...")
    detalhes_map = {}
    for chapa in agg["chapa"].unique():
        df_mot = df[df["chapa"] == chapa]
        detalhes_map[chapa] = json.dumps(calcular_detalhes_json(df_mot))

    agg["detalhes_json"] = agg["chapa"].map(detalhes_map)

    linha_top = (
        df.groupby(["chapa", "linha"], as_index=False)["Km"]
        .sum()
        .sort_values(["chapa", "Km"], ascending=[True, False])
    )
    linha_top = (
        linha_top.drop_duplicates("chapa", keep="first")[["chapa", "linha"]].rename(columns={"linha": "linha_mais_rodada"})
    )

    agg = agg.merge(linha_top, on="chapa", how="left")
    agg = agg.sort_values(["desp_meta_oficial", "combustivel_desperdicado"], ascending=[False, False])

    agg = (
        agg[
            [
                "chapa",
                "linha_mais_rodada",
                "km_percorrido",
                "consumo_realizado",
                "kml_realizado",
                "kml_meta",
                "combustivel_desperdicado",
                "meta_linha_oficial",
                "desp_meta_oficial",
                "motorista_nome",
                "cluster",
                "detalhes_json",
            ]
        ]
        .reset_index(drop=True)
    )

    agg = agg.dropna(subset=["chapa"])
    return agg


def salvar_sugestoes_supabase_b(df_sug: pd.DataFrame, mes_ref: str, periodo_inicio: date | None, periodo_fim: date | None):
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

        rows.append(
            {
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
                "extra": {
                    "meta_linha_oficial": float(r.get("meta_linha_oficial", 0)),
                    "desp_meta_oficial": float(r.get("desp_meta_oficial", 0)),
                },
            }
        )

    sb.table(SUGESTOES_TABLE).upsert(rows, on_conflict="mes_ref,chapa").execute()
    print(f"✅ [Sugestões] Salvas/atualizadas: {len(rows)} linhas (mes_ref={mes_ref}).")


# ==============================================================================
# 0) BUSCA DADOS SUPABASE B -> DF
# ==============================================================================
def carregar_dados_supabase_b(periodo_inicio: date | None, periodo_fim: date | None) -> pd.DataFrame:
    sb = _sb_b()

    if not periodo_inicio or not periodo_fim:
        hoje = datetime.utcnow().date()
        periodo_inicio = periodo_inicio or hoje.replace(day=1)
        periodo_fim = periodo_fim or hoje

    if periodo_inicio > periodo_fim:
        periodo_inicio, periodo_fim = periodo_fim, periodo_inicio

    select_fields = """
        dia,
        ano,
        mes,
        anomes,
        motorista,
        linha,
        prefixo,
        fabricante,
        cluster,
        km_rodado,
        litros_consumidos,
        km_l,
        meta_kml_usada,
        litros_ideais,
        minutos_em_viagem
    """

    all_rows = []
    pages = 0
    start = 0

    while True:
        end = start + REPORT_PAGE_SIZE - 1

        q = (
            sb.table(TABELA_ORIGEM)
            .select(select_fields)
            .gte("dia", str(periodo_inicio))
            .lte("dia", str(periodo_fim))
            .order("dia", desc=False)
            .order("linha", desc=False)
            .order("motorista", desc=False)
            .range(start, end)
        )

        resp = q.execute()
        rows = resp.data or []
        pages += 1
        all_rows.extend(rows)

        print(
            f"📦 [SupabaseB] período={periodo_inicio}..{periodo_fim} "
            f"page={pages} range={start}-{end} fetched={len(rows)} total={len(all_rows)}"
        )

        if len(rows) < REPORT_PAGE_SIZE:
            break

        if len(all_rows) >= REPORT_MAX_ROWS:
            all_rows = all_rows[:REPORT_MAX_ROWS]
            print(f"⚠️ [SupabaseB] REPORT_MAX_ROWS atingido: {REPORT_MAX_ROWS}")
            break

        start += REPORT_PAGE_SIZE

    if not all_rows:
        return pd.DataFrame(
            columns=[
                "Date",
                "Motorista",
                "veiculo",
                "linha",
                "kml",
                "Km",
                "Comb.",
                "Cluster",
                "Meta_Linha",
                "Litros_Esperados",
            ]
        )

    df = pd.DataFrame(all_rows)

    out = pd.DataFrame()
    out["Date"] = df.get("dia")
    out["Motorista"] = df.get("motorista").fillna("SEM_MOTORISTA")
    out["veiculo"] = df.get("prefixo")
    out["linha"] = df.get("linha")
    out["kml"] = df.get("km_l")
    out["Km"] = df.get("km_rodado")
    out["Comb."] = df.get("litros_consumidos")
    out["Cluster"] = df.get("cluster")
    out["Meta_Linha"] = df.get("meta_kml_usada")
    out["Litros_Esperados"] = df.get("litros_ideais")

    return out


# ==============================================================================
# 1) PROCESSAMENTO
# ==============================================================================
def processar_dados_gerenciais_df(df: pd.DataFrame, periodo_inicio: date | None, periodo_fim: date | None):
    print("⚙️  [Sistema] Processando dados para visão gerencial...")

    obrigatorias = ["Date", "Motorista", "veiculo", "linha", "kml", "Km", "Comb."]
    faltando = [c for c in obrigatorias if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas obrigatórias ausentes: {faltando}")

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

    if "Cluster" not in df.columns or df["Cluster"].isna().all():
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
    else:
        df["Cluster"] = df["Cluster"].astype(str).str.upper().str.strip()

    qtd_cluster_invalido = int(df["Cluster"].isna().sum())
    df = df.dropna(subset=["Cluster"])
    df = df.dropna(subset=["Km", "Comb."])
    df = df[(df["Km"] > 0) & (df["Comb."] > 0)].copy()
    df["kml"] = df["Km"] / df["Comb."]

    df_clean = df[(df["kml"] >= 1.5) & (df["kml"] <= 5)].copy()
    if df_clean.empty:
        raise ValueError("Sem dados válidos após filtros.")

    if periodo_inicio and periodo_fim:
        periodo_txt = f"{_fmt_br_date(periodo_inicio)} a {_fmt_br_date(periodo_fim)}"
    else:
        periodo_txt = f"{df_clean['Date'].min().strftime('%d/%m/%Y')} a {df_clean['Date'].max().strftime('%d/%m/%Y')}"

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
            .agg(Qtd_Contaminacoes=("kml", "count"), KML_Min=("kml", "min"), KML_Max=("kml", "max"))
            .reset_index()
            .sort_values("Qtd_Contaminacoes", ascending=False)
            .head(10)
        )
    else:
        top_veiculos_contaminados = pd.DataFrame(
            columns=["veiculo", "Cluster", "linha", "Qtd_Contaminacoes", "KML_Min", "KML_Max"]
        )

    df_clean["Mes_Ano"] = df_clean["Date"].dt.to_period("M")

    if "Meta_Linha" in df_clean.columns:
        df_clean["Meta_Linha"] = pd.to_numeric(df_clean["Meta_Linha"], errors="coerce").fillna(0.0)
    else:
        df_clean["Meta_Linha"] = 0.0

    if "Litros_Esperados" in df_clean.columns:
        df_clean["Litros_Esperados"] = pd.to_numeric(df_clean["Litros_Esperados"], errors="coerce").fillna(0.0)
    else:
        df_clean["Litros_Esperados"] = 0.0

    def calc_desp_meta(r):
        m = r.get("Meta_Linha", 0.0)
        litros_ideais = r.get("Litros_Esperados", 0.0)

        if m > 0 and litros_ideais > 0 and r["Comb."] > litros_ideais:
            return r["Comb."] - litros_ideais
        return 0.0

    df_clean["Litros_Desp_Meta"] = df_clean.apply(calc_desp_meta, axis=1)

    tabela_cluster = df_clean.groupby(["Cluster", "Mes_Ano"]).agg({"Km": "sum", "Comb.": "sum"}).reset_index()
    tabela_cluster["KML"] = tabela_cluster["Km"] / tabela_cluster["Comb."]
    tabela_pivot = tabela_cluster.pivot(index="Cluster", columns="Mes_Ano", values="KML")

    linha_agg = (
        df_clean.groupby(["linha", "Mes_Ano"])
        .agg({"Km": "sum", "Comb.": "sum", "Litros_Esperados": "sum", "Litros_Desp_Meta": "sum"})
        .reset_index()
    )
    linha_agg["KML"] = linha_agg["Km"] / linha_agg["Comb."]
    linha_agg["Meta_Ponderada"] = linha_agg.apply(
        lambda r: r["Km"] / r["Litros_Esperados"] if r["Litros_Esperados"] > 0 else 0.0,
        axis=1
    )

    meses_disponiveis = sorted(linha_agg["Mes_Ano"].unique())
    if len(meses_disponiveis) >= 2:
        mes_atual = meses_disponiveis[-1]
        mes_anterior = meses_disponiveis[-2]
    else:
        mes_atual = meses_disponiveis[-1] if meses_disponiveis else None
        mes_anterior = None

    linhas_list = []
    for linha in linha_agg["linha"].unique():
        df_l = linha_agg[linha_agg["linha"] == linha]

        row_atual = df_l[df_l["Mes_Ano"] == mes_atual]
        row_ant = df_l[df_l["Mes_Ano"] == mes_anterior] if mes_anterior else pd.DataFrame()

        kml_atual = float(row_atual["KML"].iloc[0]) if not row_atual.empty else 0.0
        kml_ant = float(row_ant["KML"].iloc[0]) if not row_ant.empty else 0.0
        meta_pond = float(row_atual["Meta_Ponderada"].iloc[0]) if not row_atual.empty else 0.0
        desp = float(row_atual["Litros_Desp_Meta"].iloc[0]) if not row_atual.empty else 0.0

        var = ((kml_atual - kml_ant) / kml_ant * 100) if kml_ant > 0 else 0.0

        linhas_list.append(
            {
                "linha": linha,
                "KML_Anterior": kml_ant,
                "KML_Atual": kml_atual,
                "Variacao_Pct": var,
                "Meta_Ponderada": meta_pond,
                "Desperdicio": desp,
            }
        )

    tabela_linhas = pd.DataFrame(linhas_list)
    tabela_linhas = tabela_linhas.sort_values("Desperdicio", ascending=False)

    ultimo_mes = df_clean["Mes_Ano"].max()
    df_atual = df_clean[df_clean["Mes_Ano"] == ultimo_mes].copy()

    ref_grupo = df_atual.groupby(["linha", "Cluster"]).agg({"Km": "sum", "Comb.": "sum"}).reset_index()
    ref_grupo["KML_Ref"] = ref_grupo["Km"] / ref_grupo["Comb."]
    ref_grupo.rename(columns={"Km": "KM_Total_Linha"}, inplace=True)
    df_atual = pd.merge(
        df_atual,
        ref_grupo[["linha", "Cluster", "KML_Ref", "KM_Total_Linha"]],
        on=["linha", "Cluster"],
        how="left"
    )

    def calc_desperdicio_ref(r):
        try:
            if r["KML_Ref"] > 0 and r["kml"] < r["KML_Ref"]:
                return r["Comb."] - (r["Km"] / r["KML_Ref"])
        except Exception:
            pass
        return 0

    df_atual["Litros_Desperdicio"] = df_atual.apply(calc_desperdicio_ref, axis=1)
    total_desperdicio = float(df_atual["Litros_Desp_Meta"].sum() or 0)

    top_veiculos = (
        df_atual.groupby(["veiculo", "Cluster", "linha"])
        .agg(
            {
                "Litros_Desperdicio": "sum",
                "Litros_Desp_Meta": "sum",
                "Km": "sum",
                "Comb.": "sum",
                "KML_Ref": "mean",
                "Meta_Linha": "mean",
            }
        )
        .reset_index()
    )
    top_veiculos["KML_Real"] = top_veiculos["Km"] / top_veiculos["Comb."]
    top_veiculos["KML_Meta"] = top_veiculos["KML_Ref"]
    top_veiculos = top_veiculos.sort_values("Litros_Desp_Meta", ascending=False).head(10)

    top_motoristas = (
        df_atual.groupby(["Motorista", "Cluster", "linha", "KM_Total_Linha"])
        .agg(
            {
                "Litros_Desperdicio": "sum",
                "Litros_Desp_Meta": "sum",
                "Km": "sum",
                "Comb.": "sum",
                "KML_Ref": "mean",
                "Meta_Linha": "mean",
            }
        )
        .reset_index()
    )
    top_motoristas["KML_Real"] = top_motoristas["Km"] / top_motoristas["Comb."]
    top_motoristas["KML_Meta"] = top_motoristas["KML_Ref"]
    top_motoristas["Impacto_Pct"] = (top_motoristas["Km"] / top_motoristas["KM_Total_Linha"]) * 100
    top_motoristas = top_motoristas.sort_values("Litros_Desp_Meta", ascending=False).head(10)

    df_acomp = carregar_acompanhamentos()
    instrutor_kpis = {
        "aguardando": 0,
        "em_andamento": 0,
        "concluidos": 0,
        "dias_com_acao": [],
        "evoluiram": 0,
        "nao_evoluiram": 0,
        "tabela_evolucao": pd.DataFrame(),
    }

    if not df_acomp.empty:
        df_acomp["status_norm"] = df_acomp["status"].astype(str).str.upper().str.strip()
        df_acomp["status_norm"] = df_acomp["status_norm"].replace(
            {
                "AGUARDANDO INSTRUTOR": "AGUARDANDO_INSTRUTOR",
                "AG_ACOMPANHAMENTO": "AGUARDANDO_INSTRUTOR",
                "TRATATIVA": "ATAS",
                "CONCLUIDO": "OK",
            }
        )

        instrutor_kpis["aguardando"] = len(df_acomp[df_acomp["status_norm"] == "AGUARDANDO_INSTRUTOR"])
        instrutor_kpis["em_andamento"] = len(df_acomp[df_acomp["status_norm"] == "EM_MONITORAMENTO"])
        instrutor_kpis["concluidos"] = len(df_acomp[df_acomp["status_norm"].isin(["OK", "ENCERRADO", "ATAS"])])

        df_acomp["dt_inicio"] = (
            pd.to_datetime(df_acomp["dt_inicio_monitoramento"], errors="coerce", utc=True)
            .dt.tz_convert(None)
        )

        ativos = df_acomp[df_acomp["status_norm"].isin(["EM_MONITORAMENTO", "OK", "ENCERRADO", "ATAS", "EM_ANALISE"])].copy()

        if periodo_inicio and periodo_fim:
            pi_ts = pd.Timestamp(periodo_inicio)
            pf_ts = pd.Timestamp(periodo_fim) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            ativos_periodo = ativos[(ativos["dt_inicio"] >= pi_ts) & (ativos["dt_inicio"] <= pf_ts)]
        else:
            ativos_periodo = ativos

        if not ativos_periodo.empty:
            dias_unicos = ativos_periodo["dt_inicio"].dt.normalize().dropna().unique()
            instrutor_kpis["dias_com_acao"] = sorted([pd.Timestamp(d).strftime("%d/%m") for d in dias_unicos])

        df_atual_kml = df_atual.groupby("Motorista").agg({"Km": "sum", "Comb.": "sum"}).reset_index()
        df_atual_kml["KML_Atual"] = df_atual_kml["Km"] / df_atual_kml["Comb."]
        df_atual_kml["chapa"] = df_atual_kml["Motorista"].apply(_extract_chapa)
        df_atual_kml = df_atual_kml.groupby("chapa")["KML_Atual"].mean().reset_index()

        if "motorista_chapa" in ativos.columns:
            evolucao = pd.merge(ativos, df_atual_kml, left_on="motorista_chapa", right_on="chapa", how="inner")
            if not evolucao.empty:
                evolucao["kml_inicial"] = pd.to_numeric(evolucao.get("kml_inicial", 0), errors="coerce").fillna(0)
                evolucao["melhoria"] = evolucao["KML_Atual"] - evolucao["kml_inicial"]

                valid_evo = evolucao[evolucao["kml_inicial"] > 0].copy()
                instrutor_kpis["evoluiram"] = len(valid_evo[valid_evo["melhoria"] > 0])
                instrutor_kpis["nao_evoluiram"] = len(valid_evo[valid_evo["melhoria"] <= 0])

                hoje_naive = pd.Timestamp(datetime.utcnow().date())
                valid_evo["dias_monitorados"] = (hoje_naive - valid_evo["dt_inicio"].dt.normalize()).dt.days
                valid_evo["dt_inicio_fmt"] = valid_evo["dt_inicio"].dt.strftime("%d/%m/%Y")

                tabela_evo = valid_evo[
                    [
                        "motorista_nome",
                        "motorista_chapa",
                        "status_norm",
                        "dt_inicio_fmt",
                        "dias_monitorados",
                        "kml_inicial",
                        "KML_Atual",
                        "melhoria",
                    ]
                ].copy()

                tabela_evo = tabela_evo.sort_values("melhoria", ascending=False)
                instrutor_kpis["tabela_evolucao"] = tabela_evo

    checkpoint_data = carregar_eventos_checkpoints(periodo_inicio, periodo_fim)

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
        "tabela_linhas": tabela_linhas,
        "top_motoristas": top_motoristas,
        "top_veiculos_contaminados": top_veiculos_contaminados,
        "instrutor_kpis": instrutor_kpis,
        "checkpoint_kpis": checkpoint_data["kpis"],
        "checkpoint_tabela": checkpoint_data["tabela_eventos"],
        "checkpoint_cards": checkpoint_data["cards_motoristas"],
        "checkpoint_resumo_por_linha": checkpoint_data["resumo_por_linha"],
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
    evolucao_cluster = df_clean.groupby(["Semana", "Cluster"]).agg({"Km": "sum", "Comb.": "sum"}).reset_index()
    evolucao_cluster["KML"] = evolucao_cluster["Km"] / evolucao_cluster["Comb."]
    pivot_chart = evolucao_cluster.pivot(index="Semana", columns="Cluster", values="KML")

    evolucao_geral = df_clean.groupby(["Semana"]).agg({"Km": "sum", "Comb.": "sum"}).reset_index()
    evolucao_geral["KML"] = evolucao_geral["Km"] / evolucao_geral["Comb."]

    plt.figure(figsize=(10, 5))
    cores = {"C11": "#e67e22", "C10": "#2ecc71", "C9": "#3498db", "C8": "#9b59b6", "C6": "#95a5a6"}
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
        plt.text(x, y + 0.05, f"{y:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color="black")

    plt.title("Evolução de Eficiência: Clusters vs Média Frota", fontsize=12, fontweight="bold")
    plt.xlabel("Semana")
    plt.ylabel("KM/L")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(caminho_img, dpi=110)
    plt.close()


# ==============================================================================
# 3) IA
# ==============================================================================
def consultar_ia_gerencial(dados_proc: dict) -> str:
    print("🧠 [Gerente] Solicitando análise estratégica à IA...")
    if not VERTEX_PROJECT_ID:
        return "<p><b>Visão Geral da Eficiência no Período</b><br>IA desativada.</p>"
    _ensure_vertex_adc_if_possible()

    try:
        df_clean = dados_proc["df_clean"].copy()
        km_total_periodo = float(df_clean["Km"].sum() or 0)
        comb_total_periodo = float(df_clean["Comb."].sum() or 0)
        kml_periodo = km_total_periodo / comb_total_periodo if comb_total_periodo > 0 else 0.0
        if "Mes_Ano" not in df_clean.columns:
            df_clean["Mes_Ano"] = df_clean["Date"].dt.to_period("M")
        mensal = df_clean.groupby("Mes_Ano").agg({"Km": "sum", "Comb.": "sum"}).reset_index()
        mensal["KML"] = mensal["Km"] / mensal["Comb."]
        mensal = mensal.sort_values("Mes_Ano")

        kml_mes_atual = float(mensal.iloc[-1]["KML"]) if len(mensal) >= 1 else 0.0
        mes_atual_label = str(mensal.iloc[-1]["Mes_Ano"]) if len(mensal) >= 1 else "N/D"
        kml_mes_anterior = float(mensal.iloc[-2]["KML"]) if len(mensal) >= 2 else 0.0
        delta_kml_mes = ((kml_mes_atual - kml_mes_anterior) / kml_mes_anterior * 100) if kml_mes_anterior > 0 else 0.0

        df_clean["Semana"] = df_clean["Date"].dt.to_period("W").apply(lambda r: r.start_time)
        semanal = df_clean.groupby("Semana").agg({"Km": "sum", "Comb.": "sum"}).reset_index()
        semanal["KML"] = semanal["Km"] / semanal["Comb."]
        semanal = semanal.sort_values("Semana")

        kml_semana_atual = float(semanal.iloc[-1]["KML"]) if len(semanal) >= 1 else 0.0
        semana_atual_inicio_txt = semanal.iloc[-1]["Semana"].strftime("%d/%m/%Y") if len(semanal) >= 1 else "N/D"
        kml_semana_anterior = float(semanal.iloc[-2]["KML"]) if len(semanal) >= 2 else 0.0
        delta_kml_semana = ((kml_semana_atual - kml_semana_anterior) / kml_semana_anterior * 100) if kml_semana_anterior > 0 else 0.0

        vertexai.init(project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION)
        model = GenerativeModel(VERTEX_MODEL)

        template_prompt = carregar_prompt_ia("gerencial_diesel")

        if not template_prompt:
            template_prompt = """Você é Diretor de Operações de uma empresa de transporte urbano, especialista em eficiência energética (KM/L).

Analise a performance de KM/L no período: {periodo}
MÊS DE REFERÊNCIA: {mes_atual_nome}

VISÃO GERAL (FROTA):
- KM/L médio período: {kml_periodo}
- KM/L mês atual ({mes_atual_label}): {kml_mes_atual}
- Variação mês atual vs anterior: {delta_kml_mes}
- KM/L última semana (início {semana_atual_inicio_txt}): {kml_semana_atual}
- Variação última semana vs anterior: {delta_kml_semana}
- Desperdício total estimado em relação à Meta Oficial: {total_desperdicio} litros

TOP OFENSORES DO MÊS:
- TOP 5 VEÍCULOS:
{top_veiculos}

- TOP 5 MOTORISTAS:
{top_motoristas}

CHECKPOINTS:
- <10 dias: {fase_lt_10}
- CP10: {fase_cp10}
- CP20: {fase_cp20}
- CP30: {fase_cp30}
- Em análise final: {fase_analise_final}

Gere um resumo executivo em HTML usando apenas: <p>, <b>, <br>, <ul>, <li>.
"""

        prompt = template_prompt
        mapeamento = {
            "{periodo}": dados_proc["periodo"],
            "{mes_atual_nome}": dados_proc["mes_atual_nome"],
            "{kml_periodo}": f"{kml_periodo:.2f}",
            "{mes_atual_label}": mes_atual_label,
            "{kml_mes_atual}": f"{kml_mes_atual:.2f}",
            "{delta_kml_mes}": f"{delta_kml_mes:+.1f}%",
            "{semana_atual_inicio_txt}": semana_atual_inicio_txt,
            "{kml_semana_atual}": f"{kml_semana_atual:.2f}",
            "{delta_kml_semana}": f"{delta_kml_semana:+.1f}%",
            "{total_desperdicio}": f"{dados_proc['total_desperdicio']:.0f}",
            "{top_veiculos}": dados_proc["top_veiculos"].head(5).to_markdown(),
            "{top_motoristas}": dados_proc["top_motoristas"].head(5).to_markdown(),
            "{fase_lt_10}": str(dados_proc["checkpoint_kpis"].get("fase_lt_10", 0)),
            "{fase_cp10}": str(dados_proc["checkpoint_kpis"].get("fase_cp10", 0)),
            "{fase_cp20}": str(dados_proc["checkpoint_kpis"].get("fase_cp20", 0)),
            "{fase_cp30}": str(dados_proc["checkpoint_kpis"].get("fase_cp30", 0)),
            "{fase_analise_final}": str(dados_proc["checkpoint_kpis"].get("fase_analise_final", 0)),
        }

        for chave, valor in mapeamento.items():
            prompt = prompt.replace(chave, str(valor))

        resp = model.generate_content(prompt)
        texto = getattr(resp, "text", None) or "Análise indisponível."
        return texto.replace("```html", "").replace("```", "")
    except DefaultCredentialsError:
        print("⚠️ [IA] Credenciais ADC não encontradas.")
        return "<p>IA desativada. Relatório gerado apenas com dados.</p>"
    except Exception as e:
        print("❌ Erro ao chamar IA:", repr(e))
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
                val_str = fmt.format(val) if isinstance(val, (int, float)) else str(val)
                style = ""

                if col == "Variacao_Pct":
                    try:
                        v = float(val)
                    except Exception:
                        v = 0
                    style = (
                        "color: #c0392b; font-weight: bold;"
                        if v < -5
                        else ("color: #e67e22;" if v < 0 else "color: #27ae60; font-weight: bold;")
                    )
                    val_str = f"{v:+.1f}%"

                elif col == "Desperdicio":
                    style = "color: #c0392b; font-weight: bold;"
                elif col == "Meta_Ponderada":
                    style = "color: #7f8c8d;"

                rows += f"<td style='{style}'>{val_str}</td>"
            rows += "</tr>"
        return rows

    def make_checkpoint_line_rows(df_resumo: pd.DataFrame):
        if df_resumo is None or df_resumo.empty:
            return "<tr><td colspan='9'>Nenhum checkpoint encontrado neste estágio no período.</td></tr>"

        rows = ""
        for _, row in df_resumo.iterrows():
            delta_kml = float(row.get("delta_kml") or 0)
            delta_desp = float(row.get("delta_desperdicio") or 0)

            cor_kml = "#27ae60" if delta_kml > 0 else "#c0392b" if delta_kml < 0 else "#7f8c8d"
            cor_desp = "#27ae60" if delta_desp < 0 else "#c0392b" if delta_desp > 0 else "#7f8c8d"

            rows += f"""
            <tr>
                <td style="text-align:left;"><b>{row.get('linha_foco', '-')}</b></td>
                <td>{int(row.get('qtd_motoristas', 0))}</td>
                <td>{float(row.get('antes_kml', 0)):.2f}</td>
                <td><b>{float(row.get('depois_kml', 0)):.2f}</b></td>
                <td style="color:{cor_kml};font-weight:bold;">{delta_kml:+.2f}</td>
                <td>{float(row.get('antes_desp', 0)):.1f} L</td>
                <td>{float(row.get('depois_desp', 0)):.1f} L</td>
                <td style="color:{cor_desp};font-weight:bold;">{delta_desp:+.1f} L</td>
                <td>
                    <span style="color:#16a34a;font-weight:700;">M {int(row.get('melhorou', 0))}</span> /
                    <span style="color:#dc2626;font-weight:700;">P {int(row.get('piorou', 0))}</span> /
                    <span style="color:#6b7280;font-weight:700;">S {int(row.get('sem_evolucao', 0))}</span>
                </td>
            </tr>
            """
        return rows

    df_clean = dados["df_clean"].copy()
    if "Mes_Ano" not in df_clean.columns:
        df_clean["Mes_Ano"] = df_clean["Date"].dt.to_period("M")
    mensal = df_clean.groupby("Mes_Ano").agg({"Km": "sum", "Comb.": "sum"}).reset_index()
    mensal["KML"] = mensal["Km"] / mensal["Comb."]
    mensal = mensal.sort_values("Mes_Ano")

    kml_mes_atual = float(mensal.iloc[-1]["KML"]) if len(mensal) >= 1 else 0.0
    kml_mes_anterior = float(mensal.iloc[-2]["KML"]) if len(mensal) >= 2 else 0.0
    delta_kml_mes = ((kml_mes_atual - kml_mes_anterior) / kml_mes_anterior * 100) if kml_mes_anterior > 0 else 0

    if delta_kml_mes < 0:
        texto_var, cor_var = f"{abs(delta_kml_mes):.1f}% de QUEDA", "#c0392b"
    elif delta_kml_mes > 0:
        texto_var, cor_var = f"{delta_kml_mes:.1f}% de MELHORA", "#27ae60"
    else:
        texto_var, cor_var = "0,0% (Estável)", "#7f8c8d"

    rows_lin = make_rows(
        dados["tabela_linhas"],
        ["linha", "KML_Anterior", "KML_Atual", "Variacao_Pct", "Meta_Ponderada", "Desperdicio"],
        {"KML_Anterior": "{:.2f}", "KML_Atual": "{:.2f}", "Meta_Ponderada": "{:.2f}", "Desperdicio": "{:.0f}"},
    )

    rows_veic = ""
    if dados["top_veiculos"] is not None and not dados["top_veiculos"].empty:
        for _, row in dados["top_veiculos"].iterrows():
            rows_veic += f"""
            <tr>
                <td style="text-align:left;">{row['veiculo']}</td>
                <td>{row['Cluster']}</td>
                <td>{row['linha']}</td>
                <td><b>{row['KML_Real']:.2f}</b></td>
                <td style="color:#7f8c8d">{row['Meta_Linha']:.2f}</td>
                <td><b style="color:#c0392b">{row['Litros_Desp_Meta']:.0f}</b></td>
            </tr>"""

    rows_mot = ""
    if dados["top_motoristas"] is not None and not dados["top_motoristas"].empty:
        for _, row in dados["top_motoristas"].iterrows():
            rows_mot += f"""
            <tr>
                <td style="text-align:left;">{row['Motorista']}</td>
                <td>{row['Cluster']}</td>
                <td>{row['linha']}</td>
                <td><b>{row['KML_Real']:.2f}</b></td>
                <td style="color:#7f8c8d">{row['Meta_Linha']:.2f}</td>
                <td><b style="color:#c0392b">{row['Litros_Desp_Meta']:.0f}</b></td>
            </tr>"""

    rows_cont = make_rows(
        dados["top_veiculos_contaminados"],
        ["veiculo", "Cluster", "linha", "Qtd_Contaminacoes", "KML_Min", "KML_Max"],
        {"Qtd_Contaminacoes": "{:.0f}", "KML_Min": "{:.2f}", "KML_Max": "{:.2f}"},
    )

    kpis_inst = dados.get("instrutor_kpis", {})
    tabela_evo = kpis_inst.get("tabela_evolucao", pd.DataFrame())

    rows_evo = ""
    if not tabela_evo.empty:
        for _, row in tabela_evo.iterrows():
            melhoria = row["melhoria"]
            cor_melhoria = "#27ae60" if melhoria > 0 else "#c0392b"
            sinal = "+" if melhoria > 0 else ""
            rows_evo += f"""
            <tr>
                <td style="text-align:left;">{row['motorista_nome']} ({row['motorista_chapa']})</td>
                <td>{row['status_norm']}</td>
                <td>{row.get('dt_inicio_fmt','')}</td>
                <td>{int(row.get('dias_monitorados', 0))}</td>
                <td>{row['kml_inicial']:.2f}</td>
                <td><b>{row['KML_Atual']:.2f}</b></td>
                <td><b style="color:{cor_melhoria}">{sinal}{melhoria:.2f}</b></td>
            </tr>"""
    else:
        rows_evo = "<tr><td colspan='7'>Nenhum dado de evolução comparativa computado neste período.</td></tr>"

    dias_acao_str = ", ".join(kpis_inst.get("dias_com_acao", []))
    if not dias_acao_str:
        dias_acao_str = "Nenhum dia com lançamentos no período."

    cp = dados.get("checkpoint_kpis", {})
    cards_cp = dados.get("checkpoint_cards", [])
    resumo_cp = dados.get("checkpoint_resumo_por_linha", {})

    rows_cp10_linha = make_checkpoint_line_rows(resumo_cp.get("PRONTUARIO_10", pd.DataFrame()))
    rows_cp20_linha = make_checkpoint_line_rows(resumo_cp.get("PRONTUARIO_20", pd.DataFrame()))
    rows_cp30_linha = make_checkpoint_line_rows(resumo_cp.get("PRONTUARIO_30", pd.DataFrame()))

    cards_motoristas_html = ""
    if cards_cp:
        for c in cards_cp:
            cor = "#27ae60" if c["conclusao"] == "MELHOROU" else "#c0392b" if c["conclusao"] == "PIOROU" else "#7f8c8d"
            dt_txt = c["data"].strftime("%d/%m/%Y") if isinstance(c["data"], pd.Timestamp) else "-"
            cards_motoristas_html += f"""
            <div class="mini-card">
                <div class="mini-title">{c['motorista_nome']}</div>
                <div class="mini-sub">{c['motorista_chapa']} • Linha {c['linha_foco']}</div>
                <div class="mini-chip">{c['tipo']}</div>
                <div class="mini-metrics">
                    <span>Δ KM/L: <b style="color:{'#27ae60' if c['delta_kml'] > 0 else '#c0392b' if c['delta_kml'] < 0 else '#7f8c8d'}">{c['delta_kml']:+.2f}</b></span>
                    <span>Δ Desp.: <b style="color:{'#27ae60' if c['delta_desperdicio'] < 0 else '#c0392b' if c['delta_desperdicio'] > 0 else '#7f8c8d'}">{c['delta_desperdicio']:+.1f} L</b></span>
                </div>
                <div class="mini-foot" style="color:{cor};">{c['conclusao']} • {dt_txt}</div>
            </div>
            """
    else:
        cards_motoristas_html = "<div class='muted'>Nenhum card de checkpoint disponível.</div>"

    html = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <title>Relatório Gerencial</title>
        <style>
            body {{
                font-family: 'Segoe UI', sans-serif;
                background:
                    radial-gradient(circle at top left, rgba(37,99,235,.08), transparent 24%),
                    radial-gradient(circle at top right, rgba(22,163,74,.08), transparent 22%),
                    linear-gradient(180deg, #eef3f8 0%, #f6f8fb 100%);
                margin: 0;
                padding: 20px;
                color: #1f2937;
            }}
            .container {{
                max-width: 1100px;
                margin: auto;
                background: white;
                padding: 34px 38px;
                box-shadow: 0 14px 36px rgba(15, 23, 42, 0.10);
                border-radius: 18px;
                border: 1px solid #e5e7eb;
            }}
            .header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 3px solid #1f2937;
                padding-bottom: 20px;
                margin-bottom: 18px;
            }}
            .title h1 {{
                margin: 0;
                color: #111827;
                text-transform: uppercase;
                letter-spacing: 1px;
                font-size: 28px;
            }}
            .month-card {{
                background: linear-gradient(135deg, #1f2937 0%, #334155 100%);
                color: white;
                padding: 12px 20px;
                border-radius: 12px;
                text-align: center;
                min-width: 170px;
                box-shadow: 0 10px 20px rgba(15, 23, 42, .18);
            }}
            .month-label {{ font-size: 10px; text-transform: uppercase; opacity: 0.8; }}
            .month-val {{ font-size: 18px; font-weight: bold; }}

            .kpi-grid {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 18px;
                margin-bottom: 22px;
            }}
            .kpi-grid-5 {{
                display: grid;
                grid-template-columns: repeat(5, 1fr);
                gap: 10px;
                margin-bottom: 20px;
            }}
            .kpi-card {{
                background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
                padding: 16px;
                border-radius: 14px;
                text-align: center;
                border: 1px solid #e5e7eb;
                box-shadow: 0 6px 14px rgba(15, 23, 42, 0.05);
            }}
            .kpi-val {{ display: block; font-size: 26px; font-weight: bold; }}
            .kpi-lbl {{ font-size: 11px; text-transform: uppercase; color: #64748b; letter-spacing: 0.6px; }}

            h2 {{
                color: #1d4ed8;
                font-size: 18px;
                border-left: 5px solid #1d4ed8;
                padding-left: 10px;
                margin-top: 30px;
                margin-bottom: 14px;
                page-break-after: avoid;
            }}

            .ai-box {{
                background: linear-gradient(180deg, #fffdf2 0%, #fff9db 100%);
                border: 1px solid #f5d76e;
                padding: 20px;
                border-radius: 12px;
                line-height: 1.7;
                font-size: 14px;
                margin-bottom: 18px;
                box-shadow: inset 0 1px 0 rgba(255,255,255,.7);
            }}

            .chart-box {{
                text-align: center;
                margin-bottom: 30px;
                border: 1px solid #e5e7eb;
                padding: 12px;
                border-radius: 14px;
                page-break-inside: avoid;
                background: linear-gradient(180deg, #fff 0%, #f9fafb 100%);
            }}
            .chart-box img {{ max-width: 100%; height: auto; }}

            .row-split {{
                display: flex;
                gap: 20px;
                margin-bottom: 20px;
            }}
            .col {{ flex: 1; }}

            .section-lead {{
                margin-top: -8px;
                margin-bottom: 12px;
                color: #6b7280;
                font-size: 12px;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 12px;
                margin-bottom: 20px;
                page-break-inside: auto;
                overflow: hidden;
                border-radius: 12px;
            }}
            th {{
                background: linear-gradient(180deg, #1f2937 0%, #334155 100%);
                color: white;
                padding: 9px 8px;
                text-align: center;
                font-weight: 600;
                border: 1px solid #1f2937;
            }}
            td {{
                border-bottom: 1px solid #e5e7eb;
                border-right: 1px solid #eef2f7;
                border-left: 1px solid #eef2f7;
                padding: 7px 8px;
                text-align: center;
                background: #fff;
            }}
            th:first-child, td:first-child {{ text-align: left; }}
            tr:nth-child(even) td {{ background-color: #f8fafc; }}

            .muted {{ font-size: 12px; color: #6b7280; }}

            .footer {{
                margin-top: 30px;
                text-align: center;
                font-size: 11px;
                color: #94a3b8;
                border-top: 1px solid #e5e7eb;
                padding-top: 14px;
            }}

            .cards-motoristas {{
                display:grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 12px;
                margin-bottom:20px;
            }}
            .mini-card {{
                border:1px solid #e5e7eb;
                border-radius:14px;
                background:linear-gradient(180deg,#ffffff 0%,#f8fafc 100%);
                padding:12px;
                box-shadow: 0 6px 12px rgba(15,23,42,.04);
            }}
            .mini-title {{ font-weight:700; color:#111827; font-size:13px; }}
            .mini-sub {{ font-size:11px; color:#6b7280; margin-top:4px; }}
            .mini-chip {{
                display:inline-block;
                margin-top:8px;
                padding:4px 8px;
                background:#eff6ff;
                color:#1d4ed8;
                border-radius:999px;
                font-size:10px;
                font-weight:700;
            }}
            .mini-metrics {{
                margin-top:10px;
                display:flex;
                flex-direction:column;
                gap:4px;
                font-size:11px;
            }}
            .mini-foot {{ margin-top:8px; font-size:11px; font-weight:700; }}

            .checkpoint-header {{
                display:flex;
                justify-content:space-between;
                gap:14px;
                align-items:stretch;
                margin-bottom:14px;
            }}
            .checkpoint-badge {{
                flex:1;
                background: linear-gradient(135deg, #eff6ff 0%, #ffffff 100%);
                border: 1px solid #dbeafe;
                border-radius: 14px;
                padding: 12px 14px;
                box-shadow: 0 6px 12px rgba(37, 99, 235, 0.06);
            }}
            .checkpoint-badge .t {{
                font-size: 11px;
                text-transform: uppercase;
                color: #64748b;
                margin-bottom: 6px;
                font-weight: 700;
                letter-spacing: .4px;
            }}
            .checkpoint-badge .v {{
                font-size: 22px;
                font-weight: 800;
                color: #0f172a;
            }}
            .checkpoint-badge .s {{
                font-size: 11px;
                color: #64748b;
                margin-top: 4px;
            }}

            @page {{ size: A4; margin: 10mm; }}
            @media print {{
              html, body {{
                background: #fff !important;
                padding: 0 !important;
                margin: 0 !important;
              }}
              .container {{
                max-width: none !important;
                margin: 0 !important;
                padding: 0 !important;
                box-shadow: none !important;
                border-radius: 0 !important;
                border: none !important;
              }}
              .header, .kpi-grid, .kpi-grid-5 {{
                break-inside: avoid;
                page-break-inside: avoid;
              }}
              .row-split {{ display: block !important; }}
              .col {{
                width: 100% !important;
                page-break-inside: avoid;
                margin-bottom: 20px;
              }}
              .chart-box, .cards-motoristas, .checkpoint-header {{
                break-inside: avoid;
                page-break-inside: avoid;
              }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="title">
                    <h1>Relatório Gerencial</h1>
                    <div class="muted" style="margin-top:5px;">Eficiência Energética de Frota</div>
                    <div class="muted" style="margin-top:6px;"><b>Período:</b> {dados['periodo']}</div>
                </div>
                <div class="month-card">
                    <div class="month-label">MÊS DE REFERÊNCIA</div>
                    <div class="month-val">{dados['mes_atual_nome']}</div>
                </div>
            </div>

            <div class="kpi-grid">
                <div class="kpi-card">
                    <span class="kpi-val" style="color:#0f172a">{kml_mes_atual:.2f}</span>
                    <span class="kpi-lbl">KM/L MÊS BASE</span>
                </div>
                <div class="kpi-card">
                    <span class="kpi-val" style="color:{cor_var}">{texto_var}</span>
                    <span class="kpi-lbl">VARIAÇÃO VS MÊS ANTERIOR</span>
                </div>
                <div class="kpi-card">
                    <span class="kpi-val" style="color:#c0392b">{dados['total_desperdicio']:.0f} L</span>
                    <span class="kpi-lbl">DESPERDÍCIO (META LINHA)</span>
                </div>
            </div>

            <h2>1. Inteligência Executiva</h2>
            <div class="ai-box">{texto_ia}</div>

            <h2>2. Evolução de Eficiência</h2>
            <div class="chart-box"><img src="{img_src}"></div>

            <h2>3. Análise de Eficiência por Linha</h2>
            <p class="section-lead">
                Comparativo de performance entre o mês atual e o mês anterior.
                A <b>Meta Ponderada</b> considera a mistura entre veículos que operaram na linha.
            </p>
            <table>
                <thead>
                    <tr>
                        <th>Linha</th>
                        <th>Mês Ant. (KM/L)</th>
                        <th>Mês Atual (KM/L)</th>
                        <th>Variação</th>
                        <th>Meta Ponderada</th>
                        <th>Desperdício (L)</th>
                    </tr>
                </thead>
                <tbody>{rows_lin}</tbody>
            </table>

            <div class="row-split">
                <div class="col">
                    <h2>4. Top 10 Veículos (Perdas na Meta)</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Veículo</th><th>Clust.</th><th>Linha</th>
                                <th>Real</th><th>Meta</th><th>Perda (L)</th>
                            </tr>
                        </thead>
                        <tbody>{rows_veic}</tbody>
                    </table>
                </div>
                <div class="col">
                    <h2>5. Top 10 Motoristas (Perdas na Meta)</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Motorista</th><th>Clust.</th><th>Linha</th>
                                <th>Real</th><th>Meta</th><th>Perda (L)</th>
                            </tr>
                        </thead>
                        <tbody>{rows_mot}</tbody>
                    </table>
                </div>
            </div>

            <h2>6. Auditoria de Dados KML (Fora do Padrão)</h2>
            <p class="section-lead">
                Veículos com leituras ignoradas (kml &lt; 1,5 ou kml &gt; 5,0) por contaminação de abastecimento.
            </p>
            <table>
                <thead>
                    <tr>
                        <th>Veículo</th><th>Cluster</th><th>Linha</th>
                        <th>Qtd. Contaminações</th><th>KML Mín.</th><th>KML Máx.</th>
                    </tr>
                </thead>
                <tbody>{rows_cont}</tbody>
            </table>

            <h2>7. Pipeline de Monitoramento e Checkpoints</h2>

            <div class="kpi-grid-5">
                <div class="kpi-card">
                    <span class="kpi-val" style="color:#f59e0b;">{cp.get('fase_lt_10', 0)}</span>
                    <span class="kpi-lbl">&lt; 10 dias</span>
                </div>
                <div class="kpi-card">
                    <span class="kpi-val" style="color:#2563eb;">{cp.get('fase_cp10', 0)}</span>
                    <span class="kpi-lbl">Pront. 10 dias</span>
                </div>
                <div class="kpi-card">
                    <span class="kpi-val" style="color:#7c3aed;">{cp.get('fase_cp20', 0)}</span>
                    <span class="kpi-lbl">Pront. 20 dias</span>
                </div>
                <div class="kpi-card">
                    <span class="kpi-val" style="color:#059669;">{cp.get('fase_cp30', 0)}</span>
                    <span class="kpi-lbl">Pront. 30 dias</span>
                </div>
                <div class="kpi-card">
                    <span class="kpi-val" style="color:#dc2626;">{cp.get('fase_analise_final', 0)}</span>
                    <span class="kpi-lbl">Análise Final</span>
                </div>
            </div>

            <h2>8. Resumo Visual dos Checkpoints</h2>
            <div class="cards-motoristas">
                {cards_motoristas_html}
            </div>

            <h2>9. Checkpoint 10 Dias por Linha</h2>
            <div class="checkpoint-header">
                <div class="checkpoint-badge">
                    <div class="t">Total de checkpoints</div>
                    <div class="v">{cp.get('cp10_total', 0)}</div>
                    <div class="s">Base do período filtrado</div>
                </div>
                <div class="checkpoint-badge">
                    <div class="t">Litros recuperados</div>
                    <div class="v" style="color:#16a34a;">{cp.get('cp10_litros_recuperados', 0.0):.1f} L</div>
                    <div class="s">Melhora consolidada do estágio</div>
                </div>
                <div class="checkpoint-badge">
                    <div class="t">Delta médio KM/L</div>
                    <div class="v" style="color:{'#16a34a' if cp.get('cp10_delta_kml_medio', 0.0) > 0 else '#dc2626' if cp.get('cp10_delta_kml_medio', 0.0) < 0 else '#64748b'};">{cp.get('cp10_delta_kml_medio', 0.0):+.2f}</div>
                    <div class="s">Comparação antes x pós acompanhamento</div>
                </div>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Linha</th>
                        <th>Qtd.</th>
                        <th>KM/L Anterior</th>
                        <th>KM/L Pós</th>
                        <th>Δ KM/L</th>
                        <th>Desp. Anterior</th>
                        <th>Desp. Pós</th>
                        <th>Δ Desp.</th>
                        <th>M / P / S</th>
                    </tr>
                </thead>
                <tbody>{rows_cp10_linha}</tbody>
            </table>

            <h2>10. Checkpoint 20 Dias por Linha</h2>
            <div class="checkpoint-header">
                <div class="checkpoint-badge">
                    <div class="t">Total de checkpoints</div>
                    <div class="v">{cp.get('cp20_total', 0)}</div>
                    <div class="s">Base do período filtrado</div>
                </div>
                <div class="checkpoint-badge">
                    <div class="t">Litros recuperados</div>
                    <div class="v" style="color:#16a34a;">{cp.get('cp20_litros_recuperados', 0.0):.1f} L</div>
                    <div class="s">Melhora consolidada do estágio</div>
                </div>
                <div class="checkpoint-badge">
                    <div class="t">Delta médio KM/L</div>
                    <div class="v" style="color:{'#16a34a' if cp.get('cp20_delta_kml_medio', 0.0) > 0 else '#dc2626' if cp.get('cp20_delta_kml_medio', 0.0) < 0 else '#64748b'};">{cp.get('cp20_delta_kml_medio', 0.0):+.2f}</div>
                    <div class="s">Comparação antes x pós acompanhamento</div>
                </div>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Linha</th>
                        <th>Qtd.</th>
                        <th>KM/L Anterior</th>
                        <th>KM/L Pós</th>
                        <th>Δ KM/L</th>
                        <th>Desp. Anterior</th>
                        <th>Desp. Pós</th>
                        <th>Δ Desp.</th>
                        <th>M / P / S</th>
                    </tr>
                </thead>
                <tbody>{rows_cp20_linha}</tbody>
            </table>

            <h2>11. Checkpoint 30 Dias por Linha</h2>
            <div class="checkpoint-header">
                <div class="checkpoint-badge">
                    <div class="t">Total de checkpoints</div>
                    <div class="v">{cp.get('cp30_total', 0)}</div>
                    <div class="s">Base do período filtrado</div>
                </div>
                <div class="checkpoint-badge">
                    <div class="t">Litros recuperados</div>
                    <div class="v" style="color:#16a34a;">{cp.get('cp30_litros_recuperados', 0.0):.1f} L</div>
                    <div class="s">Melhora consolidada do estágio</div>
                </div>
                <div class="checkpoint-badge">
                    <div class="t">Delta médio KM/L</div>
                    <div class="v" style="color:{'#16a34a' if cp.get('cp30_delta_kml_medio', 0.0) > 0 else '#dc2626' if cp.get('cp30_delta_kml_medio', 0.0) < 0 else '#64748b'};">{cp.get('cp30_delta_kml_medio', 0.0):+.2f}</div>
                    <div class="s">Comparação antes x pós acompanhamento</div>
                </div>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Linha</th>
                        <th>Qtd.</th>
                        <th>KM/L Anterior</th>
                        <th>KM/L Pós</th>
                        <th>Δ KM/L</th>
                        <th>Desp. Anterior</th>
                        <th>Desp. Pós</th>
                        <th>Δ Desp.</th>
                        <th>M / P / S</th>
                    </tr>
                </thead>
                <tbody>{rows_cp30_linha}</tbody>
            </table>

            <h2>12. Atuação do Instrutor (Acompanhamentos)</h2>
            <p class="section-lead">
                Dias com inícios de acompanhamento no período:
                <b>{dias_acao_str}</b> ({len(kpis_inst.get("dias_com_acao", []))} dias únicos de campo)
            </p>

            <div class="kpi-grid-5">
                <div class="kpi-card" style="padding: 10px;">
                    <span class="kpi-val" style="color:#e67e22; font-size: 22px;">{kpis_inst.get('aguardando', 0)}</span>
                    <span class="kpi-lbl">Aguardando</span>
                </div>
                <div class="kpi-card" style="padding: 10px;">
                    <span class="kpi-val" style="color:#2980b9; font-size: 22px;">{kpis_inst.get('em_andamento', 0)}</span>
                    <span class="kpi-lbl">Monitoramento</span>
                </div>
                <div class="kpi-card" style="padding: 10px;">
                    <span class="kpi-val" style="color:#8e44ad; font-size: 22px;">{kpis_inst.get('concluidos', 0)}</span>
                    <span class="kpi-lbl">Concluídos</span>
                </div>
                <div class="kpi-card" style="padding: 10px; border-left: 3px solid #27ae60;">
                    <span class="kpi-val" style="color:#27ae60; font-size: 22px;">{kpis_inst.get('evoluiram', 0)}</span>
                    <span class="kpi-lbl">Evoluíram</span>
                </div>
                <div class="kpi-card" style="padding: 10px; border-left: 3px solid #c0392b;">
                    <span class="kpi-val" style="color:#c0392b; font-size: 22px;">{kpis_inst.get('nao_evoluiram', 0)}</span>
                    <span class="kpi-lbl">S/ Evolução</span>
                </div>
            </div>

            <table>
                <thead>
                    <tr>
                        <th>Motorista</th>
                        <th>Status</th>
                        <th>Data Início</th>
                        <th>Dias Monitorados</th>
                        <th>KM/L Inicial</th>
                        <th>KM/L Atual</th>
                        <th>Evolução</th>
                    </tr>
                </thead>
                <tbody>{rows_evo}</tbody>
            </table>

            <div class="footer">
                Relatório Gerado Automaticamente pelo Agente Diesel AI.<br>
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
        periodo_inicio, periodo_fim = hoje.replace(day=1), hoje

    atualizar_status_relatorio(
        "PROCESSANDO",
        tipo=REPORT_TIPO,
        periodo_inicio=str(periodo_inicio) if periodo_inicio else None,
        periodo_fim=str(periodo_fim) if periodo_fim else None,
    )

    try:
        df_base = carregar_dados_supabase_b(periodo_inicio, periodo_fim)
        dados = processar_dados_gerenciais_df(df_base, periodo_inicio, periodo_fim)
        mes_ref = str(dados["df_clean"]["Date"].max().to_period("M"))

        df_sug = gerar_sugestoes_acompanhamento(dados)
        salvar_sugestoes_supabase_b(df_sug, mes_ref, periodo_inicio, periodo_fim)

        out_dir = Path(PASTA_SAIDA)
        img_path = out_dir / "cluster_evolution_unificado.png"
        html_path = out_dir / "Relatorio_Gerencial.html"
        pdf_path = out_dir / "Relatorio_Gerencial.pdf"

        gerar_grafico_geral(dados["df_clean"], img_path)
        texto_ia = consultar_ia_gerencial(dados)
        gerar_html_gerencial(dados, texto_ia, img_path, html_path)
        gerar_pdf_do_html(html_path, pdf_path)

        base_folder = f"{REMOTE_BASE_PREFIX}/{mes_ref}/report_{REPORT_ID}"
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
        print("✅ [OK] Relatório concluído e enviado.")

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
