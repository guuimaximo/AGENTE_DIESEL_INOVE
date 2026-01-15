import os
import io
import json
import math
import uuid
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# PDF (ReportLab)
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# Supabase
from supabase import create_client

# Vertex AI (Gemini)
import vertexai
from vertexai.generative_models import GenerativeModel


# =========================
# CONFIG (via ENV)
# =========================

# Supabase ORIGEM (onde consulta os dados)
SUPABASE_SRC_URL = os.getenv("SUPABASE_SRC_URL", "").strip()
SUPABASE_SRC_KEY = os.getenv("SUPABASE_SRC_KEY", "").strip()  # service role recomendado

# Supabase DESTINO (onde salva os relatórios)
SUPABASE_DST_URL = os.getenv("SUPABASE_DST_URL", "").strip()
SUPABASE_DST_KEY = os.getenv("SUPABASE_DST_KEY", "").strip()  # service role recomendado
SUPABASE_DST_BUCKET = os.getenv("SUPABASE_DST_BUCKET", "relatorios").strip()
SUPABASE_DST_TABLE = os.getenv("SUPABASE_DST_TABLE", "relatorios_gerados").strip()

# Vertex
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID", "").strip()
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1").strip()
VERTEX_MODEL = os.getenv("VERTEX_MODEL", "gemini-2.5-pro").strip()

# Consulta
SRC_TABLE = os.getenv("SRC_TABLE", "premiacao_diaria").strip()
DATE_COL = os.getenv("SRC_DATE_COL", "dia_date").strip()

# Intervalo (opcional)
DATE_START = os.getenv("REPORT_DATE_START", "").strip()  # "YYYY-MM-DD"
DATE_END = os.getenv("REPORT_DATE_END", "").strip()      # "YYYY-MM-DD"

# Saída
OUTPUT_DIR = os.getenv("REPORT_OUTPUT_DIR", "Relatorios_Diesel_Final").strip()

# Nome base do relatório
REPORT_TITLE = os.getenv("REPORT_TITLE", "Relatório Gerencial — Diesel / KM/L").strip()


# =========================
# HELPERS
# =========================

def _must_env(name: str, value: str):
    if not value:
        raise RuntimeError(f"Variável de ambiente obrigatória não definida: {name}")


def now_sp_iso() -> str:
    # Para simplificar: usa UTC do ambiente. Se você quiser SP real, deixe o front mandar o range.
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def parse_date_or_none(s: str):
    if not s:
        return None
    return datetime.strptime(s, "%Y-%m-%d").date()


def safe_float(x):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def df_require_cols(df: pd.DataFrame, cols: list[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Colunas ausentes no dataframe: {missing}. Colunas disponíveis: {list(df.columns)}")


# =========================
# SUPABASE: FETCH
# =========================

def fetch_data_from_supabase() -> pd.DataFrame:
    _must_env("SUPABASE_SRC_URL", SUPABASE_SRC_URL)
    _must_env("SUPABASE_SRC_KEY", SUPABASE_SRC_KEY)

    sb = create_client(SUPABASE_SRC_URL, SUPABASE_SRC_KEY)

    # Campos esperados (baseado no que você gravou)
    # dia_date, motorista, veiculo, linha, km_rodado, combustivel_consumido, minutos_em_viagem, km/l, mes, ano
    # Alguns podem vir com nome diferente (por exemplo "km_l" etc). Ajuste ENV se necessário.
    select_fields = os.getenv(
        "SRC_SELECT",
        f"{DATE_COL},motorista,veiculo,linha,km_rodado,combustivel_consumido,minutos_em_viagem,km/l,mes,ano"
    ).strip()

    q = sb.from_(SRC_TABLE).select(select_fields)

    d0 = parse_date_or_none(DATE_START)
    d1 = parse_date_or_none(DATE_END)

    if d0:
        q = q.gte(DATE_COL, d0.isoformat())
    if d1:
        q = q.lte(DATE_COL, d1.isoformat())

    res = q.execute()
    data = res.data or []
    df = pd.DataFrame(data)

    if df.empty:
        # Sem dados no range: ainda gera relatório com aviso
        return df

    # Normaliza nomes problemáticos: "km/l" vira "kml"
    if "km/l" in df.columns:
        df = df.rename(columns={"km/l": "kml"})
    elif "kml" not in df.columns and "km_l" in df.columns:
        df = df.rename(columns={"km_l": "kml"})

    # Tipos
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce").dt.date

    for c in ["km_rodado", "combustivel_consumido", "minutos_em_viagem", "kml"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# =========================
# ANALYTICS
# =========================

def build_kpis(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "has_data": False,
            "rows": 0,
            "kml_medio": None,
            "km_total": None,
            "comb_total": None,
            "motoristas": 0,
            "veiculos": 0,
            "linhas": 0,
        }

    rows = len(df)
    km_total = safe_float(df.get("km_rodado", pd.Series(dtype=float)).sum())
    comb_total = safe_float(df.get("combustivel_consumido", pd.Series(dtype=float)).sum())

    # kml médio ponderado por km (mais correto do que média simples)
    if "kml" in df.columns and "km_rodado" in df.columns:
        tmp = df.dropna(subset=["kml", "km_rodado"])
        if len(tmp) > 0 and tmp["km_rodado"].sum() > 0:
            kml_medio = float((tmp["kml"] * tmp["km_rodado"]).sum() / tmp["km_rodado"].sum())
        else:
            kml_medio = safe_float(df["kml"].mean())
    else:
        kml_medio = None

    return {
        "has_data": True,
        "rows": rows,
        "kml_medio": kml_medio,
        "km_total": km_total,
        "comb_total": comb_total,
        "motoristas": int(df["motorista"].nunique()) if "motorista" in df.columns else 0,
        "veiculos": int(df["veiculo"].nunique()) if "veiculo" in df.columns else 0,
        "linhas": int(df["linha"].nunique()) if "linha" in df.columns else 0,
    }


def make_monthly_evolution_chart(df: pd.DataFrame, out_png: str) -> str:
    """
    Gera um gráfico mensal simples:
      - KM total por mês
      - KM/L ponderado por mês
    Salva como PNG e retorna o caminho.
    """
    ensure_dir(Path(out_png).parent.as_posix())

    if df.empty or DATE_COL not in df.columns:
        # cria uma imagem vazia com aviso
        plt.figure(figsize=(10, 4))
        plt.title("Sem dados para gerar gráfico")
        plt.text(0.5, 0.5, "Nenhum dado encontrado no intervalo selecionado.", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        return out_png

    dfx = df.copy()
    dfx["mes_ref"] = pd.to_datetime(dfx[DATE_COL], errors="coerce").dt.to_period("M").astype(str)

    # KM total por mês
    km_mes = dfx.groupby("mes_ref")["km_rodado"].sum(min_count=1)

    # KM/L ponderado por mês
    if "kml" in dfx.columns and "km_rodado" in dfx.columns:
        tmp = dfx.dropna(subset=["kml", "km_rodado"])
        if not tmp.empty:
            kml_mes = tmp.groupby("mes_ref").apply(
                lambda g: float((g["kml"] * g["km_rodado"]).sum() / g["km_rodado"].sum())
                if g["km_rodado"].sum() > 0 else float("nan")
            )
        else:
            kml_mes = pd.Series(dtype=float)
    else:
        kml_mes = pd.Series(dtype=float)

    # Ordena por mês
    order = sorted(km_mes.index.tolist())
    km_mes = km_mes.reindex(order)
    kml_mes = kml_mes.reindex(order) if not kml_mes.empty else kml_mes

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(order, km_mes.values, marker="o")
    plt.title("Evolução mensal — KM total")
    plt.xlabel("Mês")
    plt.ylabel("KM")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    return out_png


# =========================
# VERTEX: INSIGHTS
# =========================

def generate_ai_summary(kpis: dict, df: pd.DataFrame) -> str:
    """
    Gera um texto objetivo, com base nos KPIs.
    Você pode ajustar o prompt depois, mas a metodologia é: dados -> IA -> texto.
    """
    _must_env("VERTEX_PROJECT_ID", VERTEX_PROJECT_ID)
    _must_env("VERTEX_LOCATION", VERTEX_LOCATION)

    vertexai.init(project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION)
    model = GenerativeModel(VERTEX_MODEL)

    # Top 5 motoristas (por kml ponderado / km)
    top_info = ""
    if not df.empty and {"motorista", "kml", "km_rodado"}.issubset(df.columns):
        tmp = df.dropna(subset=["motorista", "kml", "km_rodado"]).copy()
        if not tmp.empty:
            grp = tmp.groupby("motorista").apply(
                lambda g: float((g["kml"] * g["km_rodado"]).sum() / g["km_rodado"].sum())
                if g["km_rodado"].sum() > 0 else float("nan")
            ).dropna().sort_values(ascending=False).head(5)
            if not grp.empty:
                top_info = "\n".join([f"- {idx}: {val:.2f} km/l" for idx, val in grp.items()])

    prompt = f"""
Você é um analista de performance operacional (diesel/KM-L). Gere um resumo curto e direto, em português, com foco gerencial.

KPIs do período:
- Linhas analisadas: {kpis.get("linhas")}
- Veículos analisados: {kpis.get("veiculos")}
- Motoristas analisados: {kpis.get("motoristas")}
- Registros: {kpis.get("rows")}
- KM total: {kpis.get("km_total")}
- Combustível total (L): {kpis.get("comb_total")}
- KM/L médio (ponderado): {kpis.get("kml_medio")}

Top motoristas por KM/L (se existir):
{top_info if top_info else "Sem ranking disponível."}

Regras do texto:
- 6 a 10 linhas.
- Sem floreio, sem emojis.
- Cite 1 ponto de atenção e 1 recomendação objetiva.
- Se não houver dados, responda apenas: "Sem dados no período selecionado."
""".strip()

    if not kpis.get("has_data"):
        return "Sem dados no período selecionado."

    resp = model.generate_content(prompt)
    text = getattr(resp, "text", None) or ""
    return text.strip() or "Resumo não disponível."


# =========================
# PDF GENERATION (ReportLab)
# =========================

def build_pdf(
    pdf_path: str,
    title: str,
    kpis: dict,
    ai_text: str,
    chart_png_path: str,
    meta: dict,
):
    ensure_dir(str(Path(pdf_path).parent))

    c = canvas.Canvas(pdf_path, pagesize=A4)
    w, h = A4

    # Header
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2 * cm, h - 2 * cm, title)

    c.setFont("Helvetica", 9)
    c.drawString(2 * cm, h - 2.6 * cm, f"Gerado em: {now_sp_iso()}")

    # KPIs
    y = h - 3.6 * cm
    c.setFont("Helvetica-Bold", 11)
    c.drawString(2 * cm, y, "KPIs do período")
    y -= 0.6 * cm

    c.setFont("Helvetica", 10)
    lines = [
        f"Registros: {kpis.get('rows')}",
        f"Motoristas: {kpis.get('motoristas')} | Veículos: {kpis.get('veiculos')} | Linhas: {kpis.get('linhas')}",
        f"KM total: {kpis.get('km_total')}",
        f"Combustível total (L): {kpis.get('comb_total')}",
        f"KM/L médio (ponderado): {None if kpis.get('kml_medio') is None else round(kpis.get('kml_medio'), 3)}",
    ]
    for ln in lines:
        c.drawString(2 * cm, y, ln)
        y -= 0.5 * cm

    # Chart
    y -= 0.3 * cm
    c.setFont("Helvetica-Bold", 11)
    c.drawString(2 * cm, y, "Gráfico")
    y -= 0.4 * cm

    try:
        img = ImageReader(chart_png_path)
        img_w = w - 4 * cm
        img_h = 7.0 * cm
        c.drawImage(img, 2 * cm, y - img_h, width=img_w, height=img_h, preserveAspectRatio=True, anchor="sw")
        y -= (img_h + 0.8 * cm)
    except Exception:
        c.setFont("Helvetica", 10)
        c.drawString(2 * cm, y, "Não foi possível renderizar o gráfico.")
        y -= 0.8 * cm

    # AI Summary
    c.setFont("Helvetica-Bold", 11)
    c.drawString(2 * cm, y, "Resumo (IA)")
    y -= 0.6 * cm

    c.setFont("Helvetica", 10)
    # quebra simples
    for paragraph in ai_text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            y -= 0.3 * cm
            continue
        # quebra em linhas por largura aproximada
        words = paragraph.split()
        line = ""
        for wds in words:
            test = (line + " " + wds).strip()
            if c.stringWidth(test, "Helvetica", 10) > (w - 4 * cm):
                c.drawString(2 * cm, y, line)
                y -= 0.45 * cm
                line = wds
            else:
                line = test
        if line:
            c.drawString(2 * cm, y, line)
            y -= 0.45 * cm

        if y < 3 * cm:
            c.showPage()
            y = h - 2 * cm
            c.setFont("Helvetica", 10)

    # Rodapé / meta
    c.setFont("Helvetica", 8)
    c.drawString(2 * cm, 1.2 * cm, f"meta: {json.dumps(meta, ensure_ascii=False)[:180]}")

    c.save()


# =========================
# SUPABASE DEST: UPLOAD + REGISTER
# =========================

def upload_pdf_to_dest(pdf_path: str, report_id: str) -> dict:
    """
    Faz upload no Supabase DESTINO (Storage) e registra linha na tabela.
    Retorna {uploaded: bool, path: str, db_row: dict|None}
    """
    if not SUPABASE_DST_URL or not SUPABASE_DST_KEY:
        # Sem destino configurado: não faz upload, só retorna local
        return {"uploaded": False, "path": None, "db_row": None}

    sb = create_client(SUPABASE_DST_URL, SUPABASE_DST_KEY)

    # caminho no bucket
    dt = date.today().isoformat()
    filename = Path(pdf_path).name
    storage_path = f"diesel/{dt}/{report_id}_{filename}"

    with open(pdf_path, "rb") as f:
        content = f.read()

    # upsert = True evita falha se reexecutar
    sb.storage.from_(SUPABASE_DST_BUCKET).upload(
        storage_path,
        content,
        {"content-type": "application/pdf", "x-upsert": "true"},
    )

    # registra na tabela destino (se existir)
    row = None
    try:
        payload = {
            "id": report_id,
            "created_at": now_sp_iso(),
            "tipo": "diesel_kml",
            "arquivo_path": storage_path,
            "status": "GERADO",
        }
        ins = sb.from_(SUPABASE_DST_TABLE).insert(payload).execute()
        row = (ins.data or [None])[0]
    except Exception:
        # se a tabela/colunas forem diferentes, não derruba o processo
        row = None

    return {"uploaded": True, "path": storage_path, "db_row": row}


# =========================
# MAIN
# =========================

def main():
    ensure_dir(OUTPUT_DIR)

    # 1) Fetch
    df = fetch_data_from_supabase()

    # 2) KPIs
    kpis = build_kpis(df)

    # 3) Chart
    chart_path = str(Path(OUTPUT_DIR) / "cluster_evolution_unificado.png")
    make_monthly_evolution_chart(df, chart_path)

    # 4) IA
    ai_text = generate_ai_summary(kpis, df)

    # 5) PDF
    report_id = str(uuid.uuid4())
    pdf_name = f"Relatorio_Gerencial_{date.today().isoformat()}.pdf"
    pdf_path = str(Path(OUTPUT_DIR) / pdf_name)

    meta = {
        "source_table": SRC_TABLE,
        "date_start": DATE_START or None,
        "date_end": DATE_END or None,
        "rows": kpis.get("rows"),
    }

    build_pdf(
        pdf_path=pdf_path,
        title=REPORT_TITLE,
        kpis=kpis,
        ai_text=ai_text,
        chart_png_path=chart_path,
        meta=meta,
    )

    # 6) Upload destino (opcional)
    up = upload_pdf_to_dest(pdf_path, report_id)

    # Saída no stdout (para o endpoint capturar)
    result = {
        "ok": True,
        "report_id": report_id,
        "pdf_local_path": pdf_path,
        "chart_local_path": chart_path,
        "uploaded": up.get("uploaded"),
        "storage_path": up.get("path"),
        "kpis": kpis,
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
