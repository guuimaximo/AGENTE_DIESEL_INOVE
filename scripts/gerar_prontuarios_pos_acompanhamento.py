# -*- coding: utf-8 -*-
import os
import json
from datetime import datetime, timedelta, date
from pathlib import Path

import pandas as pd
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
# CONFIG
# ==============================================================================
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID") or os.getenv("PROJECT_ID")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
VERTEX_MODEL = os.getenv("VERTEX_MODEL", "gemini-2.5-pro")
VERTEX_SA_JSON = os.getenv("VERTEX_SA_JSON")

SUPABASE_B_URL = os.getenv("SUPABASE_B_URL")
SUPABASE_B_SERVICE_ROLE_KEY = os.getenv("SUPABASE_B_SERVICE_ROLE_KEY")

PRONTUARIO_BATCH_ID = os.getenv("PRONTUARIO_BATCH_ID") or datetime.utcnow().strftime("batch_%Y%m%d_%H%M%S")

TABELA_FILA = "v_diesel_fila_prontuarios"
TABELA_DADOS = "fato_kml_meta_ponderada_dia"
TABELA_ACOMP = "diesel_acompanhamentos"
TABELA_EVENTOS = "diesel_acompanhamento_eventos"
TABELA_LOG = "acompanhamento_lotes"

BUCKET = os.getenv("REPORT_BUCKET", "relatorios")
REMOTE_PREFIX = os.getenv("REPORT_REMOTE_PREFIX", "acompanhamento_pos")
PASTA_SAIDA = Path(os.getenv("REPORT_OUTPUT_DIR", "Prontuarios_Pos_Acompanhamento"))

REPORT_PAGE_SIZE = int(os.getenv("REPORT_PAGE_SIZE", "500"))


# ==============================================================================
# HELPERS
# ==============================================================================
def _sb_b():
    return create_client(SUPABASE_B_URL, SUPABASE_B_SERVICE_ROLE_KEY)


def _ensure_vertex_adc_if_possible():
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        return
    if not VERTEX_SA_JSON:
        return
    try:
        tmp = Path("/tmp/vertex_sa.json")
        tmp.write_text(VERTEX_SA_JSON, encoding="utf-8")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(tmp)
    except Exception:
        pass


def _safe_filename(name: str) -> str:
    import re
    name = str(name or "").strip()
    name = re.sub(r"[^\w\-.() ]+", "_", name, flags=re.UNICODE)
    name = re.sub(r"\s+", "_", name)
    return name[:140] or "arquivo"


def n(v):
    try:
        x = float(v)
        return x if x == x else 0.0
    except Exception:
        return 0.0


def _fmt_num(v, dec=2):
    try:
        return f"{float(v):,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return f"0,{''.join(['0'] * dec)}"


def _fmt_int(v):
    try:
        return f"{int(round(float(v))):,}".replace(",", ".")
    except Exception:
        return "0"


def _esc(s: str) -> str:
    s = "" if s is None else str(s)
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
         .replace("'", "&#39;")
    )


def _parse_date(val):
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except Exception:
        return None


# AJUSTADO: não grava mais em acompanhamento_lotes usando run_id numérico como UUID
def atualizar_log_lote(status: str, msg: str = None, extra: dict = None):
    print(f"[LOTE] status={status} msg={msg} extra={extra}")
    return


def upload_storage(local_path: Path, remote_name: str, content_type: str):
    if not local_path.exists():
        return (None, None)

    sb = _sb_b()
    remote_path = f"{REMOTE_PREFIX}/{PRONTUARIO_BATCH_ID}/{remote_name}"

    with open(local_path, "rb") as f:
        sb.storage.from_(BUCKET).upload(
            path=remote_path,
            file=f,
            file_options={"content-type": content_type, "upsert": "true"},
        )

    public_url = f"{SUPABASE_B_URL}/storage/v1/object/public/{BUCKET}/{remote_path}"
    return remote_path, public_url


def carregar_prompt_ia(prompt_id: str) -> str:
    try:
        sb = _sb_b()
        resp = sb.table("ia_prompts").select("prompt_text").eq("id", prompt_id).execute()
        if resp.data and len(resp.data) > 0:
            return resp.data[0]["prompt_text"]
    except Exception:
        pass
    return ""


# ==============================================================================
# FILA
# ==============================================================================
def obter_fila_prontuarios():
    sb = _sb_b()
    resp = (
        sb.table(TABELA_FILA)
        .select("*")
        .order("ordem_fila", desc=False)
        .order("dt_inicio_monitoramento", desc=False)
        .limit(500)
        .execute()
    )
    return resp.data or []


# ==============================================================================
# DADOS DIÁRIOS
# ==============================================================================
def carregar_dados_diarios_pos(chapa: str, dt_ini: str, dt_fim: str):
    sb = _sb_b()

    resp = (
        sb.table(TABELA_DADOS)
        .select("""
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
        """)
        .ilike("motorista", f"%{chapa}%")
        .gte("dia", dt_ini)
        .lte("dia", dt_fim)
        .order("dia", desc=False)
        .execute()
    )

    rows = resp.data or []
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["dia"] = pd.to_datetime(df["dia"], errors="coerce").dt.date
    df["km"] = pd.to_numeric(df["km_rodado"], errors="coerce").fillna(0)
    df["litros"] = pd.to_numeric(df["litros_consumidos"], errors="coerce").fillna(0)
    df["kml_real"] = pd.to_numeric(df["km_l"], errors="coerce").fillna(0)
    df["kml_meta"] = pd.to_numeric(df["meta_kml_usada"], errors="coerce").fillna(0)
    df["litros_ideais_num"] = pd.to_numeric(df["litros_ideais"], errors="coerce").fillna(0)
    df["desperdicio"] = (df["litros"] - df["litros_ideais_num"]).clip(lower=0)

    df["linha"] = df["linha"].astype(str).str.strip().str.upper()
    df["cluster"] = df["cluster"].astype(str).str.strip().str.upper()
    return df


# ==============================================================================
# CÁLCULO DAS JANELAS
# ==============================================================================
def resumir_periodo(df):
    if df is None or df.empty:
        return {
            "dias_com_dado": 0,
            "km": 0.0,
            "litros": 0.0,
            "kml_real": 0.0,
            "kml_meta": 0.0,
            "litros_ideais": 0.0,
            "desperdicio": 0.0,
        }

    km = float(df["km"].sum())
    litros = float(df["litros"].sum())
    litros_ideais = float(df["litros_ideais_num"].sum())
    desperdicio = float(df["desperdicio"].sum())
    kml_real = km / litros if litros > 0 else 0.0
    kml_meta = km / litros_ideais if litros_ideais > 0 else 0.0

    return {
        "dias_com_dado": int(df["dia"].nunique()),
        "km": km,
        "litros": litros,
        "kml_real": kml_real,
        "kml_meta": kml_meta,
        "litros_ideais": litros_ideais,
        "desperdicio": desperdicio,
    }


def calcular_janela_comparativa(df, dt_inicio_monitoramento: date, dias_janela: int):
    dt_antes_ini = dt_inicio_monitoramento - timedelta(days=dias_janela)
    dt_antes_fim = dt_inicio_monitoramento - timedelta(days=1)

    dt_depois_ini = dt_inicio_monitoramento
    dt_depois_fim = dt_inicio_monitoramento + timedelta(days=dias_janela - 1)

    df_antes = df[(df["dia"] >= dt_antes_ini) & (df["dia"] <= dt_antes_fim)].copy()
    df_depois = df[(df["dia"] >= dt_depois_ini) & (df["dia"] <= dt_depois_fim)].copy()

    antes = resumir_periodo(df_antes)
    depois = resumir_periodo(df_depois)

    delta_kml = depois["kml_real"] - antes["kml_real"]
    delta_desp = depois["desperdicio"] - antes["desperdicio"]

    if antes["desperdicio"] > 0:
        delta_desp_pct = ((depois["desperdicio"] - antes["desperdicio"]) / antes["desperdicio"]) * 100
    else:
        delta_desp_pct = 0.0

    if delta_desp < -1:
        conclusao = "MELHOROU"
    elif delta_desp > 1:
        conclusao = "PIOROU"
    else:
        conclusao = "SEM_EVOLUCAO"

    return {
        "dias_janela": dias_janela,
        "antes_periodo": {
            "inicio": dt_antes_ini.isoformat(),
            "fim": dt_antes_fim.isoformat(),
            **antes,
        },
        "depois_periodo": {
            "inicio": dt_depois_ini.isoformat(),
            "fim": dt_depois_fim.isoformat(),
            **depois,
        },
        "delta_kml": delta_kml,
        "delta_desperdicio": delta_desp,
        "delta_desperdicio_pct": delta_desp_pct,
        "conclusao": conclusao,
        "df_antes": df_antes,
        "df_depois": df_depois,
    }


# ==============================================================================
# IA
# ==============================================================================
def analisar_pos_acompanhamento_ia(dados: dict) -> str:
    if not VERTEX_PROJECT_ID:
        return "<p>IA desativada. Compare os indicadores antes e depois.</p>"

    _ensure_vertex_adc_if_possible()

    try:
        vertexai.init(project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION)
        model = GenerativeModel(VERTEX_MODEL)

        template_prompt = carregar_prompt_ia("prontuario_pos_acompanhamento")
        if not template_prompt:
            template_prompt = """
Você é um analista de desempenho operacional especializado em KM/L.

Analise o comparativo ANTES x DEPOIS do acompanhamento.

DADOS:
Motorista: {nome} (Chapa {chapa})
Tipo de prontuário: {tipo_prontuario}
Janela: {janela} dias

ANTES:
- KM: {antes_km}
- Litros: {antes_litros}
- KM/L Real: {antes_kml}
- KM/L Meta: {antes_meta}
- Desperdício: {antes_desp}

DEPOIS:
- KM: {depois_km}
- Litros: {depois_litros}
- KM/L Real: {depois_kml}
- KM/L Meta: {depois_meta}
- Desperdício: {depois_desp}

DELTAS:
- Delta KM/L: {delta_kml}
- Delta Desperdício: {delta_desp}
- Delta Desperdício %: {delta_desp_pct}

Responda em HTML usando apenas <p>, <b>, <ul>, <li>.
Estrutura:
1) <b>Leitura da Evolução</b>
2) <b>Diagnóstico Operacional</b>
3) <b>Encaminhamento da Etapa</b>
"""

        comp = dados["comparativo"]
        a = comp["antes_periodo"]
        d = comp["depois_periodo"]

        prompt = (
            template_prompt
            .replace("{nome}", dados["motorista_nome"])
            .replace("{chapa}", dados["motorista_chapa"])
            .replace("{tipo_prontuario}", dados["tipo_prontuario"])
            .replace("{janela}", str(comp["dias_janela"]))
            .replace("{antes_km}", f"{a['km']:.0f}")
            .replace("{antes_litros}", f"{a['litros']:.0f}")
            .replace("{antes_kml}", f"{a['kml_real']:.2f}")
            .replace("{antes_meta}", f"{a['kml_meta']:.2f}")
            .replace("{antes_desp}", f"{a['desperdicio']:.1f}")
            .replace("{depois_km}", f"{d['km']:.0f}")
            .replace("{depois_litros}", f"{d['litros']:.0f}")
            .replace("{depois_kml}", f"{d['kml_real']:.2f}")
            .replace("{depois_meta}", f"{d['kml_meta']:.2f}")
            .replace("{depois_desp}", f"{d['desperdicio']:.1f}")
            .replace("{delta_kml}", f"{comp['delta_kml']:+.2f}")
            .replace("{delta_desp}", f"{comp['delta_desperdicio']:+.1f}")
            .replace("{delta_desp_pct}", f"{comp['delta_desperdicio_pct']:+.1f}%")
        )

        resp = model.generate_content(prompt)
        return getattr(resp, "text", "Análise indisponível.").replace("```html", "").replace("```", "")
    except DefaultCredentialsError:
        return "<p>IA indisponível por credencial.</p>"
    except Exception as e:
        print(f"⚠️ Falha ao acionar Vertex AI: {e}")
        return "<p>IA indisponível no momento.</p>"


# ==============================================================================
# CHART
# ==============================================================================
def _build_svg_before_after_chart(comp):
    antes = comp["antes_periodo"]["desperdicio"]
    depois = comp["depois_periodo"]["desperdicio"]

    maxv = max(antes, depois, 1)
    h_max = 180

    h1 = (antes / maxv) * h_max if maxv > 0 else 0
    h2 = (depois / maxv) * h_max if maxv > 0 else 0

    return f"""
    <div class="barWrap">
      <div class="barTitle">Desperdício Antes x Depois ({comp['dias_janela']} dias)</div>
      <svg viewBox="0 0 520 260" width="100%" height="260">
        <line x1="70" y1="20" x2="70" y2="220" class="axis"/>
        <line x1="70" y1="220" x2="470" y2="220" class="axis"/>

        <rect x="140" y="{220 - h1:.1f}" width="80" height="{h1:.1f}" class="barBefore"/>
        <rect x="300" y="{220 - h2:.1f}" width="80" height="{h2:.1f}" class="barAfter"/>

        <text x="180" y="240" text-anchor="middle" class="axisLabel">Antes</text>
        <text x="340" y="240" text-anchor="middle" class="axisLabel">Depois</text>

        <text x="180" y="{220 - h1 - 10:.1f}" text-anchor="middle" class="barValue">{antes:.1f} L</text>
        <text x="340" y="{220 - h2 - 10:.1f}" text-anchor="middle" class="barValue">{depois:.1f} L</text>
      </svg>
    </div>
    """


# ==============================================================================
# HTML
# ==============================================================================
def gerar_html_prontuario_pos(dados: dict, texto_ia: str):
    comp = dados["comparativo"]
    a = comp["antes_periodo"]
    d = comp["depois_periodo"]

    delta_kml = comp["delta_kml"]
    delta_desp = comp["delta_desperdicio"]
    delta_desp_pct = comp["delta_desperdicio_pct"]

    cor_kml = "#16a34a" if delta_kml > 0 else "#dc2626" if delta_kml < 0 else "#475569"
    cor_desp = "#16a34a" if delta_desp < 0 else "#dc2626" if delta_desp > 0 else "#475569"

    chart_html = _build_svg_before_after_chart(comp)

    return f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8" />
<title>Prontuário Pós-Acompanhamento</title>
<style>
  :root {{
    --text:#111827;
    --muted:#6b7280;
    --line:#e5e7eb;
    --shadow: 0 2px 12px rgba(0,0,0,.06);
    --green:#16a34a;
    --red:#dc2626;
    --slate:#475569;
    --blue:#2563eb;
    --amber:#d97706;
  }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: Arial, sans-serif; background:#fff; color:var(--text); margin:0; padding:22px; }}
  .page {{ max-width: 920px; margin: 0 auto; }}
  .topbar {{ height:10px; background:#0f172a; border-radius:999px; margin-bottom:16px; }}
  .header {{ display:flex; justify-content:space-between; gap:16px; align-items:flex-start; }}
  .title {{ font-size:26px; font-weight:900; margin:0; }}
  .sub {{ color:var(--muted); font-size:13px; margin-top:6px; }}
  .badge {{
    display:inline-block; padding:8px 12px; border-radius:999px;
    font-size:12px; font-weight:900; background:#eff6ff; color:#1d4ed8; border:1px solid #bfdbfe;
  }}

  .cards {{ display:grid; grid-template-columns: repeat(4, 1fr); gap:14px; margin:18px 0; }}
  .card {{ background:#fff; border:1px solid var(--line); border-radius:14px; padding:16px 14px; box-shadow:var(--shadow); }}
  .cardBig {{ font-size:20px; font-weight:900; }}
  .cardLabel {{ margin-top:8px; font-size:10px; color:var(--muted); text-transform:uppercase; font-weight:800; }}

  .ai-box {{
    background:#fffde7; border:1px solid #fbc02d; padding:16px; border-radius:8px;
    font-size:13px; margin:18px 0; page-break-inside: avoid;
  }}

  .secTitle {{
    color:#0f172a; font-weight:900; font-size:15px; margin:24px 0 10px 0;
    border-left:4px solid #0f172a; padding-left:8px;
  }}

  table {{
    width:100%; border-collapse: collapse; border:1px solid var(--line); border-radius:8px;
    overflow:hidden; box-shadow:var(--shadow); font-size:12px; margin-bottom:20px;
  }}
  thead th {{
    background:#f8fafc; color:#475569; font-size:10px; text-transform:uppercase;
    padding:8px; border-bottom:1px solid var(--line); text-align:left;
  }}
  tbody td {{ padding:8px; border-bottom:1px solid var(--line); }}
  tbody tr:nth-child(even) td {{ background:#fafafa; }}
  .num {{ text-align:right; font-variant-numeric: tabular-nums; }}
  .strong {{ font-weight:900; }}

  .barWrap {{
    border:1px solid var(--line); border-radius:14px; padding:12px; margin:14px 0;
    box-shadow: var(--shadow); page-break-inside: avoid;
  }}
  .barTitle {{ font-weight:900; margin-bottom:10px; font-size:13px; }}
  .axis {{ stroke:#94a3b8; stroke-width:1.5; }}
  .barBefore {{ fill:#94a3b8; }}
  .barAfter {{ fill:#dc2626; }}
  .axisLabel {{ font-size:12px; fill:#475569; font-weight:700; }}
  .barValue {{ font-size:12px; fill:#111827; font-weight:900; }}

  @media print {{
    body {{ padding:0; }}
    .page {{ max-width:100%; }}
    .card, .barWrap, table {{ box-shadow:none; }}
  }}
</style>
</head>
<body>
  <div class="page">
    <div class="topbar"></div>

    <div class="header">
      <div>
        <div class="title">{_esc(dados["motorista_nome"])}</div>
        <div class="sub">
          Chapa: <b>{_esc(dados["motorista_chapa"])}</b><br>
          Início do acompanhamento: <b>{_esc(dados["dt_inicio_monitoramento"])}</b><br>
          Tipo de prontuário: <b>{_esc(dados["tipo_prontuario"])}</b>
        </div>
      </div>
      <div class="badge">{_esc(comp["conclusao"])}</div>
    </div>

    <div class="cards">
      <div class="card">
        <div class="cardBig" style="color:{cor_kml}">{delta_kml:+.2f}</div>
        <div class="cardLabel">DELTA KM/L</div>
      </div>
      <div class="card">
        <div class="cardBig" style="color:{cor_desp}">{delta_desp:+.1f} L</div>
        <div class="cardLabel">DELTA DESPERDÍCIO</div>
      </div>
      <div class="card">
        <div class="cardBig" style="color:{cor_desp}">{delta_desp_pct:+.1f}%</div>
        <div class="cardLabel">DELTA DESPERDÍCIO %</div>
      </div>
      <div class="card">
        <div class="cardBig">{comp["dias_janela"]} dias</div>
        <div class="cardLabel">JANELA ANALISADA</div>
      </div>
    </div>

    <div class="ai-box">
      <h3 style="margin-top:0; color:#b7950b; font-size:14px; margin-bottom:8px;">💡 Leitura do Agente IA</h3>
      {texto_ia}
    </div>

    <div class="secTitle">1. COMPARATIVO ANTES X DEPOIS</div>
    <table>
      <thead>
        <tr>
          <th>Período</th>
          <th>Dias c/ dado</th>
          <th class="num">KM</th>
          <th class="num">Litros</th>
          <th class="num">KM/L Real</th>
          <th class="num">KM/L Meta</th>
          <th class="num">Litros Ideais</th>
          <th class="num">Desperdício</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class="strong">Antes<br><span style="color:#6b7280">{_esc(a['inicio'])} a {_esc(a['fim'])}</span></td>
          <td>{a['dias_com_dado']}</td>
          <td class="num">{_fmt_int(a['km'])}</td>
          <td class="num">{_fmt_int(a['litros'])}</td>
          <td class="num strong">{_fmt_num(a['kml_real'])}</td>
          <td class="num">{_fmt_num(a['kml_meta'])}</td>
          <td class="num">{_fmt_int(a['litros_ideais'])}</td>
          <td class="num strong">{_fmt_num(a['desperdicio'],1)} L</td>
        </tr>
        <tr>
          <td class="strong">Depois<br><span style="color:#6b7280">{_esc(d['inicio'])} a {_esc(d['fim'])}</span></td>
          <td>{d['dias_com_dado']}</td>
          <td class="num">{_fmt_int(d['km'])}</td>
          <td class="num">{_fmt_int(d['litros'])}</td>
          <td class="num strong">{_fmt_num(d['kml_real'])}</td>
          <td class="num">{_fmt_num(d['kml_meta'])}</td>
          <td class="num">{_fmt_int(d['litros_ideais'])}</td>
          <td class="num strong">{_fmt_num(d['desperdicio'],1)} L</td>
        </tr>
      </tbody>
    </table>

    {chart_html}

    <div class="secTitle">2. LEITURA FINAL DA ETAPA</div>
    <table>
      <thead>
        <tr>
          <th>Indicador</th>
          <th class="num">Resultado</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class="strong">Conclusão da Janela</td>
          <td class="num strong">{_esc(comp["conclusao"])}</td>
        </tr>
        <tr>
          <td>Delta KM/L</td>
          <td class="num" style="color:{cor_kml}; font-weight:900;">{delta_kml:+.2f}</td>
        </tr>
        <tr>
          <td>Delta Desperdício (L)</td>
          <td class="num" style="color:{cor_desp}; font-weight:900;">{delta_desp:+.1f} L</td>
        </tr>
        <tr>
          <td>Delta Desperdício (%)</td>
          <td class="num" style="color:{cor_desp}; font-weight:900;">{delta_desp_pct:+.1f}%</td>
        </tr>
      </tbody>
    </table>
  </div>
</body>
</html>
"""


def html_to_pdf(p_html: Path, p_pdf: Path):
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page()
        page.goto(p_html.resolve().as_uri())
        page.pdf(
            path=str(p_pdf),
            format="A4",
            print_background=True,
            margin={"top": "10mm", "bottom": "10mm", "left": "10mm", "right": "10mm"},
        )
        browser.close()


# ==============================================================================
# BANCO
# ==============================================================================
def marcar_prontuario_gerado(acomp_id: str, tipo_prontuario: str, pdf_path, pdf_url, html_path, html_url, comparativo):
    sb = _sb_b()

    agora = datetime.utcnow().isoformat()
    campo = None
    if tipo_prontuario == "PRONTUARIO_10":
        campo = "prontuario_10_gerado_em"
    elif tipo_prontuario == "PRONTUARIO_20":
        campo = "prontuario_20_gerado_em"
    elif tipo_prontuario == "PRONTUARIO_30":
        campo = "prontuario_30_gerado_em"

    if not campo:
        raise RuntimeError(f"Tipo de prontuário inválido: {tipo_prontuario}")

    payload = {
        campo: agora,
        "updated_at": agora,
    }

    if tipo_prontuario == "PRONTUARIO_30":
        payload["status"] = "EM_ANALISE"

    sb.table(TABELA_ACOMP).update(payload).eq("id", acomp_id).execute()

    # Mantido tipo = PRONTUARIO_10/20/30
    # IMPORTANTE: isso exige que o constraint do banco já esteja ajustado para aceitar esses 3 valores
    evento = {
        "acompanhamento_id": acomp_id,
        "tipo": tipo_prontuario,
        "observacoes": f"{tipo_prontuario} gerado automaticamente.",
        "evidencias_urls": [u for u in [pdf_url, html_url] if u],
        "extra": {
            "comparativo": {
                "dias_janela": comparativo.get("dias_janela"),
                "antes_periodo": {
                    "inicio": comparativo.get("antes_periodo", {}).get("inicio"),
                    "fim": comparativo.get("antes_periodo", {}).get("fim"),
                    "dias_com_dado": comparativo.get("antes_periodo", {}).get("dias_com_dado"),
                    "km": comparativo.get("antes_periodo", {}).get("km"),
                    "litros": comparativo.get("antes_periodo", {}).get("litros"),
                    "kml_real": comparativo.get("antes_periodo", {}).get("kml_real"),
                    "kml_meta": comparativo.get("antes_periodo", {}).get("kml_meta"),
                    "litros_ideais": comparativo.get("antes_periodo", {}).get("litros_ideais"),
                    "desperdicio": comparativo.get("antes_periodo", {}).get("desperdicio"),
                },
                "depois_periodo": {
                    "inicio": comparativo.get("depois_periodo", {}).get("inicio"),
                    "fim": comparativo.get("depois_periodo", {}).get("fim"),
                    "dias_com_dado": comparativo.get("depois_periodo", {}).get("dias_com_dado"),
                    "km": comparativo.get("depois_periodo", {}).get("km"),
                    "litros": comparativo.get("depois_periodo", {}).get("litros"),
                    "kml_real": comparativo.get("depois_periodo", {}).get("kml_real"),
                    "kml_meta": comparativo.get("depois_periodo", {}).get("kml_meta"),
                    "litros_ideais": comparativo.get("depois_periodo", {}).get("litros_ideais"),
                    "desperdicio": comparativo.get("depois_periodo", {}).get("desperdicio"),
                },
                "delta_kml": comparativo.get("delta_kml"),
                "delta_desperdicio": comparativo.get("delta_desperdicio"),
                "delta_desperdicio_pct": comparativo.get("delta_desperdicio_pct"),
                "conclusao": comparativo.get("conclusao"),
            },
            "pdf_path": pdf_path,
            "html_path": html_path,
            "batch_id": PRONTUARIO_BATCH_ID,
        },
    }
    sb.table(TABELA_EVENTOS).insert(evento).execute()


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("🚀 Iniciando geração de prontuários pós-acompanhamento...")
    atualizar_log_lote("PROCESSANDO", extra={"started_at": datetime.utcnow().isoformat()})
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    fila = obter_fila_prontuarios()
    if not fila:
        print("ℹ️ Nenhum prontuário pendente na fila.")
        atualizar_log_lote(
            "CONCLUIDO",
            extra={"ok": 0, "erros": 0, "finished_at": datetime.utcnow().isoformat()},
        )
        return

    ok = 0
    erros = 0
    erros_list = []

    for item in fila:
        acomp_id = item.get("id")
        chapa = str(item.get("motorista_chapa") or "").strip()
        nome = str(item.get("motorista_nome") or chapa).strip().upper()
        tipo_prontuario = str(item.get("prontuario_pendente") or "").strip()
        dt_inicio = _parse_date(item.get("dt_inicio_monitoramento"))

        print(f"\n👤 Processando acompanhamento={acomp_id} | chapa={chapa} | tipo={tipo_prontuario}")

        try:
            if not acomp_id or not chapa or not tipo_prontuario or not dt_inicio:
                raise RuntimeError("Item da fila incompleto.")

            janela = 10 if tipo_prontuario == "PRONTUARIO_10" else 20 if tipo_prontuario == "PRONTUARIO_20" else 30

            dt_ini_busca = (dt_inicio - timedelta(days=janela)).isoformat()
            dt_fim_busca = (dt_inicio + timedelta(days=janela - 1)).isoformat()

            df = carregar_dados_diarios_pos(chapa, dt_ini_busca, dt_fim_busca)
            comparativo = calcular_janela_comparativa(df, dt_inicio, janela)

            dados = {
                "acomp_id": acomp_id,
                "motorista_chapa": chapa,
                "motorista_nome": nome,
                "tipo_prontuario": tipo_prontuario,
                "dt_inicio_monitoramento": dt_inicio.isoformat(),
                "comparativo": comparativo,
            }

            texto_ia = analisar_pos_acompanhamento_ia(dados)

            safe = _safe_filename(f"{nome}_{chapa}_{tipo_prontuario}")
            p_html = PASTA_SAIDA / f"{safe}.html"
            p_pdf = PASTA_SAIDA / f"{safe}.pdf"

            html = gerar_html_prontuario_pos(dados, texto_ia)
            p_html.write_text(html, encoding="utf-8")
            html_to_pdf(p_html, p_pdf)

            pdf_path, pdf_url = upload_storage(p_pdf, f"{safe}.pdf", "application/pdf")
            html_path, html_url = upload_storage(p_html, f"{safe}.html", "text/html; charset=utf-8")

            marcar_prontuario_gerado(
                acomp_id,
                tipo_prontuario,
                pdf_path,
                pdf_url,
                html_path,
                html_url,
                comparativo,
            )

            ok += 1
            print(f"✅ {tipo_prontuario} gerado com sucesso para {chapa}")

        except Exception as e:
            erros += 1
            erros_list.append({
                "acomp_id": acomp_id,
                "motorista": chapa,
                "erro": str(e)[:500],
            })
            print(f"❌ Erro ao processar {chapa}: {e}")

    finished = {
        "ok": ok,
        "erros": erros,
        "erros_list": erros_list,
        "finished_at": datetime.utcnow().isoformat(),
    }

    if erros == 0:
        atualizar_log_lote("CONCLUIDO", extra=finished)
    elif ok == 0:
        atualizar_log_lote("ERRO", msg=f"OK={ok} | ERROS={erros}", extra=finished)
    else:
        atualizar_log_lote("CONCLUIDO_COM_ERROS", msg=f"OK={ok} | ERROS={erros}", extra=finished)

    print("\n🏁 Fim da geração dos prontuários pós-acompanhamento.")


if __name__ == "__main__":
    main()
