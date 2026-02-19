import os
import re
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from supabase import create_client
from playwright.sync_api import sync_playwright

# ==============================================================================
# ENV / CONFIG
# ==============================================================================
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID") or os.getenv("PROJECT_ID")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
VERTEX_MODEL = os.getenv("VERTEX_MODEL", "gemini-2.5-pro")
VERTEX_ENABLED = (os.getenv("VERTEX_ENABLED", "0").strip() == "1")
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON", "").strip()

SUPABASE_B_URL = os.getenv("SUPABASE_B_URL")
SUPABASE_B_SERVICE_ROLE_KEY = os.getenv("SUPABASE_B_SERVICE_ROLE_KEY")

ORDEM_BATCH_ID = os.getenv("ORDEM_BATCH_ID")  # ID do lote no Supabase B

# Tabelas (Supabase B)
TABELA_LOTE = "acompanhamento_lotes"
TABELA_ITENS = "acompanhamento_lote_itens"
TABELA_ORDEM = "diesel_acompanhamentos"
TABELA_EVENTOS = "diesel_acompanhamento_eventos"
TABELA_SUGESTOES = "diesel_sugestoes_acompanhamento"  # base com detalhes_json

# Storage (Supabase B)
BUCKET = "relatorios"
REMOTE_PREFIX = "acompanhamento"
PASTA_SAIDA = Path("Ordens_Geradas")

DEFAULT_DIAS_MONITORAMENTO = int(os.getenv("DEFAULT_DIAS_MONITORAMENTO", "7"))

# ==============================================================================
# HELPERS
# ==============================================================================
def _sb_b():
    if not SUPABASE_B_URL or not SUPABASE_B_SERVICE_ROLE_KEY:
        raise RuntimeError("ENV Supabase B ausente (SUPABASE_B_URL / SUPABASE_B_SERVICE_ROLE_KEY)")
    return create_client(SUPABASE_B_URL, SUPABASE_B_SERVICE_ROLE_KEY)

def _safe_filename(name: str) -> str:
    name = str(name or "").strip()
    name = re.sub(r"[^\w\-.() ]+", "_", name, flags=re.UNICODE)
    return name[:100] or "sem_nome"

def n(v):
    try:
        x = float(v)
        return x if pd.notna(x) else 0.0
    except Exception:
        return 0.0

def atualizar_status_lote(status: str, msg: str = None, extra: dict = None):
    if not ORDEM_BATCH_ID:
        return
    sb = _sb_b()
    payload = {"status": status}
    if msg:
        payload["erro_msg"] = str(msg)[:1000]
    if extra is not None:
        payload["metadata"] = extra
    sb.table(TABELA_LOTE).update(payload).eq("id", ORDEM_BATCH_ID).execute()
    print(f"🔄 [Lote {ORDEM_BATCH_ID}] Status: {status}")

def upload_storage(local_path: Path, remote_name: str, content_type: str):
    """
    Retorna (remote_path, public_url)
    """
    if not ORDEM_BATCH_ID:
        return (None, None)
    if not local_path.exists():
        return (None, None)

    sb = _sb_b()
    remote_path = f"{REMOTE_PREFIX}/{ORDEM_BATCH_ID}/{remote_name}"

    with open(local_path, "rb") as f:
        sb.storage.from_(BUCKET).upload(
            path=remote_path,
            file=f,
            file_options={"content-type": content_type, "upsert": "true"},
        )

    public_url = f"{SUPABASE_B_URL}/storage/v1/object/public/{BUCKET}/{remote_path}"
    return (remote_path, public_url)

def _fmt_int(v):
    try:
        return f"{int(round(float(v))):,}".replace(",", ".")
    except Exception:
        return "0"

def _fmt_float(v, dec=2):
    try:
        return f"{float(v):.{dec}f}".replace(".", ",")
    except Exception:
        return f"{0:.{dec}f}".replace(".", ",")

def _esc(s: str) -> str:
    s = "" if s is None else str(s)
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
         .replace("'", "&#39;")
    )

def _parse_ia_sections(txt: str):
    base = {"diagnostico": "-", "foco": "-", "feedback": "-"}
    if not txt:
        return base

    t = txt.replace("\r\n", "\n").strip()

    def pick(tag):
        m = re.search(rf"{re.escape(tag)}\s*:\s*(.*?)(?=\n[A-ZÁÂÃÉÊÍÓÔÕÚÇ ]{{5,}}:|\Z)", t, flags=re.S)
        return m.group(1).strip() if m else None

    d = pick("DIAGNÓSTICO COMPORTAMENTAL")
    f = pick("FOCO DA MONITORIA")
    fb = pick("FEEDBACK EDUCATIVO")

    if d: base["diagnostico"] = d
    if f: base["foco"] = f
    if fb: base["feedback"] = fb

    if base["diagnostico"] == "-" and base["foco"] == "-" and base["feedback"] == "-":
        base["feedback"] = t[:4000]

    return base

# ==============================================================================
# LEITURA LOTE (Supabase B)
# ==============================================================================
def obter_motoristas_do_lote():
    if not ORDEM_BATCH_ID:
        print("❌ ORDEM_BATCH_ID não fornecido.")
        return []

    print(f"📥 Buscando itens do lote {ORDEM_BATCH_ID}...")
    sb = _sb_b()
    res = sb.table(TABELA_ITENS).select("*").eq("lote_id", ORDEM_BATCH_ID).execute()
    itens = res.data or []
    print(f"📋 {len(itens)} motoristas para processar.")
    return itens

# ==============================================================================
# DADOS DO PRONTUÁRIO (Supabase B -> detalhes_json)
# ==============================================================================
def carregar_detalhes_sugestao(chapa: str, mes_ref: str = None):
    """
    Busca detalhes_json no Supabase B (diesel_sugestoes_acompanhamento).
    Se mes_ref não vier, tenta o mais recente.
    """
    sb = _sb_b()

    q = sb.table(TABELA_SUGESTOES).select("mes_ref, motorista_nome, detalhes_json").eq("chapa", chapa)
    if mes_ref:
        q = q.eq("mes_ref", mes_ref).limit(1)
        r = q.execute().data
        return (r[0] if r else None)

    # pega a mais recente
    r = q.order("mes_ref", desc=True).limit(1).execute().data
    return (r[0] if r else None)

def normalizar_prontuario_from_detalhes(chapa: str, nome: str, mes_ref: str, detalhes_json: dict):
    """
    Converte detalhes_json (raio_x + grafico_semanal) no formato que seu HTML espera.
    """
    raio = detalhes_json.get("raio_x") or []
    weekly = detalhes_json.get("grafico_semanal") or []

    if not raio:
        return None

    rx = pd.DataFrame(raio)
    # compatibiliza nomes esperados no seu template
    # origem do JSON: linha, cluster, km, litros, kml_real, kml_meta, desperdicio
    rx.rename(
        columns={
            "cluster": "Cluster",
            "linha": "linha",
            "km": "Km",
            "litros": "Comb",
            "kml_real": "kml_real",
            "kml_meta": "kml_meta",
            "desperdicio": "desperdicio",
        },
        inplace=True,
    )

    # garante tipos
    for c in ["Km", "Comb", "kml_real", "kml_meta", "desperdicio"]:
        if c in rx.columns:
            rx[c] = pd.to_numeric(rx[c], errors="coerce").fillna(0.0)

    rx = rx.sort_values("desperdicio", ascending=False)

    total_km = float(rx["Km"].sum())
    total_litros = float(rx["Comb"].sum())
    total_desperdicio = float(rx["desperdicio"].sum())

    kml_geral_real = (total_km / total_litros) if total_litros > 0 else 0.0

    litros_teoricos = 0.0
    for _, r in rx.iterrows():
        meta = float(r.get("kml_meta") or 0)
        km = float(r.get("Km") or 0)
        if meta > 0:
            litros_teoricos += (km / meta)
    kml_geral_meta = (total_km / litros_teoricos) if litros_teoricos > 0 else 0.0

    top = rx.iloc[0] if len(rx) else None
    foco_cluster = str(top["Cluster"]) if top is not None else "OUTROS"
    linha_foco = str(top["linha"]) if top is not None else None
    foco = f"{foco_cluster} - Linha {linha_foco}" if linha_foco else "Geral"

    # período (no gerencial isso é “momento da sugestão”; aqui usamos mes_ref como referência)
    # Você pode trocar depois se quiser (ex.: armazenar periodo real no detalhes_json)
    try:
        # mes_ref = 'YYYY-MM'
        dt0 = datetime.strptime(mes_ref + "-01", "%Y-%m-%d").date()
        dt1 = (dt0 + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        periodo_txt = f"{dt0.strftime('%d/%m/%Y')} a {dt1.strftime('%d/%m/%Y')}"
        periodo_inicio = dt0.isoformat()
        periodo_fim = dt1.isoformat()
    except Exception:
        periodo_txt = mes_ref or "-"
        periodo_inicio = None
        periodo_fim = None

    # piora % (opcional): se não existir no json, deixa 0
    piora_pct = float(detalhes_json.get("piora_pct") or 0.0)
    prioridade = detalhes_json.get("prioridade") or None

    # weekly já está no formato {label, real, meta}
    weekly_points = []
    for p in weekly:
        weekly_points.append(
            {
                "label": p.get("label"),
                "real": float(p.get("real")) if p.get("real") is not None else None,
                "meta": float(p.get("meta")) if p.get("meta") is not None else None,
            }
        )

    return {
        "chapa": chapa,
        "nome": nome or chapa,
        "cargo": "MOTORISTA",

        "periodo_inicio": periodo_inicio,
        "periodo_fim": periodo_fim,
        "periodo_txt": periodo_txt,

        "raio_x": rx,
        "weekly": weekly_points,
        "piora_pct": piora_pct,
        "prioridade": prioridade or "PRIORIDADE",

        "totais": {
            "km": total_km,
            "litros": total_litros,
            "desp": total_desperdicio,
            "kml_real": kml_geral_real,
            "kml_meta": kml_geral_meta,
        },

        "foco": foco,
        "foco_cluster": foco_cluster,
        "linha_foco": linha_foco,
        "veiculo_foco": None,
        "mes_ref": mes_ref,
    }

# ==============================================================================
# IA (Opcional)
# ==============================================================================
def chamar_ia_coach(dados):
    if not VERTEX_ENABLED or not VERTEX_PROJECT_ID:
        return "DIAGNÓSTICO COMPORTAMENTAL: IA desativada.\nFOCO DA MONITORIA: -\nFEEDBACK EDUCATIVO: -"

    try:
        credentials = None
        if GOOGLE_CREDENTIALS_JSON:
            from google.oauth2 import service_account
            info = json.loads(GOOGLE_CREDENTIALS_JSON)
            credentials = service_account.Credentials.from_service_account_info(info)

        import vertexai
        from vertexai.generative_models import GenerativeModel

        vertexai.init(project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION, credentials=credentials)
        model = GenerativeModel(VERTEX_MODEL)

        top_linhas = dados["raio_x"].head(6)[["linha", "Cluster", "kml_real", "kml_meta", "desperdicio"]].to_string(index=False)

        prompt = f"""
Você é um Instrutor Técnico Master de Condução Econômica (ônibus).

ALVO:
Motorista: {dados['nome']} ({dados['chapa']})
Período (ref): {dados['periodo_txt']}
Performance (ref): {dados['totais']['kml_real']:.2f} km/l (Meta ref: {dados['totais']['kml_meta']:.2f})
Desperdício (ref): {dados['totais']['desp']:.0f} Litros
Foco: {dados['foco']}

RAIO-X (Top por desperdício):
{top_linhas}

Responda ESTRITAMENTE com 3 tópicos (tags exatas):

DIAGNÓSTICO COMPORTAMENTAL: ...
FOCO DA MONITORIA: ...
FEEDBACK EDUCATIVO: ...
"""
        return model.generate_content(prompt).text

    except Exception as e:
        print(f"Erro IA (ignorado): {e}")
        return "DIAGNÓSTICO COMPORTAMENTAL: IA indisponível.\nFOCO DA MONITORIA: -\nFEEDBACK EDUCATIVO: -"

# ==============================================================================
# HTML/PDF (PRONTUÁRIO) - (seu template, sem mudanças relevantes)
# ==============================================================================
def _build_svg_line_chart(points, title="Performance Semanal"):
    if not points:
        return f"<div class='chartEmpty'>Sem dados suficientes para gráfico.</div>"

    pts = [p for p in points if p.get("real") is not None and p.get("meta") is not None]
    if len(pts) < 2:
        return f"<div class='chartEmpty'>Sem dados suficientes para gráfico.</div>"

    W, H = 760, 260
    padL, padR, padT, padB = 52, 22, 22, 42
    innerW = W - padL - padR
    innerH = H - padT - padB

    ys = []
    for p in pts:
        ys.append(float(p["real"]))
        ys.append(float(p["meta"]))
    y_min = min(ys)
    y_max = max(ys)
    rng = (y_max - y_min) if (y_max > y_min) else 0.5
    y_min -= rng * 0.12
    y_max += rng * 0.12

    def x(i):
        return padL + (innerW * (i / (len(pts) - 1)))

    def y(v):
        return padT + (innerH * (1 - ((v - y_min) / (y_max - y_min if y_max != y_min else 1))))

    real_path = "M " + " L ".join([f"{x(i):.1f} {y(p['real']):.1f}" for i, p in enumerate(pts)])
    meta_path = "M " + " L ".join([f"{x(i):.1f} {y(p['meta']):.1f}" for i, p in enumerate(pts)])

    labels = ""
    for i, p in enumerate(pts):
        labels += f"<text x='{x(i):.1f}' y='{H-18}' text-anchor='middle' class='axisLabel'>{_esc(p['label'])}</text>"

    ticks = ""
    for j in range(5):
        v = y_min + (j * (y_max - y_min) / 4)
        yy = y(v)
        ticks += f"<line x1='{padL}' y1='{yy:.1f}' x2='{W-padR}' y2='{yy:.1f}' class='grid'/>"
        ticks += f"<text x='{padL-10}' y='{yy+4:.1f}' text-anchor='end' class='axisLabel'>{_fmt_float(v,2).replace(',','.')}</text>"

    return f"""
    <div class="chartWrap">
      <div class="chartTitle">{_esc(title)}</div>
      <svg viewBox="0 0 {W} {H}" width="100%" height="{H}">
        {ticks}
        <path d="{meta_path}" class="lineMeta"/>
        <path d="{real_path}" class="lineReal"/>
        {labels}

        <g transform="translate({padL}, {padT-6})">
          <line x1="0" y1="0" x2="26" y2="0" class="legReal"/><text x="34" y="4" class="legend">Realizado</text>
          <line x1="120" y1="0" x2="146" y2="0" class="legMeta"/><text x="154" y="4" class="legend">Meta (Ref)</text>
        </g>
      </svg>
    </div>
    """

def gerar_html_prontuario(prontuario_id: str, d, txt_ia):
    ia = _parse_ia_sections(txt_ia)
    cluster = d.get("foco_cluster") or "OUTROS"

    litros_desvio = float(d["totais"]["desp"])
    piora_pct = float(d.get("piora_pct") or 0.0)
    kml_medio = float(d["totais"]["kml_real"])

    prioridade = d.get("prioridade", "PRIORIDADE")
    tecnologia = "TECNOLOGIA"

    rx = d["raio_x"].copy()
    if rx.empty:
        rx_rows_html = "<tr><td colspan='9' class='muted'>Sem dados.</td></tr>"
    else:
        mes_ref = d.get("mes_ref") or ""
        rx = rx.head(10)
        rx_rows = []
        for _, r in rx.iterrows():
            linha = _esc(r.get("linha"))
            cl = _esc(r.get("Cluster"))
            veic = _esc(r.get("veiculo") or "")
            km = _fmt_int(r.get("Km"))
            litros = _fmt_int(r.get("Comb"))
            real = _fmt_float(r.get("kml_real"), 2)
            meta = _fmt_float(r.get("kml_meta"), 2)
            desp = _fmt_float(r.get("desperdicio"), 1)
            rx_rows.append(
                f"""
                <tr>
                  <td class="td">{_esc(mes_ref)}</td>
                  <td class="td">{veic}</td>
                  <td class="td strong">{linha}</td>
                  <td class="td badge">{cl}</td>
                  <td class="td num strong">{km}</td>
                  <td class="td num">{litros}</td>
                  <td class="td num">{real}</td>
                  <td class="td num">{meta}</td>
                  <td class="td num strong">{desp}</td>
                </tr>
                """
            )
        rx_rows_html = "\n".join(rx_rows)

    chart_html = _build_svg_line_chart(d.get("weekly", []), title=f"Evolução Semanal: {prontuario_id}")

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8" />
<title>Prontuário {prontuario_id}</title>
<style>
  :root {{
    --blue:#1f6fb2; --blue2:#0b5d9a; --red:#c74343;
    --text:#1b1f24; --muted:#6b7280; --line:#e5e7eb;
    --bg:#ffffff; --card:#ffffff; --shadow: 0 2px 10px rgba(0,0,0,.06);
  }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: Arial, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 24px; }}
  .page {{ max-width: 900px; margin: 0 auto; }}
  .topbar {{ height: 10px; background: var(--blue); border-radius: 999px; margin-bottom: 18px; }}
  .header {{ display:flex; align-items:flex-start; justify-content:space-between; gap:16px; }}
  .hTitle {{ font-size: 28px; font-weight: 800; margin: 0; }}
  .hSub {{ margin-top: 6px; color: var(--muted); font-size: 14px; }}
  .prio {{ font-weight: 800; color: var(--muted); font-size: 12px; margin-top: 6px; text-align:right; }}
  .cards {{ display:grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 16px 0 14px 0; }}
  .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px 12px; box-shadow: var(--shadow); min-height: 86px; }}
  .cardBig {{ font-size: 22px; font-weight: 800; margin: 0; color: var(--text); }}
  .cardLabel {{ margin-top: 6px; font-size: 12px; color: var(--muted); letter-spacing: .4px; text-transform: uppercase; }}
  .cardSmall {{ margin-top: 2px; font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .4px; }}
  .chartWrap {{ background: #fff; border: 1px solid var(--line); border-radius: 14px; padding: 12px 14px 8px 14px; box-shadow: var(--shadow); margin-bottom: 18px; }}
  .chartTitle {{ font-weight: 800; margin-bottom: 8px; color: var(--text); }}
  .grid {{ stroke: #eef2f7; stroke-width: 1; }}
  .lineReal {{ fill:none; stroke: var(--blue2); stroke-width: 3; }}
  .lineMeta {{ fill:none; stroke: var(--red); stroke-width: 2.5; stroke-dasharray: 7 5; }}
  .axisLabel {{ font-size: 11px; fill: #6b7280; }}
  .legend {{ font-size: 12px; fill: #374151; }}
  .legReal {{ stroke: var(--blue2); stroke-width: 3; }}
  .legMeta {{ stroke: var(--red); stroke-width: 2.5; stroke-dasharray: 7 5; }}
  .section {{ margin-top: 14px; }}
  .secTitle {{ color: var(--blue); font-weight: 900; font-size: 16px; margin: 14px 0 10px 0; }}
  .divider {{ height: 1px; background: var(--line); margin: 10px 0 12px 0; }}
  table {{ width: 100%; border-collapse: collapse; border: 1px solid var(--line); border-radius: 14px; overflow: hidden; box-shadow: var(--shadow); }}
  thead th {{ background: #f7fafc; color: #374151; font-size: 12px; text-transform: uppercase; letter-spacing: .4px; padding: 10px 10px; border-bottom: 1px solid var(--line); }}
  tbody td {{ padding: 10px 10px; border-bottom: 1px solid var(--line); font-size: 13px; vertical-align: top; }}
  tbody tr:last-child td {{ border-bottom: none; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .strong {{ font-weight: 800; }}
  .badge {{ font-weight: 900; color: #b91c1c; }}
  .muted {{ color: var(--muted); }}
  .box {{ border-left: 5px solid var(--blue); background: #f8fbff; border-radius: 12px; padding: 12px 12px; box-shadow: var(--shadow); border: 1px solid var(--line); }}
  .box.orange {{ border-left-color: #f59e0b; background: #fffbeb; }}
  .box p {{ margin: 0; line-height: 1.45; white-space: pre-wrap; }}
  .foot {{ margin-top: 14px; color: var(--muted); font-size: 11px; text-align: left; }}
</style>
</head>
<body>
  <div class="page">
    <div class="topbar"></div>

    <div class="header">
      <div>
        <div class="hTitle">PRONTUÁRIO: { _esc(prontuario_id) }</div>
        <div class="hSub">Análise Técnica | Período (ref): { _esc(d["periodo_txt"]) }</div>
      </div>
      <div class="prio">{ _esc(prioridade) }</div>
    </div>

    <div class="cards">
      <div class="card">
        <div class="cardBig">{ _esc(cluster) }</div>
        <div class="cardSmall">{ _esc(tecnologia) }</div>
      </div>
      <div class="card">
        <div class="cardBig">{ _esc(_fmt_int(litros_desvio)) } L</div>
        <div class="cardLabel">DESVIO</div>
      </div>
      <div class="card">
        <div class="cardBig">{ _esc(_fmt_float(piora_pct, 1).replace(",", ".")) }%</div>
        <div class="cardLabel">PIORA</div>
      </div>
      <div class="card">
        <div class="cardBig">{ _esc(_fmt_float(kml_medio, 2).replace(",", ".")) }</div>
        <div class="cardLabel">KM/L MÉDIO</div>
      </div>
    </div>

    {chart_html}

    <div class="section">
      <div class="secTitle">1. RAIO-X DA OPERAÇÃO</div>
      <div class="divider"></div>

      <table>
        <thead>
          <tr>
            <th>Mês</th>
            <th>Veículo</th>
            <th>Linha</th>
            <th>Cluster</th>
            <th class="num">KM Tot</th>
            <th class="num">Litros</th>
            <th class="num">Real</th>
            <th class="num">Meta</th>
            <th class="num">Desperdício</th>
          </tr>
        </thead>
        <tbody>
          {rx_rows_html}
        </tbody>
      </table>
    </div>

    <div class="section">
      <div class="secTitle">2. DIAGNÓSTICO COMPORTAMENTAL</div>
      <div class="divider"></div>
      <div class="box orange"><p>{ _esc(ia["diagnostico"]) }</p></div>
    </div>

    <div class="section">
      <div class="secTitle">3. FOCO DA MONITORIA</div>
      <div class="divider"></div>
      <div class="box"><p>{ _esc(ia["foco"]) }</p></div>
    </div>

    <div class="section">
      <div class="secTitle">4. FEEDBACK EDUCATIVO</div>
      <div class="divider"></div>
      <div class="box"><p>{ _esc(ia["feedback"]) }</p></div>
    </div>

    <div class="foot">Gerado automaticamente pelo Agente Diesel AI (base: detalhes_json do gerencial).</div>
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
# SUPABASE B: CRIAR ORDEM + EVENTO
# ==============================================================================
def criar_ordem_e_evento(sb_b, dados, lote_id, pdf_path, pdf_url, html_path, html_url, txt_ia):
    dt_inicio = datetime.utcnow().date().isoformat()
    dias = DEFAULT_DIAS_MONITORAMENTO
    dt_fim_planejado = (datetime.utcnow().date() + timedelta(days=dias - 1)).isoformat()

    raio_top = dados["raio_x"].head(10).to_dict(orient="records")

    payload = {
        "lote_id": lote_id,

        "motorista_chapa": dados["chapa"],
        "motorista_nome": dados["nome"],
        "motivo": dados["foco"],

        "status": "AGUARDANDO_INSTRUTOR",
        "dias_monitoramento": dias,
        "dt_inicio": dt_inicio,
        "dt_fim_planejado": dt_fim_planejado,

        "kml_inicial": dados["totais"]["kml_real"],
        "kml_meta": dados["totais"]["kml_meta"],
        "observacao_inicial": txt_ia[:5000],

        "arquivo_pdf_url": pdf_url,
        "arquivo_html_url": html_url,

        "metadata": {
            "versao": "V14_prontuario_from_detalhes_json",
            "lote_id": lote_id,
            "mes_ref": dados.get("mes_ref"),
            "foco": dados["foco"],
            "cluster_foco": dados["foco_cluster"],
            "linha_foco": dados["linha_foco"],
            "kpis_ref": dados["totais"],
            "raio_x_top10": raio_top,
            "weekly_points": dados.get("weekly", []),
            "piora_pct": dados.get("piora_pct", 0.0),
            "prioridade": dados.get("prioridade", None),
            "pdf_path": pdf_path,
            "html_path": html_path,
            "origem": "diesel_sugestoes_acompanhamento.detalhes_json",
        },

        "evidencias_urls": [u for u in [pdf_url, html_url] if u],
    }

    ordem = sb_b.table(TABELA_ORDEM).insert(payload).execute().data
    if not ordem:
        raise RuntimeError("Falha ao inserir diesel_acompanhamentos (ordem vazia).")
    ordem_id = ordem[0].get("id")
    if not ordem_id:
        raise RuntimeError("Falha: diesel_acompanhamentos retornou sem id.")

    evento = {
        "acompanhamento_id": ordem_id,
        "tipo": "LANCAMENTO",
        "observacoes": f"Ordem gerada automaticamente (lote {lote_id}). Foco: {dados['foco']}",
        "evidencias_urls": [u for u in [pdf_url, html_url] if u],
        "periodo_inicio": dados.get("periodo_inicio"),
        "periodo_fim": dados.get("periodo_fim"),
        "kml": dados["totais"]["kml_real"],
        "extra": {
            "kml_meta_ref": dados["totais"]["kml_meta"],
            "desperdicio_ref": dados["totais"]["desp"],
            "cluster_foco": dados["foco_cluster"],
            "linha_foco": dados["linha_foco"],
            "weekly_points": dados.get("weekly", []),
        },
    }
    sb_b.table(TABELA_EVENTOS).insert(evento).execute()

    return ordem_id

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    if not ORDEM_BATCH_ID:
        print("❌ ORDEM_BATCH_ID não definido no ambiente.")
        return

    ok = 0
    erros = 0
    erros_list = []

    atualizar_status_lote("PROCESSANDO", extra={"started_at": datetime.utcnow().isoformat()})
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    itens = obter_motoristas_do_lote()
    if not itens:
        atualizar_status_lote("ERRO", "Lote sem itens")
        return

    sb_b = _sb_b()
    print(f"🚀 Iniciando geração de {len(itens)} ordens (usando detalhes_json do gerencial)...")

    for item in itens:
        mot_chapa = str(item.get("motorista_chapa") or "").strip()
        if not mot_chapa:
            continue

        try:
            # tenta usar mes_ref do item (se tiver), senão pega a sugestão mais recente
            mes_ref = (item.get("mes_ref") or item.get("extra") or {}).get("mes_ref") if isinstance(item.get("extra"), dict) else None

            sug = carregar_detalhes_sugestao(mot_chapa, mes_ref=mes_ref)
            if not sug or not sug.get("detalhes_json"):
                raise RuntimeError("Sugestão não encontrada ou detalhes_json vazio no Supabase B.")

            mes_ref_final = sug.get("mes_ref") or (mes_ref or "")
            nome = (item.get("extra") or {}).get("motorista_nome") if isinstance(item.get("extra"), dict) else None
            nome = nome or sug.get("motorista_nome") or mot_chapa

            dados = normalizar_prontuario_from_detalhes(mot_chapa, nome, mes_ref_final, sug["detalhes_json"])
            if not dados:
                raise RuntimeError("detalhes_json sem raio_x válido (não dá para montar prontuário).")

            print(f"   ⚙️ Gerando PRONTUÁRIO PDF/HTML para {mot_chapa}...")

            prontuario_id = mot_chapa
            safe = _safe_filename(f"{prontuario_id}_Prontuario_{mes_ref_final}")
            p_html = PASTA_SAIDA / f"{safe}.html"
            p_pdf = PASTA_SAIDA / f"{safe}.pdf"

            txt_ia = chamar_ia_coach(dados)

            html = gerar_html_prontuario(prontuario_id, dados, txt_ia)
            p_html.write_text(html, encoding="utf-8")
            html_to_pdf(p_html, p_pdf)

            pdf_path, pdf_url = upload_storage(p_pdf, f"{safe}.pdf", "application/pdf")
            html_path, html_url = upload_storage(p_html, f"{safe}.html", "text/html")

            ordem_id = criar_ordem_e_evento(
                sb_b=sb_b,
                dados=dados,
                lote_id=ORDEM_BATCH_ID,
                pdf_path=pdf_path,
                pdf_url=pdf_url,
                html_path=html_path,
                html_url=html_url,
                txt_ia=txt_ia,
            )

            ok += 1
            print(f"✅ {mot_chapa}: Ordem criada (id={ordem_id}).")

        except Exception as e:
            erros += 1
            msg = str(e)
            erros_list.append({"motorista": mot_chapa, "erro": msg[:500]})
            print(f"❌ {mot_chapa}: erro: {msg}")

    finished = {"ok": ok, "erros": erros, "erros_list": erros_list, "finished_at": datetime.utcnow().isoformat()}
    if erros == 0 and ok > 0:
        atualizar_status_lote("CONCLUIDO", extra=finished)
    elif ok == 0 and erros > 0:
        atualizar_status_lote("ERRO", msg=f"OK={ok} | ERROS={erros}", extra=finished)
    else:
        atualizar_status_lote("CONCLUIDO_COM_ERROS", msg=f"OK={ok} | ERROS={erros}", extra=finished)

    print(f"🏁 Finalizado. OK={ok} | ERROS={erros}")

if __name__ == "__main__":
    main()
