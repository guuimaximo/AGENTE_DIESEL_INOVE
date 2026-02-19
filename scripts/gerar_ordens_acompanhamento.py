# -*- coding: utf-8 -*-
"""
gerar_ordens_acompanhamento.py (VERSÃO AJUSTADA PARA USAR OS DETALHES DO GERENCIAL/SUGESTÃO)

O que muda (conforme você pediu):
- Prontuário NÃO recalcula do Supabase A (evita erro 'Comb.' e mantém “igual gerencial”).
- Usa diesel_sugestoes_acompanhamento (Supabase B) e o detalhes_json:
  - raio_x (linha/cluster/km/litros/kml_real/kml_meta/desperdicio)
  - grafico_semanal (label/real/meta)
  - (opcional) periodo_inicio / periodo_fim
- Layout ajustado:
  - Nome grande, chapa pequena
  - Sem card “Tecnologia”
  - Raio-X com TOTAL no rodapé
  - Sem veículo
  - Gráfico com legenda (Real vermelho, Ref cinza tracejado) igual page
  - Período correto (se tiver no detalhes_json; senão fallback por created_at)
  - Evidencia “piora” em vermelho
  - Card “Piora” = KM/L Ref → KM/L Média

Requisitos:
- ENV: SUPABASE_B_URL, SUPABASE_B_SERVICE_ROLE_KEY, ORDEM_BATCH_ID
- Tabelas (B): acompanhamento_lotes, acompanhamento_lote_itens, diesel_acompanhamentos, diesel_acompanhamento_eventos, diesel_sugestoes_acompanhamento
- Storage (B): bucket relatorios
"""

import os
import re
import json
from datetime import datetime, timedelta, date
from pathlib import Path

from supabase import create_client
from playwright.sync_api import sync_playwright

# ==============================================================================
# ENV / CONFIG (Supabase B)
# ==============================================================================
SUPABASE_B_URL = os.getenv("SUPABASE_B_URL")
SUPABASE_B_SERVICE_ROLE_KEY = os.getenv("SUPABASE_B_SERVICE_ROLE_KEY")

ORDEM_BATCH_ID = os.getenv("ORDEM_BATCH_ID")  # ID do lote no Supabase B

# Tabelas (Supabase B)
TABELA_LOTE = "acompanhamento_lotes"
TABELA_ITENS = "acompanhamento_lote_itens"
TABELA_ORDEM = "diesel_acompanhamentos"
TABELA_EVENTOS = "diesel_acompanhamento_eventos"
TABELA_SUG = "diesel_sugestoes_acompanhamento"  # fonte do detalhes_json

# Storage (Supabase B)
BUCKET = "relatorios"
REMOTE_PREFIX = "acompanhamento"
PASTA_SAIDA = Path("Ordens_Geradas")

# padrão: ordem nasce aguardando instrutor, monitoramento padrão (instrutor redefine no LANÇAR)
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
    return name[:120] or "sem_nome"


def n(v):
    try:
        x = float(v)
        return x if x == x else 0.0
    except Exception:
        return 0.0


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


def _prioridade_por_desperdicio(litros: float):
    litros = float(litros or 0)
    if litros >= 150:
        return "PRIORIDADE ALTA"
    if litros >= 60:
        return "PRIORIDADE MÉDIA"
    return "PRIORIDADE BAIXA"


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
# BUSCA DETALHES (SUGESTÃO GERENCIAL) - Supabase B
# ==============================================================================
def buscar_sugestao_detalhada(sb, chapa: str, mes_ref: str = None):
    """
    Tenta buscar a sugestão do motorista no diesel_sugestoes_acompanhamento.
    Preferência:
      1) chapa + mes_ref (se veio do item)
      2) última sugestão do chapa (order created_at desc)
    Retorna dict com:
      - motorista_nome
      - detalhes_json
      - created_at
      - mes_ref
    """
    q = sb.table(TABELA_SUG).select("motorista_nome, detalhes_json, created_at, mes_ref").eq("chapa", chapa)

    if mes_ref:
        r = q.eq("mes_ref", mes_ref).maybe_single().execute()
        if r.data and r.data.get("detalhes_json"):
            return r.data

    r2 = (
        sb.table(TABELA_SUG)
        .select("motorista_nome, detalhes_json, created_at, mes_ref")
        .eq("chapa", chapa)
        .order("created_at", desc=True)
        .limit(1)
        .maybe_single()
        .execute()
    )
    return r2.data


def _periodo_from_detalhes(detalhes: dict, created_at_iso: str = None):
    """
    1) Se detalhes_json tiver periodo_inicio/periodo_fim (YYYY-MM-DD), usa isso.
    2) Senão, usa created_at como fim e 30 dias para trás como início.
    """
    pi = (detalhes or {}).get("periodo_inicio")
    pf = (detalhes or {}).get("periodo_fim")

    if pi and pf:
        try:
            dt0 = datetime.strptime(pi, "%Y-%m-%d").date()
            dt1 = datetime.strptime(pf, "%Y-%m-%d").date()
            return {
                "periodo_inicio": dt0.isoformat(),
                "periodo_fim": dt1.isoformat(),
                "periodo_txt": f"{dt0.strftime('%d/%m/%Y')} a {dt1.strftime('%d/%m/%Y')}",
            }
        except Exception:
            pass

    # fallback por created_at
    try:
        if created_at_iso:
            dt1 = datetime.fromisoformat(created_at_iso.replace("Z", "+00:00")).date()
        else:
            dt1 = datetime.utcnow().date()
    except Exception:
        dt1 = datetime.utcnow().date()

    dt0 = dt1 - timedelta(days=29)
    return {
        "periodo_inicio": dt0.isoformat(),
        "periodo_fim": dt1.isoformat(),
        "periodo_txt": f"{dt0.strftime('%d/%m/%Y')} a {dt1.strftime('%d/%m/%Y')}",
    }


def normalizar_prontuario_from_detalhes(chapa: str, nome: str, detalhes: dict, created_at_iso: str = None):
    """
    Converte o detalhes_json do gerencial para o “payload” do prontuário (HTML/PDF + insert).
    Esperado:
      detalhes.raio_x = [{linha, cluster, km, litros, kml_real, kml_meta, desperdicio}, ...]
      detalhes.grafico_semanal = [{label, real, meta}, ...]
    """
    if not detalhes:
        return None

    raio_x = detalhes.get("raio_x") or []
    weekly = detalhes.get("grafico_semanal") or []

    if not isinstance(raio_x, list) or len(raio_x) == 0:
        return None

    # Totais (igual sua page)
    total_km = sum(n(r.get("km")) for r in raio_x)
    total_litros = sum(n(r.get("litros")) for r in raio_x)
    total_desp = sum(n(r.get("desperdicio")) for r in raio_x)

    kml_real = (total_km / total_litros) if total_litros > 0 else 0.0

    litros_teoricos_total = sum((n(r.get("km")) / n(r.get("kml_meta"))) if n(r.get("kml_meta")) > 0 else 0.0 for r in raio_x)
    kml_meta = (total_km / litros_teoricos_total) if litros_teoricos_total > 0 else 0.0

    # Foco: maior desperdício
    top = sorted(raio_x, key=lambda r: n(r.get("desperdicio")), reverse=True)[0]
    foco_cluster = (top.get("cluster") or "OUTROS")
    foco_linha = (top.get("linha") or "-")
    foco = f"{foco_cluster} - Linha {foco_linha}"

    prioridade = _prioridade_por_desperdicio(total_desp)
    periodo = _periodo_from_detalhes(detalhes, created_at_iso=created_at_iso)

    return {
        "chapa": chapa,
        "nome": (nome or chapa).strip().upper(),
        "cargo": "MOTORISTA",

        "periodo_inicio": periodo["periodo_inicio"],
        "periodo_fim": periodo["periodo_fim"],
        "periodo_txt": periodo["periodo_txt"],

        "raio_x": raio_x,     # lista (já pronta)
        "weekly": weekly,     # lista (já pronta)

        "totais": {
            "km": float(total_km),
            "litros": float(total_litros),
            "desp": float(total_desp),
            "kml_real": float(kml_real),
            "kml_meta": float(kml_meta),
        },

        "foco": foco,
        "foco_cluster": foco_cluster,
        "linha_foco": foco_linha,
        "prioridade": prioridade,
    }


# ==============================================================================
# HTML/PDF (PRONTUÁRIO) - Layout ajustado
# ==============================================================================
def _build_svg_line_chart(points, title="Evolução Semanal"):
    """
    SVG simples com 2 linhas:
    - Real (vermelho)
    - Ref/Meta (cinza tracejado)
    Com legendas iguais à page.
    """
    if not points:
        return "<div class='chartEmpty'>Sem dados suficientes para gráfico.</div>"

    pts = [p for p in points if p.get("real") is not None and p.get("meta") is not None]
    if len(pts) < 2:
        return "<div class='chartEmpty'>Sem dados suficientes para gráfico.</div>"

    W, H = 760, 260
    padL, padR, padT, padB = 56, 20, 26, 46
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
        denom = (y_max - y_min) if (y_max != y_min) else 1.0
        return padT + (innerH * (1 - ((v - y_min) / denom)))

    real_path = "M " + " L ".join([f"{x(i):.1f} {y(p['real']):.1f}" for i, p in enumerate(pts)])
    meta_path = "M " + " L ".join([f"{x(i):.1f} {y(p['meta']):.1f}" for i, p in enumerate(pts)])

    # x labels + valores (igual page)
    labels = ""
    for i, p in enumerate(pts):
        labels += f"<text x='{x(i):.1f}' y='{H-18}' text-anchor='middle' class='axisLabel'>{_esc(p['label'])}</text>"
        labels += f"<text x='{x(i):.1f}' y='{y(p['real'])-10:.1f}' text-anchor='middle' class='valReal'>{n(p['real']):.2f}</text>"
        labels += f"<text x='{x(i):.1f}' y='{y(p['meta'])+16:.1f}' text-anchor='middle' class='valMeta'>Ref: {n(p['meta']):.2f}</text>"

    # y ticks
    ticks = ""
    for j in range(5):
        v = y_min + (j * (y_max - y_min) / 4)
        yy = y(v)
        ticks += f"<line x1='{padL}' y1='{yy:.1f}' x2='{W-padR}' y2='{yy:.1f}' class='grid'/>"
        ticks += f"<text x='{padL-10}' y='{yy+4:.1f}' text-anchor='end' class='axisLabel'>{v:.2f}</text>"

    return f"""
    <div class="chartWrap">
      <div class="chartTitle">{_esc(title)}</div>
      <svg viewBox="0 0 {W} {H}" width="100%" height="{H}">
        {ticks}
        <path d="{meta_path}" class="lineMeta"/>
        <path d="{real_path}" class="lineReal"/>
        {labels}

        <!-- legend -->
        <g transform="translate({padL}, {padT-8})">
          <line x1="0" y1="0" x2="26" y2="0" class="legMeta"/>
          <text x="34" y="4" class="legend">Ref</text>

          <line x1="90" y1="0" x2="116" y2="0" class="legReal"/>
          <text x="124" y="4" class="legend">Realizado</text>
        </g>
      </svg>
    </div>
    """


def gerar_html_prontuario(prontuario_id: str, d: dict):
    """
    d: saída de normalizar_prontuario_from_detalhes()
    """
    # cards
    cluster = d.get("foco_cluster") or "OUTROS"
    prioridade = d.get("prioridade") or "PRIORIDADE"

    litros_desvio = float(d["totais"]["desp"])
    kml_media = float(d["totais"]["kml_real"])
    kml_ref = float(d["totais"]["kml_meta"])

    # “Piora” card = Ref → Média
    piora_txt = f"{kml_ref:.2f} → {kml_media:.2f}"
    piora_is_bad = (kml_media < kml_ref)

    # Raio-x (Top 10)
    rx = list(d.get("raio_x") or [])
    rx = sorted(rx, key=lambda r: n(r.get("desperdicio")), reverse=True)[:10]

    # table rows + destaque piora
    if not rx:
        rx_rows_html = "<tr><td colspan='7' class='muted'>Sem dados.</td></tr>"
    else:
        rows = []
        for r in rx:
            linha = _esc(r.get("linha") or "-")
            cl = _esc(r.get("cluster") or "-")
            km = _fmt_int(n(r.get("km")))
            litros = _fmt_int(n(r.get("litros")))
            real = f"{n(r.get('kml_real')):.2f}"
            meta = f"{n(r.get('kml_meta')):.2f}"
            desp = f"{n(r.get('desperdicio')):.1f}"

            is_piora = (n(r.get("kml_real")) < n(r.get("kml_meta")))
            row_style = "background:#fff1f2;" if is_piora else ""
            real_style = "color:#dc2626;font-weight:900;" if is_piora else "font-weight:800;"
            desp_style = "color:#b91c1c;font-weight:900;" if n(r.get("desperdicio")) > 0 else "color:#059669;font-weight:900;"

            rows.append(
                f"""
                <tr style="{row_style}">
                  <td class="td strong">{linha}</td>
                  <td class="td badge">{cl}</td>
                  <td class="td num strong">{km}</td>
                  <td class="td num">{litros}</td>
                  <td class="td num" style="{real_style}">{real}</td>
                  <td class="td num muted">{meta}</td>
                  <td class="td num" style="{desp_style}">{desp}</td>
                </tr>
                """
            )
        rx_rows_html = "\n".join(rows)

    chart_html = _build_svg_line_chart(d.get("weekly", []), title="2. EVOLUÇÃO SEMANAL")

    # totais rodapé
    total_km = _fmt_int(d["totais"]["km"])
    total_litros = _fmt_int(d["totais"]["litros"])
    total_desperdicio = f"{float(d['totais']['desp']):.1f}"
    total_kml_real = f"{float(d['totais']['kml_real']):.2f}"
    total_kml_ref = f"{float(d['totais']['kml_meta']):.2f}"

    # cores “piora”
    piora_style = "color:#dc2626;font-weight:900;" if piora_is_bad else "color:#111827;font-weight:900;"

    return f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Prontuário { _esc(prontuario_id) }</title>
<style>
  :root {{
    --text:#111827;
    --muted:#6b7280;
    --line:#e5e7eb;
    --shadow: 0 2px 12px rgba(0,0,0,.06);
    --red:#dc2626;
    --slate:#94a3b8;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: Arial, sans-serif;
    background: #fff;
    color: var(--text);
    margin: 0;
    padding: 22px;
  }}
  .page {{ max-width: 920px; margin: 0 auto; }}
  .topbar {{ height: 10px; background: #0f172a; border-radius: 999px; margin-bottom: 16px; }}

  .header {{ display:flex; align-items:flex-start; justify-content:space-between; gap:16px; }}
  .hTitle {{ font-size: 30px; font-weight: 900; margin: 0; letter-spacing: .2px; }}
  .hSub {{ margin-top: 6px; color: var(--muted); font-size: 13px; }}
  .prio {{ font-weight: 900; font-size: 12px; color: var(--muted); text-align:right; }}

  .cards {{
    display:grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin: 18px 0 16px 0;
  }}
  .card {{
    background:#fff;
    border:1px solid var(--line);
    border-radius: 14px;
    padding: 16px 14px;
    box-shadow: var(--shadow);
    min-height: 84px;
  }}
  .cardBig {{ font-size: 20px; font-weight: 900; margin: 0; }}
  .cardLabel {{
    margin-top: 8px;
    font-size: 11px;
    color: var(--muted);
    letter-spacing: .45px;
    text-transform: uppercase;
    font-weight: 800;
  }}

  .chartWrap {{
    background:#fff;
    border:1px solid var(--line);
    border-radius: 14px;
    padding: 12px 14px 8px 14px;
    box-shadow: var(--shadow);
    margin: 10px 0 18px 0;
  }}
  .chartTitle {{ font-weight: 900; margin-bottom: 10px; color: var(--text); }}
  .grid {{ stroke: #f1f5f9; stroke-width: 1; }}
  .lineReal {{ fill:none; stroke: var(--red); stroke-width: 3; }}
  .lineMeta {{ fill:none; stroke: var(--slate); stroke-width: 2.5; stroke-dasharray: 6 6; }}
  .axisLabel {{ font-size: 11px; fill: var(--muted); }}
  .legend {{ font-size: 12px; fill: #374151; font-weight: 700; }}
  .legReal {{ stroke: var(--red); stroke-width: 3; }}
  .legMeta {{ stroke: var(--slate); stroke-width: 2.5; stroke-dasharray: 6 6; }}
  .valReal {{ font-size: 10px; fill: var(--red); font-weight: 900; }}
  .valMeta {{ font-size: 9px; fill: #64748b; font-weight: 700; }}

  .secTitle {{ color:#0f172a; font-weight: 900; font-size: 15px; margin: 18px 0 10px 0; }}
  .divider {{ height:1px; background: var(--line); margin: 10px 0 12px 0; }}

  table {{
    width:100%;
    border-collapse: collapse;
    border:1px solid var(--line);
    border-radius: 14px;
    overflow:hidden;
    box-shadow: var(--shadow);
  }}
  thead th {{
    background:#f8fafc;
    color:#475569;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .45px;
    padding: 10px 10px;
    border-bottom: 1px solid var(--line);
    text-align:left;
  }}
  tbody td {{
    padding: 10px 10px;
    border-bottom: 1px solid var(--line);
    font-size: 13px;
    vertical-align: top;
  }}
  tbody tr:nth-child(even) td {{ background:#fafafa; }}
  tbody tr:last-child td {{ border-bottom: none; }}
  .num {{ text-align:right; font-variant-numeric: tabular-nums; }}
  .strong {{ font-weight: 900; }}
  .muted {{ color: var(--muted); }}
  .badge {{ font-weight: 900; color:#b91c1c; }}

  tfoot td {{
    background: #0f172a;
    color:#fff;
    padding: 10px 10px;
    font-size: 13px;
    font-weight: 900;
    border-top: 2px solid #020617;
  }}
  .footRef {{ color:#cbd5e1; font-weight:900; }}
  .footReal {{ color:#fde047; font-weight:900; }}
  .footDesp {{ background: rgba(127,29,29,.45); color:#fecaca; font-weight:900; }}

  .footnote {{ margin-top: 14px; color: var(--muted); font-size: 11px; }}

  @media print {{
    body {{ padding: 0; }}
    .page {{ max-width: 100%; }}
    .card, .chartWrap, table {{ box-shadow: none; }}
  }}
</style>
</head>

<body>
  <div class="page">
    <div class="topbar"></div>

    <div class="header">
      <div>
        <div class="hTitle">{ _esc(d["nome"]) }</div>
        <div class="hSub">
          Chapa: <b>{ _esc(d["chapa"]) }</b> • Período: <b>{ _esc(d["periodo_txt"]) }</b>
        </div>
      </div>
      <div class="prio">{ _esc(prioridade) }</div>
    </div>

    <div class="cards">
      <div class="card">
        <div class="cardBig">{ _esc(cluster) }</div>
        <div class="cardLabel">CLUSTER FOCO</div>
      </div>

      <div class="card">
        <div class="cardBig">{ _fmt_int(litros_desvio) } L</div>
        <div class="cardLabel">DESPERDÍCIO (30D)</div>
      </div>

      <div class="card">
        <div class="cardBig" style="{piora_style}">{ _esc(piora_txt) }</div>
        <div class="cardLabel">KM/L REF → MÉDIA</div>
      </div>

      <div class="card">
        <div class="cardBig">{ _esc(f"{kml_media:.2f}") }</div>
        <div class="cardLabel">KM/L MÉDIO</div>
      </div>
    </div>

    {chart_html}

    <div class="secTitle">1. RAIO-X DA OPERAÇÃO</div>
    <div class="divider"></div>

    <table>
      <thead>
        <tr>
          <th>Linha</th>
          <th>Cluster</th>
          <th class="num">KM</th>
          <th class="num">Litros</th>
          <th class="num">Real</th>
          <th class="num">Ref</th>
          <th class="num">Desp.</th>
        </tr>
      </thead>

      <tbody>
        {rx_rows_html}
      </tbody>

      <tfoot>
        <tr>
          <td colspan="2" class="num footRef">TOTAL</td>
          <td class="num">{ _esc(total_km) }</td>
          <td class="num">{ _esc(total_litros) }</td>
          <td class="num footReal">{ _esc(total_kml_real) }</td>
          <td class="num footRef">{ _esc(total_kml_ref) }</td>
          <td class="num footDesp">{ _esc(total_desperdicio) }</td>
        </tr>
      </tfoot>
    </table>

    <div class="footnote">Gerado automaticamente pelo Agente Diesel (baseado na Sugestão/Gerencial).</div>
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
# (mantém suas chaves e o padrão do seu fluxo)
# ==============================================================================
def criar_ordem_e_evento(sb_b, dados, lote_id, pdf_path, pdf_url, html_path, html_url):
    # dt padrão (instrutor redefine no LANÇAR)
    dt_inicio = datetime.utcnow().date().isoformat()
    dias = DEFAULT_DIAS_MONITORAMENTO
    dt_fim_planejado = (datetime.utcnow().date() + timedelta(days=dias - 1)).isoformat()

    # top 10 já pronto (do raio_x)
    raio_top = list(dados.get("raio_x") or [])[:10]

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
        # aqui guardamos um resumo curto textual (sem IA agora) – pode trocar depois
        "observacao_inicial": f"Ref {dados['totais']['kml_meta']:.2f} → Média {dados['totais']['kml_real']:.2f} | Desp {dados['totais']['desp']:.1f} L | Período {dados['periodo_txt']}",

        "arquivo_pdf_url": pdf_url,
        "arquivo_html_url": html_url,

        "metadata": {
            "versao": "V14_prontuario_from_sugestao",
            "lote_id": lote_id,
            "periodo_inicio": dados["periodo_inicio"],
            "periodo_fim": dados["periodo_fim"],
            "foco": dados["foco"],
            "cluster_foco": dados["foco_cluster"],
            "linha_foco": dados.get("linha_foco"),
            "kpis": dados["totais"],
            "raio_x_top10": raio_top,
            "weekly_points": dados.get("weekly", []),
            "prioridade": dados.get("prioridade", None),
            "pdf_path": pdf_path,
            "html_path": html_path,
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
        "periodo_inicio": dados["periodo_inicio"],
        "periodo_fim": dados["periodo_fim"],
        "kml": dados["totais"]["kml_real"],
        "extra": {
            "kml_meta_ref": dados["totais"]["kml_meta"],
            "desperdicio_litros": dados["totais"]["desp"],
            "cluster_foco": dados["foco_cluster"],
            "linha_foco": dados.get("linha_foco"),
            "weekly_points": dados.get("weekly", []),
            "prioridade": dados.get("prioridade", None),
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
    print(f"🚀 Iniciando geração de {len(itens)} prontuários (diesel_acompanhamentos)...")

    for item in itens:
        chapa = str(item.get("motorista_chapa") or "").strip()
        if not chapa:
            continue

        try:
            mes_ref = (item.get("mes_ref") or item.get("extra", {}) or {}).get("mes_ref") if isinstance(item.get("extra"), dict) else None
            if not mes_ref:
                # se o item tiver kml/linha etc, mas não mes_ref, tudo bem: buscamos o último por created_at
                mes_ref = None

            # nome: preferência pelo item.extra.motorista_nome; senão pela sugestão
            nome_item = None
            if isinstance(item.get("extra"), dict):
                nome_item = item["extra"].get("motorista_nome")

            sug = buscar_sugestao_detalhada(sb_b, chapa, mes_ref=mes_ref)
            if not sug or not sug.get("detalhes_json"):
                raise RuntimeError("Sugestão/detalhes_json não encontrado para este motorista (diesel_sugestoes_acompanhamento).")

            detalhes = sug.get("detalhes_json") or {}
            created_at = sug.get("created_at")
            nome_sug = sug.get("motorista_nome")

            nome = (nome_item or nome_sug or chapa)
            dados = normalizar_prontuario_from_detalhes(chapa, nome, detalhes, created_at_iso=created_at)
            if not dados:
                raise RuntimeError("detalhes_json inválido (sem raio_x/grafico_semanal).")

            # prontuario_id: pode ser a CHAPA ou um ID real depois
            prontuario_id = chapa

            safe = _safe_filename(f"{dados['nome']}_{prontuario_id}_Prontuario")
            p_html = PASTA_SAIDA / f"{safe}.html"
            p_pdf = PASTA_SAIDA / f"{safe}.pdf"

            html = gerar_html_prontuario(prontuario_id, dados)
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
            )

            ok += 1
            print(f"✅ {chapa}: Prontuário gerado e Ordem criada (id={ordem_id}).")

        except Exception as e:
            erros += 1
            msg = str(e)
            erros_list.append({"motorista": chapa, "erro": msg[:500]})
            print(f"❌ {chapa}: erro: {msg}")

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
