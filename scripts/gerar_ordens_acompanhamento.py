# scripts/relatorio_gerencial.py
import os
import re
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

VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID") or os.getenv("PROJECT_ID")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
VERTEX_MODEL = os.getenv("VERTEX_MODEL", "gemini-2.5-pro")
VERTEX_SA_JSON = os.getenv("VERTEX_SA_JSON")

SUPABASE_A_URL = os.getenv("SUPABASE_A_URL")
SUPABASE_A_SERVICE_ROLE_KEY = os.getenv("SUPABASE_A_SERVICE_ROLE_KEY")
SUPABASE_B_URL = os.getenv("SUPABASE_B_URL")
SUPABASE_B_SERVICE_ROLE_KEY = os.getenv("SUPABASE_B_SERVICE_ROLE_KEY")

ORDEM_BATCH_ID = os.getenv("ORDEM_BATCH_ID")

TABELA_LOTE = "acompanhamento_lotes"
TABELA_ITENS = "acompanhamento_lote_itens"
TABELA_ORDEM = "diesel_acompanhamentos"
TABELA_EVENTOS = "diesel_acompanhamento_eventos"
TABELA_SUG = "diesel_sugestoes_acompanhamento"
TABELA_ORIGEM = "premiacao_diaria"

BUCKET = "relatorios"
REMOTE_PREFIX = "acompanhamento"
PASTA_SAIDA = Path("Ordens_Geradas")

DEFAULT_DIAS_MONITORAMENTO = int(os.getenv("DEFAULT_DIAS_MONITORAMENTO", "7"))

def _sb_a():
    return create_client(SUPABASE_A_URL, SUPABASE_A_SERVICE_ROLE_KEY)

def _sb_b():
    return create_client(SUPABASE_B_URL, SUPABASE_B_SERVICE_ROLE_KEY)

def _ensure_vertex_adc_if_possible():
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"): return
    if not VERTEX_SA_JSON: return
    try:
        tmp = Path("/tmp/vertex_sa.json")
        tmp.write_text(VERTEX_SA_JSON, encoding="utf-8")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(tmp)
    except Exception:
        pass

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

def _esc(s: str) -> str:
    s = "" if s is None else str(s)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;")

def atualizar_status_lote(status: str, msg: str = None, extra: dict = None):
    if not ORDEM_BATCH_ID: return
    sb = _sb_b()
    payload = {"status": status}
    if msg: payload["erro_msg"] = str(msg)[:1000]
    if extra is not None: payload["metadata"] = extra
    sb.table(TABELA_LOTE).update(payload).eq("id", ORDEM_BATCH_ID).execute()

def upload_storage(local_path: Path, remote_name: str, content_type: str):
    if not ORDEM_BATCH_ID or not local_path.exists(): return (None, None)
    sb = _sb_b()
    remote_path = f"{REMOTE_PREFIX}/{ORDEM_BATCH_ID}/{remote_name}"
    with open(local_path, "rb") as f:
        sb.storage.from_(BUCKET).upload(path=remote_path, file=f, file_options={"content-type": content_type, "upsert": "true"})
    public_url = f"{SUPABASE_B_URL}/storage/v1/object/public/{BUCKET}/{remote_path}"
    return (remote_path, public_url)

def _prioridade_por_desperdicio(litros: float):
    litros = float(litros or 0)
    if litros >= 150: return "PRIORIDADE ALTA"
    if litros >= 60: return "PRIORIDADE MÉDIA"
    return "PRIORIDADE BAIXA"

def obter_motoristas_do_lote():
    if not ORDEM_BATCH_ID: return []
    sb = _sb_b()
    res = sb.table(TABELA_ITENS).select("*").eq("lote_id", ORDEM_BATCH_ID).execute()
    return res.data or []

def buscar_sugestao_detalhada(sb, chapa: str, mes_ref: str = None):
    q = sb.table(TABELA_SUG).select("motorista_nome, detalhes_json, created_at, mes_ref").eq("chapa", chapa)
    if mes_ref:
        r = q.eq("mes_ref", mes_ref).maybe_single().execute()
        if r.data and r.data.get("detalhes_json"): return r.data
    r2 = sb.table(TABELA_SUG).select("motorista_nome, detalhes_json, created_at, mes_ref").eq("chapa", chapa).order("created_at", desc=True).limit(1).maybe_single().execute()
    return r2.data

def carregar_prompt_ia(prompt_id: str) -> str:
    try:
        sb = _sb_b()
        resp = sb.table("ia_prompts").select("prompt_text").eq("id", prompt_id).execute()
        if resp.data and len(resp.data) > 0:
            return resp.data[0]["prompt_text"]
    except Exception:
        pass
    return ""

def obter_tempo_de_casa(sb_a, chapa: str) -> str:
    try:
        # CORREÇÃO: Busca por nr_cracha
        res = sb_a.table("funcionarios").select("dt_inicio_atividade").eq("nr_cracha", chapa).maybe_single().execute()
        if res.data and res.data.get("dt_inicio_atividade"):
            dt_ini = datetime.strptime(res.data["dt_inicio_atividade"].split("T")[0], "%Y-%m-%d").date()
            dias = (datetime.utcnow().date() - dt_ini).days
            if dias < 30: return f"{dias} dias"
            meses = dias // 30
            anos = meses // 12
            meses_restantes = meses % 12
            if anos > 0:
                return f"{anos} anos e {meses_restantes} meses" if meses_restantes > 0 else f"{anos} anos"
            return f"{meses} meses"
    except Exception as e:
        print(f"⚠️ Erro ao buscar tempo de casa da chapa {chapa}: {e}")
    return "N/D"

def carregar_metas_consumo(sb):
    try:
        resp = sb.table("metas_consumo").select("*").execute()
        if resp.data: return pd.DataFrame(resp.data)
    except Exception:
        pass
    return pd.DataFrame()

def carregar_dados_diarios(sb_a, chapa: str, dt_ini: str, dt_fim: str):
    try:
        # CORREÇÃO: ilike com % nos dois lados para garantir encontrar a chapa
        res = sb_a.table(TABELA_ORIGEM).select("dia, motorista, veiculo, linha, km_rodado, combustivel_consumido, km/l").ilike("motorista", f"%{chapa}%").gte("dia", dt_ini).lte("dia", dt_fim).order("dia", desc=False).execute()
        if not res.data:
            return pd.DataFrame()
        
        df = pd.DataFrame(res.data)
        df["dia"] = pd.to_datetime(df["dia"]).dt.date
        df["km"] = pd.to_numeric(df["km_rodado"], errors="coerce").fillna(0)
        df["litros"] = pd.to_numeric(df["combustivel_consumido"], errors="coerce").fillna(0)
        df["kml_real"] = pd.to_numeric(df["km/l"], errors="coerce").fillna(0)

        df = df[(df["km"] > 0) & (df["litros"] > 0)].copy()

        def definir_cluster(v):
            v = str(v).strip().upper()
            if v.startswith("W"): return "C6"
            if v.startswith("2216"): return "C8"
            if v.startswith("2222"): return "C9"
            if v.startswith("2224"): return "C10"
            if v.startswith("2425"): return "C11"
            return None

        df["cluster"] = df["veiculo"].apply(definir_cluster)

        df_metas = carregar_metas_consumo(sb_a)
        if not df_metas.empty:
            df_metas["cluster"] = df_metas["cluster"].astype(str).str.upper()
            df = df.merge(df_metas[["linha", "cluster", "meta"]], on=["linha", "cluster"], how="left")
            df["kml_meta"] = pd.to_numeric(df["meta"], errors="coerce").fillna(0.0)
        else:
            df["kml_meta"] = 0.0

        def calc_desp(r):
            if r["kml_meta"] > 0 and r["kml_real"] < r["kml_meta"]:
                return r["litros"] - (r["km"] / r["kml_meta"])
            return 0.0

        df["desperdicio"] = df.apply(calc_desp, axis=1)
        return df
    except Exception as e:
        print(f"Erro no diário: {e}")
        return pd.DataFrame()

def carregar_mapa_nomes(caminho_csv="motoristas_rows.csv"):
    if not os.path.exists(caminho_csv):
        return {}
    try:
        df = pd.read_csv(caminho_csv, dtype=str)
        df["chapa"] = df["chapa"].str.strip()
        df["nome"] = df["nome"].str.strip().str.upper()
        return dict(zip(df["chapa"], df["nome"]))
    except Exception:
        return {}

def _periodo_from_detalhes(detalhes: dict, created_at_iso: str = None):
    # CORREÇÃO: Força sempre os últimos 30 dias contados a partir de hoje
    dt_fim = datetime.utcnow().date()
    dt_ini = dt_fim - timedelta(days=30)
    return {
        "periodo_inicio": dt_ini.isoformat(),
        "periodo_fim": dt_fim.isoformat(),
        "periodo_txt": f"{dt_ini.strftime('%d/%m/%Y')} a {dt_fim.strftime('%d/%m/%Y')}"
    }

def normalizar_prontuario(sb_a, chapa: str, nome: str, detalhes: dict, created_at_iso: str = None):
    if not detalhes: return None
    raio_x = detalhes.get("raio_x") or []
    if not isinstance(raio_x, list) or len(raio_x) == 0: return None

    periodo = _periodo_from_detalhes(detalhes, created_at_iso=created_at_iso)
    tempo_casa = obter_tempo_de_casa(sb_a, chapa)
    
    total_km = sum(n(r.get("km")) for r in raio_x)
    total_litros = sum(n(r.get("litros")) for r in raio_x)
    total_desp_ref = sum(n(r.get("desperdicio")) for r in raio_x)
    total_desp_meta = sum(n(r.get("desp_meta_oficial")) for r in raio_x)

    kml_real = (total_km / total_litros) if total_litros > 0 else 0.0
    litros_teo = sum((n(r.get("km")) / n(r.get("meta_linha_oficial"))) if n(r.get("meta_linha_oficial")) > 0 else 0.0 for r in raio_x)
    kml_meta = (total_km / litros_teo) if litros_teo > 0 else 0.0

    top = sorted(raio_x, key=lambda r: n(r.get("desp_meta_oficial")), reverse=True)[0]
    foco_cluster = (top.get("cluster") or "OUTROS")
    foco_linha = (top.get("linha") or "-")
    foco = f"{foco_cluster} - Linha {foco_linha}"

    prioridade = _prioridade_por_desperdicio(total_desp_meta)

    df_diario = carregar_dados_diarios(sb_a, chapa, periodo["periodo_inicio"], periodo["periodo_fim"])

    nome_final = nome
    if df_diario is not None and not df_diario.empty and "motorista" in df_diario.columns:
        nomes = df_diario["motorista"].dropna().unique()
        if len(nomes) > 0:
            n_raw = str(nomes[0])
            n_clean = re.sub(r'^\d+\s*[-]*\s*', '', n_raw).strip()
            if n_clean: nome_final = n_clean

    mapa_nomes = carregar_mapa_nomes()
    if chapa in mapa_nomes:
        nome_final = mapa_nomes[chapa]

    return {
        "chapa": chapa,
        "nome": (nome_final or chapa).strip().upper(),
        "cargo": "MOTORISTA",
        "tempo_casa": tempo_casa,
        "periodo_inicio": periodo["periodo_inicio"],
        "periodo_fim": periodo["periodo_fim"],
        "periodo_txt": periodo["periodo_txt"],
        "raio_x": raio_x,
        "diario": df_diario,
        "totais": {
            "km": float(total_km),
            "litros": float(total_litros),
            "desp_ref": float(total_desp_ref),
            "desp_meta": float(total_desp_meta),
            "kml_real": float(kml_real),
            "kml_meta": float(kml_meta),
        },
        "foco": foco,
        "foco_cluster": foco_cluster,
        "linha_foco": foco_linha,
        "prioridade": prioridade,
    }

def analisar_motorista_ia(dados: dict) -> str:
    if not VERTEX_PROJECT_ID: return "<p>IA desativada. Foco na tabela abaixo.</p>"
    _ensure_vertex_adc_if_possible()

    try:
        vertexai.init(project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION)
        model = GenerativeModel(VERTEX_MODEL)

        template_prompt = carregar_prompt_ia("prontuario_diesel")
        if not template_prompt:
            template_prompt = """Você é um Instrutor Master de condução econômica de ônibus urbano. Avalie o histórico recente do motorista abaixo para orientar o instrutor de campo que fará o acompanhamento.

DADOS DO MOTORISTA:
Motorista: {nome} (Chapa {chapa})
Experiência (Tempo de Casa): {tempo_casa}
Período Analisado: {periodo}

PERFORMANCE GERAL:
- KM/L Realizado: {kml_real}
- KM/L Meta Oficial: {kml_meta}
- Desperdício vs Meta: {desp_meta} Litros
- Desperdício vs Referência (Média dos colegas): {desp_ref} Litros

OFENSORES:
{raio_x}

Gere uma análise tática e direta em HTML usando apenas <p>, <b>, <ul>, <li>.
Estrutura:
1) <b>Diagnóstico Operacional</b>: Comente o tamanho do desvio. Leve em conta o tempo de casa.
2) <b>Direcionamento de Rota (Plano de Ação)</b>: 3 pontos táticos para o instrutor atuar no acompanhamento prático."""

        rx_txt = ""
        top_rx = sorted(dados["raio_x"], key=lambda r: n(r.get("desp_meta_oficial")), reverse=True)[:5]
        for r in top_rx:
            rx_txt += f"- Linha {r.get('linha')} ({r.get('cluster')}): {n(r.get('kml_real')):.2f} km/l | Perdeu {n(r.get('desp_meta_oficial')):.0f} L\n"

        prompt = template_prompt.replace("{nome}", dados["nome"]) \
                                .replace("{chapa}", dados["chapa"]) \
                                .replace("{tempo_casa}", dados["tempo_casa"]) \
                                .replace("{periodo}", dados["periodo_txt"]) \
                                .replace("{km}", f"{dados['totais']['km']:.0f}") \
                                .replace("{kml_real}", f"{dados['totais']['kml_real']:.2f}") \
                                .replace("{kml_meta}", f"{dados['totais']['kml_meta']:.2f}") \
                                .replace("{desp_meta}", f"{dados['totais']['desp_meta']:.0f}") \
                                .replace("{desp_ref}", f"{dados['totais']['desp_ref']:.0f}") \
                                .replace("{raio_x}", rx_txt)

        resp = model.generate_content(prompt)
        return getattr(resp, "text", "Análise não retornou dados.").replace("```html", "").replace("```", "")
    except Exception:
        return "<p>IA indisponível no momento.</p>"

def _build_svg_line_chart_diario(df: pd.DataFrame):
    if df is None or df.empty: return "<div class='chartEmpty'>Sem dados diários para gerar gráfico.</div>"
    
    df_grp = df.groupby("dia").agg({"km": "sum", "litros": "sum", "kml_meta": "mean"}).reset_index()
    df_grp["real"] = df_grp["km"] / df_grp["litros"]
    
    pts = []
    for _, r in df_grp.iterrows():
        pts.append({"label": r["dia"].strftime("%d/%m"), "real": r["real"], "meta": r["kml_meta"]})

    if len(pts) < 2: return "<div class='chartEmpty'>Poucos dados diários para gerar gráfico.</div>"

    W, H = 760, 260
    padL, padR, padT, padB = 56, 20, 26, 46
    innerW = W - padL - padR
    innerH = H - padT - padB

    ys = [float(p["real"]) for p in pts] + [float(p["meta"]) for p in pts]
    y_min, y_max = min(ys), max(ys)
    rng = (y_max - y_min) if (y_max > y_min) else 0.5
    y_min -= rng * 0.12
    y_max += rng * 0.12

    def x(i): return padL + (innerW * (i / (len(pts) - 1)))
    def y(v): return padT + (innerH * (1 - ((v - y_min) / (y_max - y_min if y_max != y_min else 1))))

    real_path = "M " + " L ".join([f"{x(i):.1f} {y(p['real']):.1f}" for i, p in enumerate(pts)])
    meta_path = "M " + " L ".join([f"{x(i):.1f} {y(p['meta']):.1f}" for i, p in enumerate(pts)])

    labels = ""
    for i, p in enumerate(pts):
        labels += f"<text x='{x(i):.1f}' y='{H-18}' text-anchor='middle' class='axisLabel'>{_esc(p['label'])}</text>"
        labels += f"<text x='{x(i):.1f}' y='{y(p['real'])-10:.1f}' text-anchor='middle' class='valReal'>{n(p['real']):.2f}</text>"
        labels += f"<text x='{x(i):.1f}' y='{y(p['meta'])+16:.1f}' text-anchor='middle' class='valMeta'>Ref: {n(p['meta']):.2f}</text>"

    ticks = ""
    for j in range(5):
        v = y_min + (j * (y_max - y_min) / 4)
        yy = y(v)
        ticks += f"<line x1='{padL}' y1='{yy:.1f}' x2='{W-padR}' y2='{yy:.1f}' class='grid'/>"
        ticks += f"<text x='{padL-10}' y='{yy+4:.1f}' text-anchor='end' class='axisLabel'>{v:.2f}</text>"

    return f"""
    <div class="chartWrap" style="page-break-inside: avoid;">
      <div class="chartTitle">EVOLUÇÃO DIÁRIA (KML)</div>
      <svg viewBox="0 0 {W} {H}" width="100%" height="{H}">
        {ticks}
        <path d="{meta_path}" class="lineMeta"/>
        <path d="{real_path}" class="lineReal"/>
        {labels}
        <g transform="translate({padL}, {padT-8})">
          <line x1="0" y1="0" x2="26" y2="0" class="legMeta"/>
          <text x="34" y="4" class="legend">Meta Oficial</text>
          <line x1="110" y1="0" x2="136" y2="0" class="legReal"/>
          <text x="144" y="4" class="legend">Realizado</text>
        </g>
      </svg>
    </div>
    """

def gerar_html_prontuario(prontuario_id: str, d: dict, texto_ia: str):
    cluster = d.get("foco_cluster") or "OUTROS"
    prioridade = d.get("prioridade") or "PRIORIDADE"

    litros_desv_meta = float(d["totais"]["desp_meta"])
    litros_desv_ref = float(d["totais"]["desp_ref"])
    kml_media = float(d["totais"]["kml_real"])
    kml_ref = float(d["totais"]["kml_meta"])

    piora_txt = f"{kml_ref:.2f} → {kml_media:.2f}"
    piora_style = "color:#dc2626;font-weight:900;" if kml_media < kml_ref else "color:#111827;font-weight:900;"

    rx = sorted(list(d.get("raio_x") or []), key=lambda r: n(r.get("desp_meta_oficial")), reverse=True)[:10]
    if not rx:
        rx_rows_html = "<tr><td colspan='8' class='muted'>Sem dados.</td></tr>"
    else:
        rows = []
        for r in rx:
            linha = _esc(r.get("linha") or "-")
            cl = _esc(r.get("cluster") or "-")
            km = _fmt_int(n(r.get("km")))
            real = f"{n(r.get('kml_real')):.2f}"
            
            meta_oficial = f"{n(r.get('meta_linha_oficial')):.2f}"
            desp_meta = f"{n(r.get('desp_meta_oficial')):.1f}"
            
            meta_ref = f"{n(r.get('kml_meta')):.2f}"
            desp_ref = f"{n(r.get('desperdicio')):.1f}"

            style_dm = "color:#b91c1c;font-weight:900;" if n(r.get("desp_meta_oficial")) > 0 else "color:#059669;font-weight:900;"
            style_dr = "color:#b91c1c;font-weight:900;" if n(r.get("desperdicio")) > 0 else "color:#059669;font-weight:900;"
            style_real = "color:#dc2626;font-weight:900;" if n(r.get('kml_real')) < n(r.get('meta_linha_oficial')) else "font-weight:800;"

            rows.append(f"""
                <tr>
                  <td class="td strong">{linha}</td>
                  <td class="td badge">{cl}</td>
                  <td class="td num strong">{km}</td>
                  <td class="td num" style="{style_real}">{real}</td>
                  <td class="td num muted">{meta_oficial}</td>
                  <td class="td num" style="{style_dm}">{desp_meta}</td>
                  <td class="td num muted" style="border-left: 2px solid #e5e7eb;">{meta_ref}</td>
                  <td class="td num" style="{style_dr}">{desp_ref}</td>
                </tr>
            """)
        rx_rows_html = "\n".join(rows)

    df_dia = d.get("diario")
    if df_dia is None or df_dia.empty:
        dia_rows_html = "<tr><td colspan='7' class='muted'>Sem dados diários no Supabase.</td></tr>"
    else:
        df_dia = df_dia.sort_values("dia", ascending=False).head(15)
        rows = []
        for _, r in df_dia.iterrows():
            dia_str = r['dia'].strftime("%d/%m/%Y")
            veic = _esc(r['veiculo'])
            lin = _esc(r['linha'])
            km = _fmt_int(r['km'])
            # CORREÇÃO: Coluna de Combustível em Litros
            litros = _fmt_int(r['litros'])
            mt = f"{r['kml_meta']:.2f}"
            dp = f"{r['desperdicio']:.1f}"
            style_d = "color:#b91c1c;font-weight:900;" if r['desperdicio'] > 0 else "color:#059669;font-weight:900;"
            
            rows.append(f"""
            <tr>
                <td>{dia_str}</td>
                <td>{veic}</td>
                <td>{lin}</td>
                <td class="num">{km}</td>
                <td class="num strong">{litros}</td>
                <td class="num muted">{mt}</td>
                <td class="num" style="{style_d}">{dp} L</td>
            </tr>
            """)
        dia_rows_html = "\n".join(rows)

    chart_html = _build_svg_line_chart_diario(d.get("diario"))

    return f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8" />
<title>Prontuário { _esc(prontuario_id) }</title>
<style>
  :root {{ --text:#111827; --muted:#6b7280; --line:#e5e7eb; --shadow: 0 2px 12px rgba(0,0,0,.06); --red:#dc2626; --slate:#94a3b8; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: Arial, sans-serif; background: #fff; color: var(--text); margin: 0; padding: 22px; }}
  .page {{ max-width: 920px; margin: 0 auto; }}
  .topbar {{ height: 10px; background: #0f172a; border-radius: 999px; margin-bottom: 16px; }}
  .header {{ display:flex; align-items:flex-start; justify-content:space-between; gap:16px; }}
  .hTitle {{ font-size: 26px; font-weight: 900; margin: 0; letter-spacing: .2px; }}
  .hSub {{ margin-top: 6px; color: var(--muted); font-size: 13px; }}
  .prio {{ font-weight: 900; font-size: 12px; color: var(--muted); text-align:right; }}
  
  .cards {{ display:grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 18px 0 16px 0; }}
  .card {{ background:#fff; border:1px solid var(--line); border-radius: 14px; padding: 16px 14px; box-shadow: var(--shadow); min-height: 84px; }}
  .cardBig {{ font-size: 20px; font-weight: 900; margin: 0; }}
  .cardLabel {{ margin-top: 8px; font-size: 10px; color: var(--muted); text-transform: uppercase; font-weight: 800; }}

  .ai-box {{ background-color: #fffde7; border: 1px solid #fbc02d; padding: 16px; border-radius: 8px; font-size: 13px; margin-bottom: 20px; page-break-inside: avoid; }}

  .secTitle {{ color:#0f172a; font-weight: 900; font-size: 15px; margin: 24px 0 10px 0; border-left: 4px solid #0f172a; padding-left: 8px; }}
  
  table {{ width:100%; border-collapse: collapse; border:1px solid var(--line); border-radius: 8px; overflow:hidden; box-shadow: var(--shadow); font-size: 12px; page-break-inside: auto; margin-bottom: 20px;}}
  thead th {{ background:#f8fafc; color:#475569; font-size: 10px; text-transform: uppercase; padding: 8px; border-bottom: 1px solid var(--line); text-align:left; }}
  tbody td {{ padding: 8px; border-bottom: 1px solid var(--line); vertical-align: top; }}
  tbody tr:nth-child(even) td {{ background:#fafafa; }}
  .num {{ text-align:right; font-variant-numeric: tabular-nums; }}
  .strong {{ font-weight: 900; }}
  .muted {{ color: var(--muted); }}
  .badge {{ font-weight: 900; color:#b91c1c; }}

  tfoot td {{ background: #0f172a; color:#fff; padding: 8px; font-size: 12px; font-weight: 900; }}
  
  .chartWrap {{ border:1px solid var(--line); border-radius: 14px; padding: 12px; margin: 10px 0; box-shadow: var(--shadow); }}
  .chartTitle {{ font-weight: 900; margin-bottom: 10px; font-size: 13px; color: var(--text); }}
  .grid {{ stroke: #f1f5f9; stroke-width: 1; }}
  .lineReal {{ fill:none; stroke: var(--red); stroke-width: 3; }}
  .lineMeta {{ fill:none; stroke: var(--slate); stroke-width: 2.5; stroke-dasharray: 6 6; }}
  .axisLabel {{ font-size: 11px; fill: var(--muted); }}
  .legend {{ font-size: 12px; fill: #374151; font-weight: 700; }}
  .legReal {{ stroke: var(--red); stroke-width: 3; }}
  .legMeta {{ stroke: var(--slate); stroke-width: 2.5; stroke-dasharray: 6 6; }}
  .valReal {{ font-size: 10px; fill: var(--red); font-weight: 900; }}
  .valMeta {{ font-size: 9px; fill: #64748b; font-weight: 700; }}

  @media print {{ body {{ padding: 0; }} .page {{ max-width: 100%; }} .card, .chartWrap, table {{ box-shadow: none; }} }}
</style>
</head>
<body>
  <div class="page">
    <div class="topbar"></div>

    <div class="header">
      <div>
        <div class="hTitle">{ _esc(d["nome"]) }</div>
        <div class="hSub">
          Chapa: <b>{ _esc(d["chapa"]) }</b> • Tempo de Casa: <b>{_esc(d['tempo_casa'])}</b><br>
          Período Analisado: <b>{ _esc(d["periodo_txt"]) }</b>
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
        <div class="cardBig" style="color:#b91c1c;">{ _fmt_int(litros_desv_meta) } L</div>
        <div class="cardLabel">DESPERDÍCIO OFICIAL (META)</div>
      </div>
      <div class="card">
        <div class="cardBig" style="color:#e67e22;">{ _fmt_int(litros_desv_ref) } L</div>
        <div class="cardLabel">DESP. REFERÊNCIA (COLEGAS)</div>
      </div>
      <div class="card">
        <div class="cardBig" style="{piora_style}">{ _esc(piora_txt) }</div>
        <div class="cardLabel">META → KM/L REALIZADO</div>
      </div>
    </div>

    <div class="ai-box">
      <h3 style="margin-top:0; color:#b7950b; font-size:14px; margin-bottom:8px;">💡 Orientações do Agente IA para o Instrutor</h3>
      {texto_ia}
    </div>

    <div class="secTitle">1. VISÃO GERAL (RAIO-X DAS LINHAS)</div>
    <table>
      <thead>
        <tr>
          <th>Linha</th>
          <th>Cluster</th>
          <th class="num">KM</th>
          <th class="num">Real</th>
          <th class="num" style="color:#1d4ed8;">Meta Linha</th>
          <th class="num" style="color:#1d4ed8;">Desp. Meta</th>
          <th class="num" style="border-left: 2px solid #e5e7eb; color:#6b7280;">Ref Colegas</th>
          <th class="num" style="color:#6b7280;">Desp. Ref</th>
        </tr>
      </thead>
      <tbody>{rx_rows_html}</tbody>
    </table>

    {chart_html}

    <div class="secTitle">2. DIÁRIO DE BORDO (Últimas Operações)</div>
    <table>
      <thead>
        <tr>
          <th>Data</th>
          <th>Veículo</th>
          <th>Linha</th>
          <th class="num">KM</th>
          <th class="num">Comb. (L)</th>
          <th class="num">Meta (KM/L)</th>
          <th class="num">Desperdício</th>
        </tr>
      </thead>
      <tbody>{dia_rows_html}</tbody>
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
        page.pdf(path=str(p_pdf), format="A4", print_background=True, margin={"top": "10mm", "bottom": "10mm", "left": "10mm", "right": "10mm"})
        browser.close()

def criar_ordem_e_evento(sb_b, dados, lote_id, pdf_path, pdf_url, html_path, html_url):
    dt_inicio = datetime.utcnow().date().isoformat()
    dias = DEFAULT_DIAS_MONITORAMENTO
    dt_fim_planejado = (datetime.utcnow().date() + timedelta(days=dias - 1)).isoformat()

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
        "observacao_inicial": f"Meta {dados['totais']['kml_meta']:.2f} → Real {dados['totais']['kml_real']:.2f} | Desp Meta {dados['totais']['desp_meta']:.1f} L",
        "arquivo_pdf_url": pdf_url,
        "arquivo_html_url": html_url,
        "metadata": {
            "versao": "V15_prontuario_com_diario_e_IA",
            "lote_id": lote_id,
            "periodo_inicio": dados["periodo_inicio"],
            "periodo_fim": dados["periodo_fim"],
            "foco": dados["foco"],
            "cluster_foco": dados["foco_cluster"],
            "kpis": dados["totais"],
            "raio_x_top10": raio_top,
            "prioridade": dados.get("prioridade", None),
            "pdf_path": pdf_path,
        },
        "evidencias_urls": [u for u in [pdf_url, html_url] if u],
    }

    ordem = sb_b.table(TABELA_ORDEM).insert(payload).execute().data
    ordem_id = ordem[0].get("id")

    evento = {
        "acompanhamento_id": ordem_id,
        "tipo": "LANCAMENTO",
        "observacoes": f"Ordem gerada automaticamente (lote {lote_id}).",
        "evidencias_urls": [u for u in [pdf_url, html_url] if u],
        "kml": dados["totais"]["kml_real"],
        "extra": {"prioridade": dados.get("prioridade", None)},
    }
    sb_b.table(TABELA_EVENTOS).insert(evento).execute()
    return ordem_id

def main():
    if not ORDEM_BATCH_ID: return
    ok, erros = 0, 0
    erros_list = []

    atualizar_status_lote("PROCESSANDO", extra={"started_at": datetime.utcnow().isoformat()})
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    itens = obter_motoristas_do_lote()
    if not itens:
        atualizar_status_lote("ERRO", "Lote sem itens")
        return

    sb_a, sb_b = _sb_a(), _sb_b()
    
    for item in itens:
        chapa = str(item.get("motorista_chapa") or "").strip()
        if not chapa: continue

        try:
            mes_ref = (item.get("mes_ref") or item.get("extra", {}) or {}).get("mes_ref") if isinstance(item.get("extra"), dict) else None
            nome_item = item.get("extra", {}).get("motorista_nome") if isinstance(item.get("extra"), dict) else None

            sug = buscar_sugestao_detalhada(sb_b, chapa, mes_ref=mes_ref)
            if not sug or not sug.get("detalhes_json"):
                raise RuntimeError("Sugestão não encontrada no Supabase B.")

            detalhes = sug.get("detalhes_json") or {}
            nome = (nome_item or sug.get("motorista_nome") or chapa)
            
            dados = normalizar_prontuario(sb_a, chapa, nome, detalhes, created_at_iso=sug.get("created_at"))
            if not dados: raise RuntimeError("Falha na normalização dos dados.")

            texto_ia = analisar_motorista_ia(dados)

            prontuario_id = chapa
            safe = _safe_filename(f"{dados['nome']}_{prontuario_id}_Prontuario")
            p_html, p_pdf = PASTA_SAIDA / f"{safe}.html", PASTA_SAIDA / f"{safe}.pdf"

            html = gerar_html_prontuario(prontuario_id, dados, texto_ia)
            p_html.write_text(html, encoding="utf-8")
            html_to_pdf(p_html, p_pdf)

            pdf_path, pdf_url = upload_storage(p_pdf, f"{safe}.pdf", "application/pdf")
            html_path, html_url = upload_storage(p_html, f"{safe}.html", "text/html")

            ordem_id = criar_ordem_e_evento(sb_b, dados, ORDEM_BATCH_ID, pdf_path, pdf_url, html_path, html_url)
            ok += 1

        except Exception as e:
            erros += 1
            erros_list.append({"motorista": chapa, "erro": str(e)[:500]})

    finished = {"ok": ok, "erros": erros, "erros_list": erros_list, "finished_at": datetime.utcnow().isoformat()}
    if erros == 0 and ok > 0: atualizar_status_lote("CONCLUIDO", extra=finished)
    elif ok == 0 and erros > 0: atualizar_status_lote("ERRO", msg=f"OK={ok} | ERROS={erros}", extra=finished)
    else: atualizar_status_lote("CONCLUIDO_COM_ERROS", msg=f"OK={ok} | ERROS={erros}", extra=finished)

if __name__ == "__main__":
    main()
