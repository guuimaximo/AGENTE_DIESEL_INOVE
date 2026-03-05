# -*- coding: utf-8 -*-
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

# =============================================================================
# CONFIGURAÇÕES E VARIÁVEIS DE AMBIENTE
# =============================================================================
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID") or os.getenv("PROJECT_ID")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
VERTEX_MODEL = os.getenv("VERTEX_MODEL", "gemini-2.5-pro")
VERTEX_SA_JSON = os.getenv("VERTEX_SA_JSON")

SUPABASE_A_URL = os.getenv("SUPABASE_A_URL")
SUPABASE_A_SERVICE_ROLE_KEY = os.getenv("SUPABASE_A_SERVICE_ROLE_KEY")
SUPABASE_B_URL = os.getenv("SUPABASE_B_URL")
SUPABASE_B_SERVICE_ROLE_KEY = os.getenv("SUPABASE_B_SERVICE_ROLE_KEY")

ORDEM_BATCH_ID = os.getenv("ORDEM_BATCH_ID")

# Tabelas base
TABELA_LOTE = "acompanhamento_lotes"
TABELA_ITENS = "acompanhamento_lote_itens"
TABELA_SUG = "diesel_sugestoes_acompanhamento"
TABELA_ORIGEM = "premiacao_diaria"

# NOVAS TABELAS DE DESTINO PARA TRATATIVA
TABELA_TRATATIVA = "diesel_tratativas"
TABELA_TRATATIVA_DETALHES = "diesel_tratativas_detalhes"

BUCKET = "relatorios"
REMOTE_PREFIX = "tratativas"
PASTA_SAIDA = Path("Ordens_Geradas_Tratativas")

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

def formatarDataBR(val_str):
    if not val_str: return "-"
    try:
        if "T" in str(val_str):
            dt = datetime.fromisoformat(str(val_str).replace("Z", "+00:00").split("+")[0])
            dt = dt - timedelta(hours=3)
            return dt.strftime("%d/%m/%Y")
        return str(val_str)
    except:
        return str(val_str)

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
    if litros >= 150: return "Gravíssima"
    if litros >= 60: return "Alta"
    return "Média"

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

def obter_tempo_de_casa(sb_a, chapa: str) -> str:
    print(f"      -> Consultando tempo de casa na tabela funcionarios (chapa: {chapa})...")
    try:
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
        print(f"      ⚠️ Erro ao buscar tempo de casa da chapa {chapa}: {e}")
    return "N/D"

def carregar_metas_consumo():
    try:
        sb_a = _sb_a()
        resp = sb_a.table("metas_consumo").select("*").execute()
        if resp.data: return pd.DataFrame(resp.data)
    except Exception: pass
    try:
        sb_b = _sb_b()
        resp_b = sb_b.table("metas_consumo").select("*").execute()
        if resp_b.data: return pd.DataFrame(resp_b.data)
    except Exception as e:
        print(f"      ⚠️ Erro ao carregar metas_consumo: {e}")
    return pd.DataFrame()

def carregar_dados_diarios(sb_a, chapa: str, dt_ini: str, dt_fim: str):
    try:
        res = sb_a.table(TABELA_ORIGEM).select('dia, motorista, veiculo, linha, km_rodado, combustivel_consumido, "km/l"').ilike("motorista", f"%{chapa}%").gte("dia", dt_ini).lte("dia", dt_fim).order("dia", desc=False).execute()
        if not res.data:
            return pd.DataFrame()
        
        df = pd.DataFrame(res.data)
        df["dia"] = pd.to_datetime(df["dia"]).dt.date
        df["km"] = pd.to_numeric(df["km_rodado"], errors="coerce").fillna(0)
        df["litros"] = pd.to_numeric(df["combustivel_consumido"], errors="coerce").fillna(0)
        df["kml_real"] = pd.to_numeric(df["km/l"], errors="coerce").fillna(0)
        
        df["linha"] = df["linha"].astype(str).str.strip().str.upper()
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

        df_metas = carregar_metas_consumo()
        if not df_metas.empty:
            df_metas["linha"] = df_metas["linha"].astype(str).str.strip().str.upper()
            df_metas["cluster"] = df_metas["cluster"].astype(str).str.strip().str.upper()
            
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
        print(f"      ⚠️ Erro no diário: {e}")
        return pd.DataFrame()

# NOVAS FUNÇÕES PARA TRAZER O HISTÓRICO DE ACOMPANHAMENTO E TRATATIVAS
def buscar_ultimo_acompanhamento(sb_b, chapa: str):
    try:
        # Busca o último que não seja "AGUARDANDO" (ou seja, o instrutor já atuou)
        res = sb_b.table("diesel_acompanhamentos").select("created_at, dt_inicio_monitoramento, intervencao_nota, intervencao_obs").eq("motorista_chapa", chapa).not_.is_("intervencao_nota", "null").order("created_at", desc=True).limit(1).execute()
        if res.data:
            return res.data[0]
    except Exception as e:
        print(f"Erro buscar_ultimo_acompanhamento: {e}")
    return None

def buscar_historico_tratativas(sb_b, chapa: str):
    try:
        res = sb_b.table("diesel_tratativas").select("created_at, prioridade, status, descricao").eq("motorista_chapa", chapa).eq("tipo_ocorrencia", "DIESEL_KML").order("created_at", desc=True).limit(3).execute()
        return res.data or []
    except Exception as e:
        print(f"Erro buscar_historico_tratativas: {e}")
        return []

def calc_desperdicio_periodo(sb_a, chapa: str, dt_ini: str, dt_fim: str):
    df = carregar_dados_diarios(sb_a, chapa, dt_ini, dt_fim)
    if df is not None and not df.empty and "desperdicio" in df.columns:
        return float(df["desperdicio"].sum())
    return 0.0

def carregar_mapa_nomes():
    mapa = {}
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
            ch = str(row.get("nr_cracha") or "").strip()
            nm = str(row.get("nm_funcionario") or "").strip().upper()
            if ch:
                mapa[ch] = nm
    except Exception as e:
        print(f"      ⚠️ Erro ao ler tabela funcionarios: {e}")
    return mapa

def _periodo_from_detalhes(detalhes: dict, created_at_iso: str = None):
    dt_fim = datetime.utcnow().date()
    dt_ini = dt_fim - timedelta(days=30)
    return {
        "periodo_inicio": dt_ini.isoformat(),
        "periodo_fim": dt_fim.isoformat(),
        "periodo_txt": f"{dt_ini.strftime('%d/%m/%Y')} a {dt_fim.strftime('%d/%m/%Y')}"
    }

def normalizar_prontuario(sb_a, sb_b, chapa: str, nome: str, detalhes: dict, created_at_iso: str = None, mapa_nomes: dict = None):
    print(f"    [Passo 3.2] Consolidando informações, buscando acompanhamentos e tratativas anteriores...")
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
    if mapa_nomes and chapa in mapa_nomes:
        nome_final = mapa_nomes[chapa]
    elif df_diario is not None and not df_diario.empty and "motorista" in df_diario.columns:
        nomes = df_diario["motorista"].dropna().unique()
        if len(nomes) > 0:
            n_raw = str(nomes[0])
            n_clean = re.sub(r'^\d+\s*[-]*\s*', '', n_raw).strip()
            if n_clean: nome_final = n_clean

    # ==============================================================
    # RECUPERAÇÃO DE HISTÓRICO (ACOMPANHAMENTOS E TRATATIVAS)
    # ==============================================================
    ultimo_acomp = buscar_ultimo_acompanhamento(sb_b, chapa)
    tratativas = buscar_historico_tratativas(sb_b, chapa)
    
    acomp_data = None
    if ultimo_acomp:
        dt_ref_str = str(ultimo_acomp.get("dt_inicio_monitoramento") or ultimo_acomp.get("created_at"))
        try:
            # Pega a data pura
            dt_ref = datetime.fromisoformat(dt_ref_str.split("T")[0][:10]).date()
            
            # 15 dias antes
            dt_antes_ini = (dt_ref - timedelta(days=15)).isoformat()
            dt_antes_fim = (dt_ref - timedelta(days=1)).isoformat()
            
            # 15 dias depois
            dt_depois_ini = dt_ref.isoformat()
            dt_depois_fim = (dt_ref + timedelta(days=15)).isoformat()
            
            # Calcula desperdício nesses períodos específicos
            desp_antes = calc_desperdicio_periodo(sb_a, chapa, dt_antes_ini, dt_antes_fim)
            desp_depois = calc_desperdicio_periodo(sb_a, chapa, dt_depois_ini, dt_depois_fim)
            
            acomp_data = {
                "data": dt_ref.strftime("%d/%m/%Y"),
                "nota": ultimo_acomp.get("intervencao_nota"),
                "obs": ultimo_acomp.get("intervencao_obs") or "Sem observações registradas.",
                "desp_antes": desp_antes,
                "desp_depois": desp_depois
            }
        except Exception as e:
            print(f"      ⚠️ Erro ao processar datas do acompanhamento: {e}")

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
        "acomp_data": acomp_data,
        "tratativas_anteriores": tratativas,
    }

def analisar_motorista_ia(dados: dict) -> str:
    print(f"    [Passo 3.3] Solicitando insights curtos e diretos da Vertex AI...")
    if not VERTEX_PROJECT_ID: return "<p>IA desativada. Foco na tabela abaixo.</p>"
    _ensure_vertex_adc_if_possible()

    try:
        vertexai.init(project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION)
        model = GenerativeModel(VERTEX_MODEL)

        nota_acomp = f"{dados.get('acomp_data', {}).get('nota', 'N/A')}" if dados.get("acomp_data") else "N/A"

        # NOVO PROMPT MAIS OBJETIVO (Curto e Grosso)
        template_prompt = """Você é um Gestor de Operações. Avalie o histórico recente deste motorista para embasar uma TRATATIVA DISCIPLINAR. Seja objetivo, curto e grosso.

DADOS DO INFRATOR:
Motorista: {nome} (Chapa {chapa})
Tempo de Casa: {tempo_casa}
Nota Último Acompanhamento (Checklist do Instrutor): {nota_acomp}/100

PERFORMANCE GERAL:
- KM/L Meta Oficial: {kml_meta} | Realizado: {kml_real}
- Desperdício vs Meta: {desp_meta} Litros (Custo jogado fora)

OFENSORES (Piores Linhas):
{raio_x}

Gere um resumo executivo em HTML usando apenas <p>, <b>, <ul>, <li>.
Estrutura exigida:
1) <b>Gravidade do Desvio</b>: No máximo 2 frases. Resuma o impacto em litros e cite a nota do último acompanhamento se houver.
2) <b>Pontos de Cobrança</b>: 3 bullet points extremamente curtos e diretos ordenando o que deve ser exigido do motorista na mesa de tratativa."""

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
                                .replace("{nota_acomp}", str(nota_acomp)) \
                                .replace("{raio_x}", rx_txt)

        resp = model.generate_content(prompt)
        return getattr(resp, "text", "Análise não retornou dados.").replace("```html", "").replace("```", "")
    except Exception as e:
        print(f"      ⚠️ Falha ao acionar Vertex AI: {e}")
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
    prioridade = d.get("prioridade") or "Média"

    litros_desv_meta = float(d["totais"]["desp_meta"])
    litros_desv_ref = float(d["totais"]["desp_ref"])
    kml_media = float(d["totais"]["kml_real"])
    kml_ref = float(d["totais"]["kml_meta"])

    piora_txt = f"{kml_ref:.2f} → {kml_media:.2f}"
    piora_style = "color:#dc2626;font-weight:900;" if kml_media < kml_ref else "color:#111827;font-weight:900;"

    # =========================================================
    # RENDERIZAÇÃO DO HISTÓRICO DE ACOMPANHAMENTO E TRATATIVAS
    # =========================================================
    acomp = d.get("acomp_data")
    trats = d.get("tratativas_anteriores") or []

    if acomp:
        nota_str = f"{n(acomp['nota']):.0f}" if acomp['nota'] is not None else "N/A"
        acomp_html = f"""
        <div style="font-size:12px; line-height:1.6; color:#374151;">
            <b>Data do Acompanhamento:</b> {acomp['data']} <br>
            <b>Nota do Checklist:</b> <span style="color:#059669; font-weight:900; background:#d1fae5; padding:2px 6px; border-radius:4px;">{nota_str}/100</span> <br>
            <div style="margin-top:8px; margin-bottom:8px; display:flex; gap:8px;">
                <div style="background:#fee2e2; border:1px solid #fca5a5; padding:6px 8px; border-radius:4px; flex:1;">
                    <div style="font-size:9px; color:#991b1b; font-weight:bold; text-transform:uppercase;">15 Dias Antes (Desp.)</div>
                    <div style="color:#7f1d1d; font-weight:900; font-size:13px;">{_fmt_int(acomp['desp_antes'])} L</div>
                </div>
                <div style="background:#ffedd5; border:1px solid #fdba74; padding:6px 8px; border-radius:4px; flex:1;">
                    <div style="font-size:9px; color:#9a3412; font-weight:bold; text-transform:uppercase;">15 Dias Depois (Desp.)</div>
                    <div style="color:#7c2d12; font-weight:900; font-size:13px;">{_fmt_int(acomp['desp_depois'])} L</div>
                </div>
            </div>
            <b>Observação do Instrutor:</b> <i style="color:#64748b;">"{_esc(acomp['obs'])}"</i>
        </div>
        """
    else:
        acomp_html = "<div class='muted' style='font-size:12px; padding-top:4px;'>Nenhum acompanhamento prático recente registrado.</div>"

    if trats:
        lis = []
        for t in trats:
            dt_t = formatarDataBR(t.get("created_at"))
            st = _esc(t.get("status") or "-")
            pr = _esc(t.get("prioridade") or "-")
            ds = _esc(t.get("descricao") or "")[:120] + "..."
            lis.append(f"<li style='margin-bottom:8px; border-bottom:1px dashed #cbd5e1; padding-bottom:6px;'><b>{dt_t}</b> - Status: <b>{st}</b> (Prio: {pr})<br><i style='color:#64748b;'>{ds}</i></li>")
        trats_html = f"<ul style='margin:0; padding-left:16px; font-size:11px; color:#374151; list-style-type:square;'>{''.join(lis)}</ul>"
    else:
        trats_html = "<div class='muted' style='font-size:12px; padding-top:4px;'>Nenhuma tratativa anterior de KM/L.</div>"

    hist_secao_html = f"""
    <div class="secTitle" style="margin-top: 10px;">1. HISTÓRICO DISCIPLINAR E ACOMPANHAMENTOS PREGRESSOS</div>
    <div style="display:flex; gap:16px; margin-bottom:20px; page-break-inside: avoid;">
      <div style="flex:1; border:1px solid var(--line); border-radius:8px; padding:14px; background:#f8fafc; box-shadow: var(--shadow);">
        <h4 style="margin:0 0 10px 0; color:#0f172a; font-size:12px; border-bottom:2px solid #e2e8f0; padding-bottom:6px;">ÚLTIMO ACOMPANHAMENTO PRÁTICO</h4>
        {acomp_html}
      </div>
      <div style="flex:1; border:1px solid var(--line); border-radius:8px; padding:14px; background:#fef2f2; box-shadow: var(--shadow);">
        <h4 style="margin:0 0 10px 0; color:#991b1b; font-size:12px; border-bottom:2px solid #fca5a5; padding-bottom:6px;">TRATATIVAS DE KM/L ANTERIORES</h4>
        {trats_html}
      </div>
    </div>
    """

    # TABELAS DE DADOS
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

    # RE-ADICIONADO PARA CORRIGIR O ERRO "chart_html is not defined"
    chart_html = _build_svg_line_chart_diario(d.get("diario"))

    return f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8" />
<title>Prontuário de Tratativa { _esc(prontuario_id) }</title>
<style>
  :root {{ --text:#111827; --muted:#6b7280; --line:#e5e7eb; --shadow: 0 2px 12px rgba(0,0,0,.06); --red:#b91c1c; --slate:#94a3b8; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: Arial, sans-serif; background: #fff; color: var(--text); margin: 0; padding: 22px; border-top: 12px solid var(--red);}}
  .page {{ max-width: 920px; margin: 0 auto; }}
  .header {{ display:flex; align-items:flex-start; justify-content:space-between; gap:16px; margin-top: 10px;}}
  .hTitle {{ font-size: 26px; font-weight: 900; margin: 0; letter-spacing: .2px; color: var(--red); text-transform: uppercase; }}
  .hSub {{ margin-top: 6px; color: var(--text); font-size: 14px; font-weight: bold; }}
  .prio {{ font-weight: 900; font-size: 14px; color: #fff; background: var(--red); padding: 4px 12px; border-radius: 4px; text-align:right; }}
  
  .cards {{ display:grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 18px 0 16px 0; }}
  .card {{ background:#fff; border:1px solid var(--line); border-radius: 10px; padding: 16px 14px; box-shadow: var(--shadow); min-height: 84px; border-left: 4px solid var(--red); }}
  .cardBig {{ font-size: 20px; font-weight: 900; margin: 0; }}
  .cardLabel {{ margin-top: 8px; font-size: 10px; color: var(--muted); text-transform: uppercase; font-weight: 800; }}

  .ai-box {{ background-color: #fef2f2; border: 1px solid #f87171; padding: 16px; border-radius: 8px; font-size: 13px; margin-bottom: 20px; page-break-inside: avoid; }}

  .secTitle {{ color:var(--red); font-weight: 900; font-size: 15px; margin: 24px 0 10px 0; border-bottom: 2px solid var(--line); padding-bottom: 6px; text-transform: uppercase; }}
  
  table {{ width:100%; border-collapse: collapse; border:1px solid var(--line); border-radius: 8px; overflow:hidden; box-shadow: var(--shadow); font-size: 12px; page-break-inside: auto; margin-bottom: 20px;}}
  thead th {{ background:#fef2f2; color:#7f1d1d; font-size: 10px; text-transform: uppercase; padding: 8px; border-bottom: 2px solid #fca5a5; text-align:left; font-weight: bold;}}
  tbody td {{ padding: 8px; border-bottom: 1px solid var(--line); vertical-align: top; }}
  tbody tr:nth-child(even) td {{ background:#fafafa; }}
  .num {{ text-align:right; font-variant-numeric: tabular-nums; }}
  .strong {{ font-weight: 900; }}
  .muted {{ color: var(--muted); }}
  .badge {{ font-weight: 900; color:#b91c1c; }}

  tfoot td {{ background: var(--red); color:#fff; padding: 8px; font-size: 12px; font-weight: 900; }}
  
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

  @media print {{ body {{ padding: 0; border-top: none;}} .page {{ max-width: 100%; }} .card, .chartWrap, table {{ box-shadow: none; }} }}
</style>
</head>
<body>
  <div class="page">
    <div class="header">
      <div>
        <div class="hTitle">PRONTUÁRIO DE TRATATIVA</div>
        <div class="hSub">
          Infrator: <b>{ _esc(d["nome"]) } (Chapa: { _esc(d["chapa"]) })</b><br>
          Tempo de Empresa: <b>{_esc(d['tempo_casa'])}</b><br>
          Período Analisado: <b>{ _esc(d["periodo_txt"]) }</b>
        </div>
      </div>
      <div class="prio">PRIORIDADE: { _esc(prioridade).upper() }</div>
    </div>

    <div class="cards">
      <div class="card">
        <div class="cardBig">{ _esc(cluster) }</div>
        <div class="cardLabel">CLUSTER DE ATUAÇÃO</div>
      </div>
      <div class="card">
        <div class="cardBig" style="color:#b91c1c;">{ _fmt_int(litros_desv_meta) } L</div>
        <div class="cardLabel">DESPERDÍCIO FINANCEIRO (META)</div>
      </div>
      <div class="card">
        <div class="cardBig" style="color:#e67e22;">{ _fmt_int(litros_desv_ref) } L</div>
        <div class="cardLabel">DESP. REFERÊNCIA (COLEGAS)</div>
      </div>
      <div class="card">
        <div class="cardBig" style="{piora_style}">{ _esc(piora_txt) }</div>
        <div class="cardLabel">META OFICIAL → KM/L REALIZADO</div>
      </div>
    </div>

    <div class="ai-box">
      <h3 style="margin-top:0; color:#991b1b; font-size:14px; margin-bottom:8px;">⚠️ Resumo Executivo da Ocorrência</h3>
      {texto_ia}
    </div>

    {hist_secao_html}

    <div class="secTitle">2. GRAVIDADE POR LINHA (TOP OFENSORES)</div>
    <table>
      <thead>
        <tr>
          <th>Linha</th>
          <th>Cluster</th>
          <th class="num">KM Rodado</th>
          <th class="num">KM/L Real</th>
          <th class="num" style="color:#1d4ed8;">Meta Exigida</th>
          <th class="num" style="color:#1d4ed8;">Desperdício (Meta)</th>
          <th class="num" style="border-left: 2px solid #fca5a5; color:#6b7280;">Média Colegas</th>
          <th class="num" style="color:#6b7280;">Desp. (Colegas)</th>
        </tr>
      </thead>
      <tbody>{rx_rows_html}</tbody>
    </table>

    {chart_html}

    <div class="secTitle">3. HISTÓRICO DE OCORRÊNCIAS DIÁRIAS</div>
    <table>
      <thead>
        <tr>
          <th>Data</th>
          <th>Prefixo</th>
          <th>Linha</th>
          <th class="num">KM</th>
          <th class="num">Combustível (L)</th>
          <th class="num">Meta (KM/L)</th>
          <th class="num">Desperdício do Dia</th>
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

def criar_tratativa_e_evento(sb_b, dados, lote_id, pdf_path, pdf_url, html_path, html_url):
    print(f"    [Passo 3.6] Atualizando registro na Central de Tratativas com o PDF do Robô...")
    
    # 1. Tentar encontrar se a tratativa já foi criada pelo frontend (pelo lote_id no histórico ou data recente)
    # Como o front acabou de criar, nós precisamos fazer um UPDATE na coluna evidencias_urls da tratativa correta
    
    # Busca a tratativa mais recente deste motorista que está pendente
    res_trat = sb_b.table(TABELA_TRATATIVA).select("id, evidencias_urls").eq("motorista_chapa", dados["chapa"]).eq("status", "Pendente").order("created_at", desc=True).limit(1).execute()
    
    if res_trat.data:
        trat_existente = res_trat.data[0]
        tratativa_id = trat_existente["id"]
        
        # Junta os PDFs antigos (se houver) com os novos do robô
        urls_antigas = trat_existente.get("evidencias_urls") or []
        if isinstance(urls_antigas, str):
            urls_antigas = [urls_antigas]
            
        novas_urls = urls_antigas + [u for u in [pdf_url, html_url] if u]
        
        # Atualiza a tratativa existente
        sb_b.table(TABELA_TRATATIVA).update({
            "evidencias_urls": novas_urls
        }).eq("id", tratativa_id).execute()
        
        # Insere o evento do robô na linha do tempo
        payload_evento = {
            "tratativa_id": tratativa_id,
            "acao_aplicada": "ABERTURA_AUTOMATICA", # Robô inseriu dados
            "observacoes": f"🤖 Prontuário Inteligente gerado e anexado aos autos da tratativa. Foco da análise: {dados['foco']}.",
            "extra": {
                "evidencias_urls": [u for u in [pdf_url, html_url] if u],
                "lote_id": lote_id
            }
        }
        sb_b.table(TABELA_TRATATIVA_DETALHES).insert(payload_evento).execute()
        return tratativa_id

    else:
        print("    ⚠️ Tratativa não encontrada para atualizar. Criando uma nova...")
        # Caso o frontend não tenha criado por algum motivo, o robô cria uma do zero
        payload_tratativa = {
            "motorista_chapa": dados["chapa"],
            "motorista_nome": dados["nome"],
            "origem": "BOT_DIESEL",
            "tipo_ocorrencia": "DIESEL_KML",
            "prioridade": dados.get("prioridade", "Média"),
            "status": "Pendente",
            "descricao": f"Tratativa gerada exclusivamente via BOT. Meta: {dados['totais']['kml_meta']:.2f} | Real: {dados['totais']['kml_real']:.2f} | Desperdício Total: {dados['totais']['desp_meta']:.1f} Litros.",
            "linha": dados.get("linha_foco", None),
            "cluster": dados.get("foco_cluster", None),
            "periodo_inicio": dados.get("periodo_inicio"),
            "periodo_fim": dados.get("periodo_fim"),
            "evidencias_urls": [u for u in [pdf_url, html_url] if u]
        }

        trat = sb_b.table(TABELA_TRATATIVA).insert(payload_tratativa).execute().data
        tratativa_id = trat[0].get("id")

        payload_evento = {
            "tratativa_id": tratativa_id,
            "acao_aplicada": "ABERTURA_AUTOMATICA",
            "observacoes": f"Prontuário de Tratativa anexado automaticamente. Foco: {dados['foco']}.",
            "extra": {
                "evidencias_urls": [u for u in [pdf_url, html_url] if u],
                "lote_id": lote_id
            }
        }
        
        sb_b.table(TABELA_TRATATIVA_DETALHES).insert(payload_evento).execute()
        return tratativa_id

def main():
    print("🚀 [Passo 1] Iniciando script de geração de TRATATIVAS (Medidas Disciplinares)...")
    if not ORDEM_BATCH_ID: 
        print("❌ ORDEM_BATCH_ID não definido no ambiente.")
        return

    ok, erros = 0, 0
    erros_list = []

    atualizar_status_lote("PROCESSANDO", extra={"started_at": datetime.utcnow().isoformat()})
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    print("📦 [Passo 2] Buscando lote de motoristas infratores no Supabase B...")
    itens = obter_motoristas_do_lote()
    if not itens:
        atualizar_status_lote("ERRO", "Lote sem itens")
        return

    sb_a, sb_b = _sb_a(), _sb_b()
    
    print("👨‍💻 [Passo 2.1] Carregando mapa de nomes de funcionários...")
    mapa_nomes = carregar_mapa_nomes()
    
    for item in itens:
        chapa = str(item.get("motorista_chapa") or "").strip()
        if not chapa: continue

        print(f"\n👤 [Passo 3] Processando Tratativa para motorista chapa: {chapa}...")
        try:
            mes_ref = (item.get("mes_ref") or item.get("extra", {}) or {}).get("mes_ref") if isinstance(item.get("extra"), dict) else None
            nome_item = item.get("extra", {}).get("motorista_nome") if isinstance(item.get("extra"), dict) else None

            print(f"    [Passo 3.1] Buscando detalhes (raio-x) da telemetria...")
            sug = buscar_sugestao_detalhada(sb_b, chapa, mes_ref=mes_ref)
            if not sug or not sug.get("detalhes_json"):
                raise RuntimeError("Dados de telemetria não encontrados.")

            detalhes = sug.get("detalhes_json") or {}
            nome = (nome_item or sug.get("motorista_nome") or chapa)
            
            dados = normalizar_prontuario(sb_a, sb_b, chapa, nome, detalhes, created_at_iso=sug.get("created_at"), mapa_nomes=mapa_nomes)
            if not dados: raise RuntimeError("Falha na normalização dos dados do motorista.")

            texto_ia = analisar_motorista_ia(dados)

            print(f"    [Passo 3.4] Gerando HTML e convertendo para PDF de Tratativa...")
            prontuario_id = chapa
            safe = _safe_filename(f"{dados['nome']}_{prontuario_id}_Prontuario_Tratativa")
            p_html, p_pdf = PASTA_SAIDA / f"{safe}.html", PASTA_SAIDA / f"{safe}.pdf"

            html = gerar_html_prontuario(prontuario_id, dados, texto_ia)
            p_html.write_text(html, encoding="utf-8")
            html_to_pdf(p_html, p_pdf)

            print(f"    [Passo 3.5] Fazendo upload do Prontuário para o Storage...")
            pdf_path, pdf_url = upload_storage(p_pdf, f"{safe}.pdf", "application/pdf")
            html_path, html_url = upload_storage(p_html, f"{safe}.html", "text/html")

            # CRIA A TRATATIVA E NÃO O ACOMPANHAMENTO
            tratativa_id = criar_tratativa_e_evento(sb_b, dados, ORDEM_BATCH_ID, pdf_path, pdf_url, html_path, html_url)
            
            ok += 1
            print(f"✅ [Passo 3.7] Tratativa do motorista {chapa} atualizada com sucesso! (ID: {tratativa_id})")

        except Exception as e:
            erros += 1
            erros_list.append({"motorista": chapa, "erro": str(e)[:500]})
            print(f"❌ Erro ao processar motorista {chapa}: {e}")

    print("\n🏁 [Passo 4] Processamento do Lote de Tratativas Finalizado.")
    finished = {"ok": ok, "erros": erros, "erros_list": erros_list, "finished_at": datetime.utcnow().isoformat()}
    if erros == 0 and ok > 0: atualizar_status_lote("CONCLUIDO", extra=finished)
    elif ok == 0 and erros > 0: atualizar_status_lote("ERRO", msg=f"OK={ok} | ERROS={erros}", extra=finished)
    else: atualizar_status_lote("CONCLUIDO_COM_ERROS", msg=f"OK={ok} | ERROS={erros}", extra=finished)

if __name__ == "__main__":
    main()
