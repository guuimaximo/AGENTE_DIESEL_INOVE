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

SUPABASE_A_URL = os.getenv("SUPABASE_A_URL")
SUPABASE_A_SERVICE_ROLE_KEY = os.getenv("SUPABASE_A_SERVICE_ROLE_KEY")

SUPABASE_B_URL = os.getenv("SUPABASE_B_URL")
SUPABASE_B_SERVICE_ROLE_KEY = os.getenv("SUPABASE_B_SERVICE_ROLE_KEY")

ORDEM_BATCH_ID = os.getenv("ORDEM_BATCH_ID")  # ID do lote no Supabase B
TABELA_ORIGEM = os.getenv("DIESEL_SOURCE_TABLE", "premiacao_diaria")

# Tabelas (Supabase B)
TABELA_LOTE = "acompanhamento_lotes"
TABELA_ITENS = "acompanhamento_lote_itens"
TABELA_ORDEM = "diesel_acompanhamentos"
TABELA_EVENTOS = "diesel_acompanhamento_eventos"

# Storage (Supabase B)
BUCKET = "relatorios"
REMOTE_PREFIX = "acompanhamento"
PASTA_SAIDA = Path("Ordens_Geradas")

# Regras
KML_MIN = 1.5
KML_MAX = 5.0
FETCH_DAYS = 90
JANELA_DIAS = 7
PAGE_SIZE = 2000

# padrão: ordem nasce aguardando instrutor, monitoramento padrão 7d (o instrutor redefine no LANÇAR)
DEFAULT_DIAS_MONITORAMENTO = int(os.getenv("DEFAULT_DIAS_MONITORAMENTO", "7"))

# Metas por cluster (igual seu script)
METAS_CLUSTER = {"C6": 2.51, "C8": 2.60, "C9": 2.73, "C10": 2.80, "C11": 2.90, "OUTROS": 2.50}


# ==============================================================================
# HELPERS
# ==============================================================================
def _sb_a():
    if not SUPABASE_A_URL or not SUPABASE_A_SERVICE_ROLE_KEY:
        raise RuntimeError("ENV Supabase A ausente (SUPABASE_A_URL / SUPABASE_A_SERVICE_ROLE_KEY)")
    return create_client(SUPABASE_A_URL, SUPABASE_A_SERVICE_ROLE_KEY)


def _sb_b():
    if not SUPABASE_B_URL or not SUPABASE_B_SERVICE_ROLE_KEY:
        raise RuntimeError("ENV Supabase B ausente (SUPABASE_B_URL / SUPABASE_B_SERVICE_ROLE_KEY)")
    return create_client(SUPABASE_B_URL, SUPABASE_B_SERVICE_ROLE_KEY)


def _safe_filename(name: str) -> str:
    name = str(name or "").strip()
    name = re.sub(r"[^\w\-.() ]+", "_", name, flags=re.UNICODE)
    return name[:100] or "sem_nome"


def to_num(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    s = series.astype(str).str.strip()
    s = s.str.replace(",", ".", regex=False)
    s = s.str.replace(r"[^0-9\.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def n(v):
    try:
        x = float(v)
        return x if pd.notna(x) else 0.0
    except Exception:
        return 0.0


def get_cluster(v):
    v = str(v).strip()
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
    """
    Retorna dict com chaves: diagnostico, foco, feedback.
    Se não achar, usa fallback com texto inteiro em feedback.
    """
    base = {"diagnostico": "-", "foco": "-", "feedback": "-"}
    if not txt:
        return base

    # normaliza
    t = txt.replace("\r\n", "\n").strip()

    # tenta tags exatas
    def pick(tag):
        m = re.search(rf"{re.escape(tag)}\s*:\s*(.*?)(?=\n[A-ZÁÂÃÉÊÍÓÔÕÚÇ ]{{5,}}:|\Z)", t, flags=re.S)
        return m.group(1).strip() if m else None

    d = pick("DIAGNÓSTICO COMPORTAMENTAL")
    f = pick("FOCO DA MONITORIA")
    fb = pick("FEEDBACK EDUCATIVO")

    if d: base["diagnostico"] = d
    if f: base["foco"] = f
    if fb: base["feedback"] = fb

    # fallback: se veio tudo em texto corrido
    if base["diagnostico"] == "-" and base["foco"] == "-" and base["feedback"] == "-":
        base["feedback"] = t[:4000]

    return base


# ==============================================================================
# NOMES (CSV)
# ==============================================================================
def carregar_mapa_nomes(caminho_csv="motoristas_rows.csv"):
    if not os.path.exists(caminho_csv):
        print("⚠️ CSV de nomes não encontrado. Usando chapa como nome.")
        return {}
    try:
        df = pd.read_csv(caminho_csv, dtype=str)
        df["chapa"] = df["chapa"].astype(str).str.strip()

        mapa = {}
        for _, row in df.iterrows():
            mapa[row["chapa"]] = {
                "nome": str(row.get("nome", "")).strip().upper() or row["chapa"],
                "cargo": str(row.get("cargo", "MOTORISTA")).strip().upper() or "MOTORISTA",
            }
        print(f"📋 Mapa de nomes carregado: {len(mapa)} registros.")
        return mapa
    except Exception as e:
        print(f"❌ Erro ao ler CSV de nomes: {e}")
        return {}


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
# DADOS BRUTOS (Supabase A)
# ==============================================================================
def carregar_historico_motorista(chapa):
    print(f"   ↳ Buscando dados brutos: {chapa}...")
    sb = _sb_a()

    agora = datetime.utcnow()
    limite = agora - timedelta(days=FETCH_DAYS)
    cursor = agora
    all_rows = []

    while cursor > limite:
        inicio_janela = cursor - timedelta(days=JANELA_DIAS)
        s_fim = cursor.strftime("%Y-%m-%d")
        s_ini = inicio_janela.strftime("%Y-%m-%d")

        start = 0
        while True:
            resp = (
                sb.table(TABELA_ORIGEM)
                .select('dia, motorista, veiculo, linha, "km/l", km_rodado, combustivel_consumido')
                .eq("motorista", chapa)
                .gte("dia", s_ini)
                .lte("dia", s_fim)
                .order("dia", desc=True)
                .range(start, start + PAGE_SIZE - 1)
                .execute()
            )

            rows = resp.data or []
            all_rows.extend(rows)
            if len(rows) < PAGE_SIZE:
                break
            start += PAGE_SIZE

        cursor = inicio_janela - timedelta(days=1)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df.rename(
        columns={
            "dia": "Date",
            "motorista": "Motorista",
            "km_rodado": "Km",
            "combustivel_consumido": "Comb.",
            "km/l": "kml_db",
        },
        inplace=True,
    )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Km"] = to_num(df["Km"])
    df["Comb."] = to_num(df["Comb."])

    df = df.dropna(subset=["Date", "Km", "Comb."])
    df = df[(df["Comb."] > 0) & (df["Km"] > 0)].copy()

    df["Cluster"] = df["veiculo"].apply(get_cluster)

    # outliers pelo kml calculado
    kml_calc = df["Km"] / df["Comb."]
    df = df[(kml_calc >= KML_MIN) & (kml_calc <= KML_MAX)].copy()

    return df


# ==============================================================================
# PROCESSAMENTO (gera raio_x + periodo + weekly chart)
# ==============================================================================
def _semana_ref(dt: pd.Timestamp):
    # Week label dd/mm do inicio da semana (segunda)
    if pd.isna(dt):
        return None
    d = pd.Timestamp(dt).normalize()
    monday = d - pd.Timedelta(days=(d.weekday()))  # monday=0
    return monday


def _weekly_series(df_30d: pd.DataFrame):
    """
    Retorna lista de pontos semanais (ordenado):
    [{label:'27/10', real:2.32, meta:2.73}, ...]
    meta aqui é "meta ref" da semana com base em litros_meta (meta ponderada)
    """
    if df_30d.empty:
        return []

    tmp = df_30d.copy()
    tmp["week"] = tmp["Date"].apply(_semana_ref)

    agg = (
        tmp.groupby("week")
        .agg(Km=("Km", "sum"), Comb=("Comb.", "sum"))
        .reset_index()
        .sort_values("week")
    )
    if agg.empty:
        return []

    # km/l real por semana
    agg["real"] = agg["Km"] / agg["Comb"]

    # meta ref semanal: ponderada por cluster (usando metas por cluster e litros_meta)
    # litros_meta semana = soma(Km_cluster / meta_cluster)
    def week_meta_ref(w):
        dfw = tmp[tmp["week"] == w]
        if dfw.empty:
            return None
        # soma litros teóricos com meta por cluster
        litros_teor = 0.0
        km_tot = float(dfw["Km"].sum())
        for c, dfc in dfw.groupby("Cluster"):
            meta = float(METAS_CLUSTER.get(str(c), METAS_CLUSTER["OUTROS"]))
            kmc = float(dfc["Km"].sum())
            if meta > 0:
                litros_teor += (kmc / meta)
        if litros_teor <= 0:
            return None
        return km_tot / litros_teor  # kml meta ponderada

    metas = []
    for w in agg["week"].tolist():
        metas.append(week_meta_ref(w))
    agg["meta"] = metas

    points = []
    for _, r in agg.iterrows():
        w = r["week"]
        label = pd.Timestamp(w).strftime("%d/%m")
        points.append(
            {
                "label": label,
                "real": float(r["real"]) if pd.notna(r["real"]) else None,
                "meta": float(r["meta"]) if pd.notna(r["meta"]) else None,
            }
        )
    # mantém no máximo 6 pontos (visual)
    return points[-6:]


def _calc_piora_percent(df_30d: pd.DataFrame, kml_geral_real: float):
    """
    Piora %: compara últimos 15 dias vs 15 dias anteriores dentro dos 30d.
    Se faltar dados, retorna 0.
    """
    if df_30d.empty:
        return 0.0

    maxd = df_30d["Date"].max()
    if pd.isna(maxd):
        return 0.0

    cut = maxd - timedelta(days=15)
    a = df_30d[df_30d["Date"] < cut]
    b = df_30d[df_30d["Date"] >= cut]

    def kml(df):
        km = float(df["Km"].sum())
        li = float(df["Comb."].sum())
        return (km / li) if li > 0 else None

    kml_a = kml(a)
    kml_b = kml(b)

    if kml_a is None or kml_b is None or kml_a <= 0:
        return 0.0

    # piora = queda relativa (se melhorou, 0)
    delta = (kml_b - kml_a) / kml_a
    piora = abs(delta) * 100.0 if delta < 0 else 0.0
    return float(piora)


def _prioridade_por_desperdicio(litros: float):
    # regra simples: ajuste depois se quiser
    if litros >= 150:
        return "PRIORIDADE ALTA"
    if litros >= 60:
        return "PRIORIDADE MÉDIA"
    return "PRIORIDADE BAIXA"


def processar_dados_prontuario(df, chapa, info_nome):
    if df.empty:
        return None

    max_date = df["Date"].max()
    if pd.isna(max_date):
        return None

    min_date_30 = max_date - timedelta(days=30)
    df_30d = df[df["Date"] >= min_date_30].copy()
    if df_30d.empty:
        return None

    # Raio-X por linha + cluster (30d)
    raio_x = (
        df_30d.groupby(["linha", "Cluster"])
        .agg(
            Km=("Km", "sum"),
            Comb=("Comb.", "sum"),
            veiculo=("veiculo", lambda x: list(pd.Series(x).dropna().unique())[0] if len(pd.Series(x).dropna().unique()) else None),
        )
        .reset_index()
    )

    raio_x["kml_meta"] = raio_x["Cluster"].map(METAS_CLUSTER).fillna(METAS_CLUSTER["OUTROS"])
    raio_x["kml_real"] = raio_x["Km"] / raio_x["Comb"]

    def calc_desp(r):
        litros_meta = r["Km"] / r["kml_meta"] if r["kml_meta"] > 0 else 0
        desp = (r["Comb"] - litros_meta) if (r["kml_real"] < r["kml_meta"]) else 0
        return litros_meta, desp

    raio_x[["litros_meta", "desperdicio"]] = raio_x.apply(lambda r: pd.Series(calc_desp(r)), axis=1)
    raio_x = raio_x.sort_values("desperdicio", ascending=False)

    total_km = float(raio_x["Km"].sum())
    total_litros = float(raio_x["Comb"].sum())
    total_desperdicio = float(raio_x["desperdicio"].sum())

    kml_geral_real = (total_km / total_litros) if total_litros > 0 else 0.0
    soma_litros_teoricos = float(raio_x["litros_meta"].sum())
    kml_geral_meta = (total_km / soma_litros_teoricos) if soma_litros_teoricos > 0 else 0.0

    top = raio_x.iloc[0] if not raio_x.empty else None
    foco = f"{top['Cluster']} - Linha {top['linha']}" if top is not None else "Geral"
    foco_cluster = top["Cluster"] if top is not None else "OUTROS"

    # semanal
    weekly_points = _weekly_series(df_30d)

    # piora %
    piora_pct = _calc_piora_percent(df_30d, kml_geral_real)

    # prioridade
    prioridade = _prioridade_por_desperdicio(total_desperdicio)

    return {
        "chapa": chapa,
        "nome": info_nome.get("nome", chapa),
        "cargo": info_nome.get("cargo", "MOTORISTA"),

        # periodo
        "periodo_inicio": min_date_30.date().isoformat(),
        "periodo_fim": max_date.date().isoformat(),
        "periodo_txt": f"{min_date_30.strftime('%d/%m/%Y')} a {max_date.strftime('%d/%m/%Y')}",

        # dados
        "raio_x": raio_x,
        "weekly": weekly_points,
        "piora_pct": piora_pct,
        "prioridade": prioridade,

        "totais": {
            "km": total_km,
            "litros": total_litros,
            "desp": total_desperdicio,
            "kml_real": kml_geral_real,
            "kml_meta": kml_geral_meta,
        },

        "foco": foco,
        "foco_cluster": foco_cluster,
        "linha_foco": top["linha"] if top is not None else None,
        "veiculo_foco": top["veiculo"] if top is not None else None,
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

        top_linhas = dados["raio_x"].head(4)[["linha", "Cluster", "kml_real", "kml_meta", "desperdicio"]].to_string(index=False)

        prompt = f"""
Você é um Instrutor Técnico Master de Condução Econômica (ônibus).

ALVO:
Motorista: {dados['nome']} ({dados['chapa']})
Período: {dados['periodo_txt']}
Performance 30d: {dados['totais']['kml_real']:.2f} km/l (Meta ref: {dados['totais']['kml_meta']:.2f})
Desperdício 30d: {dados['totais']['desp']:.0f} Litros
Foco: {dados['foco']}

RAIO-X (Top 4 por desperdício):
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
# HTML/PDF (PRONTUÁRIO)
# ==============================================================================
def _build_svg_line_chart(points, title="Performance Semanal"):
    """
    Gera um SVG simples (sem libs externas) com 2 linhas:
    - Real (azul)
    - Meta (vermelho tracejado)
    """
    if not points:
        return f"<div class='chartEmpty'>Sem dados suficientes para gráfico.</div>"

    # filtra None
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
    # margem
    rng = (y_max - y_min) if (y_max > y_min) else 0.5
    y_min -= rng * 0.12
    y_max += rng * 0.12

    def x(i):
        return padL + (innerW * (i / (len(pts) - 1)))

    def y(v):
        return padT + (innerH * (1 - ((v - y_min) / (y_max - y_min if y_max != y_min else 1))))

    # linhas
    real_path = "M " + " L ".join([f"{x(i):.1f} {y(p['real']):.1f}" for i, p in enumerate(pts)])
    meta_path = "M " + " L ".join([f"{x(i):.1f} {y(p['meta']):.1f}" for i, p in enumerate(pts)])

    # labels x
    labels = ""
    for i, p in enumerate(pts):
        labels += f"<text x='{x(i):.1f}' y='{H-18}' text-anchor='middle' class='axisLabel'>{_esc(p['label'])}</text>"

    # y ticks (4)
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

        <!-- legend -->
        <g transform="translate({padL}, {padT-6})">
          <line x1="0" y1="0" x2="26" y2="0" class="legReal"/><text x="34" y="4" class="legend">Realizado</text>
          <line x1="120" y1="0" x2="146" y2="0" class="legMeta"/><text x="154" y="4" class="legend">Meta (Ref)</text>
        </g>
      </svg>
    </div>
    """


def gerar_html_prontuario(prontuario_id: str, d, txt_ia):
    """
    Gera HTML no estilo do seu exemplo (cards + gráfico + raio-x + seções IA).
    Mantém suas propriedades/fluxo do script: não altera payload, só o template.
    """
    ia = _parse_ia_sections(txt_ia)
    cluster = d.get("foco_cluster") or "OUTROS"

    # cards
    litros_desvio = float(d["totais"]["desp"])
    piora_pct = float(d.get("piora_pct") or 0.0)
    kml_medio = float(d["totais"]["kml_real"])

    prioridade = d.get("prioridade", "PRIORIDADE")
    tecnologia = "TECNOLOGIA"  # mantém igual do exemplo

    # raio-x tabela (Top 10) com mês (YYYY-MM) estimado pelo Date max (30d)
    rx = d["raio_x"].copy()
    if rx.empty:
        rx_rows_html = "<tr><td colspan='9' class='muted'>Sem dados.</td></tr>"
    else:
        # % do impacto
        total_d = float(rx["desperdicio"].sum()) if float(rx["desperdicio"].sum()) > 0 else 1.0
        rx["pct"] = (rx["desperdicio"] / total_d) * 100.0

        # mes ref: usa periodo_fim
        mes_ref = ""
        try:
            mes_ref = pd.to_datetime(d["periodo_fim"]).strftime("%Y-%m")
        except Exception:
            mes_ref = ""

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

    chart_html = _build_svg_line_chart(d.get("weekly", []), title=f"Performance Semanal: {prontuario_id}")

    return f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Prontuário {prontuario_id}</title>
<style>
  :root {{
    --blue:#1f6fb2;
    --blue2:#0b5d9a;
    --red:#c74343;
    --text:#1b1f24;
    --muted:#6b7280;
    --line:#e5e7eb;
    --bg:#ffffff;
    --card:#ffffff;
    --shadow: 0 2px 10px rgba(0,0,0,.06);
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    margin: 0;
    padding: 24px;
  }}
  .page {{
    max-width: 900px;
    margin: 0 auto;
  }}
  .topbar {{
    height: 10px;
    background: var(--blue);
    border-radius: 999px;
    margin-bottom: 18px;
  }}
  .header {{
    display:flex;
    align-items:flex-start;
    justify-content:space-between;
    gap:16px;
  }}
  .hTitle {{
    font-size: 28px;
    font-weight: 800;
    margin: 0;
  }}
  .hSub {{
    margin-top: 6px;
    color: var(--muted);
    font-size: 14px;
  }}
  .prio {{
    font-weight: 800;
    color: var(--muted);
    font-size: 12px;
    margin-top: 6px;
    text-align:right;
  }}
  .cards {{
    display:grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 16px 0 14px 0;
  }}
  .card {{
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 14px 12px;
    box-shadow: var(--shadow);
    min-height: 86px;
  }}
  .cardBig {{
    font-size: 22px;
    font-weight: 800;
    margin: 0;
    color: var(--text);
  }}
  .cardLabel {{
    margin-top: 6px;
    font-size: 12px;
    color: var(--muted);
    letter-spacing: .4px;
    text-transform: uppercase;
  }}
  .cardSmall {{
    margin-top: 2px;
    font-size: 12px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: .4px;
  }}
  .chartWrap {{
    background: #fff;
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 12px 14px 8px 14px;
    box-shadow: var(--shadow);
    margin-bottom: 18px;
  }}
  .chartTitle {{
    font-weight: 800;
    margin-bottom: 8px;
    color: var(--text);
  }}
  .grid {{
    stroke: #eef2f7;
    stroke-width: 1;
  }}
  .lineReal {{
    fill:none;
    stroke: var(--blue2);
    stroke-width: 3;
  }}
  .lineMeta {{
    fill:none;
    stroke: var(--red);
    stroke-width: 2.5;
    stroke-dasharray: 7 5;
  }}
  .axisLabel {{
    font-size: 11px;
    fill: #6b7280;
  }}
  .legend {{
    font-size: 12px;
    fill: #374151;
  }}
  .legReal {{
    stroke: var(--blue2);
    stroke-width: 3;
  }}
  .legMeta {{
    stroke: var(--red);
    stroke-width: 2.5;
    stroke-dasharray: 7 5;
  }}
  .section {{
    margin-top: 14px;
  }}
  .secTitle {{
    color: var(--blue);
    font-weight: 900;
    font-size: 16px;
    margin: 14px 0 10px 0;
  }}
  .divider {{
    height: 1px;
    background: var(--line);
    margin: 10px 0 12px 0;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    border: 1px solid var(--line);
    border-radius: 14px;
    overflow: hidden;
    box-shadow: var(--shadow);
  }}
  thead th {{
    background: #f7fafc;
    color: #374151;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: .4px;
    padding: 10px 10px;
    border-bottom: 1px solid var(--line);
  }}
  tbody td {{
    padding: 10px 10px;
    border-bottom: 1px solid var(--line);
    font-size: 13px;
    vertical-align: top;
  }}
  tbody tr:last-child td {{ border-bottom: none; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .strong {{ font-weight: 800; }}
  .badge {{
    font-weight: 900;
    color: #b91c1c;
  }}
  .muted {{ color: var(--muted); }}
  .box {{
    border-left: 5px solid var(--blue);
    background: #f8fbff;
    border-radius: 12px;
    padding: 12px 12px;
    box-shadow: var(--shadow);
    border: 1px solid var(--line);
  }}
  .box.orange {{
    border-left-color: #f59e0b;
    background: #fffbeb;
  }}
  .box p {{
    margin: 0;
    line-height: 1.45;
    white-space: pre-wrap;
  }}
  .foot {{
    margin-top: 14px;
    color: var(--muted);
    font-size: 11px;
    text-align: left;
  }}

  @media print {{
    body {{ padding: 0; }}
    .page {{ max-width: 100%; }}
    .card, .chartWrap, table, .box {{ box-shadow: none; }}
  }}
</style>
</head>

<body>
  <div class="page">
    <div class="topbar"></div>

    <div class="header">
      <div>
        <div class="hTitle">PRONTUÁRIO: { _esc(prontuario_id) }</div>
        <div class="hSub">Análise Técnica | Período: { _esc(d["periodo_txt"]) }</div>
      </div>
      <div class="prio">{ _esc(prioridade) }</div>
    </div>

    <div class="cards">
      <div class="card">
        <div class="cardBig">{ _esc(cluster) }</div>
        <div class="cardSmall">{ _esc(tecnologia) }</div>
      </div>
      <div class="card">
        <div class="cardBig">{ _fmt_int(litros_desvio) } L</div>
        <div class="cardLabel">DESVIO</div>
      </div>
      <div class="card">
        <div class="cardBig">{ _fmt_float(piora_pct, 1).replace(",", ".") }%</div>
        <div class="cardLabel">PIORA</div>
      </div>
      <div class="card">
        <div class="cardBig">{ _fmt_float(kml_medio, 2).replace(",", ".") }</div>
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

    <div class="foot">Gerado automaticamente pelo Agente Diesel AI.</div>
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
# (NÃO REMOVER/ALTERAR CHAVES EXISTENTES – mantém suas propriedades)
# ==============================================================================
def criar_ordem_e_evento(sb_b, dados, lote_id, pdf_path, pdf_url, html_path, html_url, txt_ia):
    # dt padrão (instrutor redefine no LANÇAR)
    dt_inicio = datetime.utcnow().date().isoformat()
    dias = DEFAULT_DIAS_MONITORAMENTO
    dt_fim_planejado = (datetime.utcnow().date() + timedelta(days=dias - 1)).isoformat()

    raio_top = dados["raio_x"].head(10).to_dict(orient="records")

    payload = {
        # vínculo lote (se existir coluna; se não existir, remove)
        "lote_id": lote_id,

        "motorista_chapa": dados["chapa"],
        "motorista_nome": dados["nome"],
        "motivo": dados["foco"],

        # ✅ fluxo certo
        "status": "AGUARDANDO_INSTRUTOR",
        "dias_monitoramento": dias,
        "dt_inicio": dt_inicio,
        "dt_fim_planejado": dt_fim_planejado,

        # baseline
        "kml_inicial": dados["totais"]["kml_real"],
        "kml_meta": dados["totais"]["kml_meta"],
        "observacao_inicial": txt_ia[:5000],

        # arquivos / evidências
        "arquivo_pdf_url": pdf_url,
        "arquivo_html_url": html_url,

        # opcional: salvar paths também (se você quiser)
        "metadata": {
            "versao": "V13_prontuario_layout",
            "lote_id": lote_id,
            "periodo_inicio_30d": dados["periodo_inicio"],
            "periodo_fim_30d": dados["periodo_fim"],
            "foco": dados["foco"],
            "cluster_foco": dados["foco_cluster"],
            "linha_foco": dados["linha_foco"],
            "veiculo_foco": dados["veiculo_foco"],
            "kpis_30d": dados["totais"],
            "raio_x_top10": raio_top,
            "weekly_points": dados.get("weekly", []),
            "piora_pct": dados.get("piora_pct", 0.0),
            "prioridade": dados.get("prioridade", None),
            "pdf_path": pdf_path,
            "html_path": html_path,
        },

        # se sua coluna for jsonb array
        "evidencias_urls": [u for u in [pdf_url, html_url] if u],
    }

    # 1) cria ORDEM (diesel_acompanhamentos)
    ordem = sb_b.table(TABELA_ORDEM).insert(payload).execute().data
    if not ordem:
        raise RuntimeError("Falha ao inserir diesel_acompanhamentos (ordem vazia).")
    ordem_id = ordem[0].get("id")
    if not ordem_id:
        raise RuntimeError("Falha: diesel_acompanhamentos retornou sem id.")

    # 2) cria EVENTO LANCAMENTO
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
            "desperdicio_litros_30d": dados["totais"]["desp"],
            "cluster_foco": dados["foco_cluster"],
            "linha_foco": dados["linha_foco"],
            "veiculo_foco": dados["veiculo_foco"],
            "weekly_points": dados.get("weekly", []),
            "piora_pct": dados.get("piora_pct", 0.0),
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

    mapa_nomes = carregar_mapa_nomes()
    itens = obter_motoristas_do_lote()
    if not itens:
        atualizar_status_lote("ERRO", "Lote sem itens")
        return

    sb_b = _sb_b()

    print(f"🚀 Iniciando geração de {len(itens)} ordens (diesel_acompanhamentos)...")

    for item in itens:
        mot_chapa = str(item.get("motorista_chapa") or "").strip()
        if not mot_chapa:
            continue

        try:
            df_full = carregar_historico_motorista(mot_chapa)
            if df_full.empty:
                print(f"⚠️ Sem dados para {mot_chapa}")
                continue

            info_nome = mapa_nomes.get(mot_chapa, {"nome": mot_chapa, "cargo": "MOTORISTA"})
            dados = processar_dados_prontuario(df_full, mot_chapa, info_nome)
            if not dados:
                print(f"⚠️ Sem janela válida para {mot_chapa}")
                continue

            print(f"   ⚙️ Gerando PRONTUÁRIO PDF/HTML para {mot_chapa}...")

            # prontuario_id: mantém compatível (se você tiver um número real, pode trocar depois)
            prontuario_id = mot_chapa

            safe = _safe_filename(f"{prontuario_id}_Prontuario")
            p_html = PASTA_SAIDA / f"{safe}.html"
            p_pdf = PASTA_SAIDA / f"{safe}.pdf"

            txt_ia = chamar_ia_coach(dados)

            html = gerar_html_prontuario(prontuario_id, dados, txt_ia)
            p_html.write_text(html, encoding="utf-8")

            html_to_pdf(p_html, p_pdf)

            # upload Storage (Supabase B)
            pdf_path, pdf_url = upload_storage(p_pdf, f"{safe}.pdf", "application/pdf")
            html_path, html_url = upload_storage(p_html, f"{safe}.html", "text/html")

            # cria ordem+evento (mantendo propriedades)
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

    # status final do lote
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
