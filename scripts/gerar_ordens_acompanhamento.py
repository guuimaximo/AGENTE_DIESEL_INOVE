# scripts/relatorio_gerencial.py
import os
import re
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

import vertexai
from vertexai.generative_models import GenerativeModel

from supabase import create_client
from playwright.sync_api import sync_playwright

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
REPORT_PERIODO_INICIO = os.getenv("REPORT_PERIODO_INICIO")  # YYYY-MM-DD
REPORT_PERIODO_FIM = os.getenv("REPORT_PERIODO_FIM")        # YYYY-MM-DD

PASTA_SAIDA = os.getenv("REPORT_OUTPUT_DIR", "Relatorios_Diesel_Final")
TABELA_ORIGEM = os.getenv("DIESEL_SOURCE_TABLE", "premiacao_diaria")
BUCKET_RELATORIOS = os.getenv("REPORT_BUCKET", "relatorios")

# Padroniza o "prefix" remoto (só path no bucket)
REMOTE_BASE_PREFIX = os.getenv("REPORT_REMOTE_PREFIX", "diesel")


# ==============================================================================
# Helpers
# ==============================================================================
def _assert_env():
    missing = []
    for k in [
        "SUPABASE_A_URL",
        "SUPABASE_A_SERVICE_ROLE_KEY",
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

    # Sem upsert: se existir, falha e você enxerga no log
    storage.upload(
        path=remote_path,
        file=data,
        file_options={"content-type": content_type},
    )
    return len(data)


def _fmt_br_date(d: date | None) -> str:
    return d.strftime("%d/%m/%Y") if d else ""


# ==============================================================================
# 0) BUSCA DADOS SUPABASE A -> DF padrão interno
# ==============================================================================
def carregar_dados_supabase_a(periodo_inicio: date | None, periodo_fim: date | None) -> pd.DataFrame:
    sb = _sb_a()
    PAGE_SIZE = int(os.getenv("REPORT_PAGE_SIZE", "1000"))
    MAX_ROWS = int(os.getenv("REPORT_MAX_ROWS", "250000"))

    base_q = sb.table(TABELA_ORIGEM).select(
        'id_premiacao_diaria, dia, motorista, veiculo, linha, km_rodado, combustivel_consumido, minutos_em_viagem, "km/l"'
    )

    if periodo_inicio:
        base_q = base_q.gte("dia", str(periodo_inicio))
    if periodo_fim:
        base_q = base_q.lte("dia", str(periodo_fim))

    base_q = base_q.order("dia", desc=False).order("id_premiacao_diaria", desc=False)

    all_rows = []
    start = 0
    pages = 0

    while True:
        end = start + PAGE_SIZE - 1
        resp = base_q.range(start, end).execute()
        rows = resp.data or []
        pages += 1
        all_rows.extend(rows)
        print(f"📦 [SupabaseA] page={pages} range={start}-{end} fetched={len(rows)} total={len(all_rows)}")

        if len(rows) < PAGE_SIZE:
            break

        if len(all_rows) >= MAX_ROWS:
            all_rows = all_rows[:MAX_ROWS]
            print(f"⚠️ [SupabaseA] MAX_ROWS atingido: {MAX_ROWS}")
            break

        start += PAGE_SIZE

    if not all_rows:
        return pd.DataFrame(columns=["Date", "Motorista", "veiculo", "linha", "kml", "Km", "Comb."])

    df = pd.DataFrame(all_rows)

    if "km/l" in df.columns and "kml" not in df.columns:
        df["kml"] = df["km/l"]

    out = pd.DataFrame()
    out["Date"] = df.get("dia")
    out["Motorista"] = df.get("motorista")
    out["veiculo"] = df.get("veiculo")
    out["linha"] = df.get("linha")
    out["kml"] = df.get("kml")
    out["Km"] = df.get("km_rodado")
    out["Comb."] = df.get("combustivel_consumido")
    return out


# ==============================================================================
# 1) PROCESSAMENTO (mantive o seu)
# ==============================================================================
def processar_dados_gerenciais_df(df: pd.DataFrame, periodo_inicio: date | None, periodo_fim: date | None):
    print("⚙️  [Sistema] Processando dados para visão gerencial...")

    obrigatorias = ["Date", "Motorista", "veiculo", "linha", "kml", "Km", "Comb."]
    faltando = [c for c in obrigatorias if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas obrigatórias ausentes: {faltando}. Colunas atuais: {df.columns.tolist()}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    for col in ["kml", "Km", "Comb."]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    bruto_min = df["Date"].min()
    bruto_max = df["Date"].max()
    bruto_min_txt = bruto_min.strftime("%d/%m/%Y") if pd.notna(bruto_min) else "N/D"
    bruto_max_txt = bruto_max.strftime("%d/%m/%Y") if pd.notna(bruto_max) else "N/D"
    qtd_bruto = len(df)

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
    qtd_cluster_invalido = int(df["Cluster"].isna().sum())
    df = df.dropna(subset=["Cluster"])

    df_clean = df[(df["kml"] >= 1.5) & (df["kml"] <= 5)].copy()
    if df_clean.empty:
        raise ValueError("Sem dados válidos após filtros (kml entre 1.5 e 5 e cluster válido).")

    if periodo_inicio and periodo_fim:
        periodo_txt = f"{_fmt_br_date(periodo_inicio)} a {_fmt_br_date(periodo_fim)}"
    else:
        data_ini = df_clean["Date"].min().strftime("%d/%m/%Y")
        data_fim = df_clean["Date"].max().strftime("%d/%m/%Y")
        periodo_txt = f"{data_ini} a {data_fim}"

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
            .agg(
                Qtd_Contaminacoes=("kml", "count"),
                KML_Min=("kml", "min"),
                KML_Max=("kml", "max"),
            )
            .reset_index()
            .sort_values("Qtd_Contaminacoes", ascending=False)
            .head(10)
        )
    else:
        top_veiculos_contaminados = pd.DataFrame(
            columns=["veiculo", "Cluster", "linha", "Qtd_Contaminacoes", "KML_Min", "KML_Max"]
        )

    df_clean["Mes_Ano"] = df_clean["Date"].dt.to_period("M")

    tabela_cluster = (
        df_clean.groupby(["Cluster", "Mes_Ano"])
        .agg({"Km": "sum", "Comb.": "sum"})
        .reset_index()
    )
    tabela_cluster["KML"] = tabela_cluster["Km"] / tabela_cluster["Comb."]
    tabela_pivot = tabela_cluster.pivot(index="Cluster", columns="Mes_Ano", values="KML")

    linha_trend = (
        df_clean.groupby(["linha", "Mes_Ano"])
        .agg({"Km": "sum", "Comb.": "sum"})
        .reset_index()
    )
    linha_trend["KML"] = linha_trend["Km"] / linha_trend["Comb."]
    linha_pivot = linha_trend.pivot(index="linha", columns="Mes_Ano", values="KML")

    cols_meses = linha_pivot.columns.sort_values()
    if len(cols_meses) >= 2:
        mes_atual = cols_meses[-1]
        mes_anterior = cols_meses[-2]
        linha_pivot["KML_Atual"] = linha_pivot[mes_atual]
        linha_pivot["KML_Anterior"] = linha_pivot[mes_anterior]
        linha_pivot["Variacao_Pct"] = (
            (linha_pivot["KML_Atual"] - linha_pivot["KML_Anterior"]) / linha_pivot["KML_Anterior"]
        ) * 100
        top_linhas_queda = (
            linha_pivot.dropna(subset=["Variacao_Pct"])
            .sort_values("Variacao_Pct", ascending=True)
            .head(5)[["KML_Anterior", "KML_Atual", "Variacao_Pct"]]
            .reset_index()
        )
    else:
        linha_pivot["KML_Atual"] = linha_pivot[cols_meses[-1]]
        linha_pivot["KML_Anterior"] = 0
        linha_pivot["Variacao_Pct"] = 0
        top_linhas_queda = linha_pivot.sort_values("KML_Atual", ascending=True).head(5).reset_index()

    ultimo_mes = df_clean["Mes_Ano"].max()
    df_atual = df_clean[df_clean["Mes_Ano"] == ultimo_mes].copy()

    ref_grupo = (
        df_atual.groupby(["linha", "Cluster"])
        .agg({"Km": "sum", "Comb.": "sum"})
        .reset_index()
    )
    ref_grupo["KML_Ref"] = ref_grupo["Km"] / ref_grupo["Comb."]
    ref_grupo.rename(columns={"Km": "KM_Total_Linha"}, inplace=True)

    df_atual = pd.merge(
        df_atual,
        ref_grupo[["linha", "Cluster", "KML_Ref", "KM_Total_Linha"]],
        on=["linha", "Cluster"],
        how="left",
    )

    def calc_desperdicio(r):
        try:
            if r["KML_Ref"] > 0 and r["kml"] < r["KML_Ref"]:
                return r["Comb."] - (r["Km"] / r["KML_Ref"])
            return 0
        except Exception:
            return 0

    df_atual["Litros_Desperdicio"] = df_atual.apply(calc_desperdicio, axis=1)
    total_desperdicio = float(df_atual["Litros_Desperdicio"].sum())

    top_veiculos = (
        df_atual.groupby(["veiculo", "Cluster", "linha"])
        .agg({"Litros_Desperdicio": "sum", "Km": "sum", "Comb.": "sum", "KML_Ref": "mean"})
        .reset_index()
    )
    top_veiculos["KML_Real"] = top_veiculos["Km"] / top_veiculos["Comb."]
    top_veiculos["KML_Meta"] = top_veiculos["KML_Ref"]
    top_veiculos = top_veiculos.sort_values("Litros_Desperdicio", ascending=False).head(5)

    top_motoristas = (
        df_atual.groupby(["Motorista", "Cluster", "linha", "KM_Total_Linha"])
        .agg({"Litros_Desperdicio": "sum", "Km": "sum", "Comb.": "sum", "KML_Ref": "mean"})
        .reset_index()
    )
    top_motoristas["KML_Real"] = top_motoristas["Km"] / top_motoristas["Comb."]
    top_motoristas["KML_Meta"] = top_motoristas["KML_Ref"]
    top_motoristas["Impacto_Pct"] = (top_motoristas["Km"] / top_motoristas["KM_Total_Linha"]) * 100
    top_motoristas = top_motoristas.sort_values("Litros_Desperdicio", ascending=False).head(5)

    clean_min = df_clean["Date"].min()
    clean_max = df_clean["Date"].max()

    return {
        "df_clean": df_clean,
        "df_atual": df_atual,
        "qtd_excluidos": int(qtd_excluidos),
        "total_desperdicio": total_desperdicio,
        "top_veiculos": top_veiculos,
        "top_linhas_queda": top_linhas_queda,
        "top_motoristas": top_motoristas,
        "top_veiculos_contaminados": top_veiculos_contaminados,
        "periodo": periodo_txt,
        "mes_atual_nome": mes_atual_txt,
        "tabela_pivot": tabela_pivot,
        "cobertura": {
            "bruto_min": bruto_min_txt,
            "bruto_max": bruto_max_txt,
            "qtd_bruto": int(qtd_bruto),
            "qtd_cluster_invalido": int(qtd_cluster_invalido),
            "qtd_contaminacao": int(qtd_excluidos),
            "clean_min": clean_min.strftime("%d/%m/%Y") if pd.notna(clean_min) else "N/D",
            "clean_max": clean_max.strftime("%d/%m/%Y") if pd.notna(clean_max) else "N/D",
            "qtd_clean": int(len(df_clean)),
        },
    }


# ==============================================================================
# 2) GRÁFICO
# ==============================================================================
def gerar_grafico_geral(df_clean: pd.DataFrame, caminho_img: Path):
    df_clean = df_clean.copy()
    df_clean["Semana"] = df_clean["Date"].dt.to_period("W").apply(lambda r: r.start_time)

    evolucao_cluster = (
        df_clean.groupby(["Semana", "Cluster"])
        .agg({"Km": "sum", "Comb.": "sum"})
        .reset_index()
    )
    evolucao_cluster["KML"] = evolucao_cluster["Km"] / evolucao_cluster["Comb."]
    pivot_chart = evolucao_cluster.pivot(index="Semana", columns="Cluster", values="KML")

    evolucao_geral = (
        df_clean.groupby(["Semana"])
        .agg({"Km": "sum", "Comb.": "sum"})
        .reset_index()
    )
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
# 3) IA (mantida)
# ==============================================================================
def consultar_ia_gerencial(dados_proc: dict) -> str:
    print("🧠 [Gerente] Solicitando análise estratégica à IA...")

    if not VERTEX_PROJECT_ID:
        return (
            "<p><b>Visão Geral da Eficiência no Período</b><br>"
            "IA desativada (VERTEX_PROJECT_ID não configurado). Relatório gerado apenas com dados.</p>"
        )

    try:
        df_clean = dados_proc["df_clean"].copy()

        km_total_periodo = df_clean["Km"].sum()
        comb_total_periodo = df_clean["Comb."].sum()
        kml_periodo = km_total_periodo / comb_total_periodo if comb_total_periodo > 0 else 0

        if "Mes_Ano" not in df_clean.columns:
            df_clean["Mes_Ano"] = df_clean["Date"].dt.to_period("M")

        mensal = df_clean.groupby("Mes_Ano").agg({"Km": "sum", "Comb.": "sum"}).reset_index()
        mensal["KML"] = mensal["Km"] / mensal["Comb."]
        mensal = mensal.sort_values("Mes_Ano")

        tabela_mensal_md = mensal[["Mes_Ano", "Km", "Comb.", "KML"]].to_markdown(index=False)

        if len(mensal) >= 1:
            mes_atual_row = mensal.iloc[-1]
            kml_mes_atual = float(mes_atual_row["KML"])
            mes_atual_label = str(mes_atual_row["Mes_Ano"])
        else:
            kml_mes_atual = 0.0
            mes_atual_label = "N/D"

        if len(mensal) >= 2:
            mes_ant_row = mensal.iloc[-2]
            kml_mes_anterior = float(mes_ant_row["KML"])
            mes_ant_label = str(mes_ant_row["Mes_Ano"])
            delta_kml_mes = ((kml_mes_atual - kml_mes_anterior) / kml_mes_anterior * 100) if kml_mes_anterior > 0 else 0.0
        else:
            kml_mes_anterior = 0.0
            mes_ant_label = "N/D"
            delta_kml_mes = 0.0

        df_clean["Semana"] = df_clean["Date"].dt.to_period("W").apply(lambda r: r.start_time)
        semanal = df_clean.groupby("Semana").agg({"Km": "sum", "Comb.": "sum"}).reset_index()
        semanal["KML"] = semanal["Km"] / semanal["Comb."]
        semanal = semanal.sort_values("Semana")

        if len(semanal) >= 1:
            semana_atual_row = semanal.iloc[-1]
            kml_semana_atual = float(semana_atual_row["KML"])
            semana_atual_inicio_txt = semana_atual_row["Semana"].strftime("%d/%m/%Y")
        else:
            kml_semana_atual = 0.0
            semana_atual_inicio_txt = "N/D"

        if len(semanal) >= 2:
            semana_ant_row = semanal.iloc[-2]
            kml_semana_anterior = float(semana_ant_row["KML"])
            semana_ant_inicio_txt = semana_ant_row["Semana"].strftime("%d/%m/%Y")
            delta_kml_semana = ((kml_semana_atual - kml_semana_anterior) / kml_semana_anterior * 100) if kml_semana_anterior > 0 else 0.0
        else:
            kml_semana_anterior = 0.0
            semana_ant_inicio_txt = "N/D"
            delta_kml_semana = 0.0

        vertexai.init(project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION)
        model = GenerativeModel(VERTEX_MODEL)

        cob = dados_proc.get("cobertura", {}) or {}

        prompt = f"""
Você é Diretor de Operações de uma empresa de transporte urbano, especialista em eficiência energética (KM/L).

Analise a performance de KM/L no período: {dados_proc['periodo']}
Foco:
- MÊS DE REFERÊNCIA: {dados_proc['mes_atual_nome']}
- ÚLTIMA SEMANA do período

TRANSPARÊNCIA DA BASE:
- Base bruta no range: {cob.get('bruto_min','N/D')} a {cob.get('bruto_max','N/D')} | registros: {cob.get('qtd_bruto','?')}
- Removidos por cluster inválido: {cob.get('qtd_cluster_invalido','?')}
- Removidos por contaminação (kml < 1.5 ou > 5): {cob.get('qtd_contaminacao','?')}
- Base limpa analisada: {cob.get('clean_min','N/D')} a {cob.get('clean_max','N/D')} | registros: {cob.get('qtd_clean','?')}

VISÃO GERAL (FROTA):
- KM/L médio período: {kml_periodo:.2f}
- KM/L mês atual ({mes_atual_label}): {kml_mes_atual:.2f}
- KM/L mês anterior ({mes_ant_label}): {kml_mes_anterior:.2f}
- Variação mês atual vs anterior: {delta_kml_mes:+.1f}%
- KM/L última semana (início {semana_atual_inicio_txt}): {kml_semana_atual:.2f}
- KM/L semana anterior (início {semana_ant_inicio_txt}): {kml_semana_anterior:.2f}
- Variação última semana vs anterior: {delta_kml_semana:+.1f}%
- Desperdício total estimado no mês atual: {dados_proc['total_desperdicio']:.0f} litros

TABELA MENSAL (base limpa):
{tabela_mensal_md}

CLUSTER (KML por mês):
{dados_proc['tabela_pivot'].to_markdown()}

TOP ALVOS DO MÊS:
VEÍCULOS:
{dados_proc['top_veiculos'].to_markdown()}

LINHAS EM QUEDA:
{dados_proc['top_linhas_queda'].to_markdown()}

MOTORISTAS:
{dados_proc['top_motoristas'].to_markdown()}

FORMATO DE RESPOSTA:
Gere um resumo executivo em HTML (sem markdown, sem ```), usando apenas: <p>, <b>, <br>, <ul>, <li>.

Estrutura obrigatória:
1) <b>Visão Geral da Eficiência no Período</b>
2) <b>Zoom na Última Semana</b>
3) <b>Recomendações Práticas para o Próximo Ciclo</b>
   - No mínimo 8 itens, agrupados por:
     (a) Operação
     (b) Manutenção
     (c) Dados/Sistema
   - Cada item: objetivo, alvo (cite linhas/clusters/veículos/motoristas do dado), métrica de sucesso.

Regras:
- Não invente fatos.
- Seja direto e acionável.
""".strip()

        resp = model.generate_content(prompt)
        texto = getattr(resp, "text", None) or "Análise indisponível."
        return texto.replace("```html", "").replace("```", "")

    except Exception as e:
        import traceback
        print("❌ Erro ao chamar IA:", repr(e))
        print(traceback.format_exc())
        return "<p>Análise indisponível (erro na IA).</p>"


# ==============================================================================
# 4) HTML + PDF (SUA FUNÇÃO COMPLETA – agora sem NotImplementedError)
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
                if isinstance(val, (int, float)):
                    val_str = fmt.format(val)
                else:
                    val_str = str(val)

                style = ""
                if col == "Variacao_Pct":
                    try:
                        v = float(val)
                    except Exception:
                        v = 0
                    if v < -5:
                        style = "color: #c0392b; font-weight: bold;"
                    elif v < 0:
                        style = "color: #e67e22;"
                    else:
                        style = "color: #27ae60;"
                    val_str = f"{v:.1f}%"

                rows += f"<td style='{style}'>{val_str}</td>"
            rows += "</tr>"
        return rows

    df_clean = dados["df_clean"].copy()
    if "Mes_Ano" not in df_clean.columns:
        df_clean["Mes_Ano"] = df_clean["Date"].dt.to_period("M")

    mensal = (
        df_clean.groupby("Mes_Ano")
        .agg({"Km": "sum", "Comb.": "sum"})
        .reset_index()
    )
    mensal["KML"] = mensal["Km"] / mensal["Comb."]
    mensal = mensal.sort_values("Mes_Ano")

    if len(mensal) >= 1:
        kml_mes_atual = float(mensal.iloc[-1]["KML"])
    else:
        kml_mes_atual = 0.0

    if len(mensal) >= 2:
        kml_mes_anterior = float(mensal.iloc[-2]["KML"])
        delta_kml_mes = ((kml_mes_atual - kml_mes_anterior) / kml_mes_anterior * 100) if kml_mes_anterior > 0 else 0
    else:
        delta_kml_mes = 0

    if delta_kml_mes < 0:
        texto_var = f"{abs(delta_kml_mes):.1f}% de QUEDA"
        cor_var = "#c0392b"
    elif delta_kml_mes > 0:
        texto_var = f"{delta_kml_mes:.1f}% de MELHORA"
        cor_var = "#27ae60"
    else:
        texto_var = "0,0% (Estável)"
        cor_var = "#7f8c8d"

    rows_veic = make_rows(
        dados["top_veiculos"],
        ["veiculo", "Cluster", "linha", "KML_Real", "KML_Meta", "Litros_Desperdicio"],
        {"KML_Real": "{:.2f}", "KML_Meta": "{:.2f}", "Litros_Desperdicio": "{:.0f}"},
    )

    rows_lin = make_rows(
        dados["top_linhas_queda"],
        ["linha", "KML_Anterior", "KML_Atual", "Variacao_Pct"],
        {"KML_Anterior": "{:.2f}", "KML_Atual": "{:.2f}", "Variacao_Pct": "{:.1f}"},
    )

    rows_mot = ""
    if dados["top_motoristas"] is not None and not dados["top_motoristas"].empty:
        for _, row in dados["top_motoristas"].iterrows():
            rows_mot += f"""
            <tr>
                <td>{row['Motorista']}</td>
                <td>{row['Cluster']}</td>
                <td>{row['linha']}</td>
                <td><b>{row['KML_Real']:.2f}</b></td>
                <td style="color:#777">{row['KML_Meta']:.2f}</td>
                <td><b>{row['Litros_Desperdicio']:.0f}</b></td>
                <td><span class="badge">{row['Impacto_Pct']:.1f}%</span></td>
            </tr>"""

    rows_cont = make_rows(
        dados["top_veiculos_contaminados"],
        ["veiculo", "Cluster", "linha", "Qtd_Contaminacoes", "KML_Min", "KML_Max"],
        {"Qtd_Contaminacoes": "{:.0f}", "KML_Min": "{:.2f}", "KML_Max": "{:.2f}"},
    )

    cob = dados.get("cobertura", {}) or {}

    html = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <title>Relatório Gerencial</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background-color: #f0f2f5; margin: 0; padding: 20px; }}
            .container {{ max-width: 1000px; margin: auto; background: white; padding: 40px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); border-radius: 8px; }}
            .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 3px solid #2c3e50; padding-bottom: 20px; margin-bottom: 18px; }}
            .title h1 {{ margin: 0; color: #2c3e50; text-transform: uppercase; letter-spacing: 1px; }}
            .month-card {{ background: #2c3e50; color: white; padding: 10px 20px; border-radius: 6px; text-align: center; }}
            .month-label {{ font-size: 10px; text-transform: uppercase; opacity: 0.8; }}
            .month-val {{ font-size: 18px; font-weight: bold; }}
            .kpi-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 18px; }}
            .kpi-card {{ background: #f8f9fa; padding: 18px; border-radius: 8px; text-align: center; border: 1px solid #e0e0e0; }}
            .kpi-val {{ display: block; font-size: 26px; font-weight: bold; }}
            .kpi-lbl {{ font-size: 12px; text-transform: uppercase; color: #666; letter-spacing: 1px; }}
            h2 {{ color: #2980b9; font-size: 18px; border-left: 5px solid #2980b9; padding-left: 10px; margin-top: 22px; margin-bottom: 14px; }}
            .ai-box {{ background-color: #fffde7; border: 1px solid #fbc02d; padding: 18px; border-radius: 6px; line-height: 1.6; color: #333; margin-bottom: 18px; }}
            .chart-box {{ text-align: center; margin-bottom: 18px; border: 1px solid #eee; padding: 10px; border-radius: 8px; }}
            .chart-box img {{ max-width: 100%; height: auto; }}
            .row-split {{ display: flex; gap: 22px; }}
            .col {{ flex: 1; }}
            table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
            th {{ background-color: #34495e; color: white; padding: 10px; text-align: left; }}
            td {{ border-bottom: 1px solid #eee; padding: 8px; vertical-align: top; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .badge {{ background: #e67e22; color: white; padding: 3px 8px; border-radius: 10px; font-size: 11px; font-weight: bold; }}
            .muted {{ font-size: 12px; color: #666; }}
            .footer {{ margin-top: 26px; text-align: center; font-size: 11px; color: #aaa; border-top: 1px solid #eee; padding-top: 14px; }}

            @page {{ size: A4; margin: 10mm; }}
            @media print {{
              html, body {{ background: #fff !important; padding: 0 !important; margin: 0 !important; }}
              .container {{ max-width: none !important; margin: 0 !important; padding: 0 !important; box-shadow: none !important; border-radius: 0 !important; }}
              .header, .kpi-grid {{ break-inside: avoid; page-break-inside: avoid; }}
              .row-split {{ display: block !important; }}
              .col {{ width: 100% !important; }}
              .kpi-card, .ai-box, .chart-box, table {{ break-inside: avoid; page-break-inside: avoid; }}
              h2 {{ break-after: avoid; page-break-after: avoid; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="title">
                    <h1>Relatório Gerencial</h1>
                    <div class="muted" style="margin-top:5px;">Eficiência Energética de Frota</div>
                    <div class="muted" style="margin-top:6px;">
                      <b>Período de Análise:</b> {dados['periodo']}
                    </div>
                </div>
                <div class="month-card">
                    <div class="month-label">MÊS DE REFERÊNCIA</div>
                    <div class="month-val">{dados['mes_atual_nome']}</div>
                </div>
            </div>

            <div class="kpi-grid">
                <div class="kpi-card">
                    <span class="kpi-val" style="color:#2c3e50">{kml_mes_atual:.2f}</span>
                    <span class="kpi-lbl">KM/L MÊS BASE (SEM CONTAMINAÇÕES)</span>
                </div>
                <div class="kpi-card">
                    <span class="kpi-val" style="color:{cor_var}">{texto_var}</span>
                    <span class="kpi-lbl">VARIAÇÃO VS MÊS ANTERIOR</span>
                </div>
                <div class="kpi-card">
                    <span class="kpi-val" style="color:#27ae60">AGENTE AI</span>
                    <span class="kpi-lbl">MONITORAMENTO ATIVO</span>
                </div>
            </div>

            <div class="muted" style="margin-bottom:16px;">
              <b>Cobertura (transparência):</b><br>
              Base bruta no range: {cob.get('bruto_min','N/D')} → {cob.get('bruto_max','N/D')} ({cob.get('qtd_bruto','?')} registros)<br>
              Removidos por cluster inválido: {cob.get('qtd_cluster_invalido','?')}<br>
              Removidos por contaminação (kml &lt; 1,5 ou &gt; 5): {cob.get('qtd_contaminacao','?')}<br>
              Base limpa analisada: {cob.get('clean_min','N/D')} → {cob.get('clean_max','N/D')} ({cob.get('qtd_clean','?')} registros)
            </div>

            <h2>1. Inteligência Executiva</h2>
            <div class="ai-box">{texto_ia}</div>

            <h2>2. Evolução de Eficiência (Clusters vs Média Frota)</h2>
            <div class="chart-box"><img src="{img_src}"></div>

            <div class="row-split">
                <div class="col">
                    <h2>3. Top 5 Veículos (Máquinas)</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Veículo</th><th>Cluster</th><th>Linha</th>
                                <th>Real</th><th>Ref</th><th>Perda (L)</th>
                            </tr>
                        </thead>
                        <tbody>{rows_veic}</tbody>
                    </table>
                </div>
                <div class="col">
                    <h2>4. Top 5 Linhas em Queda (Piora de KM/L)</h2>
                    <p class="muted">Comparativo Mês Atual vs Anterior.</p>
                    <table>
                        <thead>
                            <tr>
                                <th>Linha</th><th>Mês Ant.</th><th>Mês Atual</th><th>Variação</th>
                            </tr>
                        </thead>
                        <tbody>{rows_lin}</tbody>
                    </table>
                </div>
            </div>

            <h2>5. Fator Humano (Top 5 Motoristas Críticos)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Motorista</th><th>Cluster</th><th>Linha</th>
                        <th>Real</th><th>Ref</th><th>Perda (L)</th><th>Impacto (%)</th>
                    </tr>
                </thead>
                <tbody>{rows_mot}</tbody>
            </table>

            <h2>6. Top 10 Veículos com Leituras Contaminadas (Auditoria de Dados)</h2>
            <p class="muted">
                Veículos abaixo apresentam leituras de KM/L fora da faixa aceitável (kml &lt; 1,5 ou kml &gt; 5).
            </p>
            <table>
                <thead>
                    <tr>
                        <th>Veículo</th><th>Cluster</th><th>Linha</th>
                        <th>Qtd. Contaminações</th><th>KML Mínimo</th><th>KML Máximo</th>
                    </tr>
                </thead>
                <tbody>{rows_cont}</tbody>
            </table>

            <div class="footer">
                Relatório Gerado Automaticamente pelo Agente Diesel AI.<br>
                Período de Análise: {dados['periodo']}
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

    print(f"✅ PDF (Chromium) salvo: {pdf_path}")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    _assert_env()
    Path(PASTA_SAIDA).mkdir(parents=True, exist_ok=True)

    periodo_inicio = _parse_iso(REPORT_PERIODO_INICIO)
    periodo_fim = _parse_iso(REPORT_PERIODO_FIM)

    # fallback (se API não mandou datas)
    if not periodo_inicio and not periodo_fim:
        hoje = datetime.utcnow().date()
        periodo_inicio = hoje.replace(day=1)
        periodo_fim = hoje

    atualizar_status_relatorio(
        "PROCESSANDO",
        tipo=REPORT_TIPO,
        periodo_inicio=str(periodo_inicio) if periodo_inicio else None,
        periodo_fim=str(periodo_fim) if periodo_fim else None,
        erro_msg=None,
        arquivo_pdf_path=None,
        arquivo_png_path=None,
        arquivo_html_path=None,
    )

    try:
        df_base = carregar_dados_supabase_a(periodo_inicio, periodo_fim)
        dados = processar_dados_gerenciais_df(df_base, periodo_inicio, periodo_fim)

        out_dir = Path(PASTA_SAIDA)
        img_path = out_dir / "cluster_evolution_unificado.png"
        html_path = out_dir / "Relatorio_Gerencial.html"
        pdf_path = out_dir / "Relatorio_Gerencial.pdf"

        gerar_grafico_geral(dados["df_clean"], img_path)
        texto_ia = consultar_ia_gerencial(dados)
        gerar_html_gerencial(dados, texto_ia, img_path, html_path)
        gerar_pdf_do_html(html_path, pdf_path)

        # pasta remota padronizada por mês de referência
        mes_ref = str(dados["df_clean"]["Date"].max().to_period("M"))  # ex: 2026-01
        base_folder = f"{REMOTE_BASE_PREFIX}/gerencial/{mes_ref}/report_{REPORT_ID}"

        remote_png = f"{base_folder}/{img_path.name}"
        remote_html = f"{base_folder}/{html_path.name}"
        remote_pdf = f"{base_folder}/{pdf_path.name}"

        upload_storage_b(img_path, remote_png, "image/png")
        upload_storage_b(html_path, remote_html, "text/html; charset=utf-8")
        size_pdf = upload_storage_b(pdf_path, remote_pdf, "application/pdf")

        atualizar_status_relatorio(
            "CONCLUIDO",
            arquivo_pdf_path=remote_pdf,
            arquivo_png_path=remote_png,
            arquivo_html_path=remote_html,
            arquivo_nome=_safe_filename(f"Relatorio_Gerencial_{mes_ref}.pdf"),
            mime_type="application/pdf",
            tamanho_bytes=int(size_pdf),
            erro_msg=None,
        )

        print("✅ [OK] Relatório concluído e enviado ao Supabase B (PDF + HTML + PNG).")

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
