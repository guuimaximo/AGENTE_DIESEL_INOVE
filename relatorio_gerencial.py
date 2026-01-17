# relatorio_gerencial.py
import os
import io
import re
import json
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from xhtml2pdf import pisa  # pode remover depois se quiser (não usado mais)

import vertexai
from vertexai.generative_models import GenerativeModel

from supabase import create_client


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

# Controle do relatório (criado pelo backend)
REPORT_ID = os.getenv("REPORT_ID")
REPORT_TIPO = os.getenv("REPORT_TIPO", "diesel_gerencial")
REPORT_PERIODO_INICIO = os.getenv("REPORT_PERIODO_INICIO")  # YYYY-MM-DD
REPORT_PERIODO_FIM = os.getenv("REPORT_PERIODO_FIM")        # YYYY-MM-DD

# Saída local
PASTA_SAIDA = os.getenv("REPORT_OUTPUT_DIR", "Relatorios_Diesel_Final")

# Tabela origem (Supabase A)
TABELA_ORIGEM = os.getenv("DIESEL_SOURCE_TABLE", "premiacao_diaria")

# Bucket destino (Supabase B)
BUCKET_RELATORIOS = os.getenv("REPORT_BUCKET", "relatorios")


# ==============================================================================
# Helpers
# ==============================================================================
def _assert_env():
    missing = []
    for k in [
        "VERTEX_PROJECT_ID",
        "SUPABASE_A_URL",
        "SUPABASE_A_SERVICE_ROLE_KEY",
        "SUPABASE_B_URL",
        "SUPABASE_B_SERVICE_ROLE_KEY",
        "REPORT_ID",
    ]:
        if k == "VERTEX_PROJECT_ID":
            if not VERTEX_PROJECT_ID:
                missing.append(k)
        elif k == "REPORT_ID":
            if not REPORT_ID:
                missing.append(k)
        else:
            if not os.getenv(k):
                missing.append(k)

    if missing:
        raise RuntimeError(f"Variáveis obrigatórias ausentes: {missing}")


def _parse_iso(d: str | None) -> date | None:
    if not d:
        return None
    return datetime.strptime(d, "%Y-%m-%d").date()


def _safe_filename(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^\w\-.() ]+", "_", name, flags=re.UNICODE)
    name = re.sub(r"\s+", "_", name)
    return name[:180]


def _sb_a():
    return create_client(SUPABASE_A_URL, SUPABASE_A_SERVICE_ROLE_KEY)


def _sb_b():
    return create_client(SUPABASE_B_URL, SUPABASE_B_SERVICE_ROLE_KEY)


def atualizar_status_relatorio(status: str, **fields):
    """
    Atualiza public.relatorios_gerados no Supabase B
    """
    sb = _sb_b()
    payload = {"status": status, **fields}
    sb.table("relatorios_gerados").update(payload).eq("id", REPORT_ID).execute()


def upload_storage_b(local_path: Path, remote_path: str, content_type: str) -> int:
    """
    Faz upload no Supabase B Storage (bucket relatorios)
    Retorna tamanho em bytes.
    """
    sb = _sb_b()
    storage = sb.storage.from_(BUCKET_RELATORIOS)

    data = local_path.read_bytes()
    storage.upload(
        path=remote_path,
        file=data,
        file_options={"content-type": content_type},  # ✅ sem upsert
    )
    return len(data)


# ==============================================================================
# 0) BUSCA DADOS SUPABASE A  -> DF no formato do seu "CSV"
# ==============================================================================
def carregar_dados_supabase_a(periodo_inicio: date | None, periodo_fim: date | None) -> pd.DataFrame:
    """
    Origem: Supabase A / premiacao_diaria
    Mapeamento (conforme suas colunas):
      dia -> Date
      motorista -> Motorista
      veiculo -> veiculo
      linha -> linha
      km_rodado -> Km
      combustivel_consumido -> Comb.
      km/l -> kml
    """
    sb = _sb_a()

    # ✅ IMPORTANTE: coluna com "/" precisa de aspas duplas na string do select
    q = sb.table(TABELA_ORIGEM).select(
        'dia, motorista, veiculo, linha, km_rodado, combustivel_consumido, minutos_em_viagem, "km/l"'
    )

    if periodo_inicio:
        q = q.gte("dia", str(periodo_inicio))
    if periodo_fim:
        q = q.lte("dia", str(periodo_fim))

    q = q.limit(100000)

    resp = q.execute()
    rows = resp.data or []
    if not rows:
        return pd.DataFrame(columns=["Date", "Motorista", "veiculo", "linha", "kml", "Km", "Comb."])

    df = pd.DataFrame(rows)

    # Normaliza coluna do Supabase ("km/l") para o padrão interno do script ("kml")
    if "km/l" in df.columns and "kml" not in df.columns:
        df["kml"] = df["km/l"]

    # Monta no padrão do script
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
# 1) PROCESSAMENTO (mesma lógica do seu script)
# ==============================================================================
def processar_dados_gerenciais_df(df: pd.DataFrame):
    print("⚙️  [Sistema] Processando dados para visão gerencial...")

    obrigatorias = ["Date", "Motorista", "veiculo", "linha", "kml", "Km", "Comb."]
    faltando = [c for c in obrigatorias if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas obrigatórias ausentes: {faltando}. Colunas atuais: {df.columns.tolist()}")

    # Conversões
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    for col in ["kml", "Km", "Comb."]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ======== SEU CÓDIGO ORIGINAL (mantido) ========
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
    df = df.dropna(subset=["Cluster"])

    df_clean = df[(df["kml"] >= 1.5) & (df["kml"] <= 5)].copy()

    if df_clean.empty:
        raise ValueError("Sem dados válidos após filtros (kml entre 1.5 e 5 e cluster válido).")

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
        linha_pivot["Variacao_Pct"] = ((linha_pivot["KML_Atual"] - linha_pivot["KML_Anterior"]) / linha_pivot["KML_Anterior"]) * 100

        top_linhas_queda = (
            linha_pivot.dropna(subset=["Variacao_Pct"])
            .sort_values("Variacao_Pct", ascending=True)
            .head(5)
        )
        top_linhas_queda = top_linhas_queda[["KML_Anterior", "KML_Atual", "Variacao_Pct"]].reset_index()
    else:
        print("⚠️ Aviso: Base com apenas 1 mês. Mostrando piores KML absolutos.")
        linha_pivot["KML_Atual"] = linha_pivot[cols_meses[-1]]
        linha_pivot["KML_Anterior"] = 0
        linha_pivot["Variacao_Pct"] = 0
        top_linhas_queda = (
            linha_pivot.sort_values("KML_Atual", ascending=True)
            .head(5)
            .reset_index()
        )

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
    total_desperdicio = df_atual["Litros_Desperdicio"].sum()

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

    return {
        "df_clean": df_clean,
        "df_atual": df_atual,
        "qtd_excluidos": qtd_excluidos,
        "total_desperdicio": total_desperdicio,
        "top_veiculos": top_veiculos,
        "top_linhas_queda": top_linhas_queda,
        "top_motoristas": top_motoristas,
        "top_veiculos_contaminados": top_veiculos_contaminados,
        "periodo": periodo_txt,
        "mes_atual_nome": mes_atual_txt,
        "tabela_pivot": tabela_pivot,
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

    cores = {
        "C11": "#e67e22",
        "C10": "#2ecc71",
        "C9": "#3498db",
        "C8": "#9b59b6",
        "C6": "#95a5a6",
    }

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
        plt.text(
            x,
            y + 0.05,
            f"{y:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="black",
        )

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
    try:
        df_clean = dados_proc["df_clean"].copy()

        km_total_periodo = df_clean["Km"].sum()
        comb_total_periodo = df_clean["Comb."].sum()
        kml_periodo = km_total_periodo / comb_total_periodo if comb_total_periodo > 0 else 0

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
        semanal = (
            df_clean.groupby("Semana")
            .agg({"Km": "sum", "Comb.": "sum"})
            .reset_index()
        )
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

        prompt = f"""
Você é Diretor de Operações de uma empresa de transporte urbano, especializado em eficiência energética de frotas.

Sua missão é analisar a performance de KM/L da frota no período de {dados_proc['periodo']},
com foco especial no mês de {dados_proc['mes_atual_nome']} e na ÚLTIMA SEMANA do período.

VISÃO GERAL DO PERÍODO (FROTA INTEIRA):
- KM/L médio do período completo: {kml_periodo:.2f}
- KM/L médio do mês atual ({mes_atual_label}): {kml_mes_atual:.2f}
- KM/L médio do mês anterior ({mes_ant_label}): {kml_mes_anterior:.2f}
- Variação mês atual vs anterior: {delta_kml_mes:+.1f}%
- KM/L médio da última semana (início {semana_atual_inicio_txt}): {kml_semana_atual:.2f}
- KM/L médio da semana anterior (início {semana_ant_inicio_txt}): {kml_semana_anterior:.2f}
- Variação última semana vs semana anterior: {delta_kml_semana:+.1f}%
- Desperdício total estimado no mês atual: {dados_proc['total_desperdicio']:.0f} litros.
- Registros excluídos como contaminação/erro: {dados_proc['qtd_excluidos']}.

CONTEXTO POR CLUSTER (KML por mês – tabela dinâmica):
{dados_proc['tabela_pivot'].to_markdown()}

TOP ALVOS DO MÊS ATUAL:
- Top veículos com maior perda (litros desperdiçados):
{dados_proc['top_veiculos'].to_markdown()}

- Linhas com maior queda de KM/L (mês atual vs mês anterior):
{dados_proc['top_linhas_queda'].to_markdown()}

- Top motoristas com maior desperdício no mês atual:
{dados_proc['top_motoristas'].to_markdown()}

AGORA, ESCREVA UM RESUMO EXECUTIVO EM HTML (sem markdown, sem ```).
Use apenas tags simples: <p>, <b>, <br>, <ul>, <li>.

Estruture a resposta em 3 blocos principais:
1) <b>Visão Geral da Eficiência no Período</b>
2) <b>Zoom na Última Semana</b>
3) <b>Recomendações Práticas para o Próximo Ciclo</b>

Regras finais:
- Seja direto, executivo e objetivo (linguagem para diretoria).
- Sempre baseie as conclusões nos dados apresentados, evitando frases genéricas.
""".strip()

        resp = model.generate_content(prompt)
        texto = getattr(resp, "text", None) or "Análise indisponível."
        return texto.replace("```html", "").replace("```", "")

    except Exception as e:
        import traceback
        print("❌ Erro ao chamar a IA (Gerencial):", repr(e))
        print(traceback.format_exc())
        print("DEBUG VERTEX_PROJECT_ID:", VERTEX_PROJECT_ID)
        print("DEBUG VERTEX_LOCATION:", VERTEX_LOCATION)
        print("DEBUG VERTEX_MODEL:", VERTEX_MODEL)
        print("DEBUG GOOGLE_APPLICATION_CREDENTIALS:", os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
        return "<p>Análise indisponível (erro na chamada da IA).</p>"




# ==============================================================================
# 4) HTML + PDF (Chromium/Playwright) — PDF fiel ao layout do HTML
# ==============================================================================

from playwright.sync_api import sync_playwright

def gerar_html_gerencial(dados: dict, texto_ia: str, img_path: Path, html_path: Path):
    # ✅ para abrir no Storage: HTML e PNG ficam juntos na mesma pasta
    img_src = img_path.name  # "cluster_evolution_unificado.png"

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
        kml_mes_atual = mensal.iloc[-1]["KML"]
    else:
        kml_mes_atual = 0

    if len(mensal) >= 2:
        kml_mes_anterior = mensal.iloc[-2]["KML"]
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

    # ✅ adiciona regras de impressão para PDF (Chromium)
    html = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <title>Relatório Gerencial</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background-color: #f0f2f5; margin: 0; padding: 20px; }}
            .container {{ max-width: 1000px; margin: auto; background: white; padding: 40px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); border-radius: 8px; }}
            .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 3px solid #2c3e50; padding-bottom: 20px; margin-bottom: 30px; }}
            .title h1 {{ margin: 0; color: #2c3e50; text-transform: uppercase; letter-spacing: 1px; }}
            .month-card {{ background: #2c3e50; color: white; padding: 10px 20px; border-radius: 6px; text-align: center; }}
            .month-label {{ font-size: 10px; text-transform: uppercase; opacity: 0.8; }}
            .month-val {{ font-size: 18px; font-weight: bold; }}
            .kpi-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 40px; }}
            .kpi-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; border: 1px solid #e0e0e0; }}
            .kpi-val {{ display: block; font-size: 28px; font-weight: bold; }}
            .kpi-lbl {{ font-size: 12px; text-transform: uppercase; color: #666; letter-spacing: 1px; }}
            h2 {{ color: #2980b9; font-size: 18px; border-left: 5px solid #2980b9; padding-left: 10px; margin-top: 40px; margin-bottom: 20px; }}
            .ai-box {{ background-color: #fffde7; border: 1px solid #fbc02d; padding: 20px; border-radius: 6px; line-height: 1.6; color: #333; margin-bottom: 30px; }}
            .chart-box {{ text-align: center; margin-bottom: 30px; border: 1px solid #eee; padding: 10px; border-radius: 8px; }}
            .chart-box img {{ max-width: 100%; height: auto; }}
            .row-split {{ display: flex; gap: 30px; }}
            .col {{ flex: 1; }}
            table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
            th {{ background-color: #34495e; color: white; padding: 10px; text-align: left; }}
            td {{ border-bottom: 1px solid #eee; padding: 8px; vertical-align: top; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .badge {{ background: #e67e22; color: white; padding: 3px 8px; border-radius: 10px; font-size: 11px; font-weight: bold; }}
            .footer {{ margin-top: 50px; text-align: center; font-size: 11px; color: #aaa; border-top: 1px solid #eee; padding-top: 20px; }}

            /* ===== PDF/PRINT (Chromium) ===== */
            @page {{ size: A4; margin: 10mm; }}
            @media print {{
              body {{ background: #fff !important; padding: 0 !important; }}
              .container {{
                max-width: none !important;
                margin: 0 !important;
                padding: 0 !important;
                box-shadow: none !important;
                border-radius: 0 !important;
              }}
              .row-split {{ display: block !important; }}
              .col {{ width: 100% !important; }}
              /* evita quebras feias em cards/tabelas */
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
                    <div style="font-size:14px; color:#666; margin-top:5px;">Eficiência Energética de Frota</div>
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
                    <p style="font-size:12px; color:#666;">Comparativo Mês Atual vs Anterior.</p>
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
            <p style="font-size:12px; color:#666;">
                Veículos abaixo apresentam leituras de KM/L fora da faixa aceitável (kml &lt; 1,5 ou kml &gt; 5),
                úteis para auditoria de apontamentos, erros de sistema ou comportamentos extremos.
            </p>
            <table>
                <thead>
                    <tr>
                        <th>Veículo</th>
                        <th>Cluster</th>
                        <th>Linha</th>
                        <th>Qtd. Contaminações</th>
                        <th>KML Mínimo</th>
                        <th>KML Máximo</th>
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
    """
    PDF fiel ao HTML usando Chromium (Playwright).
    Requisito: playwright instalado + chromium instalado no build.
    """
    html_path = html_path.resolve()
    pdf_path = pdf_path.resolve()

    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1280, "height": 720})

        # carrega o arquivo local e resolve o PNG no mesmo diretório
        page.goto(html_path.as_uri(), wait_until="networkidle")
        page.wait_for_timeout(300)

        page.pdf(
            path=str(pdf_path),
            format="A4",
            print_background=True,
            margin={"top": "10mm", "right": "10mm", "bottom": "10mm", "left": "10mm"},
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

    # Se nenhum período vier, usa: mês atual até hoje
    if not periodo_inicio and not periodo_fim:
        hoje = datetime.utcnow().date()
        periodo_inicio = hoje.replace(day=1)
        periodo_fim = hoje

    atualizar_status_relatorio(
        "PROCESSANDO",
        periodo_inicio=str(periodo_inicio) if periodo_inicio else None,
        periodo_fim=str(periodo_fim) if periodo_fim else None,
    )

    try:
        # 1) Busca Supabase A
        df_base = carregar_dados_supabase_a(periodo_inicio, periodo_fim)

        # 2) Processa
        dados = processar_dados_gerenciais_df(df_base)

        # 3) Gera gráfico + IA + HTML + PDF
        out_dir = Path(PASTA_SAIDA)
        img_path = out_dir / "cluster_evolution_unificado.png"
        html_path = out_dir / "Relatorio_Gerencial.html"
        pdf_path = out_dir / "Relatorio_Gerencial.pdf"

        gerar_grafico_geral(dados["df_clean"], img_path)
        texto_ia = consultar_ia_gerencial(dados)
        gerar_html_gerencial(dados, texto_ia, img_path, html_path)
        gerar_pdf_do_html(html_path, pdf_path)

        # 4) Upload para Supabase B (bucket relatorios) — HTML + PNG + PDF
        mes_ref = str(dados["df_clean"]["Date"].max().to_period("M"))  # ex: 2026-01
        base_folder = f"diesel/{mes_ref}/report_{REPORT_ID}"

        remote_img = f"{base_folder}/{img_path.name}"
        remote_html = f"{base_folder}/{html_path.name}"
        remote_pdf = f"{base_folder}/{pdf_path.name}"

        upload_storage_b(img_path, remote_img, "image/png")
        upload_storage_b(html_path, remote_html, "text/html; charset=utf-8")
        size_pdf = upload_storage_b(pdf_path, remote_pdf, "application/pdf")

        # 5) Atualiza controle no Supabase B (PDF como principal)
        atualizar_status_relatorio(
            "CONCLUIDO",
            arquivo_path=remote_pdf,
            arquivo_nome=_safe_filename(f"Relatorio_Gerencial_{mes_ref}.pdf"),
            mime_type="application/pdf",
            tamanho_bytes=size_pdf,
            erro_msg=None,
        )

        print("✅ [OK] Relatório concluído e gravado no Supabase B (PDF + HTML + PNG).")

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
