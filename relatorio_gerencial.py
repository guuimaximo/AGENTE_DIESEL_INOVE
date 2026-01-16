import os
import io
import re
import json
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from xhtml2pdf import pisa

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
        file_options={"content-type": content_type, "upsert": True},
    )
    return len(data)


# ==============================================================================
# 0) BUSCA DADOS SUPABASE A  -> DF no formato do seu "CSV"
# ==============================================================================
def carregar_dados_supabase_a(periodo_inicio: date | None, periodo_fim: date | None) -> pd.DataFrame:
    """
    Origem: Supabase A / premiacao_diaria
    Mapeamento:
      dia_date -> Date
      motorista -> Motorista
      veiculo -> veiculo
      linha -> linha
      km_rodado -> Km
      combustivel_consumido -> Comb.
      "km/l" -> kml
    """
    sb = _sb_a()

    q = sb.table(TABELA_ORIGEM).select('dia_date, motorista, veiculo, linha, km_rodado, combustivel_consumido, minutos_em_viagem, "km/l"')

    if periodo_inicio:
        q = q.gte("dia_date", str(periodo_inicio))
    if periodo_fim:
        q = q.lte("dia_date", str(periodo_fim))

    q = q.limit(100000)

    resp = q.execute()
    rows = resp.data or []
    if not rows:
        return pd.DataFrame(columns=["Date", "Motorista", "veiculo", "linha", "kml", "Km", "Comb."])

    df = pd.DataFrame(rows)

    if "km/l" in df.columns and "kml" not in df.columns:
        df["kml"] = df["km/l"]

    out = pd.DataFrame()
    out["Date"] = df.get("dia_date")
    out["Motorista"] = df.get("motorista")
    out["veiculo"] = df.get("veiculo")
    out["linha"] = df.get("linha")
    out["kml"] = df.get("kml")
    out["Km"] = df.get("km_rodado")
    out["Comb."] = df.get("combustivel_consumido")

    return out


# ==============================================================================
# 1) PROCESSAMENTO
# ==============================================================================
def processar_dados_gerenciais_df(df: pd.DataFrame):
    print("⚙️  [Sistema] Processando dados para visão gerencial...")

    obrigatorias = ["Date", "Motorista", "veiculo", "linha", "kml", "Km", "Comb."]
    faltando = [c for c in obrigatorias if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas obrigatórias ausentes: {faltando}. Colunas atuais: {df.columns.tolist()}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    for col in ["kml", "Km", "Comb."]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

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
        "january": "JANEIRO", "february": "FEVEREIRO", "march": "MARÇO",
        "april": "ABRIL", "may": "MAIO", "june": "JUNHO",
        "july": "JULHO", "august": "AGOSTO", "september": "SETEMBRO",
        "october": "OUTUBRO", "november": "NOVEMBRO", "december": "DEZEMBRO",
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
    cores = {"C11": "#e67e22", "C10": "#2ecc71", "C9": "#3498db", "C8": "#9b59b6", "C6": "#95a5a6"}

    for cluster in pivot_chart.columns:
        plt.plot(
            pivot_chart.index, pivot_chart[cluster],
            marker=".", linewidth=1.5, label=cluster,
            color=cores.get(cluster, "gray"), alpha=0.7,
        )

    plt.plot(
        evolucao_geral["Semana"], evolucao_geral["KML"],
        marker="o", linewidth=3, label="MÉDIA FROTA", color="black",
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

        kml_mes_atual = mensal.iloc[-1]["KML"] if not mensal.empty else 0
        mes_atual_label = str(mensal.iloc[-1]["Mes_Ano"]) if not mensal.empty else "N/D"

        if len(mensal) >= 2:
            kml_mes_anterior = mensal.iloc[-2]["KML"]
            mes_ant_label = str(mensal.iloc[-2]["Mes_Ano"])
            delta_kml_mes = ((kml_mes_atual - kml_mes_anterior) / kml_mes_anterior * 100) if kml_mes_anterior > 0 else 0
        else:
            kml_mes_anterior, mes_ant_label, delta_kml_mes = 0, "N/D", 0

        df_clean["Semana"] = df_clean["Date"].dt.to_period("W").apply(lambda r: r.start_time)
        semanal = (
            df_clean.groupby("Semana")
            .agg({"Km": "sum", "Comb.": "sum"})
            .reset_index()
        )
        semanal["KML"] = semanal["Km"] / semanal["Comb."]
        semanal = semanal.sort_values("Semana")

        kml_semana_atual = semanal.iloc[-1]["KML"] if not semanal.empty else 0
        semana_atual_inicio_txt = semanal.iloc[-1]["Semana"].strftime("%d/%m/%Y") if not semanal.empty else "N/D"

        if len(semanal) >= 2:
            kml_semana_anterior = semanal.iloc[-2]["KML"]
            semana_ant_inicio_txt = semanal.iloc[-2]["Semana"].strftime("%d/%m/%Y")
            delta_kml_semana = ((kml_semana_atual - kml_semana_anterior) / kml_semana_anterior * 100) if kml_semana_anterior > 0 else 0
        else:
            kml_semana_anterior, semana_ant_inicio_txt, delta_kml_semana = 0, "N/D", 0

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

        CONTEXTO POR CLUSTER:
        {dados_proc['tabela_pivot'].to_markdown()}

        TOP ALVOS DO MÊS ATUAL:
        - Veículos: {dados_proc['top_veiculos'].to_markdown()}
        - Linhas: {dados_proc['top_linhas_queda'].to_markdown()}
        - Motoristas: {dados_proc['top_motoristas'].to_markdown()}

        ESCREVA UM RESUMO EXECUTIVO EM HTML (sem markdown, sem ```). Use apenas: <p>, <b>, <br>, <ul>, <li>.
        1) <b>Visão Geral da Eficiência no Período</b>
        2) <b>Zoom na Última Semana</b>
        3) <b>Recomendações Práticas para o Próximo Ciclo</b>
        """

        resp = model.generate_content(prompt)
        texto = getattr(resp, "text", None) or "Análise indisponível."
        return texto.replace("```html", "").replace("```", "")

    except Exception as e:
        print("❌ Erro ao chamar a IA (Gerencial):", repr(e))
        return "<p>Análise indisponível (erro na chamada da IA).</p>"


# ==============================================================================
# 4) HTML + PDF
# ==============================================================================
def gerar_html_gerencial(dados: dict, texto_ia: str, img_path: Path, html_path: Path):
    def make_rows(df, cols, fmt_map):
        rows = ""
        if df is None or df.empty: return rows
        for _, row in df.iterrows():
            rows += "<tr>"
            for col in cols:
                val = row.get(col, "")
                fmt = fmt_map.get(col, "{}")
                val_str = fmt.format(val) if isinstance(val, (int, float)) else str(val)
                style = ""
                if col == "Variacao_Pct":
                    v = float(val)
                    style = "color: #c0392b; font-weight: bold;" if v < -5 else "color: #e67e22;" if v < 0 else "color: #27ae60;"
                    val_str = f"{v:.1f}%"
                rows += f"<td style='{style}'>{val_str}</td>"
            rows += "</tr>"
        return rows

    df_clean = dados["df_clean"].copy()
    if "Mes_Ano" not in df_clean.columns:
        df_clean["Mes_Ano"] = df_clean["Date"].dt.to_period("M")

    mensal = df_clean.groupby("Mes_Ano").agg({"Km": "sum", "Comb.": "sum"}).reset_index()
    mensal["KML"] = mensal["Km"] / mensal["Comb."]
    mensal = mensal.sort_values("Mes_Ano")

    kml_mes_atual = mensal.iloc[-1]["KML"] if not mensal.empty else 0
    delta_kml_mes = 0
    if len(mensal) >= 2:
        kml_mes_ant = mensal.iloc[-2]["KML"]
        delta_kml_mes = ((kml_mes_atual - kml_mes_ant) / kml_mes_ant * 100) if kml_mes_ant > 0 else 0

    if delta_kml_mes < 0: texto_var, cor_var = f"{abs(delta_kml_mes):.1f}% de QUEDA", "#c0392b"
    elif delta_kml_mes > 0: texto_var, cor_var = f"{delta_kml_mes:.1f}% de MELHORA", "#27ae60"
    else: texto_var, cor_var = "0,0% (Estável)", "#7f8c8d"

    rows_veic = make_rows(dados["top_veiculos"], ["veiculo", "Cluster", "linha", "KML_Real", "KML_Meta", "Litros_Desperdicio"], {"KML_Real": "{:.2f}", "KML_Meta": "{:.2f}", "Litros_Desperdicio": "{:.0f}"})
    rows_lin = make_rows(dados["top_linhas_queda"], ["linha", "KML_Anterior", "KML_Atual", "Variacao_Pct"], {"KML_Anterior": "{:.2f}", "KML_Atual": "{:.2f}", "Variacao_Pct": "{:.1f}"})
    rows_cont = make_rows(dados["top_veiculos_contaminados"], ["veiculo", "Cluster", "linha", "Qtd_Contaminacoes", "KML_Min", "KML_Max"], {"Qtd_Contaminacoes": "{:.0f}", "KML_Min": "{:.2f}", "KML_Max": "{:.2f}"})

    rows_mot = ""
    if dados["top_motoristas"] is not None and not dados["top_motoristas"].empty:
        for _, r in dados["top_motoristas"].iterrows():
            rows_mot += f"<tr><td>{r['Motorista']}</td><td>{r['Cluster']}</td><td>{r['linha']}</td><td><b>{r['KML_Real']:.2f}</b></td><td style='color:#777'>{r['KML_Meta']:.2f}</td><td><b>{r['Litros_Desperdicio']:.0f}</b></td><td><span class='badge'>{r['Impacto_Pct']:.1f}%</span></td></tr>"

    html = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background-color: #f0f2f5; margin: 0; padding: 20px; }}
            .container {{ max-width: 1000px; margin: auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
            .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 3px solid #2c3e50; padding-bottom: 20px; margin-bottom: 30px; }}
            .title h1 {{ margin: 0; color: #2c3e50; text-transform: uppercase; }}
            .month-card {{ background: #2c3e50; color: white; padding: 10px 20px; border-radius: 6px; text-align: center; }}
            .kpi-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 40px; }}
            .kpi-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; border: 1px solid #e0e0e0; }}
            .kpi-val {{ display: block; font-size: 28px; font-weight: bold; }}
            h2 {{ color: #2980b9; font-size: 18px; border-left: 5px solid #2980b9; padding-left: 10px; margin-top: 40px; }}
            .ai-box {{ background-color: #fffde7; border: 1px solid #fbc02d; padding: 20px; border-radius: 6px; line-height: 1.6; margin-bottom: 30px; }}
            .chart-box {{ text-align: center; margin-bottom: 30px; border: 1px solid #eee; padding: 10px; border-radius: 8px; }}
            .row-split {{ display: flex; gap: 30px; }}
            .col {{ flex: 1; }}
            table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
            th {{ background-color: #34495e; color: white; padding: 10px; text-align: left; }}
            td {{ border-bottom: 1px solid #eee; padding: 8px; }}
            .badge {{ background: #e67e22; color: white; padding: 3px 8px; border-radius: 10px; font-size: 11px; }}
            .footer {{ margin-top: 50px; text-align: center; font-size: 11px; color: #aaa; border-top: 1px solid #eee; padding-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="title"><h1>Relatório Gerencial</h1></div>
                <div class="month-card"><div>MÊS DE REFERÊNCIA</div><b>{dados['mes_atual_nome']}</b></div>
            </div>
            <div class="kpi-grid">
                <div class="kpi-card"><span class="kpi-val">{kml_mes_atual:.2f}</span><span>KM/L MÊS BASE</span></div>
                <div class="kpi-card"><span class="kpi-val" style="color:{cor_var}">{texto_var}</span><span>VARIAÇÃO VS ANTERIOR</span></div>
                <div class="kpi-card"><span class="kpi-val" style="color:#27ae60">AGENTE AI</span><span>MONITORAMENTO ATIVO</span></div>
            </div>
            <h2>1. Inteligência Executiva</h2><div class="ai-box">{texto_ia}</div>
            <h2>2. Evolução de Eficiência</h2><div class="chart-box"><img src="{img_path.name}"></div>
            <div class="row-split">
                <div class="col"><h2>3. Top 5 Veículos</h2><table><thead><tr><th>Veículo</th><th>Clust</th><th>Linha</th><th>Real</th><th>Ref</th><th>Perda</th></tr></thead><tbody>{rows_veic}</tbody></table></div>
                <div class="col"><h2>4. Top 5 Linhas em Queda</h2><table><thead><tr><th>Linha</th><th>Ant.</th><th>Atual</th><th>Var.</th></tr></thead><tbody>{rows_lin}</tbody></table></div>
            </div>
            <h2>5. Fator Humano (Top 5 Motoristas)</h2><table><thead><tr><th>Motorista</th><th>Clust</th><th>Linha</th><th>Real</th><th>Ref</th><th>Perda</th><th>Impacto</th></tr></thead><tbody>{rows_mot}</tbody></table>
            <h2>6. Auditoria de Dados</h2><table><thead><tr><th>Veículo</th><th>Clust</th><th>Linha</th><th>Qtd</th><th>Mín</th><th>Máx</th></tr></thead><tbody>{rows_cont}</tbody></table>
            <div class="footer">Gerado por Diesel AI. Período: {dados['periodo']}</div>
        </div>
    </body>
    </html>
    """
    html_path.write_text(html, encoding="utf-8")


def gerar_pdf_do_html(html_path: Path, pdf_path: Path):
    html = html_path.read_text(encoding="utf-8")
    with open(pdf_path, "wb") as f:
        pisa.CreatePDF(io.StringIO(html), dest=f)


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

    atualizar_status_relatorio("PROCESSANDO", periodo_inicio=str(periodo_inicio), periodo_fim=str(periodo_fim))

    try:
        df_base = carregar_dados_supabase_a(periodo_inicio, periodo_fim)
        dados = processar_dados_gerenciais_df(df_base)
        out_dir = Path(PASTA_SAIDA)
        img_path, html_path, pdf_path = out_dir / "cluster_evolution_unificado.png", out_dir / "Relatorio_Gerencial.html", out_dir / "Relatorio_Gerencial.pdf"

        gerar_grafico_geral(dados["df_clean"], img_path)
        texto_ia = consultar_ia_gerencial(dados)
        gerar_html_gerencial(dados, texto_ia, img_path, html_path)
        gerar_pdf_do_html(html_path, pdf_path)

        mes_ref = str(dados["df_clean"]["Date"].max().to_period("M"))
        base_folder = f"diesel/{mes_ref}/report_{REPORT_ID}"
        size_pdf = upload_storage_b(pdf_path, f"{base_folder}/{pdf_path.name}", "application/pdf")
        upload_storage_b(img_path, f"{base_folder}/{img_path.name}", "image/png")
        upload_storage_b(html_path, f"{base_folder}/{html_path.name}", "text/html; charset=utf-8")

        atualizar_status_relatorio("CONCLUIDO", arquivo_path=f"{base_folder}/{pdf_path.name}", arquivo_nome=_safe_filename(f"Relatorio_Gerencial_{mes_ref}.pdf"), mime_type="application/pdf", tamanho_bytes=size_pdf)
        print("✅ [OK] Relatório concluído.")

    except Exception as e:
        err = repr(e)
        print("❌ ERRO:", err)
        try: atualizar_status_relatorio("ERRO", erro_msg=err)
        except: pass
        raise

if __name__ == "__main__":
    main()
