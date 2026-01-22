# prontuarios_acompanhamento.py
import os
import re
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

# Matplotlib headless (evita erro em servidor)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Vertex AI
import vertexai
from vertexai.generative_models import GenerativeModel

# Supabase
from supabase import create_client

# ==============================================================================
# CONFIG (ENV FIRST)
# ==============================================================================

# Vertex
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID") or os.getenv("PROJECT_ID")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
VERTEX_MODEL = os.getenv("VERTEX_MODEL", "gemini-2.5-pro")  # fallback: gemini-2.5-flash

# Supabase A (origem)
SUPABASE_A_URL = os.getenv("SUPABASE_A_URL")
SUPABASE_A_SERVICE_ROLE_KEY = os.getenv("SUPABASE_A_SERVICE_ROLE_KEY")
TABELA_ORIGEM = os.getenv("DIESEL_SOURCE_TABLE", "premiacao_diaria")

# Supabase B (destino - storage)
SUPABASE_B_URL = os.getenv("SUPABASE_B_URL")
SUPABASE_B_SERVICE_ROLE_KEY = os.getenv("SUPABASE_B_SERVICE_ROLE_KEY")
BUCKET_RELATORIOS = os.getenv("REPORT_BUCKET", "relatorios")

# Período (opcional)
REPORT_PERIODO_INICIO = os.getenv("REPORT_PERIODO_INICIO")  # YYYY-MM-DD
REPORT_PERIODO_FIM = os.getenv("REPORT_PERIODO_FIM")        # YYYY-MM-DD

# Identificadores de execução
RUN_ID = os.getenv("RUN_ID") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
REPORT_ID = os.getenv("REPORT_ID")  # opcional (para gerencial, se você quiser)

# Saída local
PASTA_SAIDA = os.getenv("REPORT_OUTPUT_DIR", "Relatorios_Diesel_Final")

# Paginação Supabase A
PAGE_SIZE = int(os.getenv("REPORT_PAGE_SIZE", "1000"))     # Supabase costuma truncar por página
MAX_ROWS = int(os.getenv("REPORT_MAX_ROWS", "250000"))     # segurança


# ==============================================================================
# Helpers
# ==============================================================================

def _assert_env_minimo():
    missing = []
    if not VERTEX_PROJECT_ID:
        missing.append("VERTEX_PROJECT_ID (ou PROJECT_ID)")
    if not SUPABASE_A_URL:
        missing.append("SUPABASE_A_URL")
    if not SUPABASE_A_SERVICE_ROLE_KEY:
        missing.append("SUPABASE_A_SERVICE_ROLE_KEY")
    if not SUPABASE_B_URL:
        missing.append("SUPABASE_B_URL")
    if not SUPABASE_B_SERVICE_ROLE_KEY:
        missing.append("SUPABASE_B_SERVICE_ROLE_KEY")
    if missing:
        raise RuntimeError(f"Variáveis obrigatórias ausentes: {missing}")

def _parse_iso(d: str | None) -> Optional[date]:
    if not d:
        return None
    return datetime.strptime(d, "%Y-%m-%d").date()

def _sb_a():
    return create_client(SUPABASE_A_URL, SUPABASE_A_SERVICE_ROLE_KEY)

def _sb_b():
    return create_client(SUPABASE_B_URL, SUPABASE_B_SERVICE_ROLE_KEY)

def _safe_filename(name: str) -> str:
    name = str(name or "").strip()
    name = re.sub(r"[^\w\-.() ]+", "_", name, flags=re.UNICODE)
    name = re.sub(r"\s+", "_", name)
    return name[:180] if name else "arquivo"

def upload_storage_b(local_path: Path, remote_path: str, content_type: str) -> int:
    """
    Upload no Supabase B Storage com upsert (evita falhar se rodar de novo).
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


# ==============================================================================
# 0) BUSCA DADOS SUPABASE A -> DF padrão interno
# ==============================================================================

def carregar_dados_supabase_a(periodo_inicio: Optional[date], periodo_fim: Optional[date]) -> pd.DataFrame:
    sb = _sb_a()

    # Usa um ID estável para paginação/ordenação.
    # Ajuste se seu PK tiver outro nome.
    # Você já usa id_premiacao_diaria no gerencial — aqui mantive igual.
    select_cols = (
        'id_premiacao_diaria, dia, motorista, veiculo, linha, '
        'km_rodado, combustivel_consumido, minutos_em_viagem, "km/l"'
    )

    base_q = sb.table(TABELA_ORIGEM).select(select_cols)

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

    # Normaliza "km/l" -> kml
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
# 1) PROCESSAMENTO – BASE PARA PRONTUÁRIOS (agora recebe DF, não CSV)
# ==============================================================================

def processar_dados_prontuarios_df(df: pd.DataFrame):
    print("⚙️ [Sistema] Processando dados para prontuários...")

    obrig = ["Date", "Motorista", "veiculo", "linha", "kml", "Km", "Comb."]
    falt = [c for c in obrig if c not in df.columns]
    if falt:
        raise ValueError(f"Colunas obrigatórias ausentes: {falt} | colunas atuais: {df.columns.tolist()}")

    # Tipos e datas
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    for col in ["kml", "Km", "Comb."]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Cluster
    df["Cluster"] = df["veiculo"].apply(definir_cluster)
    df = df.dropna(subset=["Cluster"])

    # Limpeza (1.5 <= kml <= 5)
    df_clean = df[(df["kml"] >= 1.5) & (df["kml"] <= 5)].copy()
    if df_clean.empty:
        raise ValueError("Sem dados válidos após filtros (kml 1.5~5 e cluster válido).")

    # Período total (texto)
    data_ini = df_clean["Date"].min().strftime("%d/%m/%Y")
    data_fim = df_clean["Date"].max().strftime("%d/%m/%Y")
    periodo_txt = f"{data_ini} a {data_fim}"

    df_clean["Mes_Ano"] = df_clean["Date"].dt.to_period("M")
    ultimo_mes = df_clean["Mes_Ano"].max()

    # Trabalha com mês atual (último mês da base limpa)
    df_atual = df_clean[df_clean["Mes_Ano"] == ultimo_mes].copy()

    # Meta da linha (KML_Ref) + KM total linha
    ref_grupo = df_atual.groupby(["linha", "Cluster"]).agg({"Km": "sum", "Comb.": "sum"}).reset_index()
    ref_grupo["KML_Ref"] = ref_grupo["Km"] / ref_grupo["Comb."]
    ref_grupo.rename(columns={"Km": "KM_Total_Linha"}, inplace=True)

    df_atual = pd.merge(
        df_atual,
        ref_grupo[["linha", "Cluster", "KML_Ref", "KM_Total_Linha"]],
        on=["linha", "Cluster"],
        how="left",
    )

    # Desperdício
    def calc_desperdicio(r):
        try:
            if r["KML_Ref"] > 0 and r["kml"] < r["KML_Ref"]:
                return r["Comb."] - (r["Km"] / r["KML_Ref"])
            return 0
        except Exception:
            return 0

    df_atual["Litros_Desperdicio"] = df_atual.apply(calc_desperdicio, axis=1)

    # Impacto do motorista na linha (%)
    impacto_mot = (
        df_atual.groupby(["Motorista", "linha", "Cluster", "KM_Total_Linha"])
        .agg({"Km": "sum"})
        .reset_index()
        .rename(columns={"Km": "KM_Motorista_Linha"})
    )
    impacto_mot["Impacto_Pct_Real"] = (impacto_mot["KM_Motorista_Linha"] / impacto_mot["KM_Total_Linha"]) * 100

    df_atual = pd.merge(
        df_atual,
        impacto_mot[["Motorista", "linha", "Cluster", "Impacto_Pct_Real"]],
        on=["Motorista", "linha", "Cluster"],
        how="left",
    )

    # Melhor KML por linha/cluster (Máx)
    max_grupo = df_atual.groupby(["linha", "Cluster"])["kml"].max().reset_index()
    max_grupo.rename(columns={"kml": "KML_Max_Possivel"}, inplace=True)
    df_atual = pd.merge(df_atual, max_grupo, on=["linha", "Cluster"], how="left")

    # Estatísticas por motorista (ranking)
    stats_mot = (
        df_atual.groupby("Motorista")
        .agg(
            Litros_Desperdicio=("Litros_Desperdicio", "sum"),
            KML_Atual=("kml", "mean"),
            KML_Ref=("KML_Ref", "mean"),
            Cluster=("Cluster", lambda x: x.mode()[0] if not x.mode().empty else ""),
            linha=("linha", lambda x: ", ".join(sorted(set(map(str, x))))),
            Impacto_Pct_Real=("Impacto_Pct_Real", "max"),
        )
        .reset_index()
    )

    # Mantive sua referência fictícia (+5%), mas se quiser, removemos depois
    stats_mot["KML_Anterior"] = stats_mot["KML_Atual"] * 1.05
    stats_mot["Queda"] = ((stats_mot["KML_Anterior"] - stats_mot["KML_Atual"]) / stats_mot["KML_Anterior"]) * 100
    stats_mot["Impacto_Media"] = stats_mot["KML_Atual"] - stats_mot["KML_Ref"]

    piores_12 = stats_mot.sort_values("Litros_Desperdicio", ascending=False).head(12)

    # Linha foco (maior desperdício por motorista)
    stats_mot_agrupado = (
        df_atual.groupby(["Motorista", "Cluster", "linha"])
        .agg(Km=("Km", "sum"), Comb=("Comb.", "sum"), Litros_Desperdicio=("Litros_Desperdicio", "sum"),
             KML_Ref=("KML_Ref", "mean"), Impacto_Pct_Real=("Impacto_Pct_Real", "max"))
        .reset_index()
    )
    stats_mot_agrupado["KML_Consolidado"] = stats_mot_agrupado["Km"] / stats_mot_agrupado["Comb"]
    stats_mot_agrupado["Gap_vs_Ref"] = stats_mot_agrupado["KML_Consolidado"] - stats_mot_agrupado["KML_Ref"]

    idx_max = stats_mot_agrupado.groupby("Motorista")["Litros_Desperdicio"].idxmax()
    stats_mot_foco = stats_mot_agrupado.loc[idx_max].copy()
    stats_mot_foco.rename(columns={"linha": "Linha_Foco", "KML_Consolidado": "KML_Real_Foco"}, inplace=True)

    piores_12 = pd.merge(
        piores_12[["Motorista", "Litros_Desperdicio"]],
        stats_mot_foco[["Motorista", "Linha_Foco", "KML_Real_Foco", "KML_Ref",
                       "Impacto_Pct_Real", "Gap_vs_Ref", "Cluster"]],
        on="Motorista",
        how="left",
    )
    piores_12.rename(columns={"Cluster": "Cluster_Foco"}, inplace=True)
    piores_12.sort_values("Litros_Desperdicio", ascending=False, inplace=True)

    return {
        "df_atual": df_atual,
        "piores_12": piores_12,
        "periodo": periodo_txt,
        "mes_ref": str(ultimo_mes),  # ex: "2026-01"
    }


# ==============================================================================
# 2) GRÁFICO DO MOTORISTA
# ==============================================================================

def gerar_grafico(df_base, mot_id, caminho: Path):
    df_mot = df_base[df_base["Motorista"] == mot_id].copy()
    if df_mot.empty:
        return False

    df_mot["Semana"] = df_mot["Date"].dt.to_period("W").apply(lambda r: r.start_time)
    weekly = df_mot.groupby("Semana").agg({"Km": "sum", "Comb.": "sum"}).reset_index()
    weekly["KML"] = weekly["Km"] / weekly["Comb."]

    meta_df = df_mot.groupby("Semana").agg({"KML_Ref": "mean"}).reset_index()
    dados = pd.merge(weekly, meta_df, on="Semana", how="inner")
    if dados.empty:
        return False

    x_labels = pd.to_datetime(dados["Semana"]).dt.strftime("%d/%m")

    plt.figure(figsize=(10, 4))
    plt.plot(x_labels, dados["KML"], marker="o", linewidth=3, label="Realizado")
    plt.plot(x_labels, dados["KML_Ref"], linestyle="--", linewidth=2, label="Meta (Ref)")

    for x, y in zip(x_labels, dados["KML"]):
        try:
            plt.text(x, y + 0.05, f"{y:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        except Exception:
            pass

    plt.title(f"Performance Semanal: {mot_id}", fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(caminho, dpi=110)
    plt.close()
    return True


# ==============================================================================
# 3) TABELA RAIO-X EM HTML
# ==============================================================================

def gerar_tabela_raiox_html(df_atual, mot_id):
    df_mot = df_atual[df_atual["Motorista"] == mot_id].copy()
    if df_mot.empty:
        return "<div style='font-size:12px;color:#777'>Sem dados para este motorista no mês de referência.</div>"

    df_mot["Mes"] = df_mot["Mes_Ano"].astype(str)

    resumo = (
        df_mot.groupby(["Mes", "veiculo", "linha", "Cluster", "KML_Ref", "KML_Max_Possivel"])
        .agg(Km=("Km", "sum"), Comb=("Comb.", "sum"), Litros_Desperdicio=("Litros_Desperdicio", "sum"),
             Impacto_Pct_Real=("Impacto_Pct_Real", "max"))
        .reset_index()
    )

    resumo["KML_Real"] = resumo["Km"] / resumo["Comb"]
    resumo = resumo.sort_values("Litros_Desperdicio", ascending=False)

    html_rows = ""
    for idx, row in resumo.iterrows():
        style = ""
        if idx == resumo.index[0] and row["Litros_Desperdicio"] > 0:
            style = "background-color: #ffebee; color: #c62828; font-weight:bold;"

        html_rows += f"""
        <tr style="{style}">
            <td align="center">{row['Mes']}</td>
            <td align="center">{row['veiculo']}</td>
            <td align="center">{row['linha']}</td>
            <td align="center">{row['Cluster']}</td>
            <td align="center">{row['Km']:.0f}</td>
            <td align="center">{row['Comb']:.0f}</td>
            <td align="center">{row['KML_Real']:.2f}</td>
            <td align="center">{row['KML_Ref']:.2f}</td>
            <td align="center" style="color:#27ae60;">{row['KML_Max_Possivel']:.2f}</td>
            <td align="center" style="color:#2980b9;">{row['Impacto_Pct_Real']:.1f}%</td>
            <td align="center">{row['Litros_Desperdicio']:.1f}</td>
        </tr>"""

    return f"""
    <table style="width:100%; border-collapse: collapse; font-size: 10px; margin-top: 10px; border: 1px solid #ddd;">
        <thead>
            <tr style="background-color: #34495e; color: white;">
                <th style="padding: 6px; text-align:center;">Mês</th>
                <th style="padding: 6px; text-align:center;">Veículo</th>
                <th style="padding: 6px; text-align:center;">Linha</th>
                <th style="padding: 6px; text-align:center;">Cluster</th>
                <th style="padding: 6px; text-align:center;">KM Tot</th>
                <th style="padding: 6px; text-align:center;">Litros</th>
                <th style="padding: 6px; text-align:center;">Real</th>
                <th style="padding: 6px; text-align:center;">Ref</th>
                <th style="padding: 6px; text-align:center;">Máx</th>
                <th style="padding: 6px; text-align:center;">% Impacto</th>
                <th style="padding: 6px; text-align:center;">Perda (L)</th>
            </tr>
        </thead>
        <tbody>{html_rows}</tbody>
    </table>
    <div style="font-size:9px; color:#777; margin-top:3px; font-style:italic;">
        * <b>% Impacto</b> = % de KM rodado pelo motorista em relação ao total da linha.<br>
        * <b>Máx</b> = Melhor KML atingido nesta linha.
    </div>
    """


# ==============================================================================
# 4) IA INDIVIDUAL (Vertex)
# ==============================================================================

def consultar_vertex_individual(dados_row: dict):
    print("🧠 [Instrutor] Chamando Vertex AI para análise individual...")
    try:
        model = GenerativeModel(VERTEX_MODEL)

        tec_map = {
            "C11": "Padron VW 17.230 (Motor Dianteiro, Automático V-Tronic. Erro comum: pé fundo/kickdown).",
            "C10": "Padron MB 1721 Euro VI (Motor Dianteiro).",
            "C9":  "Padron MB 1721 (Motor Dianteiro).",
            "C8":  "Padron MB 1721 Suspensão a Ar (Motor Dianteiro).",
            "C6":  "Convencional MB 1721 (Motor Dianteiro Manual).",
        }
        tec_info = tec_map.get(dados_row.get("Cluster_Foco"), "Diesel Padrão")

        prompt = f"""
Aja como Instrutor de Treinamento de Motoristas.
ALUNO: {dados_row.get('Motorista')} | EQUIPAMENTO: {dados_row.get('Cluster_Foco')} -> {tec_info}.
DESEMPENHO: {float(dados_row.get('KML_Real_Foco') or 0):.2f} km/l (Abaixo da meta {float(dados_row.get('KML_Ref') or 0):.2f} km/l).

Considere que o veículo está em boas condições mecânicas.
O foco da correção é o MODO DE CONDUÇÃO.

ESCREVA 3 PARÁGRAFOS CURTOS (sem usar *, # ou markdown):

ANÁLISE:
- Explique, de forma técnica, qual é o erro de condução mais provável.
- Se o veículo for automático (C11 - V-Tronic), comente sobre pé fundo, kickdown e uso do acelerador.
- Se for manual, comente sobre faixas de RPM, antecipação de marcha e uso de inércia.

ROTEIRO:
- Monte um checklist prático de correções (1., 2., 3., ...), foco em melhorar KM/L nessa linha de foco.

FEEDBACK:
- Mensagem curta, educativa, mostrando o impacto da melhora de KM/L para a empresa e para a imagem do motorista.

Formato OBRIGATÓRIO:
ANÁLISE: [texto]
ROTEIRO: [texto]
FEEDBACK: [texto]
""".strip()

        response = model.generate_content(prompt)
        texto = getattr(response, "text", None) or ""
        if not texto:
            texto = "ANÁLISE: Indisponível.\nROTEIRO: Padrão.\nFEEDBACK: Genérico."

        return texto.replace("**", "").replace("*", "").replace("#", "")

    except Exception as e:
        print("❌ Erro ao chamar Vertex AI (Individual):", repr(e))
        return "ANÁLISE: Indisponível.\nROTEIRO: Padrão.\nFEEDBACK: Genérico."


# ==============================================================================
# 5) HTML INDIVIDUAL
# ==============================================================================

def gerar_html_individual(dados_row: dict, texto_ia: str, img_basename: str, tabela_raiox: str, periodo: str):
    analise, roteiro, feedback = "...", "...", "..."

    if "ANÁLISE:" in texto_ia:
        parts = texto_ia.split("ROTEIRO:")
        analise = parts[0].replace("ANÁLISE:", "").strip()
        if len(parts) > 1:
            subparts = parts[1].split("FEEDBACK:")
            roteiro = subparts[0].strip()
            if len(subparts) > 1:
                feedback = subparts[1].strip()

    kml_real = float(dados_row.get("KML_Real_Foco") or 0)
    kml_ref = float(dados_row.get("KML_Ref") or 0)
    gap = float(dados_row.get("Gap_vs_Ref") or 0)
    impacto_pct = float(dados_row.get("Impacto_Pct_Real") or 0)
    cluster_foco = str(dados_row.get("Cluster_Foco") or "")

    cor_impacto = "#c0392b" if gap < 0 else "#27ae60"
    txt_impacto = f"{gap:.2f} km/l"

    html = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <style>
    body {{ font-family: 'Segoe UI', sans-serif; padding: 20px; background: #ecf0f1; }}
    .page {{ background: white; max-width: 900px; margin: auto; padding: 30px;
             border-top: 8px solid #2980b9; box-shadow: 0 5px 15px rgba(0,0,0,0.1);
             border-radius: 5px; }}
    h1 {{ margin: 0; color: #2c3e50; font-size: 24px; }}
    .header-sub {{ color: #7f8c8d; font-size: 12px; margin-top: 5px; }}
    .tag {{ background: #c0392b; color: white; padding: 4px 12px; font-weight: bold;
            border-radius: 20px; font-size: 11px; }}
    .stats-container {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 20px 0; }}
    .stat-card {{ background: #f8f9fa; padding: 10px; border-radius: 6px; text-align: center; border: 1px solid #e0e0e0; }}
    .stat-val {{ display: block; font-size: 18px; font-weight: bold; color: #2c3e50; }}
    .stat-lbl {{ font-size: 10px; text-transform: uppercase; color: #95a5a6; }}
    h2 {{ color: #2980b9; font-size: 14px; border-bottom: 2px solid #eee; margin-top: 30px;
          padding-bottom: 5px; text-transform: uppercase; font-weight: bold; }}
    .box {{ background: #fff; border-left: 4px solid #bdc3c7; padding: 12px; margin-top: 10px;
            font-size: 13px; line-height: 1.5; color: #444; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
    .footer {{ margin-top: 40px; text-align: center; font-size: 10px; color: #aaa; border-top: 1px solid #eee; padding-top: 15px; }}
  </style>
</head>
<body>
  <div class="page">
    <div style="display:flex; justify-content:space-between; align-items:center;">
      <div>
        <h1>PRONTUÁRIO: {dados_row.get('Motorista','')}</h1>
        <div class="header-sub">Análise Técnica | Período: {periodo}</div>
      </div>
      <div class="tag">PRIORIDADE ALTA</div>
    </div>

    <div class="stats-container">
      <div class="stat-card">
        <span class="stat-val">{cluster_foco}</span>
        <span class="stat-lbl">Cluster de Foco</span>
      </div>
      <div class="stat-card">
        <span class="stat-val" style="color:#2980b9">{impacto_pct:.1f}%</span>
        <span class="stat-lbl">Participação na Linha Foco</span>
      </div>
      <div class="stat-card">
        <span class="stat-val" style="color:{cor_impacto}">{txt_impacto}</span>
        <span class="stat-lbl">Gap vs Média ({kml_ref:.2f})</span>
      </div>
      <div class="stat-card">
        <span class="stat-val">{kml_real:.2f}</span>
        <span class="stat-lbl">KM/L Real na Linha Foco</span>
      </div>
    </div>

    <img src="{img_basename}" style="width:100%; height:220px; object-fit:contain; border:1px solid #eee; border-radius:8px;">

    <h2>1. Raio-X da Operação (Detalhe por Veículo)</h2>
    {tabela_raiox}

    <h2>2. Análise de Condução</h2>
    <div class="box" style="border-color: #e67e22;">{analise}</div>

    <h2>3. Foco da Monitoria</h2>
    <div class="box" style="border-color: #3498db;">{roteiro}</div>

    <h2>4. Feedback Educativo</h2>
    <div class="box" style="border-color: #27ae60;">{feedback}</div>

    <div class="footer">Gerado automaticamente pelo Agente Diesel AI.</div>
  </div>
</body>
</html>
""".strip()

    return html


# ==============================================================================
# ACOMPANHAMENTO: Gera prontuários + upload
# ==============================================================================

def gerar_prontuarios_e_upload(df_atual: pd.DataFrame, piores_12: pd.DataFrame, periodo: str, mes_ref: str):
    out_dir = Path(PASTA_SAIDA)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_folder = f"acompanhamento/{mes_ref}/batch_{RUN_ID}"
    manifest = []

    count = 1
    for _, row in piores_12.iterrows():
        mot = str(row.get("Motorista") or "").strip()
        mot_safe = _safe_filename(mot)

        print(f"  [{count}/12] Gerando {mot}...")

        img_name = f"{mot_safe}_grafico.png"
        html_name = f"{mot_safe}_Prontuario.html"

        img_path = out_dir / img_name
        html_path = out_dir / html_name

        ok = gerar_grafico(df_atual, mot, img_path)
        tabela_raiox = gerar_tabela_raiox_html(df_atual, mot)
        texto_ia = consultar_vertex_individual(row.to_dict())
        html = gerar_html_individual(row.to_dict(), texto_ia, img_name, tabela_raiox, periodo)

        html_path.write_text(html, encoding="utf-8")

        # Upload (mesma pasta do HTML e PNG)
        remote_img = f"{base_folder}/{img_name}"
        remote_html = f"{base_folder}/{html_name}"

        if ok and img_path.exists():
            upload_storage_b(img_path, remote_img, "image/png")
        upload_storage_b(html_path, remote_html, "text/html; charset=utf-8")

        manifest.append({
            "motorista": mot,
            "html": remote_html,
            "png": remote_img if ok else None,
        })

        time.sleep(1)
        count += 1

    # Manifest (útil para listar no front)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(pd.Series(manifest).to_json(orient="values", force_ascii=False), encoding="utf-8")
    remote_manifest = f"{base_folder}/manifest.json"
    upload_storage_b(manifest_path, remote_manifest, "application/json; charset=utf-8")

    print(f"\n✅ ACOMPANHAMENTO OK. Upload em: {base_folder}/")


# ==============================================================================
# GERENCIAL (gancho) – separa em gerencial/...
# Você já tem o relatorio_gerencial.py; aqui é só um padrão de pasta.
# ==============================================================================

def pasta_gerencial(mes_ref: str) -> str:
    rid = REPORT_ID or RUN_ID
    return f"gerencial/{mes_ref}/report_{rid}"


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    _assert_env_minimo()

    periodo_inicio = _parse_iso(REPORT_PERIODO_INICIO)
    periodo_fim = _parse_iso(REPORT_PERIODO_FIM)

    # Se nenhum período vier, usa mês atual até hoje
    if not periodo_inicio and not periodo_fim:
        hoje = datetime.utcnow().date()
        periodo_inicio = hoje.replace(day=1)
        periodo_fim = hoje

    # Inicializa Vertex (ADC / service account via GOOGLE_APPLICATION_CREDENTIALS)
    vertexai.init(project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION)

    # 1) Carrega base do Supabase A
    df_base = carregar_dados_supabase_a(periodo_inicio, periodo_fim)

    # 2) Processa para prontuários
    dados = processar_dados_prontuarios_df(df_base)

    # 3) Gera + upload em acompanhamento/
    gerar_prontuarios_e_upload(
        df_atual=dados["df_atual"],
        piores_12=dados["piores_12"],
        periodo=dados["periodo"],
        mes_ref=dados["mes_ref"],
    )

    # 4) Gerencial: apenas informativo (você pluga seu relatorio_gerencial.py)
    print(f"ℹ️ Pasta sugerida para GERENCIAL (não gerado aqui): {pasta_gerencial(dados['mes_ref'])}/")


if __name__ == "__main__":
    main()
