# -*- coding: utf-8 -*-
import os
import re
import calendar
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from supabase import create_client
from playwright.sync_api import sync_playwright
from pypdf import PdfWriter, PdfReader


SUPABASE_A_URL = os.getenv("SUPABASE_A_URL")
SUPABASE_A_SERVICE_ROLE_KEY = os.getenv("SUPABASE_A_SERVICE_ROLE_KEY")

TABELA_ORIGEM = "premiacao_diaria_atualizada"
TABELA_FUNCIONARIOS = "funcionarios"
BUCKET = "parcial_meritocracia"

MES_REFERENCIA = os.getenv("MES_REFERENCIA")  # ex: 2026-03

PASTA_SAIDA = Path("Parcial_Meritocracia")
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)


def sb():
    return create_client(SUPABASE_A_URL, SUPABASE_A_SERVICE_ROLE_KEY)


def n(v):
    try:
        x = float(v)
        return x if x == x else 0.0
    except Exception:
        return 0.0


def _esc(s: str) -> str:
    s = "" if s is None else str(s)
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
         .replace("'", "&#39;")
    )


def _safe_filename(name: str) -> str:
    name = str(name or "").strip()
    name = re.sub(r"[^\w\-.() ]+", "_", name, flags=re.UNICODE)
    return name[:180] or "sem_nome"


def fmt_num(v, casas=2):
    return f"{n(v):,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_date_br(iso_date: str) -> str:
    return pd.to_datetime(iso_date).strftime("%d/%m/%Y")


def fmt_date_file(iso_date: str) -> str:
    return pd.to_datetime(iso_date).strftime("%d-%m-%Y")


def periodo_mes(mes_ref: str) -> Tuple[str, str]:
    if not mes_ref or not re.match(r"^\d{4}-\d{2}$", mes_ref):
        raise RuntimeError("Defina MES_REFERENCIA no formato YYYY-MM. Ex.: 2026-03")

    ano = int(mes_ref[:4])
    mes = int(mes_ref[5:7])
    ultimo_dia = calendar.monthrange(ano, mes)[1]
    dt_ini = f"{ano:04d}-{mes:02d}-01"
    dt_fim = f"{ano:04d}-{mes:02d}-{ultimo_dia:02d}"
    return dt_ini, dt_fim


def extrair_chapa_motorista(txt: str) -> str:
    s = str(txt or "").strip()
    m = re.match(r"^\s*(\d+)", s)
    if m:
        return m.group(1)
    return ""


def extrair_nome_motorista(txt: str) -> str:
    s = str(txt or "").strip()
    s = re.sub(r"^\s*\d+\s*[-–—]?\s*", "", s).strip()
    return s.upper() if s else ""


def obter_nomes_funcionarios() -> pd.DataFrame:
    try:
        res = (
            sb()
            .table(TABELA_FUNCIONARIOS)
            .select("nr_cracha, nm_funcionario")
            .execute()
        )
        rows = res.data or []
        if not rows:
            return pd.DataFrame(columns=["nr_cracha", "nm_funcionario"])

        df = pd.DataFrame(rows)
        df["nr_cracha"] = df["nr_cracha"].astype(str).str.strip()
        df["nm_funcionario"] = df["nm_funcionario"].astype(str).str.strip().str.upper()
        df = df.drop_duplicates(subset=["nr_cracha"], keep="first")
        return df
    except Exception as e:
        print(f"⚠️ Não foi possível carregar funcionários: {e}")
        return pd.DataFrame(columns=["nr_cracha", "nm_funcionario"])


def carregar_dados_mes(dt_ini: str, dt_fim: str) -> pd.DataFrame:
    print(f"-> Consultando {TABELA_ORIGEM} de {dt_ini} a {dt_fim}...")

    res = (
        sb()
        .table(TABELA_ORIGEM)
        .select("""
            dia,
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
        .gte("dia", dt_ini)
        .lte("dia", dt_fim)
        .order("dia", desc=False)
        .execute()
    )

    rows = res.data or []
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    df["dia"] = pd.to_datetime(df["dia"]).dt.date
    df["motorista"] = df["motorista"].astype(str).fillna("").str.strip()
    df["linha"] = df["linha"].astype(str).fillna("").str.strip().str.upper()
    df["prefixo"] = df["prefixo"].astype(str).fillna("").str.strip()
    df["fabricante"] = df["fabricante"].astype(str).fillna("").str.strip().str.upper()
    df["cluster"] = df["cluster"].astype(str).fillna("").str.strip().str.upper()

    df["km_rodado"] = pd.to_numeric(df["km_rodado"], errors="coerce").fillna(0.0)
    df["litros_consumidos"] = pd.to_numeric(df["litros_consumidos"], errors="coerce").fillna(0.0)
    df["km_l"] = pd.to_numeric(df["km_l"], errors="coerce").fillna(0.0)
    df["meta_kml_usada"] = pd.to_numeric(df["meta_kml_usada"], errors="coerce").fillna(0.0)
    df["litros_ideais"] = pd.to_numeric(df["litros_ideais"], errors="coerce").fillna(0.0)
    df["minutos_em_viagem"] = pd.to_numeric(df["minutos_em_viagem"], errors="coerce").fillna(0.0)

    df = df[(df["km_rodado"] > 0) & (df["litros_consumidos"] > 0)].copy()

    df["chapa"] = df["motorista"].apply(extrair_chapa_motorista)
    df["nome_extraido"] = df["motorista"].apply(extrair_nome_motorista)

    df["chave_motorista"] = df.apply(
        lambda r: r["chapa"] if str(r["chapa"]).strip() else str(r["motorista"]).strip().upper(),
        axis=1
    )

    def status_linha(row):
        real = n(row["km_l"])
        meta = n(row["meta_kml_usada"])
        if meta <= 0:
            return "Sem Meta"
        if real >= meta:
            return "Acima da Meta"
        if real >= (meta * 0.99):
            return "Próximo da Meta"
        return "Abaixo da Meta"

    df["status"] = df.apply(status_linha, axis=1)

    return df


def enriquecer_nomes(df: pd.DataFrame, df_func: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    if df_func is not None and not df_func.empty:
        mapa_nome = dict(zip(df_func["nr_cracha"], df_func["nm_funcionario"]))
        df["nome_oficial"] = df["chapa"].map(mapa_nome).fillna("")
    else:
        df["nome_oficial"] = ""

    df["nome_final"] = df.apply(
        lambda r: str(r["nome_oficial"]).strip().upper()
        if str(r["nome_oficial"]).strip()
        else str(r["nome_extraido"]).strip().upper(),
        axis=1
    )

    df["nome_final"] = df["nome_final"].replace("", pd.NA).fillna(df["motorista"].astype(str).str.upper())
    return df


def calcular_consolidado(df: pd.DataFrame) -> dict:
    km_total = df["km_rodado"].sum()
    litros_total = df["litros_consumidos"].sum()
    meta_litros = df["litros_ideais"].sum()
    delta_litros = meta_litros - litros_total
    kml_real = (km_total / litros_total) if litros_total > 0 else 0.0
    kml_meta = (km_total / meta_litros) if meta_litros > 0 else 0.0

    meta_base = kml_meta
    meta_3 = meta_base * 1.03
    meta_6 = meta_base * 1.06
    meta_10 = meta_base * 1.10

    if kml_real < meta_base:
        faixa = "Abaixo da Meta"
        premio = 0.0
    elif kml_real < meta_3:
        faixa = "Meta Base"
        premio = 100.0
    elif kml_real < meta_6:
        faixa = "Meta +3%"
        premio = 150.0
    elif kml_real < meta_10:
        faixa = "Meta +6%"
        premio = 200.0
    else:
        faixa = "Meta +10%"
        premio = 300.0

    qtd_dias = df["dia"].nunique()
    qtd_linhas = df["linha"].nunique()
    qtd_prefixos = df["prefixo"].nunique()
    minutos_total = df["minutos_em_viagem"].sum()

    return {
        "km_total": km_total,
        "litros_total": litros_total,
        "meta_litros": meta_litros,
        "delta_litros": delta_litros,
        "kml_real": kml_real,
        "kml_meta": kml_meta,
        "meta_base": meta_base,
        "meta_3": meta_3,
        "meta_6": meta_6,
        "meta_10": meta_10,
        "faixa": faixa,
        "premio": premio,
        "qtd_dias": qtd_dias,
        "qtd_linhas": qtd_linhas,
        "qtd_prefixos": qtd_prefixos,
        "minutos_total": minutos_total,
    }


def gerar_html_motorista(nome: str, chapa: str, dt_ini: str, dt_fim: str, df: pd.DataFrame, cons: dict) -> str:
    df = df.sort_values(["dia", "linha", "prefixo"]).copy()

    rows = []
    for _, r in df.iterrows():
        status_cls = "good" if r["status"] == "Acima da Meta" else ("mid" if r["status"] == "Próximo da Meta" else "bad")
        rows.append(f"""
        <tr>
          <td>{r['dia'].strftime('%d/%m')}</td>
          <td>{_esc(r['prefixo'])}</td>
          <td>{_esc(r['linha'])}</td>
          <td>{_esc(r['cluster'])}</td>
          <td class="num">{fmt_num(r['km_rodado'], 1)}</td>
          <td class="num">{fmt_num(r['litros_consumidos'], 1)}</td>
          <td class="num">{fmt_num(r['km_l'], 2)}</td>
          <td class="num">{fmt_num(r['meta_kml_usada'], 2)}</td>
          <td class="num">{fmt_num(r['litros_ideais'], 1)}</td>
          <td class="status {status_cls}">{_esc(r['status'])}</td>
        </tr>
        """)

    return f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8" />
  <title>Parcial Meritocracia - {_esc(nome)}</title>
  <style>
    @page {{
      size: A4 portrait;
      margin: 6mm;
    }}

    :root {{
      --bg:#ffffff;
      --line:#dbe3ef;
      --soft:#f6f9ff;
      --blue:#163a70;
      --blue2:#edf4ff;
      --text:#132033;
      --muted:#5f6f85;
      --green:#15803d;
      --yellow:#a16207;
      --red:#dc2626;
    }}

    * {{
      box-sizing: border-box;
    }}

    html, body {{
      margin: 0;
      padding: 0;
      background: #fff;
      color: var(--text);
      font-family: Arial, Helvetica, sans-serif;
      width: 100%;
    }}

    body {{
      font-size: 9px;
      line-height: 1.25;
    }}

    .page {{
      width: 100%;
      padding: 0;
    }}

    .header {{
      display: grid;
      grid-template-columns: 1fr 120px;
      gap: 8px;
      margin-bottom: 6px;
    }}

    .title {{
      font-size: 18px;
      font-weight: 800;
      color: var(--blue);
      margin: 0 0 2px 0;
      line-height: 1.05;
    }}

    .sub {{
      font-size: 8px;
      color: var(--muted);
      margin-bottom: 6px;
    }}

    .top-info {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 6px;
    }}

    .box {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 6px 7px;
      background: #fff;
      min-height: 40px;
    }}

    .label {{
      font-size: 7px;
      text-transform: uppercase;
      letter-spacing: .3px;
      color: var(--muted);
      font-weight: 700;
      margin-bottom: 2px;
    }}

    .value {{
      font-size: 9px;
      font-weight: 800;
      line-height: 1.15;
      word-break: break-word;
    }}

    .right-card {{
      border: 1px solid #bfd6ff;
      background: linear-gradient(180deg,#f3f8ff,#e6f0ff);
      border-radius: 10px;
      padding: 7px;
      text-align: center;
      display: flex;
      flex-direction: column;
      justify-content: center;
      min-height: 84px;
    }}

    .right-card .mini {{
      font-size: 7px;
      color: #4f6790;
      text-transform: uppercase;
      font-weight: 700;
      margin-bottom: 3px;
    }}

    .right-card .faixa {{
      font-size: 12px;
      font-weight: 900;
      color: var(--blue);
      line-height: 1.1;
      margin-bottom: 4px;
    }}

    .right-card .premio {{
      font-size: 18px;
      font-weight: 900;
      color: var(--green);
      line-height: 1.05;
    }}

    .metrics {{
      display: grid;
      grid-template-columns: repeat(6, 1fr);
      gap: 6px;
      margin: 7px 0 7px;
    }}

    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 6px 6px;
      background: var(--soft);
      text-align: center;
      min-height: 46px;
    }}

    .metric .t {{
      font-size: 7px;
      color: var(--muted);
      font-weight: 700;
      margin-bottom: 3px;
      text-transform: uppercase;
    }}

    .metric .v {{
      font-size: 10px;
      font-weight: 900;
      line-height: 1.1;
    }}

    .metric.good .v {{ color: var(--green); }}
    .metric.bad .v {{ color: var(--red); }}

    .panel {{
      border: 1px solid var(--line);
      border-radius: 10px;
      overflow: hidden;
      margin-bottom: 7px;
    }}

    .panel-head {{
      background: var(--blue2);
      padding: 6px 7px;
      font-size: 9px;
      font-weight: 800;
      color: var(--blue);
      border-bottom: 1px solid var(--line);
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      font-size: 7px;
    }}

    thead th {{
      background: #f8fbff;
      color: var(--blue);
      border-bottom: 1px solid var(--line);
      padding: 4px 4px;
      text-align: left;
      font-size: 6px;
      text-transform: uppercase;
      letter-spacing: .2px;
    }}

    tbody td {{
      padding: 3px 4px;
      border-bottom: 1px solid #edf2f7;
      vertical-align: middle;
    }}

    tbody tr:nth-child(even) {{
      background: #fcfdff;
    }}

    .num {{
      text-align: right;
      white-space: nowrap;
      font-variant-numeric: tabular-nums;
    }}

    .status {{
      font-size: 6px;
      font-weight: 800;
    }}

    .status.good {{ color: var(--green); }}
    .status.mid {{ color: var(--yellow); }}
    .status.bad {{ color: var(--red); }}

    .bottom {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
    }}

    .explain {{
      border: 1px solid var(--line);
      border-radius: 9px;
      padding: 7px;
      background: #fff;
      min-height: 82px;
    }}

    .explain h3 {{
      margin: 0 0 4px 0;
      font-size: 9px;
      color: var(--blue);
    }}

    .explain p {{
      margin: 0;
      font-size: 7.2px;
      line-height: 1.35;
      color: #334155;
    }}

    .hl {{
      font-weight: 800;
      color: var(--blue);
    }}

    .premios {{
      border: 1px solid var(--line);
      border-radius: 9px;
      padding: 7px;
      background: #fff;
      min-height: 82px;
    }}

    .premios h3 {{
      margin: 0 0 5px 0;
      font-size: 9px;
      color: var(--blue);
    }}

    .prem-grid {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 5px;
      margin-bottom: 5px;
    }}

    .prem-item {{
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 5px;
      background: #f8fbff;
    }}

    .prem-item .pl {{
      font-size: 6px;
      color: var(--muted);
      text-transform: uppercase;
      font-weight: 700;
      margin-bottom: 2px;
    }}

    .prem-item .pv {{
      font-size: 8px;
      font-weight: 900;
      color: var(--blue);
      line-height: 1.2;
    }}

    .foot {{
      margin-top: 5px;
      font-size: 6.5px;
      color: var(--muted);
      text-align: center;
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="header">
      <div>
        <div class="title">Parcial Meritocracia</div>
        <div class="sub">Desempenho consolidado do mês selecionado.</div>

        <div class="top-info">
          <div class="box">
            <div class="label">Motorista</div>
            <div class="value">{_esc(nome)}</div>
          </div>
          <div class="box">
            <div class="label">Chapa</div>
            <div class="value">{_esc(chapa or "-")}</div>
          </div>
          <div class="box">
            <div class="label">Período</div>
            <div class="value">{fmt_date_br(dt_ini)} a {fmt_date_br(dt_fim)}</div>
          </div>
          <div class="box">
            <div class="label">Dias / Linhas / Prefixos</div>
            <div class="value">{cons['qtd_dias']} / {cons['qtd_linhas']} / {cons['qtd_prefixos']}</div>
          </div>
        </div>
      </div>

      <div class="right-card">
        <div class="mini">Faixa Atual</div>
        <div class="faixa">{_esc(cons['faixa'])}</div>
        <div class="premio">R$ {fmt_num(cons['premio'])}</div>
      </div>
    </div>

    <div class="metrics">
      <div class="metric">
        <div class="t">KM Total</div>
        <div class="v">{fmt_num(cons['km_total'])}</div>
      </div>
      <div class="metric">
        <div class="t">Litros Total</div>
        <div class="v">{fmt_num(cons['litros_total'])}</div>
      </div>
      <div class="metric">
        <div class="t">Meta Litros</div>
        <div class="v">{fmt_num(cons['meta_litros'])}</div>
      </div>
      <div class="metric {'good' if cons['delta_litros'] >= 0 else 'bad'}">
        <div class="t">Delta Litros</div>
        <div class="v">{fmt_num(cons['delta_litros'])}</div>
      </div>
      <div class="metric">
        <div class="t">KM/L Real</div>
        <div class="v">{fmt_num(cons['kml_real'])}</div>
      </div>
      <div class="metric good">
        <div class="t">KM/L Meta</div>
        <div class="v">{fmt_num(cons['kml_meta'])}</div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-head">Detalhamento diário</div>
      <table>
        <thead>
          <tr>
            <th>Data</th>
            <th>Prefixo</th>
            <th>Linha</th>
            <th>Cluster</th>
            <th class="num">KM</th>
            <th class="num">Litros</th>
            <th class="num">KM/L</th>
            <th class="num">Meta</th>
            <th class="num">L. Ideais</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </div>

    <div class="bottom">
      <div class="explain">
        <h3>Como sua meta foi calculada</h3>
        <p>
          No período, você rodou <span class="hl">{fmt_num(cons['km_total'])} km</span>.
          Para essa operação, o consumo ideal seria de
          <span class="hl">{fmt_num(cons['meta_litros'])} litros</span>.
          Assim, sua meta consolidada ficou em
          <span class="hl">{fmt_num(cons['kml_meta'])} KM/L</span>.
        </p>
        <p style="margin-top:3px;">
          <span class="hl">Cálculo:</span>
          {fmt_num(cons['km_total'])} ÷ {fmt_num(cons['meta_litros'])}
          = <span class="hl">{fmt_num(cons['kml_meta'])} KM/L</span>
        </p>
      </div>

      <div class="premios">
        <h3>Faixas de premiação</h3>
        <div class="prem-grid">
          <div class="prem-item">
            <div class="pl">Meta Base</div>
            <div class="pv">{fmt_num(cons['meta_base'])} = R$ 100</div>
          </div>
          <div class="prem-item">
            <div class="pl">Meta +3%</div>
            <div class="pv">{fmt_num(cons['meta_3'])} = R$ 150</div>
          </div>
          <div class="prem-item">
            <div class="pl">Meta +6%</div>
            <div class="pv">{fmt_num(cons['meta_6'])} = R$ 200</div>
          </div>
          <div class="prem-item">
            <div class="pl">Meta +10%</div>
            <div class="pv">{fmt_num(cons['meta_10'])} = R$ 300</div>
          </div>
        </div>
        <p style="margin:0;font-size:7.2px;line-height:1.3;color:#334155;">
          O resultado real do mês foi <span class="hl">{fmt_num(cons['kml_real'])} KM/L</span>,
          posicionando você na faixa <span class="hl">{_esc(cons['faixa'])}</span>.
        </p>
      </div>
    </div>

    <div class="foot">
      Documento gerado automaticamente com base nos dados do mês selecionado.
    </div>
  </div>
</body>
</html>
"""


def gerar_html_resumo_geral(mes_ref: str, dt_ini: str, dt_fim: str, resumo_df: pd.DataFrame) -> str:
    total_motoristas = len(resumo_df)

    faixa_counts = resumo_df["faixa"].value_counts().to_dict()
    abaixo = int(faixa_counts.get("Abaixo da Meta", 0))
    base = int(faixa_counts.get("Meta Base", 0))
    m3 = int(faixa_counts.get("Meta +3%", 0))
    m6 = int(faixa_counts.get("Meta +6%", 0))
    m10 = int(faixa_counts.get("Meta +10%", 0))

    total_premiados = base + m3 + m6 + m10
    premio_total = resumo_df["premio"].sum()
    km_total = resumo_df["km_total"].sum()
    litros_total = resumo_df["litros_total"].sum()
    meta_litros_total = resumo_df["meta_litros"].sum()
    kml_geral = (km_total / litros_total) if litros_total > 0 else 0.0
    kml_meta_geral = (km_total / meta_litros_total) if meta_litros_total > 0 else 0.0

    resumo_ordenado = resumo_df.sort_values(["premio", "kml_real", "nome_final"], ascending=[False, False, True]).copy()

    rows = []
    for _, r in resumo_ordenado.iterrows():
        rows.append(f"""
        <tr>
          <td>{_esc(r['chapa'] or '-')}</td>
          <td>{_esc(r['nome_final'])}</td>
          <td class="num">{fmt_num(r['km_total'])}</td>
          <td class="num">{fmt_num(r['litros_total'])}</td>
          <td class="num">{fmt_num(r['kml_real'])}</td>
          <td class="num">{fmt_num(r['kml_meta'])}</td>
          <td>{_esc(r['faixa'])}</td>
          <td class="num">R$ {fmt_num(r['premio'])}</td>
        </tr>
        """)

    return f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8" />
  <title>Resumo Geral Meritocracia - {_esc(mes_ref)}</title>
  <style>
    @page {{
      size: A4 portrait;
      margin: 8mm;
    }}

    :root {{
      --line:#dbe3ef;
      --soft:#f6f9ff;
      --blue:#163a70;
      --blue2:#edf4ff;
      --text:#132033;
      --muted:#5f6f85;
    }}

    * {{ box-sizing: border-box; }}
    html, body {{
      margin: 0;
      padding: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--text);
      background: #fff;
    }}

    .title {{
      font-size: 20px;
      font-weight: 800;
      color: var(--blue);
      margin-bottom: 2px;
    }}

    .sub {{
      font-size: 10px;
      color: var(--muted);
      margin-bottom: 10px;
    }}

    .cards {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 8px;
      margin-bottom: 10px;
    }}

    .card {{
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px;
      background: var(--soft);
      min-height: 58px;
    }}

    .card .l {{
      font-size: 8px;
      color: var(--muted);
      text-transform: uppercase;
      font-weight: 700;
      margin-bottom: 4px;
    }}

    .card .v {{
      font-size: 14px;
      font-weight: 900;
      color: var(--blue);
    }}

    .grid2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-bottom: 10px;
    }}

    .panel {{
      border: 1px solid var(--line);
      border-radius: 10px;
      overflow: hidden;
      background: #fff;
    }}

    .panel-head {{
      background: var(--blue2);
      padding: 7px 8px;
      font-size: 10px;
      font-weight: 800;
      color: var(--blue);
      border-bottom: 1px solid var(--line);
    }}

    .panel-body {{
      padding: 8px;
    }}

    .faixas {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
    }}

    .fx {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 7px;
      background: #f8fbff;
    }}

    .fx .n {{
      font-size: 8px;
      color: var(--muted);
      font-weight: 700;
      margin-bottom: 2px;
    }}

    .fx .q {{
      font-size: 15px;
      font-weight: 900;
      color: var(--blue);
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      font-size: 8px;
    }}

    thead th {{
      background: #f8fbff;
      color: var(--blue);
      border-bottom: 1px solid var(--line);
      padding: 5px;
      text-align: left;
      font-size: 7px;
      text-transform: uppercase;
    }}

    tbody td {{
      padding: 4px 5px;
      border-bottom: 1px solid #edf2f7;
      vertical-align: middle;
    }}

    tbody tr:nth-child(even) {{
      background: #fcfdff;
    }}

    .num {{
      text-align: right;
      white-space: nowrap;
      font-variant-numeric: tabular-nums;
    }}

    .small {{
      font-size: 7px;
      color: var(--muted);
      margin-top: 6px;
    }}
  </style>
</head>
<body>
  <div class="title">Resumo Geral da Meritocracia</div>
  <div class="sub">Mês de referência {_esc(mes_ref)} | {fmt_date_br(dt_ini)} a {fmt_date_br(dt_fim)}</div>

  <div class="cards">
    <div class="card">
      <div class="l">Motoristas com dados</div>
      <div class="v">{total_motoristas}</div>
    </div>
    <div class="card">
      <div class="l">Motoristas premiados</div>
      <div class="v">{total_premiados}</div>
    </div>
    <div class="card">
      <div class="l">Premiação projetada</div>
      <div class="v">R$ {fmt_num(premio_total)}</div>
    </div>
    <div class="card">
      <div class="l">KM/L Geral</div>
      <div class="v">{fmt_num(kml_geral)}</div>
    </div>
  </div>

  <div class="grid2">
    <div class="panel">
      <div class="panel-head">Faixas de resultado</div>
      <div class="panel-body">
        <div class="faixas">
          <div class="fx"><div class="n">Abaixo da Meta</div><div class="q">{abaixo}</div></div>
          <div class="fx"><div class="n">Meta Base</div><div class="q">{base}</div></div>
          <div class="fx"><div class="n">Meta +3%</div><div class="q">{m3}</div></div>
          <div class="fx"><div class="n">Meta +6%</div><div class="q">{m6}</div></div>
          <div class="fx"><div class="n">Meta +10%</div><div class="q">{m10}</div></div>
          <div class="fx"><div class="n">KM/L Meta Geral</div><div class="q">{fmt_num(kml_meta_geral)}</div></div>
        </div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-head">Consolidado operacional</div>
      <div class="panel-body">
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
          <div class="fx"><div class="n">KM Total</div><div class="q">{fmt_num(km_total)}</div></div>
          <div class="fx"><div class="n">Litros Total</div><div class="q">{fmt_num(litros_total)}</div></div>
          <div class="fx"><div class="n">Meta Litros</div><div class="q">{fmt_num(meta_litros_total)}</div></div>
          <div class="fx"><div class="n">Diferença Litros</div><div class="q">{fmt_num(meta_litros_total - litros_total)}</div></div>
        </div>
        <div class="small">
          Este resumo considera apenas motoristas com movimentação válida no período selecionado.
        </div>
      </div>
    </div>
  </div>

  <div class="panel">
    <div class="panel-head">Resumo por motorista</div>
    <div class="panel-body" style="padding:0;">
      <table>
        <thead>
          <tr>
            <th style="width:8%;">Chapa</th>
            <th style="width:28%;">Motorista</th>
            <th style="width:11%;" class="num">KM</th>
            <th style="width:11%;" class="num">Litros</th>
            <th style="width:10%;" class="num">KM/L Real</th>
            <th style="width:10%;" class="num">KM/L Meta</th>
            <th style="width:12%;">Faixa</th>
            <th style="width:10%;" class="num">Prêmio</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </div>
  </div>
</body>
</html>
"""


def html_to_pdf(p_html: Path, p_pdf: Path):
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page(
            viewport={"width": 794, "height": 1123},
            device_scale_factor=1
        )
        page.goto(p_html.resolve().as_uri(), wait_until="networkidle")
        page.pdf(
            path=str(p_pdf),
            print_background=True,
            prefer_css_page_size=True,
            margin={"top": "0", "bottom": "0", "left": "0", "right": "0"}
        )
        browser.close()


def upload_storage(local_path: Path, remote_path: str, content_type: str):
    if not local_path.exists():
        return None, None

    sb_client = sb()
    with open(local_path, "rb") as f:
        sb_client.storage.from_(BUCKET).upload(
            path=remote_path,
            file=f,
            file_options={"content-type": content_type, "upsert": "true"}
        )

    public_url = f"{SUPABASE_A_URL}/storage/v1/object/public/{BUCKET}/{remote_path}"
    return remote_path, public_url


def merge_pdfs(pdf_paths: List[Path], output_path: Path):
    writer = PdfWriter()

    for p in pdf_paths:
        if not p.exists():
            continue
        reader = PdfReader(str(p))
        for page in reader.pages:
            writer.add_page(page)

    with open(output_path, "wb") as f:
        writer.write(f)


def montar_resumo_motoristas(df_enriquecido: pd.DataFrame) -> pd.DataFrame:
    registros = []

    for chave, g in df_enriquecido.groupby("chave_motorista", dropna=False):
        g = g.sort_values("dia").copy()

        chapa = ""
        if "chapa" in g.columns:
            chapas = [str(x).strip() for x in g["chapa"].dropna().tolist() if str(x).strip()]
            chapa = chapas[0] if chapas else ""

        nome_final = ""
        if "nome_final" in g.columns:
            nomes = [str(x).strip().upper() for x in g["nome_final"].dropna().tolist() if str(x).strip()]
            nome_final = nomes[0] if nomes else ""
        if not nome_final:
            nome_final = str(g["motorista"].iloc[0]).strip().upper()

        cons = calcular_consolidado(g)

        registros.append({
            "chave_motorista": chave,
            "chapa": chapa,
            "nome_final": nome_final,
            **cons
        })

    if not registros:
        return pd.DataFrame()

    return pd.DataFrame(registros)


def main():
    print("🚀 Iniciando geração mensal da Parcial Meritocracia...")

    if not SUPABASE_A_URL or not SUPABASE_A_SERVICE_ROLE_KEY:
        raise RuntimeError("Defina SUPABASE_A_URL e SUPABASE_A_SERVICE_ROLE_KEY.")

    dt_ini, dt_fim = periodo_mes(MES_REFERENCIA)
    print(f"📅 Mês selecionado: {MES_REFERENCIA} | período {dt_ini} a {dt_fim}")

    df = carregar_dados_mes(dt_ini, dt_fim)
    if df.empty:
        raise RuntimeError(f"Nenhum dado encontrado em {TABELA_ORIGEM} para o mês {MES_REFERENCIA}.")

    df_func = obter_nomes_funcionarios()
    df = enriquecer_nomes(df, df_func)

    resumo_df = montar_resumo_motoristas(df)
    if resumo_df.empty:
        raise RuntimeError("Nenhum motorista válido encontrado para geração dos prontuários.")

    resumo_df = resumo_df.sort_values(["nome_final"], ascending=[True]).reset_index(drop=True)

    pasta_mes = PASTA_SAIDA / MES_REFERENCIA
    pasta_individuais = pasta_mes / "individuais"
    pasta_mes.mkdir(parents=True, exist_ok=True)
    pasta_individuais.mkdir(parents=True, exist_ok=True)

    pdfs_individuais = []

    print(f"👥 Total de motoristas identificados: {len(resumo_df)}")

    for i, row in resumo_df.iterrows():
        chave_motorista = row["chave_motorista"]
        chapa = str(row["chapa"] or "").strip()
        nome_final = str(row["nome_final"] or "").strip().upper()

        g = df[df["chave_motorista"] == chave_motorista].copy()
        cons = calcular_consolidado(g)

        nome_base = _safe_filename(
            f"{i+1:03d}_{chapa or 'SEM_CHAPA'}_{nome_final}_{MES_REFERENCIA}"
        )

        p_html = pasta_individuais / f"{nome_base}.html"
        p_pdf = pasta_individuais / f"{nome_base}.pdf"

        html = gerar_html_motorista(nome_final, chapa, dt_ini, dt_fim, g, cons)
        p_html.write_text(html, encoding="utf-8")
        html_to_pdf(p_html, p_pdf)

        pdfs_individuais.append(p_pdf)

        remote_pdf = f"{MES_REFERENCIA}/individuais/{nome_base}.pdf"
        upload_storage(p_pdf, remote_pdf, "application/pdf")

        print(f"✅ [{i+1}/{len(resumo_df)}] Gerado: {p_pdf.name}")

    p_pdf_consolidado = pasta_mes / f"00_CONSOLIDADO_MOTORISTAS_{MES_REFERENCIA}.pdf"
    merge_pdfs(pdfs_individuais, p_pdf_consolidado)

    p_html_resumo = pasta_mes / f"00_RESUMO_GERAL_{MES_REFERENCIA}.html"
    p_pdf_resumo = pasta_mes / f"00_RESUMO_GERAL_{MES_REFERENCIA}.pdf"

    html_resumo = gerar_html_resumo_geral(MES_REFERENCIA, dt_ini, dt_fim, resumo_df)
    p_html_resumo.write_text(html_resumo, encoding="utf-8")
    html_to_pdf(p_html_resumo, p_pdf_resumo)

    _, url_consolidado = upload_storage(
        p_pdf_consolidado,
        f"{MES_REFERENCIA}/{p_pdf_consolidado.name}",
        "application/pdf"
    )

    _, url_resumo = upload_storage(
        p_pdf_resumo,
        f"{MES_REFERENCIA}/{p_pdf_resumo.name}",
        "application/pdf"
    )

    print("\n✅ PROCESSO FINALIZADO COM SUCESSO")
    print(f"📁 Pasta de saída: {pasta_mes}")
    print(f"👥 Motoristas processados: {len(resumo_df)}")
    print(f"📄 PDF consolidado: {p_pdf_consolidado}")
    print(f"📊 PDF resumo: {p_pdf_resumo}")
    print(f"🔗 URL consolidado: {url_consolidado}")
    print(f"🔗 URL resumo: {url_resumo}")

    faixa_counts = resumo_df["faixa"].value_counts().to_dict()
    print("\n📌 RESUMO DE FAIXAS")
    print(f"Abaixo da Meta: {int(faixa_counts.get('Abaixo da Meta', 0))}")
    print(f"Meta Base:      {int(faixa_counts.get('Meta Base', 0))}")
    print(f"Meta +3%:       {int(faixa_counts.get('Meta +3%', 0))}")
    print(f"Meta +6%:       {int(faixa_counts.get('Meta +6%', 0))}")
    print(f"Meta +10%:      {int(faixa_counts.get('Meta +10%', 0))}")
    print(f"💰 Total projetado de premiação: R$ {fmt_num(resumo_df['premio'].sum())}")


if __name__ == "__main__":
    main()
