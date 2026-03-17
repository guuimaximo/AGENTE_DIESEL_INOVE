# -*- coding: utf-8 -*-
import os
import re
from pathlib import Path

import pandas as pd
from supabase import create_client
from playwright.sync_api import sync_playwright


SUPABASE_A_URL = os.getenv("SUPABASE_A_URL")
SUPABASE_A_SERVICE_ROLE_KEY = os.getenv("SUPABASE_A_SERVICE_ROLE_KEY")

TABELA_ORIGEM = "premiacao_diaria_atualizada"
TABELA_FUNCIONARIOS = "funcionarios"
BUCKET = "parcial_meritocracia"

CHAPA = os.getenv("CHAPA")
PERIODO_INICIO = os.getenv("PERIODO_INICIO")
PERIODO_FIM = os.getenv("PERIODO_FIM")

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


def obter_nome_motorista(chapa: str, df: pd.DataFrame) -> str:
    try:
        r = (
            sb()
            .table(TABELA_FUNCIONARIOS)
            .select("nr_cracha, nm_funcionario")
            .eq("nr_cracha", chapa)
            .maybe_single()
            .execute()
        )
        if r.data and r.data.get("nm_funcionario"):
            return str(r.data["nm_funcionario"]).strip().upper()
    except Exception:
        pass

    if df is not None and not df.empty and "motorista" in df.columns:
        nomes = df["motorista"].dropna().astype(str).unique().tolist()
        if nomes:
            bruto = nomes[0]
            limpo = re.sub(r"^\d+\s*[-]*\s*", "", bruto).strip()
            if limpo:
                return limpo.upper()

    return chapa


def carregar_dados_motorista(chapa: str, dt_ini: str, dt_fim: str) -> pd.DataFrame:
    print(f"-> Consultando {TABELA_ORIGEM} para chapa {chapa} de {dt_ini} a {dt_fim}...")

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
        .ilike("motorista", f"%{chapa}%")
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
    df["linha"] = df["linha"].astype(str).str.strip().str.upper()
    df["prefixo"] = df["prefixo"].astype(str).str.strip()
    df["fabricante"] = df["fabricante"].astype(str).str.strip().str.upper()

    df["km_rodado"] = pd.to_numeric(df["km_rodado"], errors="coerce").fillna(0.0)
    df["litros_consumidos"] = pd.to_numeric(df["litros_consumidos"], errors="coerce").fillna(0.0)
    df["km_l"] = pd.to_numeric(df["km_l"], errors="coerce").fillna(0.0)
    df["meta_kml_usada"] = pd.to_numeric(df["meta_kml_usada"], errors="coerce").fillna(0.0)
    df["litros_ideais"] = pd.to_numeric(df["litros_ideais"], errors="coerce").fillna(0.0)
    df["minutos_em_viagem"] = pd.to_numeric(df["minutos_em_viagem"], errors="coerce").fillna(0.0)

    df = df[(df["km_rodado"] > 0) & (df["litros_consumidos"] > 0)].copy()

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
    }


def gerar_html(nome: str, chapa: str, dt_ini: str, dt_fim: str, df: pd.DataFrame, cons: dict) -> str:
    rows = []
    for _, r in df.iterrows():
        status_cls = "good" if r["status"] == "Acima da Meta" else ("mid" if r["status"] == "Próximo da Meta" else "bad")
        rows.append(f"""
        <tr>
          <td>{r['dia'].strftime('%d/%m/%Y')}</td>
          <td>{_esc(r['prefixo'])}</td>
          <td>{_esc(r['fabricante'])}</td>
          <td>{_esc(r['linha'])}</td>
          <td class="num">{fmt_num(r['km_rodado'])}</td>
          <td class="num">{fmt_num(r['litros_consumidos'])}</td>
          <td class="num">{fmt_num(r['km_l'])}</td>
          <td class="num">{fmt_num(r['meta_kml_usada'])}</td>
          <td class="num">{fmt_num(r['litros_ideais'])}</td>
          <td class="status {status_cls}">{_esc(r['status'])}</td>
        </tr>
        """)

    return f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8" />
  <title>Parcial de Desempenho KM/L</title>
  <style>
    @page {{
      size: A4 portrait;
      margin: 6mm;
    }}

    :root {{
      --bg:#f4f7fb;
      --page:#ffffff;
      --primary:#1e3a8a;
      --primary-soft:#eaf2ff;
      --text:#172033;
      --muted:#64748b;
      --line:#dbe5f1;
      --card:#f8fbff;
      --green:#16a34a;
      --yellow:#ca8a04;
      --red:#dc2626;
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    html, body {{
      width: 100%;
      height: 100%;
      font-family: Arial, Helvetica, sans-serif;
      background: var(--bg);
      color: var(--text);
    }}

    body {{
      padding: 0;
    }}

    .page {{
      width: 100%;
      background: var(--page);
      padding: 8px 9px;
    }}

    .header {{
      display: grid;
      grid-template-columns: 1fr 180px;
      gap: 8px;
      align-items: stretch;
    }}

    .header-left h1 {{
      font-size: 17px;
      line-height: 1.05;
      color: var(--primary);
      margin-bottom: 4px;
      font-weight: 800;
    }}

    .sub {{
      color: var(--muted);
      font-size: 7px;
      line-height: 1.35;
      margin-bottom: 6px;
      max-width: 360px;
    }}

    .brand {{
      background: linear-gradient(135deg, #eff6ff, #dbeafe);
      color: var(--primary);
      border: 1px solid #bfdbfe;
      border-radius: 10px;
      padding: 10px 8px;
      text-align: center;
      font-weight: 700;
      font-size: 8px;
      min-height: 72px;
      display: flex;
      align-items: flex-start;
      justify-content: center;
    }}

    .top-info {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 6px;
    }}

    .box {{
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 9px;
      padding: 7px 8px;
      min-height: 44px;
    }}

    .label {{
      font-size: 6px;
      text-transform: uppercase;
      letter-spacing: .3px;
      color: var(--muted);
      margin-bottom: 3px;
      font-weight: 700;
    }}

    .value {{
      font-size: 7.2px;
      font-weight: 800;
      line-height: 1.15;
      word-break: break-word;
    }}

    .divider {{
      height: 1px;
      background: var(--line);
      margin: 8px 0 8px;
    }}

    .section-title {{
      font-size: 9px;
      font-weight: 800;
      color: var(--primary);
      margin-bottom: 5px;
    }}

    .metrics {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 6px;
      margin-bottom: 8px;
    }}

    .metric {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 7px 6px;
      text-align: center;
      min-height: 48px;
    }}

    .metric.alert {{
      background: #fff1f2;
      border-color: #fecdd3;
    }}

    .metric.good {{
      background: #f0fdf4;
      border-color: #bbf7d0;
    }}

    .metric .title {{
      color: var(--muted);
      font-size: 5.8px;
      margin-bottom: 4px;
      font-weight: 700;
    }}

    .metric .value {{
      font-size: 8.5px;
      font-weight: 900;
      line-height: 1.1;
    }}

    .metric.alert .value {{
      color: var(--red);
    }}

    .metric.good .value {{
      color: var(--green);
    }}

    .panel {{
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 7px;
      margin-bottom: 8px;
      page-break-inside: avoid;
      break-inside: avoid;
    }}

    .panel-title {{
      font-size: 8px;
      font-weight: 800;
      color: var(--primary);
      margin-bottom: 5px;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      font-size: 6.5px;
    }}

    thead th {{
      background: #eff6ff;
      color: var(--primary);
      text-align: left;
      font-size: 5.6px;
      text-transform: uppercase;
      letter-spacing: .2px;
      padding: 5px 4px;
      border-bottom: 1px solid var(--line);
    }}

    tbody td {{
      padding: 4px 4px;
      border-bottom: 1px solid #edf2f7;
      vertical-align: middle;
    }}

    tbody tr:nth-child(even) {{
      background: #fafcff;
    }}

    .num {{
      text-align: right;
      white-space: nowrap;
      font-variant-numeric: tabular-nums;
    }}

    .status {{
      font-weight: 800;
      font-size: 5.7px;
    }}

    .status.good {{ color: var(--green); }}
    .status.mid {{ color: var(--yellow); }}
    .status.bad {{ color: var(--red); }}

    .foot {{
      margin-top: 5px;
      padding: 5px 6px;
      border-radius: 8px;
      background: #f8fafc;
      border: 1px solid var(--line);
      color: var(--muted);
      font-size: 5.7px;
      line-height: 1.3;
    }}

    .bottom-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
      align-items: start;
      page-break-inside: avoid;
      break-inside: avoid;
    }}

    .left-stack, .right-stack {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}

    .explain {{
      background: var(--primary-soft);
      border: 1px solid #c7dcff;
      border-radius: 9px;
      padding: 7px 8px;
      page-break-inside: avoid;
      break-inside: avoid;
    }}

    .explain h3 {{
      font-size: 7.2px;
      font-weight: 800;
      color: var(--primary);
      margin-bottom: 4px;
    }}

    .explain p {{
      color: #334155;
      font-size: 6.1px;
      line-height: 1.38;
    }}

    .hl {{
      font-weight: 800;
      color: var(--primary);
    }}

    .premium {{
      background: linear-gradient(180deg,#eff6ff,#dbeafe);
      border: 1px solid #bfdbfe;
      border-radius: 10px;
      padding: 7px;
      page-break-inside: avoid;
      break-inside: avoid;
    }}

    .premium h3 {{
      font-size: 8px;
      font-weight: 800;
      color: var(--primary);
      margin-bottom: 5px;
      text-align: center;
    }}

    .premium-grid {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 5px;
      margin-bottom: 6px;
    }}

    .premium-item {{
      background: #ffffff;
      border: 1px solid #dbeafe;
      border-radius: 8px;
      padding: 6px;
    }}

    .premium-item .label {{
      color: var(--muted);
      font-size: 5.2px;
      text-transform: uppercase;
      font-weight: 700;
      margin-bottom: 3px;
    }}

    .premium-item .value {{
      color: var(--primary);
      font-size: 6.7px;
      font-weight: 900;
      line-height: 1.2;
    }}

    .result-box {{
      background: #ffffff;
      border: 1.5px solid #93c5fd;
      border-radius: 9px;
      padding: 7px;
      text-align: center;
      margin-top: 6px;
    }}

    .result-box .range {{
      font-size: 7px;
      color: var(--primary);
      font-weight: 800;
      margin-bottom: 3px;
    }}

    .result-box .prize {{
      font-size: 11px;
      color: var(--green);
      font-weight: 900;
      margin-bottom: 2px;
    }}

    .obs {{
      text-align: center;
      margin-top: 5px;
      color: var(--muted);
      font-size: 5.3px;
      line-height: 1.3;
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="header">
      <div class="header-left">
        <h1>Parcial de Desempenho KM/L</h1>
        <div class="sub">
          Acompanhe sua performance no período, entenda como sua meta foi calculada
          e veja em qual faixa de premiação você está neste momento.
        </div>

        <div class="top-info">
          <div class="box">
            <div class="label">Motorista</div>
            <div class="value">{_esc(nome)}</div>
          </div>
          <div class="box">
            <div class="label">Chapa</div>
            <div class="value">{_esc(chapa)}</div>
          </div>
          <div class="box">
            <div class="label">Período</div>
            <div class="value">{fmt_date_br(dt_ini)} a {fmt_date_br(dt_fim)}</div>
          </div>
          <div class="box">
            <div class="label">Status</div>
            <div class="value">Parcial do Mês</div>
          </div>
        </div>
      </div>

      <div class="brand">Parcial Meritocracia</div>
    </div>

    <div class="divider"></div>

    <div class="section-title">Consolidado da Parcial</div>
    <div class="metrics">
      <div class="metric">
        <div class="title">KM Total</div>
        <div class="value">{fmt_num(cons['km_total'])}</div>
      </div>

      <div class="metric {'good' if cons['delta_litros'] >= 0 else 'alert'}">
        <div class="title">Delta Litros</div>
        <div class="value">{fmt_num(cons['delta_litros'])}</div>
      </div>

      <div class="metric">
        <div class="title">Litros Total</div>
        <div class="value">{fmt_num(cons['litros_total'])}</div>
      </div>

      <div class="metric">
        <div class="title">Meta Litros</div>
        <div class="value">{fmt_num(cons['meta_litros'])}</div>
      </div>

      <div class="metric">
        <div class="title">KM/L Real</div>
        <div class="value">{fmt_num(cons['kml_real'])}</div>
      </div>

      <div class="metric good">
        <div class="title">KM/L Meta</div>
        <div class="value">{fmt_num(cons['kml_meta'])}</div>
      </div>
    </div>

    <div class="panel">
      <div class="panel-title">Detalhamento da Parcial</div>
      <table>
        <thead>
          <tr>
            <th>Data</th>
            <th>Prefixo</th>
            <th>Tipo Veíc.</th>
            <th>Linha</th>
            <th class="num">KM Rodado</th>
            <th class="num">Litros</th>
            <th class="num">KM/L Real</th>
            <th class="num">Meta Linha</th>
            <th class="num">Litros Ideais</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>

      <div class="foot">
        Esta parcial considera os dados operacionais do período atual. Os números podem ser atualizados até o fechamento final do mês.
      </div>
    </div>

    <div class="bottom-grid">
      <div class="left-stack">
        <div class="explain">
          <h3>Como sua meta foi calculada</h3>
          <p>
            Sua meta do período foi calculada de forma <span class="hl">ponderada</span>,
            considerando as linhas e operações realizadas por você. No período analisado,
            você rodou <span class="hl">{fmt_num(cons['km_total'])} km</span>. Para essa operação,
            o consumo ideal seria de <span class="hl">{fmt_num(cons['meta_litros'])} litros</span>.
            Por isso, sua meta final ficou em <span class="hl">{fmt_num(cons['kml_meta'])} KM/L</span>.
          </p>
          <p style="margin-top:4px;">
            <span class="hl">Cálculo aplicado:</span>
            {fmt_num(cons['km_total'])} ÷ {fmt_num(cons['meta_litros'])} = <span class="hl">{fmt_num(cons['kml_meta'])} KM/L meta</span>
          </p>
        </div>

        <div class="explain">
          <h3>Como seu resultado foi calculado</h3>
          <p>
            Seu resultado foi calculado dividindo o total de quilômetros rodados pelo total
            de litros efetivamente consumidos. No período, você rodou <span class="hl">{fmt_num(cons['km_total'])} km</span>
            e consumiu <span class="hl">{fmt_num(cons['litros_total'])} litros</span>, chegando ao resultado de
            <span class="hl">{fmt_num(cons['kml_real'])} KM/L</span>.
          </p>
          <p style="margin-top:4px;">
            <span class="hl">Cálculo aplicado:</span>
            {fmt_num(cons['km_total'])} ÷ {fmt_num(cons['litros_total'])} = <span class="hl">{fmt_num(cons['kml_real'])} KM/L real</span>
          </p>
        </div>
      </div>

      <div class="right-stack">
        <div class="premium">
          <h3>Faixa de Premiação</h3>

          <div class="premium-grid">
            <div class="premium-item">
              <div class="label">Meta Base</div>
              <div class="value">{fmt_num(cons['meta_base'])} = R$ 100,00</div>
            </div>
            <div class="premium-item">
              <div class="label">Meta +3%</div>
              <div class="value">{fmt_num(cons['meta_3'])} = R$ 150,00</div>
            </div>
            <div class="premium-item">
              <div class="label">Meta +6%</div>
              <div class="value">{fmt_num(cons['meta_6'])} = R$ 200,00</div>
            </div>
            <div class="premium-item">
              <div class="label">Meta +10%</div>
              <div class="value">{fmt_num(cons['meta_10'])} = R$ 300,00</div>
            </div>
          </div>

          <div class="explain" style="background:#ffffff; margin-bottom:0;">
            <h3>Como sua premiação é calculada</h3>
            <p>
              A sua faixa de premiação é definida com base na <span class="hl">meta final do período</span>.
              Sobre essa meta, aplicamos os percentuais de <span class="hl">+3%</span>,
              <span class="hl">+6%</span> e <span class="hl">+10%</span> para identificar o valor correspondente.
            </p>
          </div>

          <div class="result-box">
            <div class="range">Faixa Atual: {cons['faixa']}</div>
            <div class="prize">R$ {fmt_num(cons['premio'])}</div>
          </div>

          <div class="obs">
            Valor projetado com base na parcial atual do período. O fechamento final poderá sofrer alteração conforme a consolidação mensal.
          </div>
        </div>
      </div>
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


def main():
    print("🚀 Iniciando geração da Parcial Meritocracia...")

    if not CHAPA or not PERIODO_INICIO or not PERIODO_FIM:
        raise RuntimeError("Defina CHAPA, PERIODO_INICIO e PERIODO_FIM nas variáveis de ambiente.")

    df = carregar_dados_motorista(CHAPA, PERIODO_INICIO, PERIODO_FIM)
    if df.empty:
        raise RuntimeError(f"Nenhum dado encontrado para chapa {CHAPA} no período informado.")

    nome = obter_nome_motorista(CHAPA, df)
    consolidado = calcular_consolidado(df)

    mes_ref = pd.to_datetime(PERIODO_INICIO).strftime("%Y-%m")
    periodo_pasta = f"{fmt_date_file(PERIODO_INICIO)}_a_{fmt_date_file(PERIODO_FIM)}"
    nome_base = _safe_filename(f"{CHAPA}_{fmt_date_file(PERIODO_INICIO)}_{fmt_date_file(PERIODO_FIM)}")

    p_html = PASTA_SAIDA / f"{nome_base}.html"
    p_pdf = PASTA_SAIDA / f"{nome_base}.pdf"

    html = gerar_html(nome, CHAPA, PERIODO_INICIO, PERIODO_FIM, df, consolidado)
    p_html.write_text(html, encoding="utf-8")
    html_to_pdf(p_html, p_pdf)

    remote_html = f"{mes_ref}/{periodo_pasta}/{nome_base}.html"
    remote_pdf = f"{mes_ref}/{periodo_pasta}/{nome_base}.pdf"

    html_path, html_url = upload_storage(p_html, remote_html, "text/html")
    pdf_path, pdf_url = upload_storage(p_pdf, remote_pdf, "application/pdf")

    print("✅ Parcial gerada com sucesso!")
    print(f"Nome do arquivo: {nome_base}")
    print(f"HTML: {html_path}")
    print(f"PDF: {pdf_path}")
    print(f"PDF URL: {pdf_url}")


if __name__ == "__main__":
    main()
