# scripts/gerar_ordens_acompanhamento.py
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from supabase import create_client
from playwright.sync_api import sync_playwright

# ===== ENV =====
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# controle
ORDEM_BATCH_ID = os.getenv("ORDEM_BATCH_ID")  # id do lote (tabela controle)
QTD = int(os.getenv("QTD", "10"))

# tabelas (ajuste se quiser)
TBL_METRICAS = os.getenv("TBL_METRICAS", "diesel_metricas_motorista_dia")
TBL_ACOMP = os.getenv("TBL_ACOMP", "diesel_acompanhamentos")  # onde cria acompanhamentos
TBL_LOTES = os.getenv("TBL_LOTES", "acompanhamento_lotes")    # opcional (controle)

# storage
BUCKET = os.getenv("BUCKET", "relatorios")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "Ordens_Acompanhamento")
REMOTE_PREFIX = os.getenv("REMOTE_PREFIX", "acompanhamento")

def sb():
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY obrigatórios")
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

def safe_filename(name: str) -> str:
    name = str(name or "").strip()
    name = re.sub(r"[^\w\-.() ]+", "_", name, flags=re.UNICODE)
    name = re.sub(r"\s+", "_", name)
    return name[:160] if name else "arquivo"

def update_lote(status: str, **fields):
    if not ORDEM_BATCH_ID:
        return
    sb().table(TBL_LOTES).update({"status": status, **fields}).eq("id", ORDEM_BATCH_ID).execute()

def upload(local: Path, remote_path: str, content_type: str):
    storage = sb().storage.from_(BUCKET)
    storage.upload(
        path=remote_path,
        file=local.read_bytes(),
        file_options={"content-type": content_type},
    )

def render_pdf(html_path: Path, pdf_path: Path):
    html_path = html_path.resolve()
    pdf_path = pdf_path.resolve()
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1280, "height": 720})
        page.goto(html_path.as_uri(), wait_until="networkidle")
        page.wait_for_timeout(200)
        page.pdf(
            path=str(pdf_path),
            format="A4",
            print_background=True,
            margin={"top": "0mm", "right": "0mm", "bottom": "0mm", "left": "0mm"},
            prefer_css_page_size=True,
        )
        browser.close()

def pick_alvos(qtd: int) -> pd.DataFrame:
    """
    EXEMPLO: pega últimos motoristas com pior kml médio nos últimos 7 dias.
    Ajuste a query conforme tua base real.
    """
    # tenta buscar últimos 7 dias via SQL simples: pega tudo e agrega no pandas
    resp = sb().table(TBL_METRICAS).select("chapa,nome,data,kml,km,litros").order("data", desc=True).limit(5000).execute()
    rows = resp.data or []
    if not rows:
        return pd.DataFrame(columns=["chapa", "nome", "kml_med", "dias"])

    df = pd.DataFrame(rows)
    df["kml"] = pd.to_numeric(df.get("kml"), errors="coerce")
    df = df.dropna(subset=["chapa", "kml"])

    agg = (
        df.groupby(["chapa", "nome"])
        .agg(kml_med=("kml", "mean"), dias=("data", "nunique"))
        .reset_index()
        .sort_values("kml_med", ascending=True)
        .head(qtd)
    )
    return agg

def gerar_html_ordens(alvos: pd.DataFrame, html_path: Path, titulo: str):
    rows = ""
    for i, r in alvos.reset_index(drop=True).iterrows():
        rows += f"""
          <tr>
            <td>{i+1}</td>
            <td><b>{r.get('nome','-')}</b></td>
            <td>{r.get('chapa','-')}</td>
            <td>{float(r.get('kml_med',0)):.2f}</td>
            <td>{int(r.get('dias',0))}</td>
          </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
      <meta charset="UTF-8"/>
      <title>{titulo}</title>
      <style>
        body {{ font-family: 'Segoe UI', sans-serif; margin: 0; padding: 18px; background:#f6f7fb; }}
        .wrap {{ max-width: 980px; margin: auto; background:white; border:1px solid #e5e7eb; border-radius: 12px; padding: 22px; }}
        h1 {{ margin: 0; font-size: 18px; color:#0f172a; }}
        .muted {{ margin-top: 6px; font-size: 12px; color:#64748b; }}
        table {{ width:100%; border-collapse: collapse; margin-top: 14px; font-size: 13px; }}
        th {{ background:#0f172a; color:white; text-align:left; padding:10px; }}
        td {{ border-bottom:1px solid #eee; padding:10px; vertical-align: top; }}
        tr:nth-child(even) {{ background:#fafafa; }}
        .footer {{ margin-top: 18px; font-size: 11px; color:#94a3b8; text-align:center; }}
        @page {{ size: A4; margin: 10mm; }}
      </style>
    </head>
    <body>
      <div class="wrap">
        <h1>{titulo}</h1>
        <div class="muted">Gerado em: {datetime.now().strftime("%d/%m/%Y %H:%M")}</div>
        <div class="muted">Critério: pior KM/L médio recente (exemplo) — ajuste conforme sua regra.</div>

        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Motorista</th>
              <th>Chapa</th>
              <th>KM/L Médio</th>
              <th>Dias</th>
            </tr>
          </thead>
          <tbody>
            {rows}
          </tbody>
        </table>

        <div class="footer">Ordem de Acompanhamento — Inove/Desempenho Diesel</div>
      </div>
    </body>
    </html>
    """
    html_path.write_text(html, encoding="utf-8")

def criar_acompanhamentos(alvos: pd.DataFrame):
    """
    Cria N registros em diesel_acompanhamentos (ou tabela que você definir).
    Campos mínimos (ajuste no seu schema):
    - motorista_chapa, motorista_nome, motivo, status, dias_monitoramento, dt_inicio, dt_fim_planejado
    """
    if alvos.empty:
        return []

    hoje = datetime.now().date()
    payload = []
    for _, r in alvos.iterrows():
        payload.append({
            "motorista_chapa": str(r.get("chapa")),
            "motorista_nome": str(r.get("nome") or ""),
            "motivo": "ACOMPANHAMENTO KM/L (AUTO)",
            "status": "ACOMPANHAMENTO",
            "dias_monitoramento": 7,
            "dt_inicio": str(hoje),
            "dt_fim_planejado": str(hoje),  # você pode colocar +7 aqui se quiser
        })

    resp = sb().table(TBL_ACOMP).insert(payload).select("id").execute()
    ids = [x.get("id") for x in (resp.data or []) if x.get("id")]
    return ids

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    update_lote("PROCESSANDO", qtd=QTD)

    try:
        alvos = pick_alvos(QTD)
        if alvos.empty:
            update_lote("ERRO", erro_msg="Sem alvos encontrados.")
            raise RuntimeError("Sem alvos encontrados.")

        # cria acompanhamentos no banco
        ids = criar_acompanhamentos(alvos)

        # gera HTML/PDF do lote
        lote_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        titulo = f"Ordens de Acompanhamento — Lote {lote_ts}"

        html_path = Path(OUTPUT_DIR) / "ordens_acompanhamento.html"
        pdf_path = Path(OUTPUT_DIR) / "ordens_acompanhamento.pdf"

        gerar_html_ordens(alvos, html_path, titulo)
        render_pdf(html_path, pdf_path)

        # upload storage
        remote_folder = f"{REMOTE_PREFIX}/{lote_ts}/batch_{ORDEM_BATCH_ID or lote_ts}"
        remote_html = f"{remote_folder}/{html_path.name}"
        remote_pdf = f"{remote_folder}/{pdf_path.name}"

        upload(html_path, remote_html, "text/html; charset=utf-8")
        upload(pdf_path, remote_pdf, "application/pdf")

        # atualiza lote
        update_lote(
            "CONCLUIDO",
            arquivo_path=remote_pdf,
            arquivo_nome=safe_filename(f"Ordens_Acompanhamento_{lote_ts}.pdf"),
            ids_acompanhamentos=ids,
            erro_msg=None,
        )

        print("✅ OK: lote gerado, acompanhamentos criados, PDF/HTML enviados ao bucket.")

    except Exception as e:
        err = repr(e)
        print("❌ ERRO:", err)
        update_lote("ERRO", erro_msg=err)
        raise

if __name__ == "__main__":
    main()
