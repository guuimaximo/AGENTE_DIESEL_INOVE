import os
import io
from datetime import datetime, date
from typing import Optional, List, Dict, Any

import pandas as pd
from dateutil.parser import isoparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from supabase import create_client, Client

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


# =========================
# CONFIG
# =========================
BUCKET_RELATORIOS = os.getenv("BUCKET_RELATORIOS", "relatorios")
SIGNED_URL_TTL = int(os.getenv("SIGNED_URL_TTL", "3600"))  # segundos (1h)


def get_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v


def sb_a() -> Client:
    return create_client(get_env("SUPABASE_A_URL"), get_env("SUPABASE_A_SERVICE_ROLE_KEY"))


def sb_b() -> Client:
    return create_client(get_env("SUPABASE_B_URL"), get_env("SUPABASE_B_SERVICE_ROLE_KEY"))


def parse_date_any(v: str) -> date:
    # aceita "2026-01-01" ou ISO
    try:
        if len(v) == 10 and v[4] == "-" and v[7] == "-":
            return datetime.strptime(v, "%Y-%m-%d").date()
        return isoparse(v).date()
    except Exception:
        raise ValueError(f"Data inválida: {v} (use YYYY-MM-DD)")


# =========================
# SCHEMAS
# =========================
class DieselGerencialRequest(BaseModel):
    data_inicio: str = Field(..., description="YYYY-MM-DD")
    data_fim: str = Field(..., description="YYYY-MM-DD")
    tipo: str = Field("diesel_gerencial", description="tipo do relatório")
    tabela_fonte: str = Field("premiacao_diaria", description="tabela/view no Supabase A")
    limite: int = Field(200000, ge=1, le=500000)


class DieselGerencialResponse(BaseModel):
    ok: bool
    relatorio_id: str
    status: str
    arquivo_path: Optional[str] = None
    signed_url: Optional[str] = None
    linhas: int
    kml_media: float
    km_total: float
    litros_total: float


# =========================
# PDF
# =========================
def build_pdf_bytes(
    titulo: str,
    periodo_inicio: date,
    periodo_fim: date,
    resumo: Dict[str, Any],
    df: pd.DataFrame,
) -> bytes:
    """
    Gera um PDF simples (A4) com resumo + primeiras linhas da tabela.
    """
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    margin = 15 * mm
    y = height - margin

    def line(txt: str, size: int = 11, bold: bool = False, dy: float = 6.5):
        nonlocal y
        if bold:
            c.setFont("Helvetica-Bold", size)
        else:
            c.setFont("Helvetica", size)
        c.drawString(margin, y, txt[:200])
        y -= dy * mm

    # Header
    line(titulo, size=16, bold=True, dy=8)
    line(f"Período: {periodo_inicio.strftime('%d/%m/%Y')} a {periodo_fim.strftime('%d/%m/%Y')}", size=11, dy=6)
    line(f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", size=10, dy=6)

    y -= 3 * mm
    line("Resumo", size=13, bold=True, dy=7)
    line(f"Registros: {resumo['linhas']}", size=11, dy=6)
    line(f"KM Total: {resumo['km_total']:.2f}", size=11, dy=6)
    line(f"Combustível (L): {resumo['litros_total']:.2f}", size=11, dy=6)
    line(f"KM/L Médio (ponderado): {resumo['kml_media']:.2f}", size=11, dy=6)

    y -= 4 * mm
    line("Amostra (primeiras linhas)", size=13, bold=True, dy=7)

    # Tabela simples (monoespaçada)
    c.setFont("Courier", 9)

    cols = ["dia_date", "linha", "motorista", "veiculo", "km_rodado", "combustivel_consumido", "minutos_em_viagem", "km_l"]
    cols_present = [col for col in cols if col in df.columns]

    # Cabeçalho
    header = " | ".join([col[:12].ljust(12) for col in cols_present])
    c.drawString(margin, y, header[:120])
    y -= 6 * mm

    # Linhas
    max_rows = 25
    for i, row in df.head(max_rows).iterrows():
        if y < margin + 15 * mm:
            c.showPage()
            y = height - margin
            c.setFont("Courier", 9)
        parts = []
        for col in cols_present:
            v = row.get(col, "")
            s = str(v)
            parts.append(s[:12].ljust(12))
        line_str = " | ".join(parts)
        c.drawString(margin, y, line_str[:120])
        y -= 5.5 * mm

    c.showPage()
    c.save()
    return buf.getvalue()


# =========================
# APP
# =========================
app = FastAPI(title="Agente Diesel API", version="1.0.0")


@app.get("/health")
def health():
    return {"ok": True, "service": "agentediesel-api"}


@app.post("/reports/diesel-gerencial", response_model=DieselGerencialResponse)
def gerar_relatorio_diesel(req: DieselGerencialRequest):
    """
    Fluxo:
    1) cria linha em public.relatorios_gerados (Supabase B) com PROCESSANDO
    2) consulta Supabase A (tabela_fonte) no período
    3) gera PDF
    4) upload no Storage do Supabase B (bucket relatorios)
    5) atualiza relatorios_gerados -> CONCLUIDO e arquivo_path
    6) retorna signed url
    """
    try:
        di = parse_date_any(req.data_inicio)
        df = parse_date_any(req.data_fim)
        if df < di:
            raise HTTPException(status_code=400, detail="data_fim deve ser >= data_inicio")

        client_a = sb_a()
        client_b = sb_b()

        # 1) cria registro PROCESSANDO no Supabase B
        ins = client_b.table("relatorios_gerados").insert({
            "tipo": req.tipo,
            "status": "PROCESSANDO",
            "periodo_inicio": di.isoformat(),
            "periodo_fim": df.isoformat(),
        }).execute()

        if not ins.data:
            raise HTTPException(status_code=500, detail="Falha ao criar registro em relatorios_gerados")

        rel = ins.data[0]
        relatorio_id = rel["id"]

        # 2) consulta Supabase A
        # Esperado (seu mapeamento): dia_date, linha, motorista, veiculo, km_rodado, combustivel_consumido, minutos_em_viagem, km_l, mes, ano
        q = (
            client_a
            .table(req.tabela_fonte)
            .select("*")
            .gte("dia_date", di.isoformat())
            .lte("dia_date", df.isoformat())
            .limit(req.limite)
        )
        res = q.execute()

        rows = res.data or []
        if len(rows) == 0:
            # atualiza com ERRO (sem dados)
            client_b.table("relatorios_gerados").update({
                "status": "ERRO",
                "erro_msg": "Nenhum dado encontrado no período.",
            }).eq("id", relatorio_id).execute()
            raise HTTPException(status_code=404, detail="Nenhum dado encontrado no período.")

        dfdata = pd.DataFrame(rows)

        # 3) normalizações (garante numéricos)
        # tenta cobrir variações (km/l vs km_l)
        if "km/l" in dfdata.columns and "km_l" not in dfdata.columns:
            dfdata["km_l"] = dfdata["km/l"]

        for col in ["km_rodado", "combustivel_consumido", "minutos_em_viagem", "km_l"]:
            if col in dfdata.columns:
                dfdata[col] = pd.to_numeric(dfdata[col], errors="coerce")

        # kml ponderado = soma(km) / soma(litros)
        km_total = float(dfdata["km_rodado"].fillna(0).sum()) if "km_rodado" in dfdata.columns else 0.0
        litros_total = float(dfdata["combustivel_consumido"].fillna(0).sum()) if "combustivel_consumido" in dfdata.columns else 0.0
        kml_media = (km_total / litros_total) if litros_total > 0 else 0.0

        resumo = {
            "linhas": int(len(dfdata)),
            "km_total": km_total,
            "litros_total": litros_total,
            "kml_media": float(kml_media),
        }

        # 4) gera PDF bytes
        titulo = "Relatório Gerencial Diesel (KM/L)"
        pdf_bytes = build_pdf_bytes(
            titulo=titulo,
            periodo_inicio=di,
            periodo_fim=df,
            resumo=resumo,
            df=dfdata,
        )

        # 5) upload no Storage do Supabase B
        # caminho: diesel/YYYY-MM/relatorio_<id>.pdf
        pasta_mes = f"{di.year}-{str(di.month).zfill(2)}"
        arquivo_nome = f"relatorio_{relatorio_id}.pdf"
        arquivo_path = f"diesel/{pasta_mes}/{arquivo_nome}"

        # upload (upsert=True para substituir em reprocesso)
        storage = client_b.storage.from_(BUCKET_RELATORIOS)
        storage.upload(
            path=arquivo_path,
            file=pdf_bytes,
            file_options={"content-type": "application/pdf", "upsert": "true"},
        )

        # 6) signed URL
        signed = storage.create_signed_url(arquivo_path, SIGNED_URL_TTL)
        signed_url = signed.get("signedURL") or signed.get("signedUrl") or None

        # 7) atualiza registro
        client_b.table("relatorios_gerados").update({
            "status": "CONCLUIDO",
            "arquivo_path": arquivo_path,
            "arquivo_nome": arquivo_nome,
            "tamanho_bytes": len(pdf_bytes),
            "mime_type": "application/pdf",
            "erro_msg": None,
        }).eq("id", relatorio_id).execute()

        return {
            "ok": True,
            "relatorio_id": relatorio_id,
            "status": "CONCLUIDO",
            "arquivo_path": arquivo_path,
            "signed_url": signed_url,
            "linhas": int(len(dfdata)),
            "kml_media": float(kml_media),
            "km_total": float(km_total),
            "litros_total": float(litros_total),
        }

    except HTTPException:
        raise
    except Exception as e:
        # tenta marcar ERRO se já tiver relatorio_id
        try:
            # best-effort: se falhar, só sobe erro
            pass
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint auxiliar para testar o bucket (sem consultar Supabase A)
@app.post("/debug/relatorio-pdf-teste")
def debug_pdf_teste():
    try:
        client_b = sb_b()
        ins = client_b.table("relatorios_gerados").insert({
            "tipo": "pdf_teste",
            "status": "PROCESSANDO",
        }).execute()
        relatorio_id = ins.data[0]["id"]

        df_fake = pd.DataFrame([
            {"dia_date": "2026-01-01", "linha": "04TR", "motorista": "30000001", "veiculo": "222201", "km_rodado": 100, "combustivel_consumido": 40, "minutos_em_viagem": 300, "km_l": 2.5},
            {"dia_date": "2026-01-02", "linha": "04TR", "motorista": "30000002", "veiculo": "222202", "km_rodado": 120, "combustivel_consumido": 48, "minutos_em_viagem": 320, "km_l": 2.5},
        ])
        km_total = float(df_fake["km_rodado"].sum())
        litros_total = float(df_fake["combustivel_consumido"].sum())
        kml_media = km_total / litros_total if litros_total > 0 else 0

        pdf_bytes = build_pdf_bytes(
            titulo="PDF Teste - Agente Diesel",
            periodo_inicio=date(2026, 1, 1),
            periodo_fim=date(2026, 1, 2),
            resumo={"linhas": len(df_fake), "km_total": km_total, "litros_total": litros_total, "kml_media": kml_media},
            df=df_fake,
        )

        pasta_mes = "teste"
        arquivo_nome = f"teste_{relatorio_id}.pdf"
        arquivo_path = f"diesel/{pasta_mes}/{arquivo_nome}"

        storage = client_b.storage.from_(BUCKET_RELATORIOS)
        storage.upload(
            path=arquivo_path,
            file=pdf_bytes,
            file_options={"content-type": "application/pdf", "upsert": "true"},
        )
        signed = storage.create_signed_url(arquivo_path, SIGNED_URL_TTL)
        signed_url = signed.get("signedURL") or signed.get("signedUrl") or None

        client_b.table("relatorios_gerados").update({
            "status": "CONCLUIDO",
            "arquivo_path": arquivo_path,
            "arquivo_nome": arquivo_nome,
            "tamanho_bytes": len(pdf_bytes),
            "mime_type": "application/pdf",
        }).eq("id", relatorio_id).execute()

        return {"ok": True, "relatorio_id": relatorio_id, "arquivo_path": arquivo_path, "signed_url": signed_url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
