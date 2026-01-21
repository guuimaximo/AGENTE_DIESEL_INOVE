import os
import math
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Agente Diesel API", version="1.0.0")

# =========================
# CORS (INOVE -> AGENTE)
# =========================
ALLOWED_ORIGINS = [
    "https://inovequatai.onrender.com",
    # "http://localhost:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== ENV (Render) ======
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
VERTEX_MODEL = os.getenv("VERTEX_MODEL", "gemini-2.5-pro")

# ✅ Supabase A (premiacao_diaria)
SUPABASE_A_URL = os.getenv("SUPABASE_A_URL")
SUPABASE_A_SERVICE_ROLE_KEY = os.getenv("SUPABASE_A_SERVICE_ROLE_KEY")

# ✅ Supabase B (seu sistema principal / relatorios)
SUPABASE_B_URL = os.getenv("SUPABASE_B_URL")
SUPABASE_B_SERVICE_ROLE_KEY = os.getenv("SUPABASE_B_SERVICE_ROLE_KEY")

SCRIPT_PATH = os.getenv("REPORT_SCRIPT", "relatorio_gerencial.py")
PASTA_SAIDA = os.getenv("REPORT_OUTPUT_DIR", "Relatorios_Diesel_Final")

REPORT_BUCKET = os.getenv("REPORT_BUCKET", "relatorios")


class GerarRelatorioPayload(BaseModel):
    tipo: str = Field(default="diesel_gerencial")
    periodo_inicio: str | None = Field(default=None, description="YYYY-MM-DD")
    periodo_fim: str | None = Field(default=None, description="YYYY-MM-DD")
    solicitante_login: str | None = None
    solicitante_nome: str | None = None


def _sb():
    # Supabase B (mantém seu padrão atual)
    if not SUPABASE_B_URL or not SUPABASE_B_SERVICE_ROLE_KEY:
        raise HTTPException(status_code=400, detail="SUPABASE_B_URL/SUPABASE_B_SERVICE_ROLE_KEY não definidos")
    from supabase import create_client
    return create_client(SUPABASE_B_URL, SUPABASE_B_SERVICE_ROLE_KEY)


def _sb_a():
    # Supabase A (premiacao_diaria)
    if not SUPABASE_A_URL or not SUPABASE_A_SERVICE_ROLE_KEY:
        raise HTTPException(status_code=400, detail="SUPABASE_A_URL/SUPABASE_A_SERVICE_ROLE_KEY não definidos")
    from supabase import create_client
    return create_client(SUPABASE_A_URL, SUPABASE_A_SERVICE_ROLE_KEY)


def _to_float(v):
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        x = float(v)
        return x if math.isfinite(x) else 0.0
    s = str(v).strip()
    if not s:
        return 0.0
    # BR: 1.234,56 -> 1234.56
    s2 = s.replace(".", "").replace(",", ".")
    try:
        x = float(s2)
        return x if math.isfinite(x) else 0.0
    except:
        return 0.0


@app.get("/")
def root():
    return {"ok": True, "service": "agentediesel", "status": "up"}


@app.get("/vertex/ping")
def vertex_ping():
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha import Vertex SDK: {repr(e)}")

    if not VERTEX_PROJECT_ID:
        raise HTTPException(status_code=400, detail="VERTEX_PROJECT_ID não definido")

    try:
        vertexai.init(project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION)
        model = GenerativeModel(VERTEX_MODEL)
        resp = model.generate_content("Responda apenas: OK")
        text = getattr(resp, "text", None) or "OK"
        return {
            "ok": True,
            "project": VERTEX_PROJECT_ID,
            "location": VERTEX_LOCATION,
            "model": VERTEX_MODEL,
            "reply": text.strip(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro Vertex ping: {repr(e)}")


# =========================
# ✅ NOVO: RESUMO PREMIACAO (Supabase A / premiacao_diaria)
# =========================
@app.get("/premiacao/resumo")
def premiacao_resumo(
    chapa: str = Query(..., description="Chapa do motorista"),
    inicio: str = Query(..., description="YYYY-MM-DD"),
    fim: str = Query(..., description="YYYY-MM-DD"),
):
    """
    Consulta Supabase A: premiacao_diaria
    Filtra por motorista(chapa) e período (dia entre inicio e fim)
    Retorna resumo por veículo + totais.
    """
    sb = _sb_a()

    chapa = (chapa or "").strip()
    inicio = (inicio or "").strip()
    fim = (fim or "").strip()

    if not chapa or not inicio or not fim:
        raise HTTPException(status_code=400, detail="Informe chapa, inicio e fim (YYYY-MM-DD).")

    try:
        resp = (
            sb.table("premiacao_diaria")
            .select("dia, veiculo, km_rodado, combustivel_consumido")
            .eq("motorista", chapa)
            .gte("dia", inicio)
            .lte("dia", fim)
            .limit(20000)
            .execute()
        )
        rows = resp.data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao consultar premiacao_diaria: {repr(e)}")

    by_v = {}
    dias_total = set()

    for r in rows:
        dia = r.get("dia")
        if dia:
            dias_total.add(str(dia))

        veiculo = (r.get("veiculo") or "").strip() or "SEM_VEICULO"
        km = _to_float(r.get("km_rodado"))
        litros = _to_float(r.get("combustivel_consumido"))

        cur = by_v.get(veiculo) or {"veiculo": veiculo, "km": 0.0, "litros": 0.0, "dias": set()}
        cur["km"] += km
        cur["litros"] += litros
        if dia:
            cur["dias"].add(str(dia))
        by_v[veiculo] = cur

    veiculos = []
    for veic, cur in by_v.items():
        km = cur["km"]
        litros = cur["litros"]
        veiculos.append(
            {
                "veiculo": veic,
                "dias": len(cur["dias"]),
                "km": round(km, 2),
                "litros": round(litros, 2),
                "kml": round(km / litros, 4) if litros > 0 else 0.0,
            }
        )

    veiculos.sort(key=lambda x: x["km"], reverse=True)

    total_km = sum(x["km"] for x in veiculos)
    total_litros = sum(x["litros"] for x in veiculos)
    total_kml = round(total_km / total_litros, 4) if total_litros > 0 else 0.0

    return {
        "ok": True,
        "chapa": chapa,
        "inicio": inicio,
        "fim": fim,
        "totais": {
            "dias": len(dias_total),
            "km": round(total_km, 2),
            "litros": round(total_litros, 2),
            "kml": total_kml,
        },
        "veiculos": veiculos,
    }


# =========================
# HISTÓRICO (para o INOVE)
# =========================
@app.get("/relatorios/historico")
def relatorios_historico(
    tipo: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
):
    """
    Retorna lista de relatórios já gerados (Supabase B: relatorios_gerados).
    """
    sb = _sb()
    q = (
        sb.table("relatorios_gerados")
        .select(
            "id, created_at, tipo, status, periodo_inicio, periodo_fim, "
            "arquivo_path, arquivo_nome, mime_type, tamanho_bytes, erro_msg"
        )
        .order("created_at", desc=True)
        .limit(limit)
    )
    if tipo:
        q = q.eq("tipo", tipo)

    resp = q.execute()
    return {"ok": True, "items": resp.data or []}


@app.get("/relatorios/url")
def relatorio_url(path: str = Query(..., description="arquivo_path no bucket")):
    """
    Se o bucket for público: você pode montar URL pública no front.
    Se NÃO for público: esse endpoint pode gerar signed URL (recomendado).
    """
    sb = _sb()
    try:
        signed = sb.storage.from_(REPORT_BUCKET).create_signed_url(path, 3600)
        url = signed.get("signedURL") or signed.get("signedUrl") or signed.get("signed_url") or signed.get("url")
        if not url:
            return {"ok": False, "detail": "Não foi possível gerar signed url", "raw": signed}
        return {"ok": True, "url": url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro signed url: {repr(e)}")


# =========================
# GERAR RELATÓRIO
# =========================
@app.post("/relatorios/gerar")
def gerar_relatorio(payload: GerarRelatorioPayload | None = None):
    payload = payload or GerarRelatorioPayload()

    script_file = Path(SCRIPT_PATH)
    if not script_file.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Script não encontrado: {SCRIPT_PATH}. Ajuste REPORT_SCRIPT ou inclua o arquivo no repo.",
        )

    sb = _sb()
    try:
        ins = {
            "tipo": payload.tipo,
            "status": "PROCESSANDO",
            "periodo_inicio": payload.periodo_inicio,
            "periodo_fim": payload.periodo_fim,
            "solicitante_login": payload.solicitante_login,
            "solicitante_nome": payload.solicitante_nome,
        }
        resp = sb.table("relatorios_gerados").insert(ins).execute()
        if not resp.data:
            raise RuntimeError("Insert não retornou dados.")
        report_id = resp.data[0]["id"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao criar relatorio_gerados (Supabase B): {repr(e)}")

    out_dir = Path(PASTA_SAIDA)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["REPORT_ID"] = str(report_id)
    env["REPORT_TIPO"] = payload.tipo
    if payload.periodo_inicio:
        env["REPORT_PERIODO_INICIO"] = payload.periodo_inicio
    if payload.periodo_fim:
        env["REPORT_PERIODO_FIM"] = payload.periodo_fim

    try:
        proc = subprocess.run(
            ["python", str(script_file)],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            timeout=60 * 20,
        )
    except subprocess.TimeoutExpired:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "report_id": report_id, "error": "Timeout ao executar script"},
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "report_id": report_id, "error": f"Falha ao executar script: {repr(e)}"},
        )

    if proc.returncode != 0:
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "report_id": report_id,
                "error": "Script retornou erro",
                "returncode": proc.returncode,
                "stdout": (proc.stdout or "")[-4000:],
                "stderr": (proc.stderr or "")[-4000:],
            },
        )

    arquivos = [p.name for p in sorted(out_dir.glob("*")) if p.is_file()]
    return {
        "ok": True,
        "report_id": report_id,
        "message": "Relatório solicitado e processado",
        "output_dir": str(out_dir),
        "files_local": arquivos,
        "stdout_tail": (proc.stdout or "")[-2000:],
    }
