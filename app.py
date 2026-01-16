import os
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
    # se você testar local:
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
    if not SUPABASE_B_URL or not SUPABASE_B_SERVICE_ROLE_KEY:
        raise HTTPException(status_code=400, detail="SUPABASE_B_URL/SUPABASE_B_SERVICE_ROLE_KEY não definidos")
    from supabase import create_client
    return create_client(SUPABASE_B_URL, SUPABASE_B_SERVICE_ROLE_KEY)


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
        .select("id, created_at, tipo, status, periodo_inicio, periodo_fim, arquivo_path, arquivo_nome, mime_type, tamanho_bytes, erro_msg")
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
        # signed url (expira em 1h)
        signed = sb.storage.from_(REPORT_BUCKET).create_signed_url(path, 3600)
        # libs variam: às vezes vem {"signedURL": "..."} / "signedUrl"
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

    # Valida script
    script_file = Path(SCRIPT_PATH)
    if not script_file.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Script não encontrado: {SCRIPT_PATH}. Ajuste REPORT_SCRIPT ou inclua o arquivo no repo.",
        )

    # Cria registro PROCESSANDO no Supabase B
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

    # Pasta de saída local
    out_dir = Path(PASTA_SAIDA)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Executa script
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
        return JSONResponse(status_code=500, content={"ok": False, "report_id": report_id, "error": "Timeout ao executar script"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "report_id": report_id, "error": f"Falha ao executar script: {repr(e)}"})

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
