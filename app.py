import os
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from supabase import create_client


app = FastAPI(title="Agente Diesel API", version="1.0.0")

# =========================
# CORS (INOVE -> AGENTE)
# =========================
ALLOWED_ORIGINS = [
    "https://inovequatai.onrender.com",
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
BUCKET_RELATORIOS = os.getenv("REPORT_BUCKET", "relatorios")

SCRIPT_PATH = os.getenv("REPORT_SCRIPT", "relatorio_gerencial.py")
PASTA_SAIDA = os.getenv("REPORT_OUTPUT_DIR", "Relatorios_Diesel_Final")


def sb_b():
    if not SUPABASE_B_URL or not SUPABASE_B_SERVICE_ROLE_KEY:
        raise HTTPException(
            status_code=400,
            detail="SUPABASE_B_URL/SUPABASE_B_SERVICE_ROLE_KEY não definidos",
        )
    return create_client(SUPABASE_B_URL, SUPABASE_B_SERVICE_ROLE_KEY)


class GerarRelatorioPayload(BaseModel):
    tipo: str = Field(default="diesel_gerencial")
    periodo_inicio: str | None = Field(default=None, description="YYYY-MM-DD")
    periodo_fim: str | None = Field(default=None, description="YYYY-MM-DD")
    solicitante_login: str | None = None
    solicitante_nome: str | None = None

    # NOVO: filtros do INOVE
    motorista: str | None = None
    linha: str | None = None
    veiculo: str | None = None
    cluster: str | None = None


@app.get("/")
def root():
    return {"ok": True, "service": "agentediesel", "status": "up"}


@app.post("/relatorios/gerar")
def gerar_relatorio(payload: GerarRelatorioPayload | None = None):
    payload = payload or GerarRelatorioPayload()

    script_file = Path(SCRIPT_PATH)
    if not script_file.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Script não encontrado: {SCRIPT_PATH}. Ajuste REPORT_SCRIPT ou renomeie o arquivo no repo.",
        )

    sb = sb_b()

    # 1) cria registro PROCESSANDO
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
        raise HTTPException(status_code=500, detail=f"Falha ao criar relatorio_gerados: {repr(e)}")

    # 2) roda script
    out_dir = Path(PASTA_SAIDA)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["REPORT_ID"] = str(report_id)
    env["REPORT_TIPO"] = payload.tipo

    if payload.periodo_inicio:
        env["REPORT_PERIODO_INICIO"] = payload.periodo_inicio
    if payload.periodo_fim:
        env["REPORT_PERIODO_FIM"] = payload.periodo_fim

    # NOVO: repassa filtros para o script
    env["REPORT_MOTORISTA"] = (payload.motorista or "").strip()
    env["REPORT_LINHA"] = (payload.linha or "").strip()
    env["REPORT_VEICULO"] = (payload.veiculo or "").strip()
    env["REPORT_CLUSTER"] = (payload.cluster or "").strip()

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

    # 3) resposta (o script já sobe no bucket e marca CONCLUIDO)
    arquivos = [p.name for p in sorted(out_dir.glob("*")) if p.is_file()]

    return {
        "ok": True,
        "report_id": report_id,
        "message": "Relatório gerado",
        "files_local": arquivos,
        "stdout_tail": (proc.stdout or "")[-2000:],
    }


# =========================
# LISTA (para a tela do INOVE)
# =========================
@app.get("/relatorios/listar")
def listar_relatorios(limit: int = Query(default=50, ge=1, le=200)):
    sb = sb_b()
    try:
        resp = (
            sb.table("relatorios_gerados")
            .select("id,created_at,tipo,status,periodo_inicio,periodo_fim,arquivo_path,arquivo_nome,mime_type,tamanho_bytes,erro_msg")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return {"ok": True, "items": resp.data or []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao listar relatórios: {repr(e)}")


# =========================
# SIGNED URL (abrir HTML/PDF/PNG)
# =========================
@app.get("/relatorios/{report_id}/signed-url")
def signed_url(report_id: str, path: str):
    sb = sb_b()
    try:
        # expira em 10 min
        res = sb.storage.from_(BUCKET_RELATORIOS).create_signed_url(path, 60 * 10)
        signed = (res or {}).get("signedURL") or (res or {}).get("signedUrl")
        if not signed:
            return {"ok": False, "error": "Não foi possível gerar signed url", "raw": res}
        return {"ok": True, "report_id": report_id, "path": path, "signed_url": signed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha signed url: {repr(e)}")
