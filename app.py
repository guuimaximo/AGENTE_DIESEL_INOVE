# app.py
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# =============================================================================
# APP
# =============================================================================
app = FastAPI(title="Agente Diesel API", version="2.0.0")

# =============================================================================
# CORS
# =============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://inovequatai.onrender.com",
        "http://localhost:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# ENV
# =============================================================================
SUPABASE_A_URL = os.getenv("SUPABASE_A_URL") or os.getenv("SUPABASE_URL")
SUPABASE_A_SERVICE_ROLE_KEY = os.getenv("SUPABASE_A_SERVICE_ROLE_KEY") or os.getenv(
    "SUPABASE_SERVICE_ROLE_KEY"
)

SUPABASE_B_URL = os.getenv("SUPABASE_B_URL")
SUPABASE_B_SERVICE_ROLE_KEY = os.getenv("SUPABASE_B_SERVICE_ROLE_KEY")

REPORT_SCRIPT = os.getenv("REPORT_SCRIPT", "relatorio_gerencial.py")
PRONTUARIO_SCRIPT = os.getenv("PRONTUARIO_SCRIPT", "gerar_ordens_acompanhamento.py")

REPORT_OUTPUT_DIR = os.getenv("REPORT_OUTPUT_DIR", "Relatorios_Diesel_Final")

# =============================================================================
# MODELS
# =============================================================================
class GerarPayload(BaseModel):
    tipo: str = Field(description="diesel_gerencial | prontuarios_acompanhamento")
    periodo_inicio: str | None = None
    periodo_fim: str | None = None
    qtd: int | None = None  # usado SOMENTE para prontuários
    solicitante_login: str | None = None
    solicitante_nome: str | None = None


# =============================================================================
# HELPERS
# =============================================================================
def sb_a():
    from supabase import create_client

    if not SUPABASE_A_URL or not SUPABASE_A_SERVICE_ROLE_KEY:
        raise HTTPException(400, "Supabase A não configurado")

    return create_client(SUPABASE_A_URL, SUPABASE_A_SERVICE_ROLE_KEY)


def sb_b():
    from supabase import create_client

    if not SUPABASE_B_URL or not SUPABASE_B_SERVICE_ROLE_KEY:
        raise HTTPException(400, "Supabase B não configurado")

    return create_client(SUPABASE_B_URL, SUPABASE_B_SERVICE_ROLE_KEY)


# =============================================================================
# ROOT
# =============================================================================
@app.get("/")
def root():
    return {"ok": True, "service": "agentediesel", "status": "up"}


# =============================================================================
# HISTÓRICO RELATÓRIOS
# =============================================================================
@app.get("/relatorios/historico")
def historico(tipo: str | None = None, limit: int = 50):
    sb = sb_b()
    q = (
        sb.table("relatorios_gerados")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
    )
    if tipo:
        q = q.eq("tipo", tipo)

    return {"ok": True, "items": q.execute().data or []}


# =============================================================================
# HISTÓRICO LOTES (ACOMPANHAMENTO)
# =============================================================================
@app.get("/acompanhamentos/lotes")
def listar_lotes(limit: int = 50):
    sb = sb_b()
    rows = (
        sb.table("acompanhamento_lotes")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
        .data
        or []
    )
    return {"ok": True, "items": rows}


# =============================================================================
# GERAR (GERENCIAL OU ACOMPANHAMENTO)
# =============================================================================
@app.post("/relatorios/gerar")
def gerar(payload: GerarPayload):
    tipo = payload.tipo.lower().strip()
    sb = sb_b()

    # -------------------------------------------------------------------------
    # GERENCIAL
    # -------------------------------------------------------------------------
    if tipo == "diesel_gerencial":
        script = Path(REPORT_SCRIPT)
        if not script.exists():
            raise HTTPException(400, f"Script não encontrado: {script}")

        # cria registro
        ins = sb.table("relatorios_gerados").insert(
            {
                "tipo": "diesel_gerencial",
                "status": "PROCESSANDO",
                "periodo_inicio": payload.periodo_inicio,
                "periodo_fim": payload.periodo_fim,
                "solicitante_login": payload.solicitante_login,
                "solicitante_nome": payload.solicitante_nome,
            }
        ).execute()

        report_id = ins.data[0]["id"]

        env = os.environ.copy()
        env["REPORT_ID"] = str(report_id)
        env["REPORT_PERIODO_INICIO"] = payload.periodo_inicio or ""
        env["REPORT_PERIODO_FIM"] = payload.periodo_fim or ""

        proc = subprocess.run(
            ["python", str(script)],
            capture_output=True,
            text=True,
            env=env,
            timeout=60 * 20,
        )

        if proc.returncode != 0:
            sb.table("relatorios_gerados").update(
                {"status": "ERRO", "erro_msg": proc.stderr[-4000:]}
            ).eq("id", report_id).execute()

            return JSONResponse(
                status_code=500,
                content={"ok": False, "report_id": report_id, "stderr": proc.stderr},
            )

        sb.table("relatorios_gerados").update({"status": "CONCLUIDO"}).eq(
            "id", report_id
        ).execute()

        return {"ok": True, "report_id": report_id}

    # -------------------------------------------------------------------------
    # ACOMPANHAMENTO / PRONTUÁRIOS (COM LOTE)
    # -------------------------------------------------------------------------
    if tipo in {"prontuarios_acompanhamento", "acompanhamento"}:
        if not payload.qtd or payload.qtd <= 0:
            raise HTTPException(400, "qtd é obrigatória para acompanhamento")

        script = Path(PRONTUARIO_SCRIPT)
        if not script.exists():
            raise HTTPException(400, f"Script não encontrado: {script}")

        # cria LOTE
        lote = sb.table("acompanhamento_lotes").insert(
            {"status": "PROCESSANDO", "qtd": payload.qtd}
        ).execute()

        lote_id = lote.data[0]["id"]

        env = os.environ.copy()
        env["ORDEM_BATCH_ID"] = str(lote_id)
        env["QTD_ACOMPANHAMENTOS"] = str(payload.qtd)

        proc = subprocess.run(
            ["python", str(script)],
            capture_output=True,
            text=True,
            env=env,
            timeout=60 * 30,
        )

        if proc.returncode != 0:
            sb.table("acompanhamento_lotes").update(
                {"status": "ERRO", "erro_msg": proc.stderr[-4000:]}
            ).eq("id", lote_id).execute()

            return JSONResponse(
                status_code=500,
                content={"ok": False, "lote_id": lote_id, "stderr": proc.stderr},
            )

        sb.table("acompanhamento_lotes").update({"status": "CONCLUIDO"}).eq(
            "id", lote_id
        ).execute()

        return {"ok": True, "lote_id": lote_id}

    # -------------------------------------------------------------------------
    raise HTTPException(400, f"Tipo inválido: {payload.tipo}")
