# app.py
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Agente Diesel API", version="1.0.0")

# =========================
# CORS (INOVE -> AGENTE)
# =========================
ALLOWED_ORIGINS = [
    "https://inovequatai.onrender.com",
    # local opcional:
    # "http://localhost:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# ENV (Render)
# =========================
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
VERTEX_MODEL = os.getenv("VERTEX_MODEL", "gemini-2.5-pro")

# --- Supabase A (onde existe premiacao_diaria) ---
SUPABASE_A_URL = os.getenv("SUPABASE_A_URL") or os.getenv("SUPABASE_URL")
SUPABASE_A_SERVICE_ROLE_KEY = os.getenv("SUPABASE_A_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# --- Supabase B (onde você salva relatorios_gerados e bucket relatorios) ---
SUPABASE_B_URL = os.getenv("SUPABASE_B_URL")
SUPABASE_B_SERVICE_ROLE_KEY = os.getenv("SUPABASE_B_SERVICE_ROLE_KEY")

SCRIPT_PATH = os.getenv("REPORT_SCRIPT", "relatorio_gerencial.py")
PASTA_SAIDA = os.getenv("REPORT_OUTPUT_DIR", "Relatorios_Diesel_Final")
REPORT_BUCKET = os.getenv("REPORT_BUCKET", "relatorios")


# =========================
# MODELOS
# =========================
class GerarRelatorioPayload(BaseModel):
    tipo: str = Field(default="diesel_gerencial")
    periodo_inicio: str | None = Field(default=None, description="YYYY-MM-DD")
    periodo_fim: str | None = Field(default=None, description="YYYY-MM-DD")
    solicitante_login: str | None = None
    solicitante_nome: str | None = None


# =========================
# HELPERS SUPABASE
# =========================
def _sb_a():
    if not SUPABASE_A_URL or not SUPABASE_A_SERVICE_ROLE_KEY:
        raise HTTPException(
            status_code=400,
            detail="SUPABASE_A_URL/SUPABASE_A_SERVICE_ROLE_KEY (ou SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY) não definidos",
        )
    from supabase import create_client
    return create_client(SUPABASE_A_URL, SUPABASE_A_SERVICE_ROLE_KEY)


def _sb_b():
    if not SUPABASE_B_URL or not SUPABASE_B_SERVICE_ROLE_KEY:
        raise HTTPException(status_code=400, detail="SUPABASE_B_URL/SUPABASE_B_SERVICE_ROLE_KEY não definidos")
    from supabase import create_client
    return create_client(SUPABASE_B_URL, SUPABASE_B_SERVICE_ROLE_KEY)


# =========================
# PARSER NUMÉRICO (varchar -> float)
# =========================
def parse_num(value) -> float:
    """
    Converte varchar numérico em float com segurança:
      - "102.9" -> 102.9
      - "102,9" -> 102.9
      - "1.234,56" -> 1234.56
      - "1,234.56" -> 1234.56
      - "" / None -> 0.0
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip()
    if not s:
        return 0.0

    # mantém apenas dígitos e separadores
    s = re.sub(r"[^0-9\.,\-]", "", s)

    # se tem vírgula e ponto, decide pelo último separador como decimal
    if "," in s and "." in s:
        last_comma = s.rfind(",")
        last_dot = s.rfind(".")
        if last_comma > last_dot:
            # BR: 1.234,56  -> remove milhares '.' e troca ',' por '.'
            s = s.replace(".", "").replace(",", ".")
        else:
            # US: 1,234.56 -> remove milhares ',' e mantém '.'
            s = s.replace(",", "")
    else:
        # se só tem vírgula, é decimal
        if "," in s and "." not in s:
            s = s.replace(".", "").replace(",", ".")
        # se só tem ponto, assume decimal normal

    try:
        return float(s)
    except Exception:
        return 0.0


# =========================
# ROOT / HEALTH
# =========================
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
# PREMIACAO (SUPABASE A) - RESUMO CORRETO
# =========================
@app.get("/premiacao/resumo")
def premiacao_resumo(
    chapa: str = Query(..., description="Chapa do motorista (coluna motorista na premiacao_diaria)"),
    inicio: str = Query(..., description="YYYY-MM-DD"),
    fim: str = Query(..., description="YYYY-MM-DD"),
):
    """
    Lê premiacao_diaria no Supabase A (campos varchar) e devolve:
      - totais: dias, km, litros, kml
      - veiculos: agregação por veiculo (dias, km, litros, kml)
    """
    sb = _sb_a()

    ch = str(chapa).strip()
    di = str(inicio).strip()
    df = str(fim).strip()

    resp = (
        sb.table("premiacao_diaria")
        .select("dia, veiculo, km_rodado, combustivel_consumido")
        .eq("motorista", ch)
        .gte("dia", di)
        .lte("dia", df)
        .limit(200000)
        .execute()
    )

    rows = resp.data or []

    dias_set = set()
    total_km = 0.0
    total_l = 0.0

    by_v: Dict[str, Dict[str, Any]] = {}

    for r in rows:
        dia = r.get("dia")
        if dia:
            dias_set.add(str(dia))

        veic = str(r.get("veiculo") or "").strip() or "SEM_VEICULO"
        km = parse_num(r.get("km_rodado"))
        lt = parse_num(r.get("combustivel_consumido"))

        total_km += km
        total_l += lt

        if veic not in by_v:
            by_v[veic] = {"veiculo": veic, "dias_set": set(), "km": 0.0, "litros": 0.0}

        if dia:
            by_v[veic]["dias_set"].add(str(dia))
        by_v[veic]["km"] += km
        by_v[veic]["litros"] += lt

    veiculos: List[Dict[str, Any]] = []
    for veic, agg in by_v.items():
        km = float(agg["km"])
        lt = float(agg["litros"])
        veiculos.append(
            {
                "veiculo": veic,
                "dias": len(agg["dias_set"]),
                "km": round(km, 2),
                "litros": round(lt, 2),
                "kml": round((km / lt), 2) if lt > 0 else 0.0,
            }
        )

    veiculos.sort(key=lambda x: x["km"], reverse=True)

    return {
        "ok": True,
        "chapa": ch,
        "periodo_inicio": di,
        "periodo_fim": df,
        "totais": {
            "dias": len(dias_set),
            "km": round(total_km, 2),
            "litros": round(total_l, 2),
            "kml": round((total_km / total_l), 2) if total_l > 0 else 0.0,
        },
        "veiculos": veiculos,
        "rows": len(rows),
    }


# =========================
# HISTÓRICO (SUPABASE B) - para o INOVE
# =========================
@app.get("/relatorios/historico")
def relatorios_historico(
    tipo: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
):
    """
    Retorna lista de relatórios já gerados (Supabase B: relatorios_gerados).
    """
    sb = _sb_b()
    q = (
        sb.table("relatorios_gerados")
        .select(
            "id, created_at, tipo, status, periodo_inicio, periodo_fim, arquivo_path, arquivo_nome, mime_type, tamanho_bytes, erro_msg"
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
    Se NÃO for público: esse endpoint gera signed URL (recomendado).
    """
    sb = _sb_b()
    try:
        signed = sb.storage.from_(REPORT_BUCKET).create_signed_url(path, 3600)
        url = signed.get("signedURL") or signed.get("signedUrl") or signed.get("signed_url") or signed.get("url")
        if not url:
            return {"ok": False, "detail": "Não foi possível gerar signed url", "raw": signed}
        return {"ok": True, "url": url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro signed url: {repr(e)}")


# =========================
# GERAR RELATÓRIO (SUPABASE B)
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
    sb = _sb_b()
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
