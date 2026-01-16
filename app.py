# app.py (Agente Diesel API) — COMPLETO
# - CORS para o INOVE
# - POST /relatorios/gerar (cria registro, roda o script)
# - GET  /relatorios/listar (lista últimos relatórios)
# - GET  /relatorios/{report_id}/html (retorna HTML pronto para embed no INOVE, com PNG via signed URL)
# Requisitos:
# - Variáveis: SUPABASE_B_URL, SUPABASE_B_SERVICE_ROLE_KEY, REPORT_SCRIPT, REPORT_OUTPUT_DIR
# - Bucket no Supabase B: relatorios (privado)
# - Tabela no Supabase B: public.relatorios_gerados (com arquivo_path do HTML)
#
# Observação importante:
# - O script relatorio_gerencial.py deve salvar no bucket:
#   diesel/<mes_ref>/report_<REPORT_ID>/Relatorio_Gerencial.html
#   diesel/<mes_ref>/report_<REPORT_ID>/cluster_evolution_unificado.png
# - E atualizar relatorios_gerados.arquivo_path com o caminho do HTML.

import os
import re
import subprocess
from pathlib import Path
from typing import Optional, Any, Dict, List

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field


app = FastAPI(title="Agente Diesel API", version="1.0.0")

# =========================
# CORS (INOVE -> AGENTE)
# =========================
ALLOWED_ORIGINS = [
    "https://inovequatai.onrender.com",
    "http://localhost:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,  # sem cookies/sessão
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# ENV (Render)
# =========================
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
VERTEX_MODEL = os.getenv("VERTEX_MODEL", "gemini-2.5-pro")

SUPABASE_B_URL = os.getenv("SUPABASE_B_URL")
SUPABASE_B_SERVICE_ROLE_KEY = os.getenv("SUPABASE_B_SERVICE_ROLE_KEY")

SCRIPT_PATH = os.getenv("REPORT_SCRIPT", "relatorio_gerencial.py")
PASTA_SAIDA = os.getenv("REPORT_OUTPUT_DIR", "Relatorios_Diesel_Final")

# Bucket no Supabase B onde os relatórios ficam
BUCKET_RELATORIOS = os.getenv("REPORT_BUCKET", "relatorios")

# TTL das signed URLs (segundos)
SIGNED_URL_TTL = int(os.getenv("SIGNED_URL_TTL", "600"))  # 10 min


# =========================
# Models
# =========================
class GerarRelatorioPayload(BaseModel):
    tipo: str = Field(default="diesel_gerencial")
    periodo_inicio: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    periodo_fim: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    solicitante_login: Optional[str] = None
    solicitante_nome: Optional[str] = None

    # filtros opcionais (se você decidir aplicar no script depois)
    motorista: Optional[str] = None
    linha: Optional[str] = None
    veiculo: Optional[str] = None
    cluster: Optional[str] = None


# =========================
# Supabase helpers
# =========================
def sb_b():
    if not SUPABASE_B_URL or not SUPABASE_B_SERVICE_ROLE_KEY:
        raise HTTPException(status_code=400, detail="SUPABASE_B_URL/SUPABASE_B_SERVICE_ROLE_KEY não definidos")
    from supabase import create_client
    return create_client(SUPABASE_B_URL, SUPABASE_B_SERVICE_ROLE_KEY)


def _signed_url_from_resp(resp: Any) -> Optional[str]:
    """
    Compatibilidade: libs diferentes retornam signedURL / signedUrl.
    """
    if not resp:
        return None
    if isinstance(resp, dict):
        return resp.get("signedURL") or resp.get("signedUrl") or resp.get("signed_url")
    # alguns retornos vêm como objeto com atributos
    return getattr(resp, "signedURL", None) or getattr(resp, "signedUrl", None)


def create_signed_url(path: str) -> str:
    client = sb_b()
    storage = client.storage.from_(BUCKET_RELATORIOS)
    resp = storage.create_signed_url(path, SIGNED_URL_TTL)
    url = _signed_url_from_resp(resp)
    if not url:
        raise HTTPException(status_code=500, detail=f"Não gerou signed URL para: {path}")
    return url


# =========================
# Rotas
# =========================
@app.get("/")
def root():
    return {"ok": True, "service": "agentediesel", "status": "up"}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/vertex/ping")
def vertex_ping():
    """
    Endpoint para validar credenciais e chamada do Vertex.
    Importa dentro da rota para não quebrar o boot do servidor se lib não estiver ok.
    """
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha import Vertex SDK: {repr(e)}")

    if not VERTEX_PROJECT_ID:
        raise HTTPException(status_code=400, detail="VERTEX_PROJECT_ID não definido")
    if not VERTEX_LOCATION:
        raise HTTPException(status_code=400, detail="VERTEX_LOCATION não definido")

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


@app.post("/relatorios/gerar")
def gerar_relatorio(payload: Optional[GerarRelatorioPayload] = None):
    """
    Fluxo:
    1) Cria registro PROCESSANDO no Supabase B (relatorios_gerados)
    2) Executa relatorio_gerencial.py passando REPORT_ID e filtros via ENV
    3) Script: busca Supabase A, gera arquivos, sobe no bucket relatorios, marca CONCLUIDO/ERRO
       - IMPORTANTE: script deve salvar arquivo_path apontando pro HTML (não PDF)
    4) Retorna o id e logs (tail) para debug
    """
    payload = payload or GerarRelatorioPayload()

    # Valida script
    script_file = Path(SCRIPT_PATH)
    if not script_file.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Script não encontrado: {SCRIPT_PATH}. Ajuste REPORT_SCRIPT ou renomeie o arquivo no repo.",
        )

    # Cria registro PROCESSANDO no Supabase B
    try:
        client = sb_b()
        ins = {
            "tipo": payload.tipo,
            "status": "PROCESSANDO",
            "periodo_inicio": payload.periodo_inicio,
            "periodo_fim": payload.periodo_fim,
            "solicitante_login": payload.solicitante_login,
            "solicitante_nome": payload.solicitante_nome,
        }
        resp = client.table("relatorios_gerados").insert(ins).execute()
        if not resp.data:
            raise RuntimeError("Insert não retornou dados.")
        report_id = resp.data[0]["id"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao criar relatorio_gerados (Supabase B): {repr(e)}")

    # Pasta local (debug)
    out_dir = Path(PASTA_SAIDA)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Executa script (passa params via ENV)
    env = os.environ.copy()
    env["REPORT_ID"] = str(report_id)
    env["REPORT_TIPO"] = payload.tipo
    if payload.periodo_inicio:
        env["REPORT_PERIODO_INICIO"] = payload.periodo_inicio
    if payload.periodo_fim:
        env["REPORT_PERIODO_FIM"] = payload.periodo_fim

    # filtros opcionais (se você quiser aplicar no relatorio_gerencial.py depois)
    if payload.motorista:
        env["REPORT_FILTRO_MOTORISTA"] = payload.motorista
    if payload.linha:
        env["REPORT_FILTRO_LINHA"] = payload.linha
    if payload.veiculo:
        env["REPORT_FILTRO_VEICULO"] = payload.veiculo
    if payload.cluster:
        env["REPORT_FILTRO_CLUSTER"] = payload.cluster

    try:
        proc = subprocess.run(
            ["python", str(script_file)],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            timeout=60 * 20,  # 20 min
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
                "stdout_tail": (proc.stdout or "")[-2000:],
                "stderr_tail": (proc.stderr or "")[-4000:],
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


@app.get("/relatorios/listar")
def listar_relatorios(limit: int = 50):
    """
    Lista os relatórios mais recentes do Supabase B.
    """
    try:
        client = sb_b()
        resp = (
            client.table("relatorios_gerados")
            .select("id, created_at, tipo, status, periodo_inicio, periodo_fim, arquivo_path, arquivo_nome, mime_type, tamanho_bytes, erro_msg")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return {"ok": True, "data": resp.data or []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao listar relatorios_gerados: {repr(e)}")


@app.get("/relatorios/{report_id}/html", response_class=HTMLResponse)
def abrir_html(report_id: str):
    """
    Retorna o HTML do relatório para embed no INOVE (iframe),
    já com o <img> apontando para signed URL do PNG.

    Regras:
    - relatorios_gerados.arquivo_path deve apontar para o HTML no bucket (privado)
    - PNG deve estar no mesmo folder com nome cluster_evolution_unificado.png
    """
    client = sb_b()

    # 1) Pega arquivo_path do HTML
    try:
        r = (
            client.table("relatorios_gerados")
            .select("id, status, arquivo_path, mime_type")
            .eq("id", report_id)
            .single()
            .execute()
        )
        row = r.data or None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao ler relatorios_gerados: {repr(e)}")

    if not row:
        raise HTTPException(status_code=404, detail="Relatório não encontrado")

    if row.get("status") != "CONCLUIDO":
        raise HTTPException(status_code=409, detail=f"Relatório ainda não concluído (status={row.get('status')})")

    html_path = row.get("arquivo_path")
    if not html_path:
        raise HTTPException(status_code=404, detail="Relatório sem arquivo_path")

    # 2) Deduz png_path (mesmo folder)
    folder = html_path.rsplit("/", 1)[0]
    png_path = f"{folder}/cluster_evolution_unificado.png"

    # 3) Signed URL do HTML + PNG
    html_url = create_signed_url(html_path)
    png_url = create_signed_url(png_path)

    # 4) Baixa HTML e injeta o png_url
    try:
        html_text = requests.get(html_url, timeout=30).text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao baixar HTML assinado: {repr(e)}")

    # Substitui o <img src="..."> que contém cluster_evolution_unificado.png
    # (se não encontrar, injeta mesmo assim no primeiro <img>)
    new_html = re.sub(
        r'(<img[^>]+src=")[^"]*(cluster_evolution_unificado\.png[^"]*)(")',
        r'\1' + png_url + r'\3',
        html_text,
        count=1,
        flags=re.IGNORECASE,
    )

    if new_html == html_text:
        # fallback: troca o primeiro src= de img, caso o nome não esteja no HTML
        new_html = re.sub(
            r'(<img[^>]+src=")[^"]*(")',
            r"\1" + png_url + r"\2",
            html_text,
            count=1,
            flags=re.IGNORECASE,
        )

    return HTMLResponse(content=new_html, status_code=200)


@app.get("/relatorios/{report_id}/download")
def download_arquivo_principal(report_id: str):
    """
    Retorna signed URL do arquivo principal (arquivo_path) — útil se quiser abrir em nova aba.
    """
    client = sb_b()
    try:
        r = (
            client.table("relatorios_gerados")
            .select("id, status, arquivo_path")
            .eq("id", report_id)
            .single()
            .execute()
        )
        row = r.data or None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao ler relatorios_gerados: {repr(e)}")

    if not row:
        raise HTTPException(status_code=404, detail="Relatório não encontrado")
    if row.get("status") != "CONCLUIDO":
        raise HTTPException(status_code=409, detail=f"Relatório ainda não concluído (status={row.get('status')})")

    p = row.get("arquivo_path")
    if not p:
        raise HTTPException(status_code=404, detail="Relatório sem arquivo_path")

    return {"ok": True, "signed_url": create_signed_url(p)}
