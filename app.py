import os
import json
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

# Cria o app ANTES de declarar rotas
app = FastAPI(title="Agente Diesel API", version="1.0.0")

# ====== Config padrão (usa env vars do Render) ======
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
VERTEX_MODEL = os.getenv("VERTEX_MODEL", "gemini-2.5-pro")

# Caminho do seu script (NÃO muda seu código, só chama ele)
SCRIPT_PATH = os.getenv("REPORT_SCRIPT", "relatorio_gerencial.py")

# Onde seu script grava (precisa bater com o que está no seu código)
PASTA_SAIDA = os.getenv("REPORT_OUTPUT_DIR", "Relatorios_Diesel_Final")


@app.get("/")
def root():
    return {"ok": True, "service": "agentediesel", "status": "up"}


@app.get("/vertex/ping")
def vertex_ping():
    """
    Endpoint para validar credenciais e chamada do Vertex.
    """
    # Importa dentro da rota para não quebrar o boot do servidor
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
def gerar_relatorio():
    """
    Roda seu script exatamente como está (subprocess).
    Requisitos:
      - o arquivo SCRIPT_PATH deve existir no repo
      - o script deve gerar arquivos dentro de PASTA_SAIDA (como no seu código)
    """
    script_file = Path(SCRIPT_PATH)
    if not script_file.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Script não encontrado: {SCRIPT_PATH}. Ajuste REPORT_SCRIPT ou renomeie o arquivo no repo.",
        )

    # Garante pasta de saída
    out_dir = Path(PASTA_SAIDA)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Executa o script com o Python do ambiente
    # (isso não altera nada no seu código, só roda)
    try:
        proc = subprocess.run(
            ["python", str(script_file)],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao executar script: {repr(e)}")

    # Se deu erro, devolve logs para debug
    if proc.returncode != 0:
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": "Script retornou erro",
                "returncode": proc.returncode,
                "stdout": (proc.stdout or "")[-4000:],  # corta para não explodir resposta
                "stderr": (proc.stderr or "")[-4000:],
            },
        )

    # Lista os arquivos gerados
    arquivos = []
    for p in sorted(out_dir.glob("*")):
        if p.is_file():
            arquivos.append(p.name)

    return {
        "ok": True,
        "message": "Relatório gerado com sucesso",
        "output_dir": str(out_dir),
        "files": arquivos,
        "stdout_tail": (proc.stdout or "")[-2000:],
    }
