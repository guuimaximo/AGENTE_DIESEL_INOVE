# app.py
import os
from fastapi import FastAPI, HTTPException

app = FastAPI(title="Agente Diesel API", version="1.0.0")


@app.get("/")
def root():
    return {
        "service": "agentediesel",
        "status": "ok",
        "endpoints": ["/health", "/vertex/ping"],
    }


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/vertex/ping")
def vertex_ping():
    """
    Testa:
    - se o pacote do Vertex está instalado (google-cloud-aiplatform -> vertexai)
    - se as credenciais (GOOGLE_APPLICATION_CREDENTIALS) estão acessíveis
    - se o Vertex responde gerando uma saída mínima
    """
    project = os.environ.get("VERTEX_PROJECT_ID") or os.environ.get("PROJECT_ID")
    location = os.environ.get("VERTEX_LOCATION") or os.environ.get("LOCATION") or "us-central1"
    model_name = os.environ.get("VERTEX_MODEL") or "gemini-2.5-pro"

    if not project:
        raise HTTPException(
            status_code=400,
            detail="VERTEX_PROJECT_ID (ou PROJECT_ID) não definido nas variáveis de ambiente.",
        )

    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path:
        raise HTTPException(
            status_code=400,
            detail="GOOGLE_APPLICATION_CREDENTIALS não definido. Aponte para o arquivo JSON do service account.",
        )
    if not os.path.exists(creds_path):
        raise HTTPException(
            status_code=400,
            detail=f"GOOGLE_APPLICATION_CREDENTIALS aponta para '{creds_path}', mas o arquivo não existe.",
        )

    # Import dentro da rota para retornar erro claro caso o pacote não esteja no requirements
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "Falha ao importar Vertex AI. Confirme se 'google-cloud-aiplatform' está no requirements.txt. "
                f"Erro: {repr(e)}"
            ),
        )

    try:
        vertexai.init(project=project, location=location)
        model = GenerativeModel(model_name)

        resp = model.generate_content("Responda apenas com: OK")
        text = getattr(resp, "text", None) or ""
        text = (text or "").strip()

        return {
            "ok": True,
            "project": project,
            "location": location,
            "model": model_name,
            "reply": text,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Vertex ping falhou ao gerar conteúdo. Erro: {repr(e)}",
        )
