import os
import vertexai
from vertexai.generative_models import GenerativeModel

@app.get("/vertex/ping")
def vertex_ping():
    # Lê do Render ENV
    project_id = os.environ.get("VERTEX_PROJECT_ID")
    location = os.environ.get("VERTEX_LOCATION", "us-central1")

    if not project_id:
        return {"ok": False, "error": "VERTEX_PROJECT_ID não configurado"}

    # Inicializa Vertex usando ADC (GOOGLE_APPLICATION_CREDENTIALS já está ok)
    vertexai.init(project=project_id, location=location)

    # Modelo (use o mesmo que você quer usar no relatório)
    model_name = os.environ.get("VERTEX_MODEL", "gemini-2.5-pro")
    model = GenerativeModel(model_name)

    # Prompt mínimo
    resp = model.generate_content("Responda apenas: OK")

    # Retorno simples
    text = getattr(resp, "text", None) or ""
    return {"ok": True, "model": model_name, "text": text.strip()}
