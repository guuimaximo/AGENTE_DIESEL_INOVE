from fastapi import FastAPI

app = FastAPI(title="Agente Diesel API", version="0.1.0")


@app.get("/health")
def health():
    return {"ok": True, "service": "agentediesel-api"}
