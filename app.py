import os
from fastapi import FastAPI, HTTPException
from supabase import create_client, Client

app = FastAPI(title="Agente Diesel API", version="0.2.0")

def get_supabase_b() -> Client:
    url = os.getenv("SUPABASE_B_URL")
    key = os.getenv("SUPABASE_B_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_B_URL or SUPABASE_B_SERVICE_ROLE_KEY")
    return create_client(url, key)

@app.get("/health")
def health():
    return {"ok": True, "service": "agentediesel-api"}

@app.post("/debug/supabase-b/insert")
def debug_insert():
    try:
        sb = get_supabase_b()
        payload = {
            "tipo": "teste_conexao",
            "status": "PROCESSANDO",
            "arquivo_nome": None,
            "arquivo_path": None,
            "erro_msg": None,
        }
        res = sb.table("relatorios_gerados").insert(payload).execute()
        if not res.data:
            raise HTTPException(status_code=500, detail="Insert returned no data")
        return {"ok": True, "inserted": res.data[0]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
