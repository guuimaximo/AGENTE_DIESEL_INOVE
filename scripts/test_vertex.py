import os
import sys
import traceback

def main():
    # Você pode setar via env:
    # export VERTEX_PROJECT_ID="seu-projeto"
    # export VERTEX_LOCATION="us-central1"  (se falhar, teste "global")
    # export VERTEX_MODEL="gemini-2.5-pro" (ou "gemini-2.5-flash")

    project = os.getenv("VERTEX_PROJECT_ID") or os.getenv("PROJECT_ID")
    location = (os.getenv("VERTEX_LOCATION") or "global").strip()
    model = (os.getenv("VERTEX_MODEL") or "gemini-2.5-pro").strip()

    print("=== Vertex Test ===")
    print("PROJECT :", project)
    print("LOCATION:", location)
    print("MODEL   :", model)

    if not project:
        print("\n❌ ERRO: VERTEX_PROJECT_ID (ou PROJECT_ID) não está definido.")
        sys.exit(1)

    prompt = "Responda apenas: OK (e nada mais)."

    try:
        from google import genai
        from google.genai.types import HttpOptions

        client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            http_options=HttpOptions(api_version="v1"),
        )

        resp = client.models.generate_content(
            model=model,
            contents=prompt,
        )

        text = getattr(resp, "text", None)
        print("\n✅ Sucesso! Resposta:")
        print(text if text is not None else resp)

    except Exception as e:
        print("\n❌ Falhou ao chamar Vertex/Gemini.")
        print("ERRO:", repr(e))
        print("\n--- TRACEBACK ---")
        print(traceback.format_exc())
        sys.exit(2)

if __name__ == "__main__":
    main()
