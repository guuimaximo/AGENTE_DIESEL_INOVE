import os
import time
import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

import vertexai
from vertexai.generative_models import GenerativeModel

from supabase import create_client
from playwright.sync_api import sync_playwright

# ==============================================================================
# 1. CONFIGURAÇÃO E ENV
# ==============================================================================
# Vertex AI
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
VERTEX_MODEL = os.getenv("VERTEX_MODEL", "gemini-2.5-pro")

# Supabase A (Origem dos Dados - Leitura)
SUPABASE_A_URL = os.getenv("SUPABASE_A_URL")
SUPABASE_A_SERVICE_ROLE_KEY = os.getenv("SUPABASE_A_SERVICE_ROLE_KEY")

# Supabase B (Controle e Destino - Escrita)
SUPABASE_B_URL = os.getenv("SUPABASE_B_URL")
SUPABASE_B_SERVICE_ROLE_KEY = os.getenv("SUPABASE_B_SERVICE_ROLE_KEY")

# Parâmetros do Workflow (Vindos do GitHub Action)
ORDEM_BATCH_ID = os.getenv("ORDEM_BATCH_ID")
QTD_ACOMPANHAMENTOS = int(os.getenv("QTD", "10"))

# Configurações de Arquivos e Tabelas
TABELA_ORIGEM = os.getenv("DIESEL_SOURCE_TABLE", "premiacao_diaria")
TABELA_LOTE = "acompanhamento_lotes"
TABELA_DESTINO = "diesel_acompanhamentos"
BUCKET = "relatorios"
REMOTE_PREFIX = "acompanhamento"

PASTA_SAIDA = Path("Ordens_Acompanhamento")

# ==============================================================================
# 2. CLIENTES E HELPERS
# ==============================================================================
def _sb_a():
    return create_client(SUPABASE_A_URL, SUPABASE_A_SERVICE_ROLE_KEY)

def _sb_b():
    return create_client(SUPABASE_B_URL, SUPABASE_B_SERVICE_ROLE_KEY)

def _safe_filename(name: str) -> str:
    name = str(name or "").strip()
    name = re.sub(r"[^\w\-.() ]+", "_", name, flags=re.UNICODE)
    return name[:100]

def atualizar_status_lote(status: str, msg: str = None):
    """Atualiza o status do lote pai no Supabase B."""
    if not ORDEM_BATCH_ID:
        return
    print(f"🔄 [Lote {ORDEM_BATCH_ID}] Status: {status}")
    sb = _sb_b()
    payload = {"status": status}
    if msg:
        payload["erro_msg"] = msg
    sb.table(TABELA_LOTE).update(payload).eq("id", ORDEM_BATCH_ID).execute()

def upload_storage(local_path: Path, remote_name: str, content_type: str) -> str:
    """Sobe arquivo para o Supabase Storage e retorna o Path relativo."""
    if not ORDEM_BATCH_ID:
        return None
    
    sb = _sb_b()
    # Estrutura: acompanhamento/{id_lote}/{arquivo}
    remote_path = f"{REMOTE_PREFIX}/{ORDEM_BATCH_ID}/{remote_name}"
    
    # Garante que o arquivo existe antes de subir
    if not local_path.exists():
        print(f"⚠️ Arquivo não encontrado para upload: {local_path}")
        return None

    with open(local_path, "rb") as f:
        sb.storage.from_(BUCKET).upload(
            path=remote_path,
            file=f,
            file_options={"content-type": content_type, "upsert": "true"}
        )
    return remote_path

# ==============================================================================
# 3. EXTRAÇÃO DE DADOS (SUPABASE A)
# ==============================================================================
def carregar_dados():
    print("📦 [Supabase A] Buscando dados recentes...")
    sb = _sb_a()
    
    # Pega apenas os últimos 60 dias para ser rápido e relevante
    data_corte = (datetime.utcnow() - timedelta(days=60)).strftime("%Y-%m-%d")
    
    PAGE_SIZE = 2000
    all_rows = []
    start = 0
    
    sel = 'dia, motorista, veiculo, linha, "km/l", km_rodado, combustivel_consumido'
    
    while True:
        resp = (sb.table(TABELA_ORIGEM)
                .select(sel)
                .gte("dia", data_corte)
                .range(start, start + PAGE_SIZE - 1)
                .execute())
        
        rows = resp.data or []
        all_rows.extend(rows)
        
        if len(rows) < PAGE_SIZE:
            break
        start += PAGE_SIZE
        print(f"   -> Baixados: {len(all_rows)} registros...")

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    
    df.rename(columns={
        "dia": "Date",
        "motorista": "Motorista",
        "veiculo": "veiculo",
        "linha": "linha",
        "km/l": "kml",
        "km_rodado": "Km",
        "combustivel_consumido": "Comb."
    }, inplace=True)
    
    return df

# ==============================================================================
# 4. PROCESSAMENTO E REGRAS DE NEGÓCIO
# ==============================================================================
def processar_dados(df: pd.DataFrame):
    print("⚙️ [Core] Processando regras de negócio...")
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    for col in ['kml', 'Km', 'Comb.']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(subset=['Date', 'Motorista', 'veiculo'], inplace=True)

    def get_cluster(v):
        v = str(v).strip()
        if v.startswith('2216'): return 'C8'
        if v.startswith('2222'): return 'C9'
        if v.startswith('2224'): return 'C10'
        if v.startswith('2425'): return 'C11'
        if v.startswith('W'): return 'C6'
        return None

    df['Cluster'] = df['veiculo'].apply(get_cluster)
    df = df.dropna(subset=['Cluster'])

    df = df[(df['kml'] >= 1.2) & (df['kml'] <= 6.0)].copy()
    
    data_max = df['Date'].max()
    if pd.isna(data_max):
        raise ValueError("Nenhuma data válida encontrada na base.")
        
    mes_atual = data_max.to_period('M')
    
    df_foco = df[df['Date'].dt.to_period('M') == mes_atual].copy()
    
    if df_foco.empty:
        mes_atual = (data_max - timedelta(days=30)).to_period('M')
        df_foco = df[df['Date'].dt.to_period('M') == mes_atual].copy()

    ref = df_foco.groupby(['linha', 'Cluster']).agg({'Km':'sum', 'Comb.':'sum'}).reset_index()
    ref['KML_Ref'] = ref['Km'] / ref['Comb.']
    
    df_foco = df_foco.merge(ref[['linha', 'Cluster', 'KML_Ref']], on=['linha', 'Cluster'], how='left')
    
    def calc_loss(r):
        if r['KML_Ref'] > 0 and r['kml'] < r['KML_Ref']:
            return r['Comb.'] - (r['Km'] / r['KML_Ref'])
        return 0
    
    df_foco['Litros_Desperdicio'] = df_foco.apply(calc_loss, axis=1)

    ranking = df_foco.groupby('Motorista').agg({
        'Litros_Desperdicio': 'sum',
        'Km': 'sum'
    }).reset_index()
    
    ranking = ranking.sort_values('Litros_Desperdicio', ascending=False).head(QTD_ACOMPANHAMENTOS)
    
    lista_piores = []
    for _, mot_row in ranking.iterrows():
        motorista = mot_row['Motorista']
        df_mot = df_foco[df_foco['Motorista'] == motorista]
        
        if df_mot.empty: continue
        
        pior_cenario = df_mot.groupby(['linha', 'Cluster']).agg({
            'Litros_Desperdicio': 'sum',
            'Km': 'sum',
            'Comb.': 'sum',
            'KML_Ref': 'mean'
        }).reset_index().sort_values('Litros_Desperdicio', ascending=False)

        if pior_cenario.empty: continue
        
        top_cenario = pior_cenario.iloc[0]
        
        # Evita divisão por zero
        comb = top_cenario['Comb.']
        kml_real = (top_cenario['Km'] / comb) if comb > 0 else 0
        
        lista_piores.append({
            'Motorista': motorista,
            'Litros_Total': mot_row['Litros_Desperdicio'],
            'Linha_Foco': top_cenario['linha'],
            'Cluster_Foco': top_cenario['Cluster'],
            'KML_Real': kml_real,
            'KML_Meta': top_cenario['KML_Ref'],
            'Gap': kml_real - top_cenario['KML_Ref'],
            'Dados_Completos_Mes': df_mot,
            'Periodo_Txt': str(mes_atual)
        })
        
    return lista_piores

# ==============================================================================
# 5. GERAÇÃO DE CONTEÚDO (VERTEX + GRÁFICOS + HTML)
# ==============================================================================
def chamar_vertex_ai(dados):
    # Se não tiver Project ID, aborta cedo
    if not VERTEX_PROJECT_ID:
        return "ANÁLISE: IA Desativada (Sem Project ID).\nROTEIRO: Padrão.\nFEEDBACK: Genérico."

    try:
        # Tenta inicializar. Se der erro de credencial, o try/except pega.
        vertexai.init(project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION)
        model = GenerativeModel(VERTEX_MODEL)
        
        prompt = f"""
        Atue como Instrutor Master de Motoristas.
        MOTORISTA: {dados['Motorista']}
        VEÍCULO: {dados['Cluster_Foco']}
        LINHA: {dados['Linha_Foco']}
        REAL: {dados['KML_Real']:.2f} km/l vs META: {dados['KML_Meta']:.2f} km/l.
        DESPERDÍCIO: {dados['Litros_Total']:.0f} L.
        
        Gere 3 seções curtas:
        ANÁLISE: Provável erro técnico.
        ROTEIRO: 3 passos práticos.
        FEEDBACK: Frase de impacto.
        """
        resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        # Captura erro de credencial (ADC) ou qualquer outro
        print(f"⚠️ Aviso Vertex AI: {e}")
        return "ANÁLISE: IA Indisponível.\nROTEIRO: Verificar condução manual.\nFEEDBACK: Atenção aos números."

def gerar_grafico(df_mot, caminho_img):
    daily = df_mot.groupby('Date').agg({'Km':'sum', 'Comb.':'sum'}).reset_index()
    daily['KML'] = daily['Km'] / daily['Comb.']
    daily = daily.sort_values('Date')
    
    plt.figure(figsize=(8, 3))
    plt.plot(daily['Date'], daily['KML'], marker='o', linewidth=2, color='#2c3e50', label='Diário')
    
    meta = df_mot['KML_Ref'].mean()
    plt.axhline(y=meta, color='#c0392b', linestyle='--', label=f'Meta ({meta:.2f})')
    
    plt.title("Evolução Diária", fontsize=10)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho_img, dpi=90)
    plt.close()

def gerar_html(dados, texto_ia, img_nome):
    analise, roteiro, feedback = "...", "...", "..."
    
    if "ANÁLISE:" in texto_ia:
        parts = texto_ia.split("ROTEIRO:")
        analise = parts[0].replace("ANÁLISE:", "").strip()
        if len(parts) > 1:
            sub = parts[1].split("FEEDBACK:")
            roteiro = sub[0].strip()
            if len(sub) > 1:
                feedback = sub[1].strip()

    cor_gap = "#c0392b" if dados['Gap'] < 0 else "#27ae60"
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: sans-serif; padding: 20px; color: #333; }}
            .header {{ border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 20px; }}
            h1 {{ margin: 0; font-size: 22px; text-transform: uppercase; }}
            .cards {{ display: flex; gap: 10px; margin-bottom: 20px; }}
            .card {{ border: 1px solid #ccc; padding: 10px; flex: 1; text-align: center; border-radius: 5px; }}
            .val {{ font-size: 18px; font-weight: bold; display: block; }}
            .lbl {{ font-size: 10px; text-transform: uppercase; color: #666; }}
            .box {{ background: #f9f9f9; border-left: 4px solid #333; padding: 10px; margin-top: 10px; font-size: 13px; }}
            img {{ width: 100%; height: 180px; object-fit: contain; margin-bottom: 15px; border: 1px solid #eee; }}
        </style>
    </head>
    <body>
        <div class="header">
            <span style="float:right; font-size:12px; font-weight:bold;">LOTE #{ORDEM_BATCH_ID}</span>
            <h1>Monitoria Técnica</h1>
            <div>Motorista: <b>{dados['Motorista']}</b></div>
            <div>Mês: {dados['Periodo_Txt']}</div>
        </div>
        
        <div class="cards">
            <div class="card"><span class="val">{dados['Cluster_Foco']}</span><span class="lbl">Veículo</span></div>
            <div class="card"><span class="val">{dados['Linha_Foco']}</span><span class="lbl">Linha Crítica</span></div>
            <div class="card"><span class="val" style="color:{cor_gap}">{dados['KML_Real']:.2f}</span><span class="lbl">Real</span></div>
            <div class="card"><span class="val">{dados['KML_Meta']:.2f}</span><span class="lbl">Meta</span></div>
            <div class="card"><span class="val" style="color:#c0392b">{dados['Litros_Total']:.0f} L</span><span class="lbl">Desperdício</span></div>
        </div>

        <img src="{img_nome}">
        
        <h3>Diagnóstico AI</h3>
        <div class="box" style="border-color:#e67e22"><b>Análise:</b> {analise}</div>
        <div class="box" style="border-color:#2980b9"><b>Plano:</b> {roteiro}</div>
        <div class="box" style="border-color:#27ae60"><b>Feedback:</b> {feedback}</div>
    </body>
    </html>
    """
    return html

def gerar_pdf(html_path: Path, pdf_path: Path):
    """
    Gera PDF usando Playwright.
    IMPORTANTE: Usa .resolve().as_uri() para garantir caminho absoluto.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page()
        # CORREÇÃO DO ERRO FATAL: .resolve() converte relativo em absoluto
        page.goto(html_path.resolve().as_uri())
        page.pdf(path=str(pdf_path), format="A4", margin={"top":"1cm","right":"1cm","bottom":"1cm","left":"1cm"})
        browser.close()

# ==============================================================================
# 6. EXECUTOR PRINCIPAL
# ==============================================================================
def main():
    if not ORDEM_BATCH_ID:
        print("❌ Erro: ORDEM_BATCH_ID não fornecido.")
        return

    try:
        atualizar_status_lote("PROCESSANDO")
        
        # Garante que a pasta existe
        PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
        
        # 1. Dados
        df = carregar_dados()
        if df.empty:
            raise Exception("Base de dados vazia no Supabase A.")
            
        # 2. Processamento
        lista_piores = processar_dados(df)
        print(f"🎯 Gerando {len(lista_piores)} prontuários...")
        
        sb = _sb_b()
        
        # 3. Geração Individual
        for item in lista_piores:
            mot = item['Motorista']
            safe_name = _safe_filename(mot)
            print(f"   > Gerando: {mot}...")
            
            # Paths locais
            p_img = PASTA_SAIDA / f"{safe_name}.png"
            p_html = PASTA_SAIDA / f"{safe_name}.html"
            p_pdf = PASTA_SAIDA / f"{safe_name}.pdf"
            
            # Gera Assets
            gerar_grafico(item['Dados_Completos_Mes'], p_img)
            texto_ia = chamar_vertex_ai(item) # Vai dar fallback se sem credencial
            html_content = gerar_html(item, texto_ia, p_img.name)
            
            with open(p_html, "w", encoding="utf-8") as f:
                f.write(html_content)
                
            # Gera PDF (agora com path corrigido)
            gerar_pdf(p_html, p_pdf)
            
            # Upload
            url_pdf = upload_storage(p_pdf, f"{safe_name}.pdf", "application/pdf")
            url_html = upload_storage(p_html, f"{safe_name}.html", "text/html")
            
            if url_pdf:
                # Insert no Banco - CORREÇÃO DA CHAPA (ID)
                sb.table(TABELA_DESTINO).insert({
                    "lote_id": ORDEM_BATCH_ID,
                    "motorista_nome": mot,
                    "motorista_chapa": mot, # <--- AQUI ESTÁ A CORREÇÃO
                    "veiculo_foco": item['Cluster_Foco'],
                    "linha_foco": item['Linha_Foco'],
                    "kml_real": float(item['KML_Real']),
                    "kml_meta": float(item['KML_Meta']),
                    "gap": float(item['Gap']),
                    "perda_litros": float(item['Litros_Total']),
                    "arquivo_pdf_path": url_pdf,
                    "arquivo_html_path": url_html,
                    "status": "CONCLUIDO"
                }).execute()
            
        atualizar_status_lote("CONCLUIDO")
        print("✅ Lote finalizado com sucesso.")

    except Exception as e:
        print(f"❌ Erro fatal: {e}")
        atualizar_status_lote("ERRO", str(e))
        raise

if __name__ == "__main__":
    main()
