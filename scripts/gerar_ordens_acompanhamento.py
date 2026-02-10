import os
import re
import time
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

# Parâmetros do Workflow
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
    remote_path = f"{REMOTE_PREFIX}/{ORDEM_BATCH_ID}/{remote_name}"
    
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

def extrair_bloco(texto, tag):
    """Helper para extrair seções do texto da IA"""
    if not texto or (tag + ":") not in texto:
        return "..."
    try:
        parte = texto.split(tag + ":")[1]
        tags_proximas = ["ROTEIRO:", "FEEDBACK:", "ANALISE:", "RESUMO:"]
        for t in tags_proximas:
            if t != tag + ":" and t in parte:
                parte = parte.split(t)[0]
        return parte.strip()
    except:
        return "..."

# ==============================================================================
# 3. EXTRAÇÃO DE DADOS (SUPABASE A)
# ==============================================================================
def carregar_dados():
    print("📦 [Supabase A] Buscando dados recentes (60 dias)...")
    sb = _sb_a()
    
    # Busca 60 dias para ter histórico para o gráfico
    data_corte = (datetime.utcnow() - timedelta(days=60)).strftime("%Y-%m-%d")
    
    PAGE_SIZE = 2000
    all_rows = []
    start = 0
    
    # Seleciona apenas colunas essenciais
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
    
    # Normalização de nomes
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
    
    # Conversão de Tipos
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Mes_Ano'] = df['Date'].dt.to_period('M') # Cria coluna Mes_Ano
    
    for col in ['kml', 'Km', 'Comb.']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(subset=['Date', 'Motorista', 'veiculo'], inplace=True)

    # Definição de Clusters
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

    # Filtro de sujeira
    df = df[(df['kml'] >= 1.0) & (df['kml'] <= 6.0)].copy()
    
    # --- DEFINIÇÃO DO PERÍODO DE RANKING (Último Mês Fechado ou Atual) ---
    data_max = df['Date'].max()
    mes_atual = data_max.to_period('M')
    df_ranking = df[df['Date'].dt.to_period('M') == mes_atual].copy()
    
    # Se o mês atual tiver poucos dados, usa o anterior para ranking
    if df_ranking.empty or len(df_ranking) < 100:
        mes_atual = (data_max - timedelta(days=30)).to_period('M')
        df_ranking = df[df['Date'].dt.to_period('M') == mes_atual].copy()

    print(f"   -> Foco do Ranking: {mes_atual} | Histórico Gráfico: 60 dias")

    # --- CÁLCULO DA META (Baseado no Mês de Ranking) ---
    ref = df_ranking.groupby(['linha', 'Cluster']).agg({'Km':'sum', 'Comb.':'sum'}).reset_index()
    ref['KML_Ref'] = ref['Km'] / ref['Comb.']
    
    df_ranking = df_ranking.merge(ref[['linha', 'Cluster', 'KML_Ref']], on=['linha', 'Cluster'], how='left')
    
    # Cálculo de Desperdício
    def calc_loss(r):
        if r['KML_Ref'] > 0 and r['kml'] < r['KML_Ref']:
            litros_meta = r['Km'] / r['KML_Ref']
            return r['Comb.'] - litros_meta
        return 0
    
    df_ranking['Litros_Desperdicio'] = df_ranking.apply(calc_loss, axis=1)

    # Ranking Geral
    ranking = df_ranking.groupby('Motorista').agg({
        'Litros_Desperdicio': 'sum',
        'Km': 'sum'
    }).reset_index()
    
    ranking = ranking.sort_values('Litros_Desperdicio', ascending=False).head(QTD_ACOMPANHAMENTOS)
    
    lista_piores = []
    
    for _, mot_row in ranking.iterrows():
        motorista = mot_row['Motorista']
        
        # Dados do Mês de Foco (para Raio-X e Cálculos de Perda)
        df_mot_foco = df_ranking[df_ranking['Motorista'] == motorista]
        if df_mot_foco.empty: continue
        
        # Identifica a Linha Onde Mais Perdeu
        pior_cenario = df_mot_foco.groupby(['linha', 'Cluster']).agg({
            'Litros_Desperdicio': 'sum',
            'Km': 'sum',
            'Comb.': 'sum',
            'KML_Ref': 'mean'
        }).reset_index().sort_values('Litros_Desperdicio', ascending=False)

        if pior_cenario.empty: continue
        top_cenario = pior_cenario.iloc[0]
        
        linha_foco = top_cenario['linha']
        cluster_foco = top_cenario['Cluster']
        
        # --- DADOS HISTÓRICOS (60 DIAS) PARA GRÁFICO ---
        # Filtra dados do motorista nos últimos 60 dias
        df_mot_hist = df[df['Motorista'] == motorista].copy()
        
        # Filtra dados DA LINHA nos últimos 60 dias (para comparar a meta semanal)
        df_linha_hist = df[df['linha'] == linha_foco].copy()
        
        # Veículos dirigidos nas últimas 2 semanas
        data_2_semanas = data_max - timedelta(days=14)
        veiculos_recentes = df_mot_hist[df_mot_hist['Date'] >= data_2_semanas]['veiculo'].unique()
        veiculos_str = ", ".join(sorted(veiculos_recentes)) if len(veiculos_recentes) > 0 else "Nenhum recente"

        # Métricas Consolidadas do Foco
        comb = top_cenario['Comb.']
        kml_real = (top_cenario['Km'] / comb) if comb > 0 else 0
        
        lista_piores.append({
            'Motorista': motorista,
            'Litros_Total': mot_row['Litros_Desperdicio'],
            'Linha_Foco': linha_foco,
            'Cluster_Foco': cluster_foco,
            'KML_Real': kml_real,
            'KML_Meta': top_cenario['KML_Ref'],
            'Gap': kml_real - top_cenario['KML_Ref'],
            'Dados_RaioX': df_mot_foco,      # Apenas mês atual
            'Dados_Historico_Mot': df_mot_hist, # 60 dias motorista
            'Dados_Historico_Linha': df_linha_hist, # 60 dias linha (meta dinâmica)
            'Veiculos_Recentes': veiculos_str,
            'Periodo_Txt': str(mes_atual)
        })
        
    return lista_piores

# ==============================================================================
# 5. GERAÇÃO DE CONTEÚDO (VERTEX + GRÁFICOS + HTML)
# ==============================================================================
def chamar_vertex_ai(dados):
    if not VERTEX_PROJECT_ID:
        return "ANÁLISE: IA Desativada.\nROTEIRO: Padrão.\nFEEDBACK: Genérico."

    try:
        vertexai.init(project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION)
        model = GenerativeModel(VERTEX_MODEL)
        
        tec_map = {
            "C11": "VW 17.230 Automático (Cuidado com Kickdown).",
            "C10": "MB 1721 Euro VI (Giro Verde).",
            "C9":  "MB 1721 Dianteiro.",
            "C8":  "Micro MB.",
            "C6":  "MB 1721 Manual (Esticar marchas).",
        }
        tec_info = tec_map.get(dados['Cluster_Foco'], "Veículo Padrão")

        prompt = f"""
        Instrutor Master de Eco-Driving. Análise de Motorista.
        
        DADOS:
        - Motorista: {dados['Motorista']}
        - Veículo Foco: {dados['Cluster_Foco']} ({tec_info})
        - Linha Crítica: {dados['Linha_Foco']}
        - Veículos Recentes (2 sem): {dados['Veiculos_Recentes']}
        
        RESULTADO NA LINHA CRÍTICA (Mês Atual):
        - Real: {dados['KML_Real']:.2f} km/l
        - Meta: {dados['KML_Meta']:.2f} km/l
        - Perda: {dados['Litros_Total']:.0f} Litros

        Gere 3 seções curtas e diretas:
        ANALISE: Por que esse consumo ruim? (Considere o tipo de veículo e se ele trocou muito de carro recentemente).
        ROTEIRO: 3 passos práticos para amanhã.
        FEEDBACK: Frase de impacto para o gestor.

        Formato:
        ANALISE: ...
        ROTEIRO: ...
        FEEDBACK: ...
        """
        resp = model.generate_content(prompt)
        text = resp.text.replace("**", "").replace("#", "")
        return text
    except Exception as e:
        print(f"⚠️ Aviso Vertex AI: {e}")
        return "ANÁLISE: Indisponível.\nROTEIRO: Verificar condução.\nFEEDBACK: Atenção."

def gerar_grafico(df_mot, df_linha, caminho_img):
    """
    Gera gráfico comparando KML do Motorista vs Média da Linha (Meta Dinâmica)
    Agrupado por Semana nos últimos 60 dias.
    """
    # Agrupa Motorista por Semana
    mot_weekly = df_mot.groupby(pd.Grouper(key='Date', freq='W-MON')).agg({'Km':'sum', 'Comb.':'sum'}).reset_index()
    mot_weekly['KML_Mot'] = mot_weekly['Km'] / mot_weekly['Comb.']
    
    # Agrupa Linha por Semana (Meta Dinâmica)
    lin_weekly = df_linha.groupby(pd.Grouper(key='Date', freq='W-MON')).agg({'Km':'sum', 'Comb.':'sum'}).reset_index()
    lin_weekly['KML_Linha'] = lin_weekly['Km'] / lin_weekly['Comb.']
    
    # Junta os dados
    dados = pd.merge(mot_weekly, lin_weekly[['Date', 'KML_Linha']], on='Date', how='left')
    dados = dados.sort_values('Date')
    
    # Formata data
    x_labels = dados["Date"].dt.strftime("%d/%m")
    
    plt.figure(figsize=(10, 4))
    
    # Linha do Motorista
    plt.plot(x_labels, dados['KML_Mot'], marker='o', linewidth=3, color='#2980b9', label='Motorista')
    
    # Linha da Meta Dinâmica (Média da Linha naquela semana)
    plt.plot(x_labels, dados['KML_Linha'], linestyle='--', linewidth=2, color='#c0392b', label='Média da Linha (Semana)')
    
    # Adiciona rótulos nos pontos do motorista
    for x, y in zip(x_labels, dados['KML_Mot']):
        if pd.notna(y):
            plt.text(x, y + 0.05, f"{y:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    
    plt.title("Evolução 60 Dias: Motorista vs Média da Linha", fontsize=11, fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(caminho_img, dpi=100)
    plt.close()

def gerar_tabela_raiox_html(df_mot):
    """Gera tabela do mês atual"""
    df_mot["Mes"] = df_mot["Mes_Ano"].astype(str)
    
    resumo = df_mot.groupby(["Mes", "veiculo", "linha", "Cluster", "KML_Ref"]).agg(
        Km=("Km", "sum"),
        Comb=("Comb.", "sum"),
        Litros_Desperdicio=("Litros_Desperdicio", "sum")
    ).reset_index()

    resumo["KML_Real"] = resumo["Km"] / resumo["Comb"]
    resumo = resumo.sort_values("Litros_Desperdicio", ascending=False)

    html_rows = ""
    for _, row in resumo.iterrows():
        style = ""
        if row["Litros_Desperdicio"] > 10:
            style = "background-color: #ffebee; color: #c62828; font-weight:bold;"

        html_rows += f"""
        <tr style="{style}">
            <td align="center">{row['veiculo']}</td>
            <td align="center">{row['linha']}</td>
            <td align="center">{row['Cluster']}</td>
            <td align="center">{row['Km']:.0f}</td>
            <td align="center">{row['KML_Real']:.2f}</td>
            <td align="center">{row['KML_Ref']:.2f}</td>
            <td align="center">{row['Litros_Desperdicio']:.1f} L</td>
        </tr>
        """
    return html_rows

def gerar_html_final(dados, texto_ia, img_nome, tabela_raiox):
    analise = extrair_bloco(texto_ia, "ANALISE")
    roteiro = extrair_bloco(texto_ia, "ROTEIRO")
    feedback = extrair_bloco(texto_ia, "FEEDBACK")
    
    cor_kml = "#c0392b" if dados['Gap'] < 0 else "#27ae60"

    html = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: 'Helvetica', sans-serif; background: #f4f6f7; padding: 20px; }}
            .container {{ max-width: 900px; margin: auto; background: white; padding: 30px; border-radius: 8px; border-top: 6px solid #2c3e50; }}
            h1 {{ margin: 0; color: #2c3e50; font-size: 22px; }}
            .sub-header {{ color: #7f8c8d; font-size: 12px; margin-bottom: 20px; }}
            
            .kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 20px; }}
            .kpi-card {{ background: #ecf0f1; padding: 10px; border-radius: 5px; text-align: center; }}
            .kpi-val {{ display: block; font-size: 18px; font-weight: bold; color: #2c3e50; }}
            .kpi-lbl {{ font-size: 10px; color: #7f8c8d; text-transform: uppercase; }}

            h2 {{ font-size: 14px; color: #2980b9; border-bottom: 1px solid #ddd; padding-bottom: 5px; margin-top: 25px; text-transform: uppercase; }}
            .box {{ background: #fff; border-left: 4px solid #ddd; padding: 10px 15px; margin-top: 10px; font-size: 13px; color: #444; }}
            
            table {{ width:100%; border-collapse: collapse; font-size: 11px; margin-top: 10px; border: 1px solid #ddd; }}
            th {{ background-color: #34495e; color: white; padding: 6px; }}
            
            .footer {{ margin-top: 40px; border-top: 1px solid #eee; padding-top: 10px; text-align: center; font-size: 10px; color: #aaa; }}
            .veiculos-box {{ font-size: 11px; color: #555; background: #eee; padding: 8px; border-radius: 4px; margin-bottom: 15px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div style="display:flex; justify-content:space-between;">
                <div>
                    <h1>PRONTUÁRIO: {dados['Motorista']}</h1>
                    <div class="sub-header">Lote #{ORDEM_BATCH_ID} | Mês: {dados['Periodo_Txt']}</div>
                </div>
                <div style="text-align:right;">
                    <span style="background:#e74c3c; color:white; padding:4px 8px; border-radius:4px; font-size:11px; font-weight:bold;">ALTO DESPERDÍCIO</span>
                </div>
            </div>

            <div class="kpis">
                <div class="kpi-card">
                    <span class="kpi-val">{dados['Cluster_Foco']}</span>
                    <span class="kpi-lbl">Veículo Foco</span>
                </div>
                <div class="kpi-card">
                    <span class="kpi-val">{dados['Linha_Foco']}</span>
                    <span class="kpi-lbl">Linha Crítica</span>
                </div>
                <div class="kpi-card">
                    <span class="kpi-val" style="color:{cor_kml}">{dados['KML_Real']:.2f}</span>
                    <span class="kpi-lbl">Real vs Meta ({dados['KML_Meta']:.2f})</span>
                </div>
                <div class="kpi-card">
                    <span class="kpi-val" style="color:#c0392b">{dados['Litros_Total']:.0f} L</span>
                    <span class="kpi-lbl">Diesel Perdido</span>
                </div>
            </div>

            <div class="veiculos-box">
                <b>Veículos dirigidos nas últimas 2 semanas:</b> {dados['Veiculos_Recentes']}
            </div>

            <img src="{img_nome}" style="width:100%; height:auto; border:1px solid #eee; border-radius:5px; margin-bottom: 20px;">

            <h2>1. Detalhamento da Operação (Raio-X Mês Atual)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Veículo</th><th>Linha</th><th>Cluster</th><th>KM Total</th><th>Real (km/l)</th><th>Meta (km/l)</th><th>Perda</th>
                    </tr>
                </thead>
                <tbody>{tabela_raiox}</tbody>
            </table>

            <h2>2. Diagnóstico Técnico (IA)</h2>
            <div class="box" style="border-left-color: #f39c12;"><b>Análise:</b> {analise}</div>

            <h2>3. Plano de Ação</h2>
            <div class="box" style="border-left-color: #3498db;"><b>O que fazer:</b> {roteiro}</div>

            <h2>4. Feedback do Gestor</h2>
            <div class="box" style="border-left-color: #27ae60;"><b>Mensagem:</b> {feedback}</div>

            <div class="footer">Relatório gerado automaticamente pelo Agente Diesel V4.</div>
        </div>
    </body>
    </html>
    """
    return html

def gerar_pdf(html_path: Path, pdf_path: Path):
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page()
        page.goto(html_path.resolve().as_uri())
        page.pdf(path=str(pdf_path), format="A4", margin={"top":"1cm","right":"1cm","bottom":"1cm","left":"1cm"})
        browser.close()

# ==============================================================================
# 7. EXECUTOR PRINCIPAL
# ==============================================================================
def main():
    if not ORDEM_BATCH_ID:
        print("❌ Erro: ORDEM_BATCH_ID não fornecido.")
        return

    try:
        atualizar_status_lote("PROCESSANDO")
        PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
        
        # 1. Dados
        df = carregar_dados()
        if df.empty:
            raise Exception("Base de dados vazia no Supabase A.")
            
        # 2. Processamento
        lista_piores = processar_dados(df)
        print(f"🎯 Gerando {len(lista_piores)} prontuários (Top Desperdício)...")
        
        sb = _sb_b()
        
        # 3. Geração Individual
        for item in lista_piores:
            mot = item['Motorista']
            safe_name = _safe_filename(mot)
            print(f"   > Gerando: {mot}...")
            
            # Paths
            p_img = PASTA_SAIDA / f"{safe_name}.png"
            p_html = PASTA_SAIDA / f"{safe_name}.html"
            p_pdf = PASTA_SAIDA / f"{safe_name}.pdf"
            
            # Assets
            gerar_grafico(item['Dados_Historico_Mot'], item['Dados_Historico_Linha'], p_img)
            tabela_raiox = gerar_tabela_raiox_html(item['Dados_RaioX'])
            texto_ia = chamar_vertex_ai(item)
            
            html_content = gerar_html_final(item, texto_ia, p_img.name, tabela_raiox)
            
            with open(p_html, "w", encoding="utf-8") as f:
                f.write(html_content)
                
            gerar_pdf(p_html, p_pdf)
            
            # Upload
            url_pdf = upload_storage(p_pdf, f"{safe_name}.pdf", "application/pdf")
            url_html = upload_storage(p_html, f"{safe_name}.html", "text/html")
            
            if url_pdf:
                sb.table(TABELA_DESTINO).insert({
                    "lote_id": ORDEM_BATCH_ID,
                    "motorista_nome": mot,
                    "motorista_chapa": mot,
                    "motivo": "BAIXO_DESEMPENHO",
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
