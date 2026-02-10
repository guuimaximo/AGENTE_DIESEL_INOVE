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
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
VERTEX_MODEL = os.getenv("VERTEX_MODEL", "gemini-2.5-pro")

SUPABASE_A_URL = os.getenv("SUPABASE_A_URL")
SUPABASE_A_SERVICE_ROLE_KEY = os.getenv("SUPABASE_A_SERVICE_ROLE_KEY")

SUPABASE_B_URL = os.getenv("SUPABASE_B_URL")
SUPABASE_B_SERVICE_ROLE_KEY = os.getenv("SUPABASE_B_SERVICE_ROLE_KEY")

ORDEM_BATCH_ID = os.getenv("ORDEM_BATCH_ID")
QTD_ACOMPANHAMENTOS = int(os.getenv("QTD", "10"))

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
    if not ORDEM_BATCH_ID: return
    print(f"🔄 [Lote {ORDEM_BATCH_ID}] Status: {status}")
    sb = _sb_b()
    payload = {"status": status}
    if msg: payload["erro_msg"] = msg
    sb.table(TABELA_LOTE).update(payload).eq("id", ORDEM_BATCH_ID).execute()

def upload_storage(local_path: Path, remote_name: str, content_type: str) -> str:
    if not ORDEM_BATCH_ID: return None
    sb = _sb_b()
    remote_path = f"{REMOTE_PREFIX}/{ORDEM_BATCH_ID}/{remote_name}"
    if not local_path.exists(): return None
    with open(local_path, "rb") as f:
        sb.storage.from_(BUCKET).upload(
            path=remote_path, file=f,
            file_options={"content-type": content_type, "upsert": "true"}
        )
    return remote_path

def extrair_bloco(texto, tag_chave):
    """
    Extrai texto entre tags de forma robusta (aceita markdown, maiúsculas, etc).
    """
    if not texto: return "..."
    
    # Mapeia variações comuns da tag (ex: ROTEIRO pode vir como AÇÃO, PLANO, etc)
    mapa = {
        "ANALISE": [r"AN[ÁA]LISE", r"DIAGN[ÓO]STICO", r"PROBLEMA"],
        "ROTEIRO": [r"ROTEIRO", r"PLANO", r"A[ÇC][ÕO]ES", r"O QUE FAZER"],
        "FEEDBACK": [r"FEEDBACK", r"MENSAGEM", r"GESTOR", r"CONCLUS[ÃA]O"]
    }
    
    # Cria uma regex que procura qualquer variação da chave
    chaves_possiveis = mapa.get(tag_chave, [tag_chave])
    pattern_chave = "|".join(chaves_possiveis)
    
    # Procura: (Inicio de linha ou # ou *) + (Chave) + (: ou nada) + (TEXTO ALVO) + (Até a próxima chave ou fim)
    # (?is) = Case insensitive + Dot matches newline
    regex = rf"(?:^|\n|#|\*|[\d]+\.)\s*(?:{pattern_chave})[:\s\-]*(.*?)(?=\n(?:AN[ÁA]LISE|ROTEIRO|PLANO|FEEDBACK|RESUMO)[:#\*]|$)"
    
    match = re.search(regex, texto, re.IGNORECASE | re.DOTALL)
    if match:
        conteudo = match.group(1).strip()
        # Remove caracteres de markdown sobrando no inicio (*, -)
        return re.sub(r"^[\*\-\s]+", "", conteudo)
    
    return "..."

# ==============================================================================
# 3. CARREGAMENTO DE DADOS
# ==============================================================================
def carregar_dados():
    print("📦 [Supabase A] Buscando histórico de 60 dias...")
    sb = _sb_a()
    
    data_corte = (datetime.utcnow() - timedelta(days=60)).strftime("%Y-%m-%d")
    
    all_rows = []
    start = 0
    PAGE_SIZE = 2000
    
    # Seleção otimizada
    sel = 'dia, motorista, veiculo, linha, "km/l", km_rodado, combustivel_consumido'
    
    while True:
        resp = (sb.table(TABELA_ORIGEM).select(sel).gte("dia", data_corte)
                .range(start, start + PAGE_SIZE - 1).execute())
        rows = resp.data or []
        all_rows.extend(rows)
        if len(rows) < PAGE_SIZE: break
        start += PAGE_SIZE
        print(f"   -> Lendo registros: {len(all_rows)}...")

    if not all_rows: return pd.DataFrame()
    
    df = pd.DataFrame(all_rows)
    df.rename(columns={
        "dia": "Date", "motorista": "Motorista", "veiculo": "veiculo",
        "linha": "linha", "km/l": "kml", "km_rodado": "Km",
        "combustivel_consumido": "Comb."
    }, inplace=True)
    
    return df

# ==============================================================================
# 4. PROCESSAMENTO (METODOLOGIA V4)
# ==============================================================================
def processar_dados(df: pd.DataFrame):
    print("⚙️ [Core] Calculando eficiência e desperdício...")
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Mes_Ano'] = df['Date'].dt.to_period('M')
    
    for c in ['kml', 'Km', 'Comb.']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df.dropna(subset=['Date', 'Motorista', 'veiculo', 'Km', 'Comb.'], inplace=True)
    
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
    
    # Filtro físico (remover erros de digitação extremos)
    df = df[(df['kml'] >= 0.5) & (df['kml'] <= 6.0)].copy()

    # Período de Foco (Ranking)
    data_max = df['Date'].max()
    mes_atual = data_max.to_period('M')
    
    # Se o mês atual tiver muito poucos dados (inicio de mês), olha o anterior também
    df_foco = df[df['Mes_Ano'] == mes_atual].copy()
    if len(df_foco) < 100:
        mes_anterior = (data_max - timedelta(days=30)).to_period('M')
        print(f"   -> Poucos dados em {mes_atual}, expandindo para {mes_anterior}...")
        df_foco = df[(df['Mes_Ano'] == mes_atual) | (df['Mes_Ano'] == mes_anterior)].copy()
        mes_atual = f"{mes_anterior} e {mes_atual}" # Atualiza texto para o relatório
    
    # --- CÁLCULO DA META DINÂMICA (Média Linha + Cluster) ---
    ref = df_foco.groupby(['linha', 'Cluster']).agg({'Km':'sum', 'Comb.':'sum'}).reset_index()
    ref['KML_Meta_Linha'] = ref['Km'] / ref['Comb.']
    
    df_foco = df_foco.merge(ref[['linha', 'Cluster', 'KML_Meta_Linha']], on=['linha', 'Cluster'], how='left')
    
    def calc_perda(row):
        meta = row['KML_Meta_Linha']
        real = row['kml']
        if pd.notna(meta) and meta > 0 and real < meta:
            comb_ideal = row['Km'] / meta
            return row['Comb.'] - comb_ideal
        return 0.0
    
    df_foco['Litros_Perdidos'] = df_foco.apply(calc_perda, axis=1)

    # Ranking
    ranking = df_foco.groupby('Motorista').agg({
        'Litros_Perdidos': 'sum', 'Km': 'sum'
    }).reset_index().sort_values('Litros_Perdidos', ascending=False).head(QTD_ACOMPANHAMENTOS)
    
    lista_final = []
    
    for _, r in ranking.iterrows():
        mot = r['Motorista']
        perda_total = r['Litros_Perdidos']
        
        df_mot = df_foco[df_foco['Motorista'] == mot]
        if df_mot.empty: continue
        
        # Pior Linha (Onde mais perdeu)
        pior = df_mot.groupby(['linha', 'Cluster']).agg({
            'Litros_Perdidos': 'sum', 'Km': 'sum', 'Comb.': 'sum', 'KML_Meta_Linha': 'mean'
        }).reset_index().sort_values('Litros_Perdidos', ascending=False)
        
        if pior.empty: continue
        top = pior.iloc[0]
        
        linha_foco = top['linha']
        cluster_foco = top['Cluster']
        
        # --- HISTÓRICO PARA GRÁFICO (60 DIAS) ---
        # Pega dados brutos originais (df) para ter o histórico completo
        df_hist_mot = df[df['Motorista'] == mot].copy()
        df_hist_linha = df[(df['linha'] == linha_foco) & (df['Cluster'] == cluster_foco)].copy()
        
        # Veículos Recentes
        d15 = data_max - timedelta(days=15)
        vecs = df_hist_mot[df_hist_mot['Date'] >= d15]['veiculo'].unique()
        vecs_str = ", ".join(sorted(vecs)) if len(vecs) > 0 else "Nenhum"

        kml_real = top['Km'] / top['Comb.'] if top['Comb.'] > 0 else 0
        
        lista_final.append({
            'Motorista': mot,
            'Litros_Total': perda_total,
            'Linha_Foco': linha_foco,
            'Cluster_Foco': cluster_foco,
            'KML_Real': kml_real,
            'KML_Meta': top['KML_Meta_Linha'],
            'Gap': kml_real - top['KML_Meta_Linha'],
            'Dados_RaioX': df_mot,      
            'Dados_Hist_Mot': df_hist_mot, 
            'Dados_Hist_Linha': df_hist_linha, 
            'Veiculos_Recentes': vecs_str,
            'Periodo_Txt': str(mes_atual)
        })
        
    return lista_final

# ==============================================================================
# 5. IA E ASSETS
# ==============================================================================
def chamar_vertex_ai(dados):
    if not VERTEX_PROJECT_ID: return "ANÁLISE: IA Desativada."

    try:
        vertexai.init(project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION)
        model = GenerativeModel(VERTEX_MODEL)
        
        tec = {
            "C11": "VW 17.230 Automático. DICA: Evitar kickdown (pé no fundo).",
            "C10": "MB 1721 Euro 6. DICA: Trocar marcha no verde.",
            "C6": "MB 1721 Manual. DICA: Não esticar marcha.",
            "C8": "Micro MB."
        }.get(dados['Cluster_Foco'], "Ônibus Urbano.")

        # PROMPT REFORÇADO PARA FORMATAÇÃO
        prompt = f"""
        Atue como Instrutor de Motoristas Sênior.
        Analise este motorista com ALTO DESPERDÍCIO DE COMBUSTÍVEL.

        DADOS:
        - Motorista: {dados['Motorista']}
        - Veículo: {dados['Cluster_Foco']} ({tec})
        - Linha: {dados['Linha_Foco']}
        - Carros recentes: {dados['Veiculos_Recentes']}
        
        PERFORMANCE:
        - Meta: {dados['KML_Meta']:.2f} km/l
        - Realizado: {dados['KML_Real']:.2f} km/l
        - Perda: {dados['Litros_Total']:.0f} Litros

        Responda ESTRITAMENTE neste formato (sem asteriscos, sem introdução):

        ANÁLISE:
        [Escreva aqui a provável causa técnica: RPM alto? Freio brusco? Marcha errada?]

        ROTEIRO:
        [Liste 3 ações práticas para ele fazer amanhã]

        FEEDBACK:
        [Uma frase de impacto profissional para o gestor falar]
        """
        
        resp = model.generate_content(prompt)
        # Limpa formatação markdown que a IA adora colocar
        text = resp.text.replace("**", "").replace("##", "")
        print(f"\n--- IA RESPONDENDEU ({dados['Motorista']}) ---\n{text}\n-------------------\n")
        return text
    except Exception as e:
        print(f"⚠️ Erro IA: {e}")
        return "ANÁLISE: Indisponível (Erro API)."

def gerar_grafico(df_mot, df_linha, caminho):
    # Agrupa por Semana
    mot_w = df_mot.groupby(pd.Grouper(key='Date', freq='W-MON')).agg({'Km':'sum', 'Comb.':'sum'}).reset_index()
    mot_w['KML'] = mot_w['Km'] / mot_w['Comb.']
    
    lin_w = df_linha.groupby(pd.Grouper(key='Date', freq='W-MON')).agg({'Km':'sum', 'Comb.':'sum'}).reset_index()
    lin_w['KML'] = lin_w['Km'] / lin_w['Comb.']
    
    dados = pd.merge(mot_w, lin_w[['Date', 'KML']], on='Date', how='outer', suffixes=('_Mot', '_Linha')).sort_values('Date')
    
    # Remove semanas vazias
    dados = dados.dropna(subset=['KML_Mot', 'KML_Linha'], how='all')
    
    if len(dados) == 0: return # Evita erro em gráfico vazio

    dates = dados['Date'].dt.strftime("%d/%m")
    
    plt.figure(figsize=(10, 4))
    plt.plot(dates, dados['KML_Mot'], marker='o', lw=3, color='#2980b9', label='Motorista')
    plt.plot(dates, dados['KML_Linha'], ls='--', lw=2, color='#c0392b', label='Média da Linha')
    
    for x, y in zip(dates, dados['KML_Mot']):
        if pd.notna(y): plt.text(x, y+0.05, f"{y:.2f}", ha="center", va="bottom", fontsize=9, fontweight='bold')
        
    plt.title("Evolução Semanal (Últimos 60 dias)", fontsize=11, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho, dpi=100)
    plt.close()

def gerar_tabela_html(df_mot):
    df_mot["Mes"] = df_mot["Mes_Ano"].astype(str)
    res = df_mot.groupby(["Mes", "veiculo", "linha", "Cluster", "KML_Meta_Linha"]).agg(
        Km=("Km", "sum"), Comb=("Comb.", "sum"), Loss=("Litros_Perdidos", "sum")
    ).reset_index().sort_values("Loss", ascending=False)
    
    res["KML"] = res["Km"] / res["Comb"]
    
    rows = ""
    for _, r in res.iterrows():
        style = "background:#ffebee; color:#c0392b; font-weight:bold;" if r["Loss"] > 5 else ""
        rows += f"""<tr style="{style}">
            <td>{r['veiculo']}</td><td>{r['linha']}</td>
            <td>{r['Km']:.0f}</td><td>{r['KML']:.2f}</td><td>{r['KML_Meta_Linha']:.2f}</td>
            <td>{r['Loss']:.1f} L</td></tr>"""
    return rows

def gerar_html_final(dados, texto_ia, img, tabela):
    analise = extrair_bloco(texto_ia, "ANALISE")
    roteiro = extrair_bloco(texto_ia, "ROTEIRO")
    feedback = extrair_bloco(texto_ia, "FEEDBACK")
    cor = "#c0392b" if dados['Gap'] < 0 else "#27ae60"

    return f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head><meta charset="UTF-8">
    <style>
        body {{ font-family: sans-serif; padding: 20px; background: #f4f7f6; }}
        .box {{ background: white; padding: 25px; border-radius: 8px; border-top: 5px solid #2c3e50; }}
        h1 {{ margin: 0; color: #2c3e50; font-size: 20px; }}
        .kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 20px 0; }}
        .kpi {{ background: #ecf0f1; padding: 10px; text-align: center; border-radius: 5px; }}
        .kpi b {{ display: block; font-size: 18px; color: #2c3e50; }}
        .kpi span {{ font-size: 10px; color: #7f8c8d; text-transform: uppercase; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 11px; margin-top: 10px; }}
        th {{ background: #34495e; color: white; padding: 6px; text-align: left; }}
        td {{ padding: 6px; border-bottom: 1px solid #eee; }}
        .ia {{ background: #fff; border-left: 4px solid #ddd; padding: 10px; margin-top: 8px; font-size: 13px; }}
        .obs {{ font-size: 10px; background: #eee; padding: 5px; margin-bottom: 10px; border-radius: 4px; }}
    </style>
    </head>
    <body>
    <div class="box">
        <div style="display:flex; justify-content:space-between;">
            <div><h1>PRONTUÁRIO: {dados['Motorista']}</h1><div style="font-size:12px;color:#666">Mês: {dados['Periodo_Txt']}</div></div>
            <div style="background:#c0392b; color:white; padding:5px; border-radius:3px; font-size:11px; height:fit-content;">ALTO CUSTO</div>
        </div>
        <div class="kpis">
            <div class="kpi"><b>{dados['Cluster_Foco']}</b><span>Veículo</span></div>
            <div class="kpi"><b>{dados['Linha_Foco']}</b><span>Linha</span></div>
            <div class="kpi"><b style="color:{cor}">{dados['KML_Real']:.2f}</b><span>Real (Meta {dados['KML_Meta']:.2f})</span></div>
            <div class="kpi"><b style="color:#c0392b">{dados['Litros_Total']:.0f} L</b><span>Perda</span></div>
        </div>
        <div class="obs"><b>Veículos (15d):</b> {dados['Veiculos_Recentes']}</div>
        <img src="{os.path.basename(img)}" style="width:100%; border:1px solid #ddd; margin-bottom:10px;">
        
        <h3>1. Raio-X da Perda (Onde foi o erro?)</h3>
        <table><thead><tr><th>Carro</th><th>Linha</th><th>Km</th><th>Real</th><th>Meta</th><th>Perda</th></tr></thead><tbody>{tabela}</tbody></table>
        
        <h3>2. Diagnóstico Técnico</h3>
        <div class="ia" style="border-color:#f39c12"><b>Análise:</b> {analise}</div>
        <div class="ia" style="border-color:#3498db"><b>Ação:</b> {roteiro}</div>
        <div class="ia" style="border-color:#27ae60"><b>Feedback:</b> {feedback}</div>
    </div>
    </body></html>
    """

def gerar_pdf(html_path, pdf_path):
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page()
        page.goto(html_path.resolve().as_uri())
        page.pdf(path=str(pdf_path), format="A4", margin={"top":"1cm","right":"1cm","bottom":"1cm","left":"1cm"})
        browser.close()

# ==============================================================================
# 6. MAIN
# ==============================================================================
def main():
    if not ORDEM_BATCH_ID: print("❌ Sem Batch ID"); return
    try:
        atualizar_status_lote("PROCESSANDO")
        PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
        
        df = carregar_dados()
        if df.empty: raise Exception("Base Vazia")
        
        lista = processar_dados(df)
        print(f"🎯 Gerando {len(lista)} prontuários...")
        
        sb = _sb_b()
        for item in lista:
            mot = item['Motorista']
            print(f"   > {mot}...")
            safe = _safe_filename(mot)
            p_img, p_html, p_pdf = PASTA_SAIDA/f"{safe}.png", PASTA_SAIDA/f"{safe}.html", PASTA_SAIDA/f"{safe}.pdf"
            
            gerar_grafico(item['Dados_Hist_Mot'], item['Dados_Hist_Linha'], p_img)
            tbl = gerar_tabela_html(item['Dados_RaioX'])
            txt_ia = chamar_vertex_ai(item)
            html = gerar_html_final(item, txt_ia, p_img, tbl)
            
            with open(p_html, "w", encoding="utf-8") as f: f.write(html)
            gerar_pdf(p_html, p_pdf)
            
            url_pdf = upload_storage(p_pdf, f"{safe}.pdf", "application/pdf")
            url_html = upload_storage(p_html, f"{safe}.html", "text/html")
            
            if url_pdf:
                sb.table(TABELA_DESTINO).insert({
                    "lote_id": ORDEM_BATCH_ID, "motorista_nome": mot, "motorista_chapa": mot,
                    "motivo": "BAIXO_DESEMPENHO", "veiculo_foco": item['Cluster_Foco'],
                    "linha_foco": item['Linha_Foco'], "kml_real": float(item['KML_Real']),
                    "kml_meta": float(item['KML_Meta']), "gap": float(item['Gap']),
                    "perda_litros": float(item['Litros_Total']), "arquivo_pdf_path": url_pdf,
                    "arquivo_html_path": url_html, "status": "CONCLUIDO"
                }).execute()

        atualizar_status_lote("CONCLUIDO")
        print("✅ Sucesso!")
    except Exception as e:
        print(f"❌ Erro: {e}")
        atualizar_status_lote("ERRO", str(e))
        raise

if __name__ == "__main__":
    main()
