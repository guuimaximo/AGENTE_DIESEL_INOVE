import os
import re
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import vertexai
from vertexai.generative_models import GenerativeModel

from supabase import create_client
from playwright.sync_api import sync_playwright

# ==============================================================================
# 1. CONFIGURAÇÃO E ENV
# ==============================================================================
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID") or os.getenv("PROJECT_ID")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
VERTEX_MODEL = os.getenv("VERTEX_MODEL", "gemini-2.5-pro")

SUPABASE_A_URL = os.getenv("SUPABASE_A_URL")
SUPABASE_A_SERVICE_ROLE_KEY = os.getenv("SUPABASE_A_SERVICE_ROLE_KEY")

SUPABASE_B_URL = os.getenv("SUPABASE_B_URL")
SUPABASE_B_SERVICE_ROLE_KEY = os.getenv("SUPABASE_B_SERVICE_ROLE_KEY")

ORDEM_BATCH_ID = os.getenv("ORDEM_BATCH_ID")

# Tabelas
TABELA_ORIGEM = os.getenv("DIESEL_SOURCE_TABLE", "premiacao_diaria")
TABELA_LOTE = "acompanhamento_lotes"
TABELA_ITENS = "acompanhamento_lote_itens" 
TABELA_DESTINO = "diesel_acompanhamentos" 

BUCKET = "relatorios"
REMOTE_PREFIX = "acompanhamento"
PASTA_SAIDA = Path("Ordens_Geradas")

# Configurações
KML_MIN = 1.0
KML_MAX = 6.0
FETCH_DAYS = 90  # Histórico para gráfico
JANELA_DIAS = 7  # Paginação
PAGE_SIZE = 2000

# ==============================================================================
# 2. CLIENTES E HELPERS
# ==============================================================================
def safe_num(val):
    if pd.isna(val) or val is None or val == "": return 0.0
    try: return float(val)
    except: return 0.0

def _sb_a(): return create_client(SUPABASE_A_URL, SUPABASE_A_SERVICE_ROLE_KEY)
def _sb_b(): return create_client(SUPABASE_B_URL, SUPABASE_B_SERVICE_ROLE_KEY)

def _safe_filename(name: str) -> str:
    name = str(name or "").strip()
    name = re.sub(r"[^\w\-.() ]+", "_", name, flags=re.UNICODE)
    return name[:100] or "sem_nome"

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
            file_options={"content-type": content_type, "upsert": "true"},
        )
    
    # Monta URL pública manual (ou usa get_public_url se o bucket for publico)
    return f"{SUPABASE_B_URL}/storage/v1/object/public/{BUCKET}/{remote_path}"

def to_num(series: pd.Series) -> pd.Series:
    if series is None: return pd.Series(dtype="float64")
    s = series.astype(str).str.strip()
    s = s.str.replace(",", ".", regex=False)
    s = s.str.replace(r"[^0-9\.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

def get_cluster(v):
    v = str(v).strip()
    if v.startswith("2216"): return "C8"
    if v.startswith("2222"): return "C9"
    if v.startswith("2224"): return "C10"
    if v.startswith("2425"): return "C11"
    if v.startswith("W"): return "C6"
    return "OUTROS"

# ==============================================================================
# 3. LEITURA DE NOMES (CSV)
# ==============================================================================
def carregar_mapa_nomes(caminho_csv="motoristas_rows.csv"):
    if not os.path.exists(caminho_csv):
        print("⚠️ CSV de nomes não encontrado. Usando nomes do sistema.")
        return {}
    try:
        # Lê tudo como string para não perder zeros a esquerda da chapa
        df = pd.read_csv(caminho_csv, dtype=str)
        df['chapa'] = df['chapa'].str.strip()
        
        mapa = {}
        for _, row in df.iterrows():
            mapa[row['chapa']] = {
                "nome": str(row.get('nome', '')).strip().upper(),
                "cargo": str(row.get('cargo', 'MOTORISTA')).strip().upper()
            }
        print(f"📋 Mapa de nomes carregado: {len(mapa)} registros.")
        return mapa
    except Exception as e:
        print(f"❌ Erro ao ler CSV de nomes: {e}")
        return {}

# ==============================================================================
# 4. CARREGAMENTO DE DADOS
# ==============================================================================
def obter_motoristas_do_lote():
    """Busca a lista de chapas que o React gravou no Supabase B"""
    if not ORDEM_BATCH_ID:
        print("❌ ORDEM_BATCH_ID não fornecido.")
        return []

    print(f"📥 Buscando itens do lote {ORDEM_BATCH_ID}...")
    sb = _sb_b()
    try:
        res = sb.table(TABELA_ITENS).select("*").eq("lote_id", ORDEM_BATCH_ID).execute()
        itens = res.data or []
        print(f"📋 {len(itens)} motoristas para processar.")
        return itens
    except Exception as e:
        print(f"❌ Erro ao buscar itens: {e}")
        return []

def carregar_historico_motorista(chapa):
    """Busca dados brutos no Supabase A (90 dias para gráfico, 30 para análise)"""
    print(f"   ↳ Buscando dados brutos: {chapa}...")
    sb = _sb_a()
    
    agora = datetime.utcnow()
    limite = agora - timedelta(days=FETCH_DAYS)
    cursor = agora
    all_rows = []

    # Paginação reversa por data
    while cursor > limite:
        inicio_janela = cursor - timedelta(days=JANELA_DIAS)
        s_fim = cursor.strftime("%Y-%m-%d")
        s_ini = inicio_janela.strftime("%Y-%m-%d")
        
        start = 0
        while True:
            try:
                resp = sb.table(TABELA_ORIGEM)\
                         .select('dia, motorista, veiculo, linha, "km/l", km_rodado, combustivel_consumido')\
                         .eq("motorista", chapa)\
                         .gte("dia", s_ini).lte("dia", s_fim)\
                         .order("dia", desc=True)\
                         .range(start, start + PAGE_SIZE - 1).execute()
                
                rows = resp.data or []
                all_rows.extend(rows)
                if len(rows) < PAGE_SIZE: break
                start += PAGE_SIZE
            except Exception:
                break
        
        cursor = inicio_janela - timedelta(days=1)

    if not all_rows: return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    # Padroniza colunas
    df.rename(columns={"dia": "Date", "motorista": "Motorista", "km_rodado": "Km", "combustivel_consumido": "Comb.", "km/l": "kml_db"}, inplace=True)
    
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Km"] = to_num(df["Km"])
    df["Comb."] = to_num(df["Comb."])
    
    # Limpeza básica
    df = df.dropna(subset=["Date", "Km", "Comb."])
    df = df[(df["Comb."] > 0) & (df["Km"] > 0)].copy()
    df["Cluster"] = df["veiculo"].apply(get_cluster)
    
    # Filtra outliers extremos se necessário
    df = df[(df["Km"] / df["Comb."] >= KML_MIN) & (df["Km"] / df["Comb."] <= KML_MAX)]
        
    return df

# ==============================================================================
# 5. CÁLCULOS E PROCESSAMENTO (RAIO-X E GRÁFICO CORRIGIDO)
# ==============================================================================
def processar_dados_prontuario(df, chapa, info_nome):
    if df.empty: return None
    
    # Filtra últimos 30 dias para a tabela "Raio-X" e texto da IA
    max_date = df['Date'].max()
    min_date_30 = max_date - timedelta(days=30)
    df_30d = df[df['Date'] >= min_date_30].copy()
    
    if df_30d.empty: return None

    # --- 1. RAIO-X DA OPERAÇÃO (Tabela) ---
    # Agrupa por Linha + Cluster
    raio_x = df_30d.groupby(['linha', 'Cluster']).agg({
        'Km': 'sum', 
        'Comb.': 'sum',
        'veiculo': lambda x: list(x.unique())[0] # Exemplo de veículo
    }).reset_index()
    
    # Metas (Se não tiver no banco, usa fallback)
    METAS = {"C6": 2.5, "C8": 2.6, "C9": 2.73, "C10": 2.8, "C11": 2.9, "OUTROS": 2.5}
    raio_x['KML_Meta'] = raio_x['Cluster'].map(METAS).fillna(2.5)
    raio_x['KML_Real'] = raio_x['Km'] / raio_x['Comb.']
    
    # Cálculo Desperdício
    def calc_metrics(r):
        litros_meta = r['Km'] / r['KML_Meta']
        # Só conta desperdício se Real < Meta
        desp = (r['Comb.'] - litros_meta) if r['KML_Real'] < r['KML_Meta'] else 0
        return pd.Series([litros_meta, desp])
    
    raio_x[['Litros_Meta', 'Desperdicio']] = raio_x.apply(calc_metrics, axis=1)
    raio_x = raio_x.sort_values('Desperdicio', ascending=False)

    # --- 2. TOTAIS GERAIS (Para o Rodapé) ---
    total_km = raio_x['Km'].sum()
    total_litros = raio_x['Comb.'].sum()
    total_desperdicio = raio_x['Desperdicio'].sum()
    
    kml_geral_real = total_km / total_litros if total_litros > 0 else 0
    
    # Meta Harmônica Ponderada (O jeito certo de calcular "Meta Geral")
    # Soma de Litros que DEVERIAM ter sido gastos
    soma_litros_teoricos = raio_x['Litros_Meta'].sum()
    kml_geral_meta = total_km / soma_litros_teoricos if soma_litros_teoricos > 0 else 0

    # --- 3. DADOS GRÁFICO (ORDENAÇÃO CORRETA) ---
    # Pega últimas 4-5 semanas
    min_date_chart = max_date - timedelta(days=35) 
    df_chart = df[df['Date'] >= min_date_chart].copy()
    
    # Agrupa por Data de Início da Semana (Objeto Datetime)
    df_chart['Semana_Dt'] = df_chart['Date'].dt.to_period('W').apply(lambda r: r.start_time)
    
    # Ordena Cronologicamente (Isso garante Jan -> Fev)
    grp_chart = df_chart.groupby('Semana_Dt').agg({'Km': 'sum', 'Comb.': 'sum'}).sort_index()
    
    dates = []
    real_vals = []
    meta_vals = [] 
    
    for dt, row in grp_chart.iterrows():
        kml = row['Km'] / row['Comb.'] if row['Comb.'] > 0 else 0
        dates.append(dt.strftime('%d/%m')) # Só converte pra texto aqui
        real_vals.append(kml)
        meta_vals.append(kml_geral_meta) # Usa a meta ponderada como referência

    # Define foco principal para os cards
    top_ofensor = raio_x.iloc[0] if not raio_x.empty else None

    return {
        "chapa": chapa,
        "nome": info_nome.get('nome', chapa),
        "cargo": info_nome.get('cargo', 'MOTORISTA'),
        "periodo_txt": f"{min_date_30.strftime('%d/%m')} a {max_date.strftime('%d/%m')}",
        "raio_x": raio_x,
        "totais": {
            "km": total_km, "litros": total_litros, "desp": total_desperdicio,
            "kml_real": kml_geral_real, "kml_meta": kml_geral_meta
        },
        "grafico": {"dates": dates, "real": real_vals, "meta": meta_vals},
        "foco_cluster": top_ofensor['Cluster'] if top_ofensor is not None else "-"
    }

# ==============================================================================
# 6. GERAÇÃO DE ASSETS (IA, GRÁFICO, HTML)
# ==============================================================================
def chamar_ia_coach(dados):
    if not VERTEX_PROJECT_ID: return "ANÁLISE: IA Indisponível."
    try:
        vertexai.init(project=VERTEX_PROJECT_ID, location=VERTEX_LOCATION)
        model = GenerativeModel(VERTEX_MODEL)
        
        # Prepara texto do Raio-X para a IA ler
        top_linhas = dados['raio_x'].head(3)[['linha', 'Cluster', 'KML_Real', 'KML_Meta']].to_string(index=False)
        
        prompt = f"""
        Você é um Instrutor Técnico Master de Condução Econômica.
        
        ALVO:
        Motorista: {dados['nome']}
        Performance: {dados['totais']['kml_real']:.2f} km/l (Meta: {dados['totais']['kml_meta']:.2f})
        Desperdício Mensal: {dados['totais']['desp']:.0f} Litros de Diesel
        
        ONDE ELE ESTÁ ERRANDO (Top Linhas):
        {top_linhas}
        
        Gere um PRONTUÁRIO curto e direto.
        Responda ESTRITAMENTE com estes 3 tópicos (use as tags exatas):
        
        DIAGNÓSTICO COMPORTAMENTAL: [Explique tecnicamente o provável erro: ex: giro alto, troca tardia, falta de antecipação]
        FOCO DA MONITORIA: [3 itens bullet points para o instrutor treinar na prática]
        FEEDBACK EDUCATIVO: [Um parágrafo de roteiro para o instrutor falar para o motorista, tom respeitoso mas firme sobre a meta]
        """
        return model.generate_content(prompt).text
    except Exception as e:
        print(f"Erro IA: {e}")
        return "DIAGNÓSTICO COMPORTAMENTAL: Não foi possível gerar."

def gerar_grafico_visual(dados_g, path):
    """Gera o PNG do gráfico para o PDF"""
    if not dados_g['dates']: return
    
    plt.figure(figsize=(10, 3.5))
    
    # Linha Meta (Tracejada Vermelha Suave)
    plt.plot(dados_g['dates'], dados_g['meta'], color='#c0392b', linestyle='--', linewidth=2, label='Meta (Ref)', alpha=0.6)
    
    # Linha Real (Sólida Escura com Marcadores)
    plt.plot(dados_g['dates'], dados_g['real'], color='#2c3e50', marker='o', linewidth=2.5, markersize=6, label='Realizado')
    
    # Rótulos de Valor
    for x, y in zip(dados_g['dates'], dados_g['real']):
        plt.text(x, y + 0.03, f"{y:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2c3e50')

    plt.title("Evolução Semanal (Tendência)", fontsize=10, fontweight='bold', loc='left')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(loc='lower left', fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()

def extrair_secao(texto, chaves):
    if isinstance(chaves, str): chaves = [chaves]
    pattern = "|".join(chaves)
    # Busca o conteúdo entre a chave e a próxima quebra dupla ou título
    m = re.search(rf"(?:{pattern})[:\s\-]*(.*?)(?=\n(?:DIAGN|FOCO|FEEDBACK|ROTEIRO)|$)", texto, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else "..."

def gerar_html_final(d, txt_ia, img_path):
    analise = extrair_secao(txt_ia, ["DIAGNÓSTICO COMPORTAMENTAL", "DIAGNOSTICO"])
    foco = extrair_secao(txt_ia, ["FOCO DA MONITORIA", "ROTEIRO", "AÇÃO"])
    feedback = extrair_secao(txt_ia, ["FEEDBACK EDUCATIVO", "FEEDBACK"])
    
    # Gera linhas da tabela
    rows = ""
    for _, r in d['raio_x'].head(10).iterrows():
        style_loss = "color:#c0392b; font-weight:bold" if r['Desperdicio'] > 5 else ""
        rows += f"""<tr>
            <td>{r['linha']}</td>
            <td>{r['veiculo']}</td>
            <td>{r['Cluster']}</td>
            <td align="right">{r['Km']:.0f}</td>
            <td align="right">{r['Comb.']:.0f}</td>
            <td align="right"><b>{r['KML_Real']:.2f}</b></td>
            <td align="right" style="color:#7f8c8d">{r['KML_Meta']:.2f}</td>
            <td align="right" style="{style_loss}">{r['Desperdicio']:.1f} L</td>
        </tr>"""
        
    t = d['totais']
    
    # HTML Layout Prontuário
    return f"""
    <!DOCTYPE html><html lang="pt-BR"><head><meta charset="UTF-8">
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; padding: 25px; color: #333; background: #fff; }}
        .header {{ border-bottom: 3px solid #2c3e50; padding-bottom: 15px; margin-bottom: 25px; display: flex; justify-content: space-between; }}
        .title h1 {{ margin: 0; color: #2c3e50; font-size: 24px; text-transform: uppercase; }}
        .tag {{ background:#c0392b; color:white; padding:5px 10px; border-radius:4px; font-size:12px; font-weight:bold; }}
        
        .cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 30px; }}
        .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 15px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
        .card b {{ display: block; font-size: 22px; color: #2c3e50; margin-bottom: 5px; }}
        .card span {{ font-size: 11px; text-transform: uppercase; color: #7f8c8d; font-weight: bold; }}
        
        table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin-bottom: 5px; }}
        th {{ background: #34495e; color: white; padding: 8px; text-align: left; }}
        td {{ border-bottom: 1px solid #eee; padding: 8px; }}
        tfoot {{ background: #ecf0f1; font-weight: bold; border-top: 2px solid #bdc3c7; }}
        
        h2 {{ font-size: 15px; color: #2980b9; margin-top: 30px; border-left: 5px solid #2980b9; padding-left: 10px; margin-bottom: 15px; }}
        .ia-box {{ padding: 10px 15px; border-radius: 6px; margin-bottom: 10px; font-size: 13px; line-height: 1.5; }}
        
        .footer {{ margin-top: 50px; font-size: 10px; color: #aaa; text-align: center; border-top: 1px solid #eee; padding-top: 10px; }}
    </style></head>
    <body>
        <div class="header">
            <div>
                <h1>ORDEM DE ACOMPANHAMENTO</h1>
                <div style="font-size:16px; font-weight:bold; margin-top:5px">{d['nome']}</div>
                <div style="font-size:12px; color:#666">{d['chapa']} | {d['cargo']}</div>
            </div>
            <div style="text-align:right">
                <div class="tag">PRIORIDADE ALTA</div>
                <div style="font-size:12px; margin-top:5px; color:#666">Período: {d['periodo_txt']}</div>
            </div>
        </div>

        <div class="cards">
            <div class="card"><b>{d['foco_cluster']}</b><span>Foco Tecnologia</span></div>
            <div class="card"><b style="color:#c0392b">{t['desp']:.0f} L</b><span>Desperdício Total</span></div>
            <div class="card"><b>{t['kml_real']:.2f}</b><span>KM/L Real</span></div>
            <div class="card"><b style="color:#7f8c8d">{t['kml_meta']:.2f}</b><span>Meta Ajustada</span></div>
        </div>

        <h2>1. RAIO-X DA OPERAÇÃO</h2>
        <table>
            <thead><tr><th>Linha</th><th>Veículo</th><th>Cluster</th><th align="right">KM</th><th align="right">Litros</th><th align="right">Real</th><th align="right">Meta</th><th align="right">Desp.</th></tr></thead>
            <tbody>{rows}</tbody>
            <tfoot><tr>
                <td colspan="3">TOTAIS / MÉDIAS</td>
                <td align="right">{t['km']:.0f}</td>
                <td align="right">{t['litros']:.0f}</td>
                <td align="right" style="color:#e67e22">{t['kml_real']:.2f}</td>
                <td align="right">{t['kml_meta']:.2f}</td>
                <td align="right" style="color:#c0392b">{t['desp']:.0f} L</td>
            </tr></tfoot>
        </table>

        <h2>2. EVOLUÇÃO SEMANAL</h2>
        <div style="border:1px solid #eee; padding:5px; border-radius:6px; text-align:center">
            <img src="{os.path.basename(img_path)}" style="width:100%; height:auto; max-height:200px; object-fit:contain;">
        </div>

        <h2>3. DIAGNÓSTICO E AÇÃO (IA)</h2>
        <div class="ia-box" style="background:#fff3e0; border:1px solid #ffe0b2;">
            <b>DIAGNÓSTICO COMPORTAMENTAL:</b><br>{analise}
        </div>
        <div class="ia-box" style="background:#e3f2fd; border:1px solid #bbdefb;">
            <b>FOCO DA MONITORIA:</b><br>{foco}
        </div>
        <div class="ia-box" style="background:#e8f5e9; border:1px solid #c8e6c9;">
            <b>FEEDBACK EDUCATIVO:</b><br>{feedback}
        </div>
        
        <div style="margin-top:40px; display:flex; justify-content:space-between; padding:0 20px;">
            <div style="border-top:1px solid #000; width:40%; text-align:center; font-size:11px; padding-top:5px;">Instrutor Responsável</div>
            <div style="border-top:1px solid #000; width:40%; text-align:center; font-size:11px; padding-top:5px;">Motorista</div>
        </div>

        <div class="footer">
            Gerado automaticamente pelo Agente Diesel AI em {datetime.now().strftime('%d/%m/%Y %H:%M')}
        </div>
    </body></html>
    """

# ==============================================================================
# 7. MAIN FLOW
# ==============================================================================
def main():
    if not ORDEM_BATCH_ID:
        print("❌ ORDEM_BATCH_ID não definido no ambiente.")
        return

    try:
        atualizar_status_lote("PROCESSANDO")
        PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
        
        # 1. Carrega Nomes do CSV
        mapa_nomes = carregar_mapa_nomes()
        
        # 2. Busca lista de motoristas do Lote (Vem do React)
        itens = obter_motoristas_do_lote()
        if not itens:
            print("⚠️ Lote vazio ou erro na busca.")
            atualizar_status_lote("ERRO", "Lote sem itens")
            return

        print(f"🚀 Iniciando geração de {len(itens)} ordens...")
        sb_dest = _sb_b()

        for item in itens:
            mot_chapa = item.get('motorista_chapa')
            if not mot_chapa: continue
            
            # 3. Carrega Dados Brutos (Supabase A)
            df_full = carregar_historico_motorista(mot_chapa)
            if df_full.empty:
                print(f"⚠️ Sem dados para {mot_chapa}")
                continue
            
            # 4. Processa KPIs e Gera Estrutura
            # Pega nome do mapa CSV ou usa fallback
            info_nome = mapa_nomes.get(mot_chapa, {"nome": mot_chapa, "cargo": "MOTORISTA"})
            
            dados_pront = processar_dados_prontuario(df_full, mot_chapa, info_nome)
            if not dados_pront: continue
            
            print(f"   ⚙️ Gerando assets para {mot_chapa}...")
            
            # 5. Gera Gráfico, IA e HTML
            safe = _safe_filename(mot_chapa)
            p_img = PASTA_SAIDA / f"{safe}.png"
            p_html = PASTA_SAIDA / f"{safe}.html"
            p_pdf = PASTA_SAIDA / f"{safe}.pdf"
            
            gerar_grafico_visual(dados_pront['grafico'], p_img)
            txt_ia = chamar_ia_coach(dados_pront)
            html_content = gerar_html_final(dados_pront, txt_ia, p_img)
            
            with open(p_html, "w", encoding="utf-8") as f: f.write(html_content)
            
            # 6. Gera PDF com Playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(args=["--no-sandbox"])
                page = browser.new_page()
                page.goto(p_html.resolve().as_uri())
                page.pdf(path=str(p_pdf), format="A4", print_background=True, margin={"top":"10mm", "bottom":"10mm", "left":"10mm", "right":"10mm"})
                browser.close()
            
            # 7. Upload e Registro Final
            url_pdf = upload_storage(p_pdf, f"{safe}.pdf", "application/pdf")
            url_html = upload_storage(p_html, f"{safe}.html", "text/html")
            
            # Salva na tabela final
            sb_dest.table(TABELA_DESTINO).insert({
                "lote_id": ORDEM_BATCH_ID,
                "motorista_chapa": mot_chapa,
                "motorista_nome": dados_pront['nome'],
                "status": "CONCLUIDO",
                "arquivo_pdf_path": url_pdf,
                "arquivo_html_path": url_html,
                "kml_real": dados_pront['totais']['kml_real'],
                "perda_litros": dados_pront['totais']['desp'],
                "created_at": datetime.utcnow().isoformat(),
                "metadata": {"versao": "V8_Layout_Final"}
            }).execute()
            
            print(f"✅ {mot_chapa}: Sucesso.")

        atualizar_status_lote("CONCLUIDO")
        print("🏁 Processo finalizado.")

    except Exception as e:
        print(f"❌ Erro fatal no script: {e}")
        import traceback
        traceback.print_exc()
        atualizar_status_lote("ERRO", str(e))

if __name__ == "__main__":
    main()
