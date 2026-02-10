import os
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
# 1. CONFIGURAÇÃO E ENV (MODO DIAGNÓSTICO ATIVADO)
# ==============================================================================
print("🚨 MODO DIAGNÓSTICO ATIVADO: Focando no motorista 30060983 🚨")

# --- FORÇANDO AS VARIÁVEIS PARA O TESTE ---
MOTORISTA_FOCO = "30060983"  # TRAVADO AQUI
NO_FILTERS = True            # Traz tudo, mesmo se o dado for ruim, para a gente ver
FETCH_DAYS = 120             # Busca 4 meses para garantir
JANELA_DIAS = 15             # Janelas curtas para não falhar conexão
# ------------------------------------------

VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
VERTEX_MODEL = os.getenv("VERTEX_MODEL", "gemini-2.5-pro")

SUPABASE_A_URL = os.getenv("SUPABASE_A_URL")
SUPABASE_A_SERVICE_ROLE_KEY = os.getenv("SUPABASE_A_SERVICE_ROLE_KEY")

SUPABASE_B_URL = os.getenv("SUPABASE_B_URL")
SUPABASE_B_SERVICE_ROLE_KEY = os.getenv("SUPABASE_B_SERVICE_ROLE_KEY")

ORDEM_BATCH_ID = os.getenv("ORDEM_BATCH_ID", "DEBUG_SESSION")
TABELA_ORIGEM = os.getenv("DIESEL_SOURCE_TABLE", "premiacao_diaria")
TABELA_LOTE = "acompanhamento_lotes"
TABELA_DESTINO = "diesel_acompanhamentos"

BUCKET = "relatorios"
REMOTE_PREFIX = "acompanhamento"
PASTA_SAIDA = Path("Ordens_Acompanhamento_Debug")

# Parâmetros visuais
RANKING_DIAS = 30
DETALHE_DIAS = 30
PAGE_SIZE = 2000

# ==============================================================================
# 2. CLIENTES
# ==============================================================================
def _sb_a():
    if not SUPABASE_A_URL or not SUPABASE_A_SERVICE_ROLE_KEY:
        raise Exception("⚠️ ENV faltando: SUPABASE_A_URL / SUPABASE_A_SERVICE_ROLE_KEY")
    return create_client(SUPABASE_A_URL, SUPABASE_A_SERVICE_ROLE_KEY)

def _sb_b():
    if not SUPABASE_B_URL or not SUPABASE_B_SERVICE_ROLE_KEY:
        raise Exception("⚠️ ENV faltando: SUPABASE_B_URL / SUPABASE_B_SERVICE_ROLE_KEY")
    return create_client(SUPABASE_B_URL, SUPABASE_B_SERVICE_ROLE_KEY)

def _safe_filename(name):
    return re.sub(r"[^\w\-.() ]+", "_", str(name).strip())[:100]

def upload_storage(local_path, remote_name, content_type):
    if not ORDEM_BATCH_ID: return None
    sb = _sb_b()
    remote_path = f"{REMOTE_PREFIX}/{ORDEM_BATCH_ID}/{remote_name}"
    if not local_path.exists(): return None
    with open(local_path, "rb") as f:
        sb.storage.from_(BUCKET).upload(
            path=remote_path, file=f,
            file_options={"content-type": content_type, "upsert": "true"},
        )
    return remote_path

def to_num(series):
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
    return None

# ==============================================================================
# 3. CARREGAMENTO COM DEBUG (O DETETIVE)
# ==============================================================================
def carregar_dados_focado():
    print(f"📦 [DEBUG] Iniciando varredura profunda para: {MOTORISTA_FOCO}")
    sb = _sb_a()
    
    data_final_global = datetime.utcnow()
    data_limite_global = data_final_global - timedelta(days=FETCH_DAYS)
    
    all_rows = []
    
    cursor_data = data_final_global
    
    # Loop de Janelas
    while cursor_data > data_limite_global:
        inicio_janela = cursor_data - timedelta(days=JANELA_DIAS)
        s_fim = cursor_data.strftime("%Y-%m-%d")
        s_ini = inicio_janela.strftime("%Y-%m-%d")
        
        print(f"   🔍 Investigando janela: {s_ini} até {s_fim}")
        
        start = 0
        sel = 'dia, motorista, veiculo, linha, "km/l", km_rodado, combustivel_consumido'
        
        while True:
            try:
                # AQUI: Filtramos DIRETO no banco pelo motorista para não trazer lixo
                resp = (
                    sb.table(TABELA_ORIGEM)
                    .select(sel)
                    .eq("motorista", MOTORISTA_FOCO) # TRAVA NO BANCO
                    .gte("dia", s_ini)
                    .lte("dia", s_fim)
                    .order("dia", desc=True)
                    .range(start, start + PAGE_SIZE - 1)
                    .execute()
                )
                rows = resp.data or []
                all_rows.extend(rows)
                
                if rows:
                    print(f"      ✅ Achei {len(rows)} registros nesta página.")
                    # Preview rápido dos dias achados
                    dias_achados = sorted(list(set([r['dia'] for r in rows])))
                    print(f"      🗓️  Dias encontrados: {dias_achados}")
                else:
                    print(f"      ❌ Nenhum registro nesta página.")

                if len(rows) < PAGE_SIZE:
                    break
                
                start += PAGE_SIZE
                
            except Exception as e:
                print(f"⚠️ Erro ao baixar: {e}")
                break

        cursor_data = inicio_janela - timedelta(days=1)

    print(f"📦 Total BRUTO baixado: {len(all_rows)} linhas.")
    
    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df.rename(columns={
        "dia": "Date", "motorista": "Motorista", "veiculo": "veiculo",
        "linha": "linha", "km/l": "kml_db", "km_rodado": "Km",
        "combustivel_consumido": "Comb."
    }, inplace=True)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Km"] = to_num(df["Km"])
    df["Comb."] = to_num(df["Comb."])
    
    return df

# ==============================================================================
# 4. PROCESSAMENTO COM AUDITORIA
# ==============================================================================
def preparar_base_auditada(df):
    """
    Aqui é onde descobrimos quem está sendo deletado e por quê.
    """
    print("\n🕵️ [AUDITORIA] Verificando qualidade dos dados...")
    
    df["Cluster"] = df["veiculo"].astype(str).apply(get_cluster)
    df["kml"] = df["Km"] / df["Comb."]
    
    # Check 1: Veículos sem Cluster
    sem_cluster = df[df["Cluster"].isna()]
    if not sem_cluster.empty:
        print("⚠️ [ALERTA] Registros com Veículo Desconhecido (Sem Cluster):")
        print(sem_cluster[["Date", "veiculo", "linha"]].to_string(index=False))
    
    # Check 2: Zeros ou Nulos
    zeros = df[(df["Km"] <= 0) | (df["Comb."] <= 0) | df["Km"].isna() | df["Comb."].isna()]
    if not zeros.empty:
        print("⚠️ [ALERTA] Registros com KM ou Litros Zerados/Nulos:")
        print(zeros[["Date", "veiculo", "Km", "Comb."]].to_string(index=False))

    # Check 3: Cluster válido mas KML absurdo
    kml_ruim = df[((df["kml"] < 0.5) | (df["kml"] > 6.0)) & (df["Comb."] > 0)]
    if not kml_ruim.empty:
        print("⚠️ [ALERTA] Registros com KML Absurdo (Fora de 0.5 - 6.0):")
        print(kml_ruim[["Date", "veiculo", "Km", "Comb.", "kml"]].to_string(index=False))

    # Limpeza efetiva
    df_clean = df.dropna(subset=["Date", "Km", "Comb."])
    df_clean = df_clean[(df_clean["Comb."] > 0) & (df_clean["Km"] > 0)].copy()
    
    print(f"\n📊 Linhas Originais: {len(df)} -> Linhas Válidas: {len(df_clean)}")
    return df_clean

# ==============================================================================
# FUNÇÕES DE RELATÓRIO (SIMPLIFICADAS PARA O DEBUG)
# ==============================================================================
def gerar_relatorio_debug(df, mot):
    # Pega os dados dos últimos 30 dias REAIS encontrados
    data_max = df["Date"].max()
    ini = data_max - timedelta(days=45) # Pega uma janela maior para ver o buraco
    
    df_view = df[(df["Date"] >= ini) & (df["Date"] <= data_max)].sort_values("Date", ascending=False)
    
    html_rows = ""
    for _, r in df_view.iterrows():
        # Destaca linhas suspeitas
        style = ""
        obs = ""
        
        if pd.isna(r["Cluster"]):
            style = "background: #ffcccc;" # Vermelho claro
            obs += " [SEM CLUSTER]"
        
        html_rows += f"""
        <tr style="{style}">
            <td>{r['Date'].strftime('%d/%m/%Y')}</td>
            <td>{r['veiculo']} {obs}</td>
            <td>{r['linha']}</td>
            <td>{r['Km']}</td>
            <td>{r['Comb.']}</td>
            <td>{r['kml']:.2f}</td>
            <td>{r['Cluster']}</td>
        </tr>
        """
        
    html = f"""
    <html>
    <head><style>table {{width:100%; border-collapse:collapse; font-family:sans-serif;}} th, td {{border:1px solid #ccc; padding:8px; text-align:center;}} th {{background:#eee;}}</style></head>
    <body>
        <h2>Relatório de Diagnóstico: {mot}</h2>
        <p><b>Período Analisado:</b> {ini.strftime('%d/%m')} até {data_max.strftime('%d/%m')}</p>
        <p>Se houver linhas <span style="background:#ffcccc">vermelhas</span>, é porque o veículo não tem Cluster mapeado.</p>
        <table>
            <tr><th>Data</th><th>Veículo</th><th>Linha</th><th>KM</th><th>Litros</th><th>KML</th><th>Cluster</th></tr>
            {html_rows}
        </table>
    </body>
    </html>
    """
    return html

def gerar_pdf(html_path, pdf_path):
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page()
        page.goto(html_path.resolve().as_uri())
        page.pdf(path=str(pdf_path), format="A4")
        browser.close()

# ==============================================================================
# MAIN
# ==============================================================================
def main():
    try:
        PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
        
        # 1. Carrega TUDO do motorista
        df_raw = carregar_dados_focado()
        
        if df_raw.empty:
            print("❌ ERRO CRÍTICO: O banco de dados retornou ZERO linhas para este motorista em todas as datas.")
            print("   -> Verifique se o número da chapa está correto na tabela 'premiacao_diaria'.")
            return

        # 2. Audita os dados
        df_proc = preparar_base_auditada(df_raw)
        
        if df_proc.empty:
            print("❌ ERRO CRÍTICO: Havia dados brutos, mas todos foram filtrados (Zeros ou Nulos).")
            return

        # 3. Gera PDF de Diagnóstico
        safe_name = _safe_filename(MOTORISTA_FOCO)
        p_html = PASTA_SAIDA / f"DEBUG_{safe_name}.html"
        p_pdf = PASTA_SAIDA / f"DEBUG_{safe_name}.pdf"
        
        html = gerar_relatorio_debug(df_raw, MOTORISTA_FOCO) # Passa o RAW para ver os erros no PDF
        
        with open(p_html, "w", encoding="utf-8") as f:
            f.write(html)
            
        gerar_pdf(p_html, p_pdf)
        
        print(f"\n✅ Relatório de Diagnóstico gerado em: {p_pdf}")
        print("   -> Abra este PDF. Ele vai mostrar dia a dia o que o sistema enxergou.")
        
        # Upload (opcional, só para vc pegar o link se rodar na nuvem)
        if ORDEM_BATCH_ID:
            url = upload_storage(p_pdf, f"DEBUG_{safe_name}.pdf", "application/pdf")
            print(f"   -> Upload feito: {url}")

    except Exception as e:
        print(f"❌ Erro fatal: {e}")

if __name__ == "__main__":
    main()
