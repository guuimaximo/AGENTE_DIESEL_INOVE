import os
from datetime import datetime
from supabase import create_client

# ==============================================================================
# CONFIGURAÇÃO DE BANCOS (Mesmas variáveis do relatorio_gerencial.py)
# ==============================================================================
SUPABASE_A_URL = os.getenv("SUPABASE_A_URL")
SUPABASE_A_KEY = os.getenv("SUPABASE_A_SERVICE_ROLE_KEY")
SUPABASE_B_URL = os.getenv("SUPABASE_B_URL")
SUPABASE_B_KEY = os.getenv("SUPABASE_B_SERVICE_ROLE_KEY")

TABELA_TELEMETRIA = os.getenv("DIESEL_SOURCE_TABLE", "premiacao_diaria")
TABELA_APP = "diesel_acompanhamentos"

sb_telemetria = create_client(SUPABASE_A_URL, SUPABASE_A_KEY)
sb_app = create_client(SUPABASE_B_URL, SUPABASE_B_KEY)

def to_float(val):
    try:
        if val is None or str(val).strip() == "": return 0.0
        return float(val)
    except Exception:
        return 0.0

# ==============================================================================
# MOTOR DE FECHAMENTO
# ==============================================================================
def executar_fechamento_diario():
    hoje = datetime.utcnow().strftime("%Y-%m-%d")
    print(f"🔄 Iniciando varredura de monitoramentos vencidos até {hoje}...")

    # 1. BUSCA ORDENS VENCIDAS (Banco B)
    res = sb_app.table(TABELA_APP).select("*") \
        .eq("status", "EM_MONITORAMENTO") \
        .lte("dt_fim_previsao", hoje) \
        .execute()
    
    ordens = res.data or []
    if not ordens:
        print("✅ Nenhuma ordem pendente de fechamento hoje.")
        return

    print(f"📌 Encontradas {len(ordens)} ordens para análise técnica.")

    # 2. PROCESSA CADA ORDEM INDIVIDUALMENTE
    for ordem in ordens:
        id_ordem = ordem.get("id")
        chapa = str(ordem.get("motorista_chapa", "")).strip()
        dt_ini = ordem.get("dt_inicio_monitoramento")
        dt_fim = ordem.get("dt_fim_previsao")
        
        meta = to_float(ordem.get("kml_meta"))
        kml_inicial = to_float(ordem.get("kml_inicial")) # Média de quando foi flagrado
        obs_atual = ordem.get("intervencao_obs") or ""

        # 3. BUSCA DADOS DE TELEMETRIA DO PERÍODO EXATO (Banco A)
        telemetria_res = sb_telemetria.table(TABELA_TELEMETRIA) \
            .select("km_rodado, combustivel_consumido") \
            .ilike("motorista", f"%{chapa}%") \
            .gte("dia", dt_ini) \
            .lte("dia", dt_fim) \
            .execute()
        
        dados = telemetria_res.data or []
        if not dados:
            print(f"⚠️ [IGNORADO] Chapa {chapa}: Sem dados de telemetria no período {dt_ini} a {dt_fim}.")
            continue

        # 4. MATEMÁTICA GERENCIAL (Agregação Pura)
        soma_km = sum(to_float(r.get("km_rodado")) for r in dados)
        soma_comb = sum(to_float(r.get("combustivel_consumido")) for r in dados)

        if soma_comb == 0:
            continue

        kml_realizado = soma_km / soma_comb
        
        desperdicio = 0.0
        if meta > 0 and kml_realizado < meta:
            desperdicio = soma_comb - (soma_km / meta)

        evolucao_pct = 0.0
        if kml_inicial > 0:
            evolucao_pct = ((kml_realizado - kml_inicial) / kml_inicial) * 100

        # 5. REGRA DE NEGÓCIO E STATUS FINAL
        atingiu_meta = kml_realizado >= meta
        novo_status = "ENCERRADO" if atingiu_meta else "ATAS"
        decisao_txt = "Meta Atingida - Monitoramento Encerrado" if atingiu_meta else "Abaixo da Meta - Direcionado para Atas Disciplinares"
        sinal = "+" if evolucao_pct > 0 else ""

        # 6. LOG DE AUDITORIA (Appends na observação do Instrutor)
        log_sistema = (
            f"\n\n[ANÁLISE SISTÊMICA AUTOMÁTICA - {hoje}]\n"
            f"Período Analisado: {dt_ini} a {dt_fim}\n"
            f"KM/L Meta: {meta:.2f} | Inicial: {kml_inicial:.2f} | Realizado: {kml_realizado:.2f}\n"
            f"Evolução: {sinal}{evolucao_pct:.1f}%\n"
            f"Desperdício no Período: {desperdicio:.1f} Litros\n"
            f"Decisão: {decisao_txt}"
        ).strip()

        nova_obs = f"{obs_atual}\n\n{log_sistema}".strip()

        # 7. ATUALIZAÇÃO FINAL (Banco B)
        try:
            sb_app.table(TABELA_APP).update({
                "status": novo_status,
                "kml_final_realizado": round(kml_realizado, 2),
                "desperdicio_final_litros": round(desperdicio, 2),
                "evolucao_pct": round(evolucao_pct, 2),
                "intervencao_obs": nova_obs,
                "updated_at": datetime.utcnow().isoformat()
            }).eq("id", id_ordem).execute()

            print(f"✔️ [FECHADO] Chapa {chapa} -> Status: {novo_status} | Evolução: {sinal}{evolucao_pct:.1f}%")
        except Exception as e:
            print(f"❌ [ERRO] Falha ao atualizar Chapa {chapa} no banco: {e}")

if __name__ == "__main__":
    executar_fechamento_diario()
