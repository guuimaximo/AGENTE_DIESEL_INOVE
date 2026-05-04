from __future__ import annotations


TABELA_FUNCIONARIOS_BCNT = "funcionarios_atualizada"
STATUS_FUNCIONARIO_ATIVO = "ativo"


def fetch_funcionarios_ativos_paginated(sb, select_cols: str, page_size: int = 1000):
    all_rows = []
    start = 0

    while True:
        end = start + page_size - 1
        resp = (
            sb.table(TABELA_FUNCIONARIOS_BCNT)
            .select(select_cols)
            .eq("status", STATUS_FUNCIONARIO_ATIVO)
            .range(start, end)
            .execute()
        )
        rows = resp.data or []
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        start += page_size

    return all_rows


def obter_data_inicio_atividade(sb, chapa: str):
    registro = str(chapa or "").strip()
    if not registro:
      return None

    resp = (
        sb.table(TABELA_FUNCIONARIOS_BCNT)
        .select("dt_inicio_atividade")
        .eq("status", STATUS_FUNCIONARIO_ATIVO)
        .eq("nr_cracha", registro)
        .maybe_single()
        .execute()
    )
    return (resp.data or {}).get("dt_inicio_atividade")
