"""
ils_perturbacao_v19.py
========================
Busca local iterada (ILS) usada nas 33 instancias PCGTSPLIB do artigo, com
reinicio adaptativo.

Para cada componente conexo do grafo de grupos:
  1. Descida de busca local ate convergir (`_otimizar_ate_convergir_v3`) a
     partir da ordem construtiva de `heuristica_direta_pcgtsp`; a escolha das
     cidades representantes e feita por programacao dinamica
     (`dp_selecao_cidade`).
  2. Ate esgotar `time_budget`: perturbacao double-bridge que respeita Pi
     (`busca_local_v3.double_bridge_v2`), reotimizacao
     (`_otimizar_ate_convergir_v5`) e aceitacao somente se a FO melhora.
  3. Apos `falhas_para_restart` perturbacoes consecutivas sem melhora, reinicia
     a partir de um pool de ate POOL_SIZE ancoras: o otimo local inicial mais
     ancoras diversificadas, criadas sob demanda (kicks de 2-5, 6-9 e 10-13
     double-bridges, nessa ordem, seguidos de descida) e percorridas em
     round-robin.
  4. O limiar de reinicio e adaptativo, por componente: comeca em
     FALHAS_PARA_RESTART_SONDA (25); depois de N_SONDAGENS (2) ciclos de
     reinicio, compara o espalhamento relativo (max-min)/min das FOs desses
     ciclos. Se for >= LIMIAR_BIMODAL (0,25), o componente e tratado como
     bimodal (bacias de atracao de qualidade muito diferente) e passa a
     reiniciar com mais frequencia (FALHAS_PARA_RESTART_BIMODAL = 12); caso
     contrario volta ao padrao (FALHAS_PARA_RESTART_PADRAO = 40).

Uso: python ils_perturbacao_v19.py [instancia ...]   (padrao: p43.2 p43.3)
"""
import random
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pcgtsp_comparison"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import networkx as nx
from heuristica_direta_pcgtsp import (
    parse_pcglns, construir_grafo_grupos, construir_arvore_grupos,
    ordem_de_visita_grupos, resolver_direto, _dist, INF, _viol_ciclica,
)
from dp_selecao_cidade import escolher_cidades_e_fo_dp
from busca_local_v3 import double_bridge_v2
from ils_perturbacao_v6 import _otimizar_ate_convergir_v3
from ils_perturbacao_v12 import _otimizar_ate_convergir_v5

PASTA = (Path(__file__).resolve().parent.parent / "external_tools/pcglns/PCGLNS_PCGTSP")

POOL_SIZE = 4
FALHAS_PARA_RESTART_SONDA = 25
FALHAS_PARA_RESTART_PADRAO = 40
FALHAS_PARA_RESTART_BIMODAL = 12
LIMIAR_BIMODAL = 0.25
N_SONDAGENS = 2


def resolver_com_ils_v19(path, time_budget=3.0, seed=0):
    t0 = time.time()
    rng = random.Random(seed)
    parsed = parse_pcglns(path)
    G = construir_grafo_grupos(parsed)
    fo_total, tour_total = 0.0, []
    n_aceitas, n_tentadas, n_restarts, n_ancoras_criadas = 0, 0, 0, 0
    viol = 0
    n_comps_bimodais_detectados = 0

    for comp in nx.connected_components(G):
        sub = G.subgraph(comp)
        mst = nx.minimum_spanning_tree(sub, weight="weight")
        raiz = parsed["start_group"] if parsed["start_group"] in comp else next(iter(comp))
        arvore = construir_arvore_grupos(mst, raiz)
        pares_comp = [(f, p) for f, p in parsed["pares"] if f in comp and p in comp]
        ordem_inicial = ordem_de_visita_grupos(arvore, raiz, pares_comp)

        t_comp0 = time.time()
        ordem_partida, cidades_partida, fo_partida = _otimizar_ate_convergir_v3(
            ordem_inicial, parsed, pares_comp, 5.0, raiz=raiz)

        pool = [(ordem_partida, cidades_partida, fo_partida)]

        ordem_trabalho, cidades_trabalho, fo_trabalho = ordem_partida, cidades_partida, fo_partida
        ordem_melhor, cidades_melhor, fo_melhor = ordem_partida, cidades_partida, fo_partida

        falhas_para_restart = FALHAS_PARA_RESTART_SONDA
        decisao_tomada = False
        resultados_por_ciclo = []
        falhas_consecutivas = 0
        while time.time() - t_comp0 < time_budget:
            nova_ordem = double_bridge_v2(ordem_trabalho, pares_comp, rng, raiz=raiz)
            if nova_ordem is None:
                falhas_consecutivas += 1
            else:
                n_tentadas += 1
                restante = max(time_budget - (time.time() - t_comp0), 0.05)
                try:
                    ordem_cand, cidades_cand, fo_cand = _otimizar_ate_convergir_v5(
                        nova_ordem, parsed, pares_comp, min(restante, 5.0), raiz=raiz)
                    falhou = False
                except RuntimeError:
                    falhou = True

                if not falhou and fo_cand < fo_trabalho - 1e-9:
                    ordem_trabalho, cidades_trabalho, fo_trabalho = ordem_cand, cidades_cand, fo_cand
                    n_aceitas += 1
                    falhas_consecutivas = 0
                    if fo_trabalho < fo_melhor - 1e-9:
                        ordem_melhor, cidades_melhor, fo_melhor = ordem_trabalho, cidades_trabalho, fo_trabalho
                else:
                    falhas_consecutivas += 1

            if falhas_consecutivas > 0 and falhas_consecutivas % falhas_para_restart == 0:
                n_restarts += 1
                resultados_por_ciclo.append(fo_trabalho)
                if not decisao_tomada and len(resultados_por_ciclo) >= N_SONDAGENS:
                    espalhamento = (max(resultados_por_ciclo) - min(resultados_por_ciclo)) / min(resultados_por_ciclo)
                    decisao_tomada = True
                    if espalhamento >= LIMIAR_BIMODAL:
                        falhas_para_restart = FALHAS_PARA_RESTART_BIMODAL
                        n_comps_bimodais_detectados += 1
                    else:
                        falhas_para_restart = FALHAS_PARA_RESTART_PADRAO

                if len(pool) < POOL_SIZE and time.time() - t_comp0 < time_budget:
                    idx_nova_ancora = len(pool)
                    forca_min = 2 + 4 * (idx_nova_ancora - 1)
                    forca_max = forca_min + 3
                    base = ordem_partida
                    for _ in range(rng.randint(forca_min, forca_max)):
                        kicked = double_bridge_v2(base, pares_comp, rng, raiz=raiz)
                        if kicked is not None:
                            base = kicked
                    restante = max(time_budget - (time.time() - t_comp0), 0.05)
                    nova_ancora = _otimizar_ate_convergir_v3(
                        base, parsed, pares_comp, min(restante, 3.0), raiz=raiz)
                    pool.append(nova_ancora)
                    n_ancoras_criadas += 1
                    escolhida = nova_ancora
                else:
                    escolhida = pool[n_restarts % len(pool)]
                ordem_trabalho, cidades_trabalho, fo_trabalho = escolhida
                falhas_consecutivas = 0

        fo_total += fo_melhor
        tour_total.extend(cidades_melhor)
        viol += _viol_ciclica(cidades_melhor, parsed["city_group"], pares_comp, raiz)

    t_total = time.time() - t0

    return {"fo": fo_total, "tour": tour_total, "viol": viol, "t": t_total,
            "perturbacoes_tentadas": n_tentadas, "perturbacoes_aceitas": n_aceitas,
            "restarts": n_restarts, "ancoras_criadas": n_ancoras_criadas,
            "comps_bimodais_detectados": n_comps_bimodais_detectados}


if __name__ == "__main__":
    testes = sys.argv[1:] or ["p43.2", "p43.3"]
    for nome in testes:
        fp = PASTA / f"{nome}.pcglns"
        r0 = resolver_direto(str(fp))
        rl = resolver_com_ils_v19(str(fp), time_budget=3.0)
        ganho = 100 * (r0["fo"] - rl["fo"]) / r0["fo"]
        print(f"{nome:<10} guloso={r0['fo']:>10.1f}  ils_v19={rl['fo']:>10.1f}  ganho={ganho:>6.2f}%  "
              f"viol={rl['viol']}  restarts={rl['restarts']}  ancoras={rl['ancoras_criadas']}  "
              f"bimodal_detectado={rl['comps_bimodais_detectados']}  t={rl['t']:.2f}s")
