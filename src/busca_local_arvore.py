"""
busca_local_arvore.py
======================
Busca local para a construcao da arvore/MST da heuristica LCPP nas instancias
geometricas (282 sinteticas e 50 de literatura).

Diagnostico que motiva este modulo:

1. `construir_grafo_mst` (heuristic_core.py -> quadtree.py) NAO calcula uma
   MST de verdade sobre um grafo rico -- e uma FLORESTA GULOSA de vizinho
   mais proximo (cada poligono conecta ao seu unico vizinho mais proximo,
   escolhido entre so 5 candidatos por vertice via KDTree). Isso ja sai com
   E = N-1 (uma arvore), sem NENHUMA aresta alternativa sobrando para
   qualquer busca local escolher -- confirmado empiricamente: E(mst) = N-1
   exato em 100% de uma amostra testada.
2. Uma MST verdadeira sobre o grafo COMPLETO (todos os N*(N-1)/2 pares de
   poligonos, ja implementado em `heuristic_core._build_full_graph`) reduz o
   peso total da arvore em ate ~16% em alguns casos -- ha folga real que a
   construcao atual deixa na mesa.
3. MAS o ganho em FO_total disso e pequeno (0.00-0.06% nos casos testados)
   porque o deslocamento ("d"/ar) e so ~0.42% do FO total em media (amostra
   de 30 instancias: min 0.20%, max 0.74%) -- o corte ("c"/perimetro) domina
   e e CONSTANTE por construcao (independente de ordem/topologia). Esse e o
   TETO TEORICO de qualquer busca local que mexa so em
   topologia/ordem de visita aqui: nao ha como o deslocamento melhorar mais
   do que ~0.4-0.7 pontos percentuais do FO total, nao importa quao boa seja
   a busca -- ao contrario das instancias PCGTSPLIB abstratas, onde o
   deslocamento E o objetivo inteiro.
4. `construir_grafo_mst_edge_swap` (heuristic_core.py, ja existente) faz
   busca de troca de aresta, mas e O(N^3) por RODADA de melhora (escaneia
   TODAS as N-1 arestas da MST x TODAS as N*(N-1)/2 arestas do grafo
   completo, reconstruindo o caminho INTEIRO -- Passos 5-7 -- a cada
   candidato) -- inviavel em escala (estourou 30s sem convergir numa
   instancia de N=45).

Este modulo reimplementa de forma muito mais barata:
  - `mst_completa(positions)`      : MST verdadeira sobre grafo completo.
  - `avaliador(positions, raiz, pares, hybrid)` : fabrica uma funcao
     `G -> (fo, caminho)` que reconstroi o caminho completo (Passos 5-7) e
     calcula a FO, despachando para o mesmo mecanismo usado na comparacao
     baseline (`cortar_arvore_hibrida_precedencia_v2` se a instancia tem
     pares de precedencia Pi, `cortar_arvore` caso contrario -- replica
     exatamente `_ORIGINAL_CFG`/`_HYBRID_CFG` de run_comparison.py).
  - `edge_swap_barato(...)`        : 1-troca de aresta, mas limitando os
     candidatos de reconexao aos k mais baratos por PESO BRUTO (nao aos
     N*(N-1)/2 todos) -- so paga reconstrucao completa para os poucos
     candidatos realmente promissores.
  - `perturbar(...)`               : kick estilo ILS -- remove/reconecta
     algumas arestas ALEATORIAMENTE (nao so as mais baratas) para escapar
     de otimos locais.
  - `busca_local_forte(...)`       : loop principal, orcamento de tempo
     fixo -- MST completa -> edge_swap_barato (convergencia) ->
     {perturbar, edge_swap_barato, aceita se melhora} ate esgotar o tempo.
"""
from __future__ import annotations

import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import networkx as nx
from shapely.geometry import Polygon

import heuristic_core as hc
from graph_routing import construir_arvore_direcionada, cortar_arvore


# ===========================================================================
# Construcao: MST verdadeira sobre o grafo completo
# ===========================================================================

def mst_completa(positions: List[Polygon]) -> Tuple[nx.Graph, nx.Graph]:
    """Retorna (mst, G_full). G_full e reaproveitado pelo edge-swap como
    fonte de candidatos de reconexao (ja tem TODOS os pares, entao qualquer
    aresta de reconexao necessaria ja esta presente -- sem precisar
    recalcular distancias geometricas de novo a cada troca)."""
    G_full = hc._build_full_graph(positions)
    mst = nx.minimum_spanning_tree(G_full, weight="weight")
    return mst, G_full


# ===========================================================================
# Avaliador de FO para uma arvore candidata (reusa Passos 5-7 originais)
# ===========================================================================

def fabricar_avaliador(
    positions: List[Polygon],
    raiz: int,
    pares_precedencia: List[Tuple[int, int]],
    child_order_fn=hc.child_order_num_filhos,
) -> Callable[[nx.Graph], Tuple[float, List]]:
    """
    Fabrica avaliador(G) -> (fo, caminho) para uma MST candidata G (grafo
    NAO-direcionado com atributos weight/connection nas arestas e poligono
    nos nos). Despacha para o mesmo mecanismo usado no baseline: se a
    instancia tem pares Pi, usa `cortar_arvore_hibrida_precedencia_v2_perf2`
    (identico a enforce_precedence=True, precedence_strategy='hybrid' em
    heuristic_core.run_heuristic; a variante `_perf2` produz FO e caminho
    identicos aos da original em 118/118 comparacoes de validacao (59
    instancias com pares_pi>0 x 2 topologias) e reduz o custo do teste de
    toque por segmento via caminho rapido em coordenadas cruas -- ver o bloco
    de comentario antes de `caminha_arvore_hibrida_lca_perf2` em
    heuristic_core.py); senao usa `cortar_arvore` original.
    """
    usa_hybrid = len(pares_precedencia) > 0

    def avaliador(G: nx.Graph) -> Tuple[float, List]:
        arvore = construir_arvore_direcionada(G, raiz)
        if usa_hybrid:
            caminho_raw = hc.cortar_arvore_hibrida_precedencia_v2_perf2(
                arvore, raiz, pares_precedencia, child_order_fn
            )
        else:
            caminho_raw = cortar_arvore(arvore, raiz)
        caminho = hc.reclassificar_tags(caminho_raw)
        fo = hc.calcular_fo(caminho)
        return fo, caminho

    return avaliador


# ===========================================================================
# Edge-swap barato (candidatos limitados por peso bruto)
# ===========================================================================

def _componentes_apos_remocao(mst: nx.Graph, u: int, v: int) -> Tuple[set, set]:
    """Remove aresta (u,v) e devolve os dois componentes resultantes (sem
    modificar `mst`)."""
    tmp = mst.copy()
    tmp.remove_edge(u, v)
    comp_u = nx.node_connected_component(tmp, u)
    comp_v = nx.node_connected_component(tmp, v)
    return comp_u, comp_v


def _candidatos_reconexao(
    G_full: nx.Graph, comp_a: set, comp_b: set, k: int
) -> List[Tuple[float, int, int]]:
    """k arestas mais baratas (peso bruto) que reconectam comp_a a comp_b,
    usando o grafo completo (garante que QUALQUER par tem aresta)."""
    # Escaneia a partir do menor componente por eficiencia.
    if len(comp_a) > len(comp_b):
        comp_a, comp_b = comp_b, comp_a
    candidatos = []
    for a in comp_a:
        for b in comp_b:
            w = G_full.edges[a, b]["weight"]
            candidatos.append((w, a, b))
    candidatos.sort(key=lambda x: x[0])
    return candidatos[:k]


def edge_swap_barato(
    mst: nx.Graph,
    G_full: nx.Graph,
    avaliador: Callable[[nx.Graph], Tuple[float, List]],
    fo_atual: float,
    time_limit: float,
    k_candidatos: int = 4,
    rng: Optional[random.Random] = None,
) -> Tuple[nx.Graph, float, Dict[str, int]]:
    """
    Varredura completa: para cada aresta da arvore atual, tenta substitui-la
    pelo melhor entre os `k_candidatos` mais baratos (peso bruto) que
    reconectam os dois pedacos. Primeira melhora aceita, segue para a
    proxima aresta (nao reinicia a varredura do zero -- mais barato que o
    "restart apos toda melhora" do `construir_grafo_mst_edge_swap` antigo).
    Repete rodadas completas ate uma rodada sem nenhuma melhora, ou ate
    `time_limit` esgotar.
    """
    rng = rng or random.Random(0)
    best = mst.copy()
    best_fo = fo_atual
    n_evals = 0
    n_improvements = 0
    t0 = time.time()

    melhorou_rodada = True
    while melhorou_rodada and (time.time() - t0) < time_limit:
        melhorou_rodada = False
        for u, v in list(best.edges):
            if (time.time() - t0) >= time_limit:
                break
            comp_u, comp_v = _componentes_apos_remocao(best, u, v)
            candidatos = _candidatos_reconexao(G_full, comp_u, comp_v, k_candidatos)
            for w, a, b in candidatos:
                if (a, b) == (u, v) or (b, a) == (u, v):
                    continue
                # NAO podamos por peso bruto: numa MST verdadeira (ja
                # otima por peso, propriedade do corte de Kruskal/Prim),
                # NENHUM candidato mais barato que a aresta removida pode
                # existir -- podar por peso tornaria esta funcao um no-op
                # completo sempre que chamada sobre uma MST verdadeira
                # (confirmado empiricamente: edge_swap_evals=0 em 100% dos
                # testes iniciais). O ganho de FO aqui so pode vir de trocas
                # com peso BRUTO igual/pior mas que abrem melhor
                # "carona"/interleaving -- por isso
                # avaliamos os k candidatos mais baratos incondicionalmente,
                # nao so os que melhoram peso.
                tentativa = best.copy()
                tentativa.remove_edge(u, v)
                conn = G_full.edges[a, b]["connection"]
                tentativa.add_edge(a, b, weight=w, connection=conn)
                if not nx.is_connected(tentativa):
                    continue
                fo, _ = avaliador(tentativa)
                n_evals += 1
                if fo < best_fo - 1e-9:
                    best = tentativa
                    best_fo = fo
                    n_improvements += 1
                    melhorou_rodada = True
                    break
            if melhorou_rodada:
                break

    meta = {"edge_swap_evals": n_evals, "edge_swap_improvements": n_improvements}
    return best, best_fo, meta


# ===========================================================================
# Perturbacao estilo ILS (kick aleatorio de m arestas)
# ===========================================================================

def perturbar(
    mst: nx.Graph,
    G_full: nx.Graph,
    m_trocas: int,
    rng: random.Random,
) -> nx.Graph:
    """Remove `m_trocas` arestas aleatorias (uma de cada vez) e reconecta
    cada uma via uma aresta ALEATORIA (nao necessariamente barata) entre os
    dois pedacos resultantes -- kick de diversificacao, analogo ao
    double-bridge usado nas instancias PCGTSPLIB, mas adaptado para arvores (nao ha nocao de
    "ordem" linear aqui, so topologia)."""
    atual = mst.copy()
    edges = list(atual.edges)
    if not edges:
        return atual
    alvos = rng.sample(edges, k=min(m_trocas, len(edges)))
    for u, v in alvos:
        if (u, v) not in atual.edges:
            continue
        comp_u, comp_v = _componentes_apos_remocao(atual, u, v)
        # amostra ate 10 candidatos aleatorios (nao so o mais barato)
        candidatos = _candidatos_reconexao(G_full, comp_u, comp_v, k=10)
        if not candidatos:
            continue
        w, a, b = rng.choice(candidatos)
        atual.remove_edge(u, v)
        conn = G_full.edges[a, b]["connection"]
        atual.add_edge(a, b, weight=w, connection=conn)
    return atual


# ===========================================================================
# Loop principal: construcao fortalecida + ILS
# ===========================================================================

def busca_local_forte(
    positions: List[Polygon],
    raiz: int,
    pares_precedencia: List[Tuple[int, int]],
    time_limit: float = 3.0,
    k_candidatos: int = 4,
    m_trocas_perturbacao: int = 2,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Pipeline completo: MST completa -> edge-swap barato ate
    convergencia -> loop ILS (perturbar + reotimizar + aceita se melhora)
    ate esgotar `time_limit`. Retorna dict com fo, caminho, meta.
    """
    t0 = time.time()
    rng = random.Random(seed)
    avaliador = fabricar_avaliador(positions, raiz, pares_precedencia)

    mst, G_full = mst_completa(positions)
    fo0, caminho0 = avaliador(mst)

    orcamento_construcao = time_limit * 0.5
    mst_opt, fo_opt, meta1 = edge_swap_barato(
        mst, G_full, avaliador, fo0, orcamento_construcao, k_candidatos, rng
    )

    best_mst, best_fo = mst_opt, fo_opt
    n_kicks = 0
    n_kick_improvements = 0

    while (time.time() - t0) < time_limit:
        restante = time_limit - (time.time() - t0)
        if restante < 0.05:
            break
        kicked = perturbar(best_mst, G_full, m_trocas_perturbacao, rng)
        fo_kicked, _ = avaliador(kicked)
        refinado, fo_refinado, meta2 = edge_swap_barato(
            kicked, G_full, avaliador, fo_kicked, min(restante, orcamento_construcao),
            k_candidatos, rng,
        )
        n_kicks += 1
        if fo_refinado < best_fo - 1e-9:
            best_mst = refinado
            best_fo = fo_refinado
            n_kick_improvements += 1

    fo_final, caminho_final = avaliador(best_mst)

    return {
        "fo_total": fo_final,
        "caminho": caminho_final,
        "arvore_mst": best_mst,
        "raiz": raiz,
        "time_total": time.time() - t0,
        "meta": {
            "fo_inicial_mst_completa": fo0,
            "fo_apos_edge_swap_inicial": fo_opt,
            "n_kicks": n_kicks,
            "n_kick_improvements": n_kick_improvements,
            **meta1,
        },
    }
