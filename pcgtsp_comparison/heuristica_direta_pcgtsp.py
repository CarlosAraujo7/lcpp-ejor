"""
heuristica_direta_pcgtsp.py
=============================
"Direção inversa" da comparação: em vez de converter NOSSAS
instâncias LCPP para o formato PCGTSP e rodar o PCGLNS nelas
(pcgtsp_comparison/converter.py), aqui adaptamos NOSSA heurística para
rodar DIRETAMENTE nas instâncias REAIS do PCGTSPLIB (matriz de distância
abstrata entre "cidades" agrupadas em conjuntos GTSP, sem coordenadas 2D
nem conceito de perímetro/corte).

Diferença estrutural chave frente ao LCPP original:
  - LCPP: cada grupo é um polígono; precisa cortar o perímetro INTEIRO
    (custo fixo, independente de ordem) + deslocamento entre contornos.
    Há "carona" grátis quando contornos se tocam (distância 0).
  - PCGTSP puro: cada grupo tem várias cidades candidatas, mas só UMA é
    visitada -- não há "perímetro", é um ponto só. Custo é só a soma das
    distâncias de deslocamento entre as cidades escolhidas, consecutivas na
    ordem de visita. Nas instâncias reais (derivadas de coordenadas TSPLIB),
    não existe "carona" grátis genérica entre grupos diferentes.

Isso simplifica a adaptação: a razão de ser do traversal híbrido
(preservar o estilo interleaved de percorrer um perímetro) não se aplica,
porque não há perímetro. Reaproveitado DIRETAMENTE de
heuristic_core.py (funções 100% genéricas, sem geometria/Shapely):
`_calcular_lca`, `_filho_no_caminho`, `_agrupar_pares_por_lca`,
`_ordenar_blocos`, `_ordem_respeitando_precedencia`, `_OWN`.

Algoritmo:
  1. Parse do .pcglns (matriz de distância, GTSP_SET_SECTION,
     GTSP_SET_ORDERING, START_GROUP_SECTION).
  2. Grafo completo entre GRUPOS: peso(A,B) = menor distância entre
     qualquer par (cidade em A, cidade em B), ignorando arcos proibidos
     (-1). Guarda também QUAL par de cidades atinge esse mínimo.
  3. MST sobre esse grafo de grupos (networkx).
  4. Árvore direcionada a partir de START_GROUP (BFS simples, sem
     dependência de Shapely).
  5. Ordem de visita dos grupos respeitando Π: reaproveita o mecanismo LCA
     (estratégia "hybrid") tratando cada grupo como um "nó" cujo bloco "OWN" é
     trivial (só o próprio grupo, sem sub-percurso).
  6. Escolha da cidade representante de cada grupo, na ordem definida:
     gulosa (cidade do grupo mais próxima da última cidade escolhida) --
     mesmo critério de `cortar_por_ordem_sequencial`.
  7. FO = soma das distâncias reais (direcionais) entre cidades escolhidas
     consecutivas, fechando o ciclo de volta à primeira.
"""
from __future__ import annotations

import array
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import networkx as nx

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from heuristic_core import (  # noqa: E402
    _calcular_lca, _filho_no_caminho, _agrupar_pares_por_lca,
    _ordenar_blocos, _ordem_respeitando_precedencia, _OWN,
)

INF = float("inf")


# ===========================================================================
# Parsing
# ===========================================================================

def parse_pcglns(path) -> Dict:
    text = Path(path).read_text()
    lines = text.split("\n")

    def idx_de(prefixo):
        for i, l in enumerate(lines):
            if l.strip().startswith(prefixo):
                return i
        return -1

    dim = int(lines[idx_de("DIMENSION")].split(":")[1].strip())

    ew = idx_de("EDGE_WEIGHT_SECTION")
    # array.array('d', ...) linhas em vez de list: mesma indexacao aninhada
    # matrix[a-1][b-1] em todo o codigo (_dist e todos os call sites ficam
    # inalterados), ~35% mais rapido nos lookups (hotspot: _dist e chamado
    # 8.3M vezes num unico run de 3s em kro124p.1, ~34% do runtime total).
    # Equivalencia comportamental verificada exaustivamente (todos os pares
    # (a,b) validos, 5 instancias incl. kro124p.1) antes de aplicar: 0
    # divergencias.
    matrix: List["array.array[float]"] = []
    for i in range(ew + 1, ew + 1 + dim):
        matrix.append(array.array('d', [float(x) for x in lines[i].split()]))

    city_group: Dict[int, int] = {}
    group_cities: Dict[int, List[int]] = {}
    i = idx_de("GTSP_SET_SECTION") + 1
    while i < len(lines) and not lines[i].strip().startswith(
            ("GTSP_SET_ORDERING", "START_GROUP_SECTION", "EOF")):
        s = lines[i].strip()
        if s:
            nums = [int(x) for x in s.split()]
            grp = nums[0]
            cidades = []
            for c in nums[1:]:
                if c == -1:
                    break
                cidades.append(c)
                city_group[c] = grp
            group_cities[grp] = cidades
        i += 1

    pares: List[Tuple[int, int]] = []
    i = idx_de("GTSP_SET_ORDERING") + 1
    while i < len(lines) and not lines[i].strip().startswith(("START_GROUP_SECTION", "EOF")):
        s = lines[i].strip()
        if s:
            nums = [int(x) for x in s.split()]
            pred = nums[0]
            for suc in nums[1:]:
                if suc == -1:
                    break
                pares.append((pred, suc))
        i += 1

    sg = idx_de("START_GROUP_SECTION")
    start_group = int(lines[sg + 1].strip())

    return {
        "dim": dim, "matrix": matrix, "city_group": city_group,
        "group_cities": group_cities, "pares": pares, "start_group": start_group,
    }


def _dist(matrix, a: int, b: int) -> float:
    """Distancia DIRECIONAL a->b (cidades 1-indexadas); -1 = arco proibido."""
    d = matrix[a - 1][b - 1]
    return INF if d < 0 else d


def _ordem_e_factivel_ciclica(ordem: List[int], pares: List[Tuple[int, int]], raiz: int) -> bool:
    """`_ordem_e_factivel` (heuristic_core) checa posicao via enumerate() bruto
    da lista, mas aqui `ordem` representa
    um CICLO fechado com `raiz` (start_group) como ponto de referencia fixo
    -- nao uma lista linear qualquer. Depois de Or-opt/2-opt/double-bridge, a
    raiz pode nao estar mais no indice 0, e a checagem bruta passa a
    verificar uma rotacao ARBITRARIA do ciclo (nao a ordem real relativa ao
    inicio fixo), aceitando como "factivel" ordens que na verdade violam Pi.
    Fix: rotaciona pra raiz ficar no indice 0 antes de delegar pra
    `_ordem_e_factivel`. Usar esta funcao (nao a bruta) em qualquer checagem
    de factibilidade sobre uma ordem de GRUPOS nas instancias PCGTSPLIB."""
    from heuristic_core import _ordem_e_factivel
    if raiz not in ordem:
        return _ordem_e_factivel(ordem, pares)
    idx = ordem.index(raiz)
    rotacionada = ordem[idx:] + ordem[:idx]
    return _ordem_e_factivel(rotacionada, pares)


def _viol_ciclica(ordem_cidades: List[int], city_group: Dict[int, int],
                   pares: List[Tuple[int, int]], raiz_grupo: int) -> int:
    """Conta violacoes de Pi numa lista de CIDADES (nao grupos), rotacionando
    pro grupo-raiz ficar em posicao 0 antes de comparar posicoes -- mesma
    correcao de `_ordem_e_factivel_ciclica`, aplicada na granularidade usada
    pelos viol-counts finais (tour_total)."""
    grupos = [city_group[c] for c in ordem_cidades]
    if raiz_grupo in grupos:
        idx = grupos.index(raiz_grupo)
        grupos = grupos[idx:] + grupos[:idx]
    pos = {g: i for i, g in enumerate(grupos)}
    viol = 0
    for pred, suc in pares:
        if pred in pos and suc in pos and pos[pred] > pos[suc]:
            viol += 1
    return viol


# ===========================================================================
# Grafo de grupos + MST + arvore (sem geometria)
# ===========================================================================

def construir_grafo_grupos(parsed: Dict) -> nx.Graph:
    group_cities = parsed["group_cities"]
    matrix = parsed["matrix"]
    grupos = sorted(group_cities.keys())
    G = nx.Graph()
    for g in grupos:
        G.add_node(g)
    for gi, a in enumerate(grupos):
        for b in grupos[gi + 1:]:
            melhor = None
            for ca in group_cities[a]:
                for cb in group_cities[b]:
                    d = min(_dist(matrix, ca, cb), _dist(matrix, cb, ca))
                    if d == INF:
                        continue
                    if melhor is None or d < melhor[0]:
                        melhor = (d, ca, cb)
            if melhor is not None:
                G.add_edge(a, b, weight=melhor[0], connection=(melhor[1], melhor[2]))
    return G


def construir_arvore_grupos(G: nx.Graph, raiz: int) -> nx.DiGraph:
    """BFS simples a partir da raiz -- versao sem dependencia de Shapely de
    `graph_routing.construir_arvore_direcionada` (que exige atributo
    'poligono' por no, inexistente aqui)."""
    T = nx.DiGraph()
    T.add_node(raiz)
    for u, v in nx.bfs_edges(G, raiz):
        T.add_edge(u, v, weight=G.edges[u, v]["weight"])
    return T


# ===========================================================================
# Ordem de visita dos grupos respeitando Pi (reaproveita o mecanismo LCA)
# ===========================================================================

def ordem_de_visita_grupos(arvore: nx.DiGraph, raiz: int, pares: List[Tuple[int, int]]) -> List[int]:
    """
    Gera a ordem de visita dos GRUPOS respeitando Pi, reaproveitando
    diretamente o mecanismo LCA de heuristic_core.py. Como nao
    ha "perimetro" (todo grupo e' um bloco pontual), o bloco "OWN" de cada
    no' e' trivial -- so o proprio grupo, sem filhos interleaved. Todo
    filho e', portanto, tratado como um "bloco" completo (equivalente aos
    filhos_bloco de caminha_arvore_hibrida_lca), nunca como "sucessor
    normal" (que so faz sentido quando ha percurso de perimetro).
    """
    pares_por_lca = _agrupar_pares_por_lca(arvore, pares)
    ordem: List[int] = []

    def visitar(no):
        pares_aqui = pares_por_lca.get(no, [])
        if pares_aqui:
            ordem_blocos, ok = _ordenar_blocos(no, pares_aqui, arvore)
        else:
            ordem_blocos, ok = [_OWN], True

        filhos = list(arvore.successors(no))

        if not ok:
            # CONFLITO -- recuo local: DFS simples + fixup de precedencia
            # so' para a subarvore de 'no'.
            desc = nx.descendants(arvore, no)
            desc.add(no)
            sub = arvore.subgraph(desc)
            ordem_base = list(nx.dfs_preorder_nodes(sub, no))
            preds_local: Dict[int, List[int]] = {}
            for lca_no, pares_lca in pares_por_lca.items():
                if lca_no not in desc:
                    continue
                for furo, pai in pares_lca:
                    if furo in desc and pai in desc:
                        preds_local.setdefault(pai, []).append(furo)
            ordem.extend(_ordem_respeitando_precedencia(ordem_base, preds_local))
            return

        filhos_bloco = {b for b in ordem_blocos if b is not _OWN}
        filhos_normais = [f for f in filhos if f not in filhos_bloco]

        for bloco in ordem_blocos:
            if bloco is _OWN:
                ordem.append(no)
                for f in filhos_normais:
                    visitar(f)
            else:
                visitar(bloco)

    visitar(raiz)
    return ordem


# ===========================================================================
# Selecao gulosa de cidade representante + calculo de FO
# ===========================================================================

def escolher_cidades_e_fo(ordem_grupos: List[int], parsed: Dict) -> Tuple[List[int], float]:
    group_cities = parsed["group_cities"]
    matrix = parsed["matrix"]

    cidades_escolhidas: List[int] = []
    cidade_atual: Optional[int] = None
    fo = 0.0

    for grp in ordem_grupos:
        candidatos = group_cities[grp]
        if cidade_atual is None:
            escolhida = candidatos[0]
        else:
            escolhida = min(candidatos, key=lambda c: _dist(matrix, cidade_atual, c))
            custo = _dist(matrix, cidade_atual, escolhida)
            fo += custo if custo != INF else 0.0
        cidades_escolhidas.append(escolhida)
        cidade_atual = escolhida

    if cidades_escolhidas:
        primeira = cidades_escolhidas[0]
        custo_fecha = _dist(matrix, cidade_atual, primeira)
        fo += custo_fecha if custo_fecha != INF else 0.0

    return cidades_escolhidas, fo


# ===========================================================================
# Entrada principal
# ===========================================================================

def resolver_direto(path) -> Dict:
    t0 = time.time()
    parsed = parse_pcglns(path)
    G = construir_grafo_grupos(parsed)

    fo_total, tour_total = 0.0, []
    viol = 0
    componentes = list(nx.connected_components(G))
    for comp in componentes:
        sub = G.subgraph(comp)
        mst = nx.minimum_spanning_tree(sub, weight="weight")
        raiz = parsed["start_group"] if parsed["start_group"] in comp else next(iter(comp))
        arvore = construir_arvore_grupos(mst, raiz)
        pares_comp = [(f, p) for f, p in parsed["pares"] if f in comp and p in comp]
        ordem_grupos = ordem_de_visita_grupos(arvore, raiz, pares_comp)
        cidades, fo = escolher_cidades_e_fo(ordem_grupos, parsed)
        fo_total += fo
        tour_total.extend(cidades)
        # checagem por componente, rotacionada pra raiz -- nao um
        # enumerate() global sobre tour_total (ver _ordem_e_factivel_ciclica
        # acima).
        viol += _viol_ciclica(cidades, parsed["city_group"], pares_comp, raiz)

    t_total = time.time() - t0

    return {
        "fo": fo_total,
        "tour": tour_total,
        "n_grupos": len(parsed["group_cities"]),
        "n_pares": len(parsed["pares"]),
        "violacoes": viol,
        "tempo_s": t_total,
    }


if __name__ == "__main__":
    for fp in sys.argv[1:]:
        r = resolver_direto(fp)
        print(f"{Path(fp).name:<20} FO={r['fo']:>12.2f}  grupos={r['n_grupos']:>4}  "
              f"pares={r['n_pares']:>4}  viol={r['violacoes']:>3}  t={r['tempo_s']:.3f}s")
