"""
vns_zhang_fiel.py
==================
Implementacao do VNS de Zhang, Yao, Liu, Wei & Zhang (2024, Expert Systems
with Applications v237, "Heuristic approaches for the cutting path problem"),
usada como segundo baseline (modelo GTSP / "node routing") em todos os
conjuntos de instancias.

Fidelidade ao artigo de Zhang et al. (Secoes 3.2/4.2, Algoritmos 3-6):
  - Representacao da solucao: sequencia de PONTOS S = (S_1, ..., S_|P|), um
    ponto (cidade) por peca, ONDE A ORDEM DA SEQUENCIA E' A ORDEM DE VISITA
    -- exatamente como descrito no artigo (Secao 3.2), NAO uma ordem de
    grupos separada de uma escolha de cidade via DP.
  - 7 operadores exatamente como descritos (Secao 4.2, Fig. 6):
    Change, Swap, Reverse, Shift (usados no Shake, Algoritmo 5) e Nearest
    replace, Farthest replace, Double replace (usados so' no LocalSearch,
    Algoritmo 6, junto com os 4 primeiros).
  - Shake (Algoritmo 5): max_iter_shake=10 aplicacoes de UM operador
    aleatorio entre os 4 primeiros, escolhido com igual probabilidade, com
    pontos a/b escolhidos exatamente pela regra do artigo.
  - LocalSearch (Algoritmo 6): embaralha os 7 operadores UMA vez; para cada
    operador da ordem embaralhada, tenta TODOS os movimentos possiveis
    daquele operador (melhoria-otima dentro da vizinhanca), aplica o melhor
    se for melhora e REINICIA do primeiro operador da MESMA ordem
    embaralhada; senao passa pro proximo operador da ordem.
  - Construcao inicial NearestFirst (Algoritmo 4): guloso por ponto mais
    proximo.
  - VNS externo (Algoritmo 3): contador de nao-melhoria vs max_iter_vns.

Desvio deliberado e documentado do artigo original (que nao lida com
precedencia alguma -- o GTSP deles e' irrestrito):
  - Extensao para respeitar Pi (pares furo-antes-contorno do PCGTSPLIB):
    Change/NearestReplace/FarthestReplace/DoubleReplace nunca alteram a
    ORDEM de visita (so' trocam qual ponto representa uma peca na MESMA
    posicao), logo nunca violam Pi por construcao. Swap/Reverse/Shift PODEM
    alterar a ordem; usamos o MESMO mecanismo de reparo topologico ja'
    do projeto (extracao topologica estavel priorizando a ordem desejada;
    ver `_reparar_se_infactivel`).
  - Orcamento de tempo: o artigo usa 1200s / 10000 iteracoes sem melhoria;
    aqui usamos o MESMO orcamento de 3s por instancia do protocolo deste
    trabalho para todos os metodos comparados (heuristica propria, PCGLNS,
    VNS), para manter a comparacao justa em tempo -- mantendo
    max_iter_vns=10000 como teto adicional, quase nunca atingido antes do
    teto de tempo nas instancias deste projeto.
  - NearestFirst adaptado para respeitar Pi na construcao (escolhe entre
    pecas "prontas", com predecessores ja' visitados); o artigo original nao
    precisa disso.

Infraestrutura compartilhada com heuristica_direta_pcgtsp.py (nao faz parte
do algoritmo de busca):
  - parse_pcglns, _dist: parsing do .pcglns e distancia direcional.
  - _ordem_e_factivel_ciclica, _viol_ciclica: checagem de factibilidade de Pi.
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pcgtsp_comparison"))
from heuristica_direta_pcgtsp import parse_pcglns, _dist, _ordem_e_factivel_ciclica, _viol_ciclica  # noqa: E402

INF = float("inf")

OP_SHAKE = ["change", "swap", "reverse", "shift"]              # OP_1..OP_4
OP_ALL = OP_SHAKE + ["nearest_replace", "farthest_replace", "double_replace"]  # OP_1..OP_7


# ===========================================================================
# Custo de uma sequencia de pontos (ciclo fechado, como no artigo: GTSP e'
# um tour Hamiltoniano fechado sobre as pecas)
# ===========================================================================

def _custo_total(seq: List[int], matrix) -> float:
    n = len(seq)
    total = 0.0
    for i in range(n):
        d = _dist(matrix, seq[i], seq[(i + 1) % n])
        if d == INF:
            return INF
        total += d
    return total


def _delta_troca_posicoes(seq: List[int], i: int, j: int, matrix) -> float:
    """Delta de custo (novo - velho) ao trocar seq[i] <-> seq[j] (i != j),
    olhando so' as arestas incidentes as posicoes i e j (O(1))."""
    n = len(seq)
    if i > j:
        i, j = j, i
    idxs = sorted({(i - 1) % n, i, (i + 1) % n, (j - 1) % n, j, (j + 1) % n})
    antes = sum(_dist(matrix, seq[k], seq[(k + 1) % n]) for k in idxs if (k + 1) % n not in () )
    # Mais simples e seguro: recomputa so' as arestas tocadas, evitando dupla-conta.
    arestas = set()
    for p in (i, j):
        arestas.add((p - 1) % n)
        arestas.add(p % n)
    antes = sum(_dist(matrix, seq[k], seq[(k + 1) % n]) for k in arestas)
    nova = seq[:]
    nova[i], nova[j] = nova[j], nova[i]
    depois = sum(_dist(matrix, nova[k], nova[(k + 1) % n]) for k in arestas)
    if antes == INF or depois == INF:
        return INF
    return depois - antes


# ===========================================================================
# Construcao inicial: NearestFirst (Algoritmo 4), adaptado p/ respeitar Pi
# ===========================================================================

def _nearest_first(parsed: Dict, rng: random.Random) -> List[int]:
    group_cities = parsed["group_cities"]
    matrix = parsed["matrix"]
    pares = parsed["pares"]
    raiz = parsed["start_group"]
    grupos = list(group_cities.keys())

    preds: Dict[int, set] = {g: set() for g in grupos}
    for pred, suc in pares:
        preds[suc].add(pred)

    visitados = {raiz}
    seq = [group_cities[raiz][0]]
    atual = seq[0]

    while len(visitados) < len(grupos):
        prontos = [g for g in grupos if g not in visitados and preds[g] <= visitados]
        if not prontos:
            prontos = [g for g in grupos if g not in visitados]
        melhor_g, melhor_c, melhor_d = None, None, INF
        for g in prontos:
            for c in group_cities[g]:
                d = _dist(matrix, atual, c)
                if d < melhor_d:
                    melhor_d, melhor_g, melhor_c = d, g, c
        if melhor_g is None:
            melhor_g = rng.choice(prontos)
            melhor_c = group_cities[melhor_g][0]
        seq.append(melhor_c)
        visitados.add(melhor_g)
        atual = melhor_c

    return seq


# ===========================================================================
# Reparo de precedencia (so' necessario apos Swap/Reverse/Shift, que mudam
# a ORDEM de visita das pecas -- Change/replace-* nunca mudam ordem)
# ===========================================================================

def _seq_para_ordem_grupos(seq: List[int], city_group: Dict[int, int]) -> List[int]:
    return [city_group[c] for c in seq]


def _reparar_se_infactivel(seq: List[int], parsed: Dict, preds: Dict[int, set]) -> List[int]:
    """Se a ordem de grupos implicita em `seq` violar Pi, repara via
    extracao topologica estavel, preservando o ponto (cidade) ja' escolhido para cada peca -- so' a
    ORDEM das pecas e' re-arranjada, nao os pontos representantes."""
    city_group = parsed["city_group"]
    pares = parsed["pares"]
    raiz = parsed["start_group"]
    ordem_grupos = _seq_para_ordem_grupos(seq, city_group)
    if _ordem_e_factivel_ciclica(ordem_grupos, pares, raiz):
        return seq

    ponto_do_grupo = {city_group[c]: c for c in seq}
    posicao_desejada = {g: i for i, g in enumerate(ordem_grupos)}
    visitado = {raiz}
    resultado_grupos = [raiz]
    restantes = set(ordem_grupos) - {raiz}
    while restantes:
        prontos = [g for g in restantes if preds[g] <= visitado]
        if not prontos:
            prontos = list(restantes)
        escolhido = min(prontos, key=lambda g: posicao_desejada[g])
        resultado_grupos.append(escolhido)
        visitado.add(escolhido)
        restantes.discard(escolhido)
    return [ponto_do_grupo[g] for g in resultado_grupos]


# ===========================================================================
# Os 7 operadores (Secao 4.2 / Fig. 6). Cada um recebe a sequencia e um par
# de pontos (a, b) ja' escolhidos e devolve uma NOVA sequencia.
# ===========================================================================

def _op_change(seq: List[int], a_idx: int, parsed: Dict, rng: random.Random) -> List[int] | None:
    """Muda o ponto da peca de `a` para OUTRO ponto da MESMA peca (ordem
    inalterada -- nunca pode violar Pi)."""
    city_group = parsed["city_group"]
    group_cities = parsed["group_cities"]
    a = seq[a_idx]
    g = city_group[a]
    candidatos = [c for c in group_cities[g] if c != a]
    if not candidatos:
        return None
    novo_ponto = rng.choice(candidatos)
    nova = seq[:]
    nova[a_idx] = novo_ponto
    return nova


def _op_swap(seq: List[int], i: int, j: int) -> List[int] | None:
    if i == j:
        return None
    nova = seq[:]
    nova[i], nova[j] = nova[j], nova[i]
    return nova


def _op_reverse(seq: List[int], i: int, j: int) -> List[int] | None:
    if i == j:
        return None
    if i > j:
        i, j = j, i
    return seq[:i] + seq[i:j + 1][::-1] + seq[j + 1:]


def _op_shift(seq: List[int], a_idx: int, b_idx: int) -> List[int] | None:
    """Desloca o ponto na posicao a_idx para imediatamente ANTES do ponto
    na posicao b_idx (ordem dos demais preservada)."""
    if a_idx == b_idx:
        return None
    n = len(seq)
    ponto_a = seq[a_idx]
    resto = seq[:a_idx] + seq[a_idx + 1:]
    b_val = seq[b_idx]
    j = resto.index(b_val)
    return resto[:j] + [ponto_a] + resto[j:]


def _op_nearest_replace(seq: List[int], a_idx: int, parsed: Dict, farthest: bool = False) -> List[int] | None:
    """Acha o ponto b (peca Pb != peca de a) mais proximo (ou mais
    distante, se farthest=True) de a, dentre TODAS as pecas restantes, e
    substitui o ponto atualmente usado por Pb pelo ponto b (na MESMA
    posicao de Pb na sequencia -- ordem inalterada)."""
    city_group = parsed["city_group"]
    group_cities = parsed["group_cities"]
    matrix = parsed["matrix"]
    a = seq[a_idx]
    ga = city_group[a]

    melhor_b, melhor_d = None, (-INF if farthest else INF)
    for g, cidades in group_cities.items():
        if g == ga:
            continue
        for c in cidades:
            d = _dist(matrix, a, c)
            if d == INF:
                continue
            if farthest:
                if d > melhor_d:
                    melhor_d, melhor_b = d, c
            else:
                if d < melhor_d:
                    melhor_d, melhor_b = d, c
    if melhor_b is None:
        return None
    gb = city_group[melhor_b]
    pos_b = next(k for k, c in enumerate(seq) if city_group[c] == gb)
    nova = seq[:]
    nova[pos_b] = melhor_b
    return nova


def _op_double_replace(seq: List[int], a_idx: int, parsed: Dict) -> List[int] | None:
    """b1, b2 = vizinhos de a NA SEQUENCIA (posicoes a_idx-1 e a_idx+1,
    ciclico). Para cada um, escolhe -- dentro da MESMA peca de b1/b2 -- o
    ponto que minimiza a distancia total local (arestas com `a` e com o
    outro vizinho do lado de fora), mantendo a ordem."""
    n = len(seq)
    group_cities = parsed["group_cities"]
    city_group = parsed["city_group"]
    matrix = parsed["matrix"]
    a = seq[a_idx]
    i1, i2 = (a_idx - 1) % n, (a_idx + 1) % n
    nova = seq[:]
    for pos_b, outro_lado in ((i1, (i1 - 1) % n), (i2, (i2 + 1) % n)):
        if pos_b == a_idx:
            continue
        g_b = city_group[seq[pos_b]]
        c_outro = nova[outro_lado]
        melhor_c, melhor_d = nova[pos_b], INF
        for c in group_cities[g_b]:
            d1 = _dist(matrix, a, c)
            d2 = _dist(matrix, c, c_outro) if outro_lado != a_idx else 0.0
            if d1 == INF or d2 == INF:
                continue
            tot = d1 + d2
            if tot < melhor_d:
                melhor_d, melhor_c = tot, c
        nova[pos_b] = melhor_c
    if nova == seq:
        return None
    return nova


# ===========================================================================
# Shake (Algoritmo 5): 10 aplicacoes de 1 operador aleatorio (OP_1..OP_4)
# ===========================================================================

def _shake(seq: List[int], parsed: Dict, preds: Dict[int, set], rng: random.Random,
           max_iter_shake: int = 10) -> List[int]:
    atual = seq[:]
    n = len(atual)
    for _ in range(max_iter_shake):
        op = rng.choice(OP_SHAKE)
        a_idx = rng.randrange(n)
        if op == "change":
            nova = _op_change(atual, a_idx, parsed, rng)
        else:
            b_idx = rng.randrange(n)
            while b_idx == a_idx and n > 1:
                b_idx = rng.randrange(n)
            if op == "swap":
                nova = _op_swap(atual, a_idx, b_idx)
            elif op == "reverse":
                nova = _op_reverse(atual, a_idx, b_idx)
            else:  # shift
                nova = _op_shift(atual, a_idx, b_idx)
        if nova is None:
            continue
        if op in ("swap", "reverse", "shift"):
            nova = _reparar_se_infactivel(nova, parsed, preds)
        atual = nova
    return atual


# ===========================================================================
# LocalSearch (Algoritmo 6): embaralha os 7 operadores 1 vez; para cada um,
# tenta TODOS os movimentos possiveis (melhor-melhoria), reinicia no 1o
# operador da MESMA ordem embaralhada se melhorou.
# ===========================================================================

def _melhor_movimento_operador(seq: List[int], op: str, parsed: Dict, preds: Dict[int, set],
                                custo_atual: float, matrix, rng: random.Random,
                                deadline: float) -> Tuple[List[int], float] | None:
    """Tenta TODOS os movimentos possiveis do operador `op`, retorna o
    (seq, custo) do melhor movimento que MELHORA custo_atual, ou None se
    nenhum melhora (ou se o tempo acabou no meio -- devolve o melhor achado
    ate' entao, mesmo que a varredura nao tenha sido 100% completa, pra
    nunca estourar o orcamento de 3s)."""
    n = len(seq)
    melhor_seq, melhor_custo = None, custo_atual

    def considerar(cand):
        nonlocal melhor_seq, melhor_custo
        if cand is None:
            return
        c = _custo_total(cand, matrix)
        if c < melhor_custo - 1e-9:
            melhor_custo, melhor_seq = c, cand

    if op == "change":
        for a_idx in range(n):
            if time.time() > deadline:
                break
            city_group = parsed["city_group"]
            group_cities = parsed["group_cities"]
            a = seq[a_idx]
            g = city_group[a]
            for c in group_cities[g]:
                if c == a:
                    continue
                nova = seq[:]
                nova[a_idx] = c
                considerar(nova)
    elif op == "swap":
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                cand = _op_swap(seq, i, j)
                cand = _reparar_se_infactivel(cand, parsed, preds) if cand else None
                considerar(cand)
    elif op == "reverse":
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                cand = _op_reverse(seq, i, j)
                cand = _reparar_se_infactivel(cand, parsed, preds) if cand else None
                considerar(cand)
    elif op == "shift":
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(n):
                if i == j:
                    continue
                cand = _op_shift(seq, i, j)
                cand = _reparar_se_infactivel(cand, parsed, preds) if cand else None
                considerar(cand)
    elif op == "nearest_replace":
        for a_idx in range(n):
            if time.time() > deadline:
                break
            considerar(_op_nearest_replace(seq, a_idx, parsed, farthest=False))
    elif op == "farthest_replace":
        for a_idx in range(n):
            if time.time() > deadline:
                break
            considerar(_op_nearest_replace(seq, a_idx, parsed, farthest=True))
    elif op == "double_replace":
        for a_idx in range(n):
            if time.time() > deadline:
                break
            considerar(_op_double_replace(seq, a_idx, parsed))

    if melhor_seq is None:
        return None
    return melhor_seq, melhor_custo


def _local_search(seq: List[int], parsed: Dict, preds: Dict[int, set], rng: random.Random,
                   deadline: float) -> Tuple[List[int], float]:
    matrix = parsed["matrix"]
    atual = seq[:]
    custo_atual = _custo_total(atual, matrix)
    ordem_ops = OP_ALL[:]
    rng.shuffle(ordem_ops)
    k = 0
    while k < len(ordem_ops):
        if time.time() > deadline:
            break
        op = ordem_ops[k]
        resultado = _melhor_movimento_operador(atual, op, parsed, preds, custo_atual, matrix, rng, deadline)
        if resultado is not None:
            atual, custo_atual = resultado
            k = 0
        else:
            k += 1
    return atual, custo_atual


# ===========================================================================
# VNS principal (Algoritmo 3)
# ===========================================================================

def resolver_com_vns_fiel(path, time_budget: float = 3.0, seed: int = 0,
                           max_iter_vns: int = 10000) -> Dict:
    t0 = time.time()
    deadline = t0 + time_budget
    rng = random.Random(seed)
    parsed = parse_pcglns(path)
    matrix = parsed["matrix"]
    pares = parsed["pares"]
    raiz_grupo = parsed["start_group"]

    preds: Dict[int, set] = {g: set() for g in parsed["group_cities"]}
    for pred, suc in pares:
        preds[suc].add(pred)

    seq = _nearest_first(parsed, rng)
    # NearestFirst (Algoritmo 4) e' o algoritmo ORIGINAL do Zhang2024, sem
    # nocao de precedencia -- pode gerar uma ordem inicial infactivel. O
    # reparo e' aplicado depois aos candidatos gerados por Shake/LocalSearch
    # (Swap/Reverse/Shift); se a busca nunca encontrar um movimento de
    # MELHORIA de custo que tambem corrija Pi (plausivel em instancias
    # pequenas, com pouco espaco de busca), a infactibilidade inicial
    # persistiria no resultado final. Por isso a construcao inicial e'
    # reparada antes de virar a referencia `melhor_seq`.
    seq = _reparar_se_infactivel(seq, parsed, preds)
    custo = _custo_total(seq, matrix)
    melhor_seq, melhor_custo = seq, custo

    non_imp = 0
    shakes_tentados = 0
    shakes_aceitos = 0

    while non_imp <= max_iter_vns and time.time() < deadline:
        shakes_tentados += 1
        candidata = _shake(melhor_seq, parsed, preds, rng)
        candidata, custo_cand = _local_search(candidata, parsed, preds, rng, deadline)

        if custo_cand < melhor_custo - 1e-9:
            melhor_seq, melhor_custo = candidata, custo_cand
            shakes_aceitos += 1
            non_imp = 0
        else:
            non_imp += 1

    t_total = time.time() - t0
    ordem_grupos_final = _seq_para_ordem_grupos(melhor_seq, parsed["city_group"])
    viol = _viol_ciclica(melhor_seq, parsed["city_group"], pares, raiz_grupo)

    return {
        "tour": melhor_seq,
        "fo": melhor_custo,
        "t": t_total,
        "shakes_tentados": shakes_tentados,
        "shakes_aceitos": shakes_aceitos,
        "viol": viol,
    }


if __name__ == "__main__":
    for fp in sys.argv[1:]:
        r = resolver_com_vns_fiel(fp, time_budget=3.0, seed=0)
        print(f"{Path(fp).name:<20} FO={r['fo']:>12.2f}  viol={r['viol']}  "
              f"shakes={r['shakes_aceitos']}/{r['shakes_tentados']}  t={r['t']:.2f}s")
