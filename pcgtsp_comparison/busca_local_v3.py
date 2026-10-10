"""
busca_local_v3.py
====================
Operadores de busca local sobre a ORDEM DE VISITA DOS GRUPOS (instancias
PCGTSPLIB), com a checagem de factibilidade Pi embutida no proprio operador.
A abordagem "gerar e testar" paga, por candidato, uma checagem
O(n+|pares|) (`_ordem_e_factivel_ciclica`) so' pra descobrir, na maioria das
vezes, que o candidato era invalido.

Ideia central: para cada operador, computar DIRETAMENTE o conjunto de
movimentos Pi-factiveis (a partir dos pares de precedencia diretos), em vez
de gerar candidatos as cegas e descartar os que violam Pi. Isso (a) nunca
desperdica avaliacao em candidatos infactiveis e (b) elimina a checagem cara
de ordem inteira para os movimentos ja garantidamente factiveis por
construcao -- os dois efeitos aceleram cada rodada, o que devolve iteracoes
extras ao orcamento de tempo fixo (o orcamento de tempo NAO e aumentado frente
ao PCGLNS: so' se faz mais com o mesmo orcamento).

Prova de correcao de cada operador (raciocinio, nao so' teste empirico --
mas TODO resultado final ainda passa pela verificacao independente ja
padrao no projeto, _viol_ciclica sobre o tour final, como rede de seguranca):

1) Or-opt (relocar 1 grupo g): mover SO' g, mantendo a ordem relativa de
   todos os outros grupos intacta, so' pode violar pares que envolvem o
   proprio g (pares entre outros dois grupos quaisquer nao mudam de ordem
   relativa entre si). Logo, g so' pode ir para uma posicao j tal que todo
   predecessor direto de g (em `pares`) fique ANTES de j e todo sucessor
   direto fique DEPOIS -- uma janela [L, R] computada em O(|pares|), sem
   nenhum candidato fora dela precisar ser gerado ou checado.

2) 2-opt (inverter segmento [i,j]): inverter um segmento contiguo so' troca
   a ordem relativa de pares de grupos AMBOS dentro do segmento (um grupo
   antes do segmento continua antes; um grupo depois continua depois --
   so' a ordem interna inverte). Logo, o segmento [i,j] e' Pi-seguro sse
   nenhum par de precedencia tem os dois extremos dentro de [i,j]. Isso da'
   um limite superior `limite[i]` por posicao inicial i, prE-computado em
   O(n+|pares|) via sweep, usado para podar o laco j sem checagem por
   candidato -- MAS esse argumento assume que a ancora (raiz, posicao 0)
   fica fixa; para segmentos que incluem i=0 (a propria raiz se move), o
   limite[] nao vale e usa-se a checagem exaustiva em vez da janela (ver docstring de
   `two_opt_uma_rodada_v3`).

3) Double-bridge (A+C+B+D): so' troca a ordem relativa de pares com um
   extremo em B e outro em C (A fica sempre primeiro, D sempre por ultimo,
   entao pares tocando A ou D nunca mudam de relacao). Isso reduz a checagem
   de O(n+|pares|) (rotacionar lista inteira + reconstruir dict de posicao)
   para O(|pares|) (so' testar se algum par tem um extremo no intervalo
   B e outro no intervalo C) -- mesma corretude, custo por tentativa menor,
   o que permite mais tentativas no mesmo orcamento.

Todos os 3 operadores trabalham sempre no referencial rotacionado (raiz na
posicao 0) -- rotacionam `ordem` para esse referencial no INICIO da chamada
(nao assumem que o chamador ja rotacionou, mesma defesa que
`_ordem_e_factivel_ciclica` ja faz) e devolvem a lista nesse mesmo
referencial (rotacao != mudanca de tour, so' de representacao -- o
chamador seguinte re-rotaciona a partir de onde a raiz estiver, como o
pipeline inteiro ja faz).

Tratamento da raiz: a raiz (posicao 0) e' um caso especial nos dois
operadores. Mover a raiz muda o PONTO DE CORTE do ciclo, o que pode inverter
a ordem aparente de pares que nao a envolvem; por isso a janela direta (valida
so' para grupos cujo deslocamento afeta unicamente pares que os envolvem) nao
e' usada para ela, e sim a checagem exaustiva `_ordem_e_factivel_ciclica`
(ver as docstrings de `or_opt_uma_rodada_v3` e `two_opt_uma_rodada_v3`).
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pcgtsp_comparison"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from heuristica_direta_pcgtsp import _dist, INF  # noqa: E402


def _rotacionar(ordem, raiz):
    if raiz in ordem:
        k = ordem.index(raiz)
        return ordem[k:] + ordem[:k]
    return list(ordem)


def _preds_succs(pares):
    preds_de, succs_de = {}, {}
    for p, s in pares:
        succs_de.setdefault(p, set()).add(s)
        preds_de.setdefault(s, set()).add(p)
    return preds_de, succs_de


# ===========================================================================
# 1) Or-opt v3: janela de insercao computada diretamente (sem gerar-e-testar)
# ===========================================================================

def or_opt_uma_rodada_v3(ordem_grupos, cidade_de, matrix, pares, time_limit, raiz=None):
    """Or-opt de 1 grupo, com janela de insercao computada diretamente.

    Para todo grupo que NAO e' a raiz, a janela [L, R] de posicoes validas e'
    computada a partir dos predecessores/sucessores diretos do grupo (ver
    docstring do modulo). A RAIZ e' um caso especial: mover a raiz muda o
    PONTO DE CORTE do ciclo, o que pode inclusive inverter a ordem aparente
    de OUTROS pares que nao a envolvem (contra-exemplo: ciclo raiz,X,Y,Z --
    mover a raiz pra entre Y e Z e re-rotacionar da' Z,X,Y, invertendo a
    ordem aparente de X e Z). Por isso a "janela direta" (valida para
    qualquer outro grupo, cujo deslocamento so' afeta pares que o envolvem)
    NAO e' segura para a raiz: ela e' tratada testando TODAS as posicoes com
    a checagem exaustiva `_ordem_e_factivel_ciclica`. Isso custa O(n+|pares|)
    por candidato so' para este 1 grupo (nao para os outros n-1), entao o
    ganho de velocidade sobre os demais grupos se mantem. Excluir a raiz da
    vizinhanca nao e' uma opcao: em ESC25, por exemplo, mover a raiz e' o
    primeiro movimento aceito pela busca exaustiva (delta=-639, o maior
    ganho isolado de toda a convergencia) e nao e' substituivel por nenhuma
    combinacao de mover outros grupos."""
    from heuristica_direta_pcgtsp import _ordem_e_factivel_ciclica
    t0 = time.time()
    if raiz is None:
        raiz = ordem_grupos[0]
    ordem = _rotacionar(ordem_grupos, raiz)
    n = len(ordem)
    preds_de, succs_de = _preds_succs(pares)

    melhorou_algo = False
    melhorou_nesta_passada = True
    while melhorou_nesta_passada and time.time() - t0 < time_limit:
        melhorou_nesta_passada = False
        for i in range(n):
            if time.time() - t0 > time_limit:
                break
            g = ordem[i]
            eh_raiz = (i == 0)
            a = ordem[i - 1] if i > 0 else ordem[-1]
            b = ordem[(i + 1) % n]
            if a == g or b == g:
                continue
            d_ag = _dist(matrix, cidade_de[a], cidade_de[g])
            d_gb = _dist(matrix, cidade_de[g], cidade_de[b])
            d_ab = _dist(matrix, cidade_de[a], cidade_de[b])
            if d_ab == INF:
                continue
            custo_remove = d_ag + d_gb - d_ab

            sem_g = ordem[:i] + ordem[i + 1:]

            if eh_raiz:
                j_min, j_max = 0, len(sem_g) - 1
            else:
                pos_sem_g = {grp: idx for idx, grp in enumerate(sem_g)}
                L = max((pos_sem_g[p] for p in preds_de.get(g, ()) if p in pos_sem_g), default=-1)
                R = min((pos_sem_g[s] for s in succs_de.get(g, ()) if s in pos_sem_g), default=len(sem_g))
                j_min, j_max = max(0, L), min(len(sem_g) - 1, R - 1)
                if j_min > j_max:
                    continue  # sem janela viavel (nao deveria acontecer se ordem de entrada e' factivel)

            melhor_delta, melhor_pos = 0.0, None
            for j in range(j_min, j_max + 1):
                c = sem_g[j]
                d = sem_g[(j + 1) % len(sem_g)]
                if c == g or d == g:
                    continue
                d_cg = _dist(matrix, cidade_de[c], cidade_de[g])
                d_gd = _dist(matrix, cidade_de[g], cidade_de[d])
                if d_cg == INF or d_gd == INF:
                    continue
                d_cd = _dist(matrix, cidade_de[c], cidade_de[d])
                custo_insere = d_cg + d_gd - d_cd
                delta = custo_insere - custo_remove
                if delta < melhor_delta - 1e-9:
                    candidata = sem_g[:j + 1] + [g] + sem_g[j + 1:]
                    if eh_raiz and not _ordem_e_factivel_ciclica(candidata, pares, raiz):
                        continue  # so' a raiz precisa dessa checagem extra (ver docstring)
                    melhor_delta = delta
                    melhor_pos = candidata
            if melhor_pos is not None:
                if eh_raiz:
                    # re-rotaciona pra raiz voltar ao indice 0 (convencao do
                    # modulo inteiro) antes de continuar o resto da passada
                    idx_raiz = melhor_pos.index(raiz)
                    melhor_pos = melhor_pos[idx_raiz:] + melhor_pos[:idx_raiz]
                ordem = melhor_pos
                melhorou_algo = True
                melhorou_nesta_passada = True
    return ordem, melhorou_algo


# ===========================================================================
# 1-bis) Or-opt de BLOCO v3: generaliza o Or-opt pra relocar um bloco
# contiguo de `tamanho_bloco` grupos de uma vez
# ===========================================================================

def or_opt_bloco_uma_rodada_v3(ordem_grupos, cidade_de, matrix, pares, time_limit,
                                tamanho_bloco=2, raiz=None):
    """Generaliza `or_opt_uma_rodada_v3` pra relocar um BLOCO CONTIGUO de
    `tamanho_bloco` grupos de uma vez (nao so' 1 grupo por vez), mantendo a
    ordem interna do bloco e a ordem relativa de todos os outros grupos.
    Motivacao: em instancias grandes com POUCOS pares Pi (espaco de ordens
    validas enorme, ex.: kro124p.1/2/3 e prob.100), mover 1 grupo de cada
    vez pode nao ser suficiente pra escapar de otimos locais; mover um bloco
    de 2-3 de uma vez e' um movimento estruturalmente diferente, nao
    alcancavel por nenhuma sequencia de movimentos de 1 grupo que passe por
    estados PIORES no meio (que o first-improvement elitista nunca
    aceitaria).

    Prova de correcao (mesmo raciocinio do Or-opt de 1 grupo, generalizado):
    o bloco B=[g_1..g_k] e' inserido INTEIRO (nunca se separa, ordem interna
    fixa) em outra posicao. Um grupo externo h com um par direto tocando
    QUALQUER g_i do bloco so' pode ficar de um lado do bloco inteiro (nao
    "no meio dele", ja que o bloco e' atomico) -- entao a janela de insercao
    e' o MESMO tipo de calculo do Or-opt de 1 grupo, so' que agregando os
    predecessores/sucessores diretos de TODOS os membros do bloco (uniao),
    nao so' de 1 grupo.

    ESCOPO LIMITADO: blocos que incluiriam a raiz sao pulados (i comeca em
    1, nao em 0) -- a raiz continua sendo relocada apenas como bloco de
    tamanho 1 via `or_opt_uma_rodada_v3` (que ja trata seu caso especial
    corretamente).

    Varredura: faz uma varredura COMPLETA de todo `i`, aplicando a melhor
    melhoria em CADA posicao antes de seguir pra proxima, sem parar cedo --
    mesmo padrao de `or_opt_uma_rodada_v3`. (Interromper a passada na
    primeira posicao com melhoria nao exclui nenhum movimento, mas converge
    sistematicamente para otimos locais piores; esta versao foi verificada
    contra uma referencia gerar-e-testar, sem divergencias.)"""
    t0 = time.time()
    if raiz is None:
        raiz = ordem_grupos[0]
    ordem = _rotacionar(ordem_grupos, raiz)
    n = len(ordem)
    k = tamanho_bloco
    if n < k + 3:
        return ordem, False
    preds_de, succs_de = _preds_succs(pares)

    melhorou_algo = False
    melhorou_nesta_passada = True
    while melhorou_nesta_passada and time.time() - t0 < time_limit:
        melhorou_nesta_passada = False
        for i in range(1, n - k + 1):  # i=0 (bloco incluiria a raiz) fora do escopo
            if time.time() - t0 > time_limit:
                break
            bloco = ordem[i:i + k]
            if raiz in bloco:
                continue
            a = ordem[i - 1]
            b = ordem[(i + k) % n]
            if a in bloco or b in bloco:
                continue
            d_a_g0 = _dist(matrix, cidade_de[a], cidade_de[bloco[0]])
            d_gk_b = _dist(matrix, cidade_de[bloco[-1]], cidade_de[b])
            d_a_b = _dist(matrix, cidade_de[a], cidade_de[b])
            if d_a_b == INF:
                continue
            custo_remove = d_a_g0 + d_gk_b - d_a_b

            sem_bloco = ordem[:i] + ordem[i + k:]
            pos_sem_bloco = {grp: idx for idx, grp in enumerate(sem_bloco)}
            L = max((pos_sem_bloco[p] for g in bloco for p in preds_de.get(g, ())
                     if p in pos_sem_bloco), default=-1)
            R = min((pos_sem_bloco[s] for g in bloco for s in succs_de.get(g, ())
                     if s in pos_sem_bloco), default=len(sem_bloco))
            j_min, j_max = max(0, L), min(len(sem_bloco) - 1, R - 1)
            if j_min > j_max:
                continue

            melhor_delta, melhor_pos = 0.0, None
            for j in range(j_min, j_max + 1):
                c = sem_bloco[j]
                d = sem_bloco[(j + 1) % len(sem_bloco)]
                if c in bloco or d in bloco:
                    continue
                d_c_g0 = _dist(matrix, cidade_de[c], cidade_de[bloco[0]])
                d_gk_d = _dist(matrix, cidade_de[bloco[-1]], cidade_de[d])
                if d_c_g0 == INF or d_gk_d == INF:
                    continue
                d_c_d = _dist(matrix, cidade_de[c], cidade_de[d])
                custo_insere = d_c_g0 + d_gk_d - d_c_d
                delta = custo_insere - custo_remove
                if delta < melhor_delta - 1e-9:
                    melhor_delta = delta
                    melhor_pos = sem_bloco[:j + 1] + bloco + sem_bloco[j + 1:]
            if melhor_pos is not None:
                ordem = melhor_pos
                melhorou_algo = True
                melhorou_nesta_passada = True
                # SEM break aqui (ver docstring) -- continua a varredura de
                # `i` no mesmo padrao de or_opt_uma_rodada_v3.
    return ordem, melhorou_algo


# ===========================================================================
# 2) 2-opt v3: limite[i] pre-computado (poda o laco j, sem checagem cara)
# ===========================================================================

def _computar_limites(ordem, pares):
    n = len(ordem)
    pos = {g: i for i, g in enumerate(ordem)}
    hi_em = {}  # lo -> menor hi entre pares cujo span comeca em lo
    for p, s in pares:
        if p not in pos or s not in pos:
            continue
        i1, i2 = pos[p], pos[s]
        lo, hi = (i1, i2) if i1 < i2 else (i2, i1)
        if lo == hi:
            continue
        if lo not in hi_em or hi < hi_em[lo]:
            hi_em[lo] = hi
    limite = [n - 1] * n
    corrente = n - 1
    for i in range(n - 1, -1, -1):
        if i in hi_em:
            corrente = min(corrente, hi_em[i] - 1)
        limite[i] = min(corrente, n - 1)
    return limite


def two_opt_uma_rodada_v3(ordem_grupos, cidade_de, matrix, pares, time_limit, max_seg=None, raiz=None):
    """2-opt (reversao do segmento [i,j]) com limite[i] pre-computado.

    Para i>=1 (a raiz nao se move) usa-se o caminho rapido via `limite[i]`
    (ver `_computar_limites` e a docstring do modulo). Segmentos que incluem
    a RAIZ ([0,j]) sao permitidos e tratados como caso especial: como mover
    a raiz muda o ponto de corte do ciclo, o argumento de limite[] (baseado
    em ancora fixa) NAO vale aqui, e usa-se a checagem exaustiva
    `_ordem_e_factivel_ciclica` (que rotaciona internamente pra raiz, entao
    aceita `candidata` em qualquer rotacao) so' quando o custo ja indica
    melhora -- mesma logica do Or-opt v3 pra sua propria raiz.

    Excluir a raiz da vizinhanca nao e' neutro: numa comparacao
    deterministica contra um 2-opt sem essa restricao (mesmo ponto de
    partida, convergencia completa, sem aleatoriedade), a exclusao levava a
    otimos locais ESTRITAMENTE PIORES em 5/24 instancias (ESC07: 2137 vs
    1730; ESC25: 6540 vs 6220; ESC63: 101 vs 95; ft53.1: 9516 vs 9383;
    ft53.4: 16518 vs 16308), sem nunca violar Pi (por isso nao apareceria em
    contagem de violacoes)."""
    from heuristica_direta_pcgtsp import _ordem_e_factivel_ciclica
    t0 = time.time()
    if raiz is None:
        raiz = ordem_grupos[0]
    ordem = _rotacionar(ordem_grupos, raiz)
    n = len(ordem)
    if n < 5:
        return ordem, False
    if max_seg is None:
        max_seg = n

    melhorou_algo = False
    melhorou_nesta_passada = True
    while melhorou_nesta_passada and time.time() - t0 < time_limit:
        melhorou_nesta_passada = False
        limite = _computar_limites(ordem, pares)
        for i in range(0, n - 1):
            if time.time() - t0 > time_limit:
                break
            eh_raiz = (i == 0)
            if eh_raiz:
                j_teto = min(n - 2, i + max_seg - 1)  # exclui [0,n-1]: ciclo inteiro, degenerado
            else:
                j_teto = min(n - 1, limite[i], i + max_seg - 1)
            for j in range(i + 1, j_teto + 1):

                idx_antes = (i - 1) % n
                idx_depois = (j + 1) % n
                antes = ordem[idx_antes]
                depois = ordem[idx_depois]
                if i <= idx_antes <= j or i <= idx_depois <= j:
                    continue

                custo_antigo = (_custo_segmento(ordem, cidade_de, matrix, i, j)
                                + _dist(matrix, cidade_de[antes], cidade_de[ordem[i]])
                                + _dist(matrix, cidade_de[ordem[j]], cidade_de[depois]))
                if custo_antigo == INF:
                    continue

                segmento_invertido = ordem[i:j + 1][::-1]
                candidata = ordem[:i] + segmento_invertido + ordem[j + 1:]
                custo_novo = (_custo_segmento(candidata, cidade_de, matrix, i, j)
                              + _dist(matrix, cidade_de[antes], cidade_de[candidata[i]])
                              + _dist(matrix, cidade_de[candidata[j]], cidade_de[depois]))

                # i>=1: sem checagem extra -- j <= limite[i] ja' garante (por
                # construcao) que nenhum par tem os dois extremos em [i,j].
                # i=0 (raiz no segmento): limite[] nao vale (ancora se move),
                # exige a checagem exaustiva antes de aceitar.
                if custo_novo < custo_antigo - 1e-9:
                    if eh_raiz:
                        if not _ordem_e_factivel_ciclica(candidata, pares, raiz):
                            continue
                        idx_raiz = candidata.index(raiz)
                        candidata = candidata[idx_raiz:] + candidata[:idx_raiz]
                    ordem = candidata
                    melhorou_algo = True
                    melhorou_nesta_passada = True
                    break
            if melhorou_nesta_passada:
                break
    return ordem, melhorou_algo


def _custo_segmento(ordem, cidade_de, matrix, i, j):
    total = 0.0
    for k in range(i, j):
        d = _dist(matrix, cidade_de[ordem[k]], cidade_de[ordem[k + 1]])
        if d == INF:
            return INF
        total += d
    return total


# ===========================================================================
# 3) Double-bridge v2: checagem O(|pares|) direta (so' pares B<->C importam)
# ===========================================================================

def double_bridge_v2(ordem, pares, rng, raiz=None, tentativas=25):
    n = len(ordem)
    if n < 8:
        return None
    if raiz is None:
        raiz = ordem[0]
    ordem = _rotacionar(ordem, raiz)
    for _ in range(tentativas):
        cortes = sorted(rng.sample(range(1, n), 3))
        p1, p2, p3 = cortes
        if p1 == p2 or p2 == p3:
            continue
        # B = ordem[p1:p2], C = ordem[p2:p3] -- so' pares com um pe em B e
        # outro em C podem ser violados pela troca A+C+B+D (ver docstring).
        B = set(ordem[p1:p2])
        C = set(ordem[p2:p3])
        cruza = any((p in B and s in C) or (p in C and s in B) for p, s in pares)
        if cruza:
            continue
        A, Bl, Cl, D = ordem[:p1], ordem[p1:p2], ordem[p2:p3], ordem[p3:]
        return A + Cl + Bl + D
    return None
