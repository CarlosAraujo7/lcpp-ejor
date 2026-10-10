import matplotlib.pyplot as plt
import matplotlib.cm as cm
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import nearest_points
import networkx as nx


def distancia_e_aresta_ate_raiz(ponto_origem: Point, poligono_raiz: Polygon):
    """
    Dado um ponto de origem genérico e um polígono raiz (inicial),
    calcula e retorna o ponto na borda do polígono mais próximo à origem.
    Retorna também uma linha reta (LineString) conectando esses dois pontos
    (caso eles não estejam perfeitamente sobrepostos).
    """
    # nearest_points retorna uma tupla (ponto_na_origem, ponto_no_poligono)
    ponto_proximo = nearest_points(ponto_origem, poligono_raiz)[1]

    # Se a origem não for igual ao ponto próximo, desenha a linha de transição/deslocamento
    return [
        LineString([ponto_origem, ponto_proximo])
        ] if ponto_origem != ponto_proximo else [], ponto_proximo


def chebyshev_distance(p1: Point, p2: Point) -> float:
    """
    Calcula a distância de Chebyshev entre dois pontos.
    (Onde a distância é o maior valor absoluto das diferenças ao longo de qualquer eixo coordenado).
    """
    return max(abs(p1.x - p2.x), abs(p1.y - p2.y))


def construir_grafo(poligonos):
    """
    A partir de uma lista de Polígonos (Shapely), constrói um Grafo não direcionado
    onde cada polígono é um nó e as colisões/interseções entre eles formam as arestas.
    """
    G = nx.Graph()
    # Adiciona todos os polígonos como nós independentes
    for i, pol in enumerate(poligonos):
        G.add_node(i, poligono=pol)

    # Itera comparando todos contra todos (evitando repetição i e j)
    for i, pol1 in enumerate(poligonos):
        for j in range(i + 1, len(poligonos)):
            pol2 = poligonos[j]
            # Se os limites do polígono 1 interceptarem com os do polígono 2, e não for vazio
            if pol1.intersects(pol2) and not pol1.intersection(pol2).is_empty:
                # Conecta os dois no grafo
                G.add_edge(i, j)

    return G


def construir_arvore_direcionada(G, raiz):
    """
    Recebe o Grafo completo construído e aterra ele escolhendo um nó raiz,
    convertendo o grafo não-direcionado em uma Árvore Direcionada (DiGraph)
    usando a estratégia de Busca em Largura (BFS - Breadth-First Search).
    """
    T = nx.DiGraph()
    T.add_node(raiz)

    # bfs_edges mapeia o grafo espalhando a partir da raiz nível por nível
    for u, v in nx.bfs_edges(G, raiz):
        # Captura as características/atributos estendidos que possam estar na aresta
        weight = G.edges[u, v]['weight']
        connection = G.edges[u, v]['connection']

        # Garante que a conexão seja formalmente um LineString
        if type(connection) != LineString:
            connection = LineString([Point(p) for p in G.edges[u, v]['connection']])

        T.add_edge(u, v, weight=weight, connection=connection)

    # Após a construção da árvore, contabiliza a hierarquia (quantos filhos cada nó engloba)
    for node in T.nodes:
        # Puxa o número imediato de sucessores diretos desse nó
        num_filhos = len(list(T.successors(node)))
        T.nodes[node]['num_filhos'] = num_filhos

        # Repassa o atributo geométrico principal pro nó dessa árvore
        T.nodes[node]['poligono'] = G.nodes[node]['poligono']
    return T


def imprimir_arvore(T, G, no, nivel=0):
    """
    Função auxiliar de debug para imprimir a árvore recursivamente no terminal.
    A endentação reflete a profundidade do nó (nível da árvore).
    """
    indent = "  " * nivel
    coords = list(G.nodes[no]['poligono'].exterior.coords)
    print(f"{indent}Polígono {no}: {coords}")

    for filho in T.successors(no):
        imprimir_arvore(T, G, filho, nivel + 1)


def layout_hierarquico(G, root, nivel_y_gap=1.5, nivel_x_gap=1.5):
    """
    Gera o dicionário de posições 'x' e 'y' estáticas para que a árvore (DiGraph)
    possa ser desenhada na tela em um formato verticalizado bem distribuído.
    """
    pos = {}
    nivel_dict = {}

    # Busca em Profundidade (DFS) que agrupa os nós por nível de profundidade
    def dfs(node, nivel):
        if nivel not in nivel_dict:
            nivel_dict[nivel] = []
        nivel_dict[nivel].append(node)
        for filho in G.successors(node):
            dfs(filho, nivel + 1)

    dfs(root, 0)

    # Itera por cada nível posicionando os nós horizontalmente dependendo
    # do tamanho (largura) do nível atual e do espaçamento (nivel_x_gap).
    for nivel in nivel_dict:
        largura = len(nivel_dict[nivel])
        for i, node in enumerate(nivel_dict[nivel]):
            pos[node] = (
                i * nivel_x_gap - (largura - 1) * nivel_x_gap / 2,  # Coordenada x centrada
                -nivel * nivel_y_gap  # Coordenada y negativa pra desenhar do topo para baixo
            )

    return pos


def linhas_iguais(l1: LineString, l2: LineString) -> bool:
    """Verifica se duas linhas (Shapeley LineString) são iguais (ida ou volta)."""
    return (list(l1.coords) == list(l2.coords)) or (list(l1.coords) == list(l2.coords)[::-1])


def linha_esta_na_lista(linha, lista_linhas):
    """Verifica se a linha requisitada existe numa matriz/lista de linhas e IDs."""
    for l in [i[0] for i in lista_linhas]:
        if linhas_iguais(linha, l):
            return True
    return False


def plotar_poligonos_e_arvore(G, caminho=None, raiz=None, hierarquia=True, indices=True):
    """
    Usa o Matplotlib para montar um quadro a quadro contendo a topologia 2D dos
    polígonos gerados e, opcionalmente, desenhar a árvore lógica de tomada de decisão.
    """
    # Em um grafo normal isso varreria componentes conexas separadas, aqui é mantido simples
    componentes = [G]
    cores = cm.get_cmap('tab10', len(componentes))

    for idx, componente in enumerate(componentes):
        # Puxa o pedaço atual
        subgrafo = G.subgraph(componente)
        # Transforma o pedaço em árvore referenciando a raiz original
        arvore = construir_arvore_direcionada(subgrafo, raiz)

        # Prepara a tela de plotagem (1 janela para os gráficos, a outra para a árvore se requisitado)
        fig, axs = plt.subplots(1, 2 if hierarquia else 1, figsize=(10, 6))
        ax1, ax2 = axs if '__len__' in dir(axs) else [axs, None]

        # Desenha fisicamente as formas dos polígonos
        for i in subgrafo.nodes:
            pol = G.nodes[i]['poligono']
            x, y = pol.exterior.xy
            # Preenche o polígono com uma cor
            ax1.fill(x, y, alpha=0.5, edgecolor='black', facecolor=cores(idx))
            # Plota o ID numérico bem no centro de gravidade dele
            centroide = pol.centroid
            ax1.text(centroide.x, centroide.y, str(i), fontsize=12, ha='center', va='center', weight='bold')

        # Se houver um caminho (rota) gerado, desenha a linha física
        if caminho is not None:
            # Fatia inicial e final pra não desenhar movimentos 'vazios' (quando c[1] == 'd') se existirem nas pontas
            caminho = caminho[0 if caminho[0][1] == 'c' else 1:len(caminho) if caminho[-1][1] == 'c' else len(caminho)-1]

            for i, c in enumerate(caminho):
                line = c[0]
                if isinstance(line, LineString):
                    x, y = line.xy
                    if not hierarquia:
                        # Cor 'vermelha' para corte, 'azul' para deslocamentos
                        ax1.plot(x, y, 'r-' if c[1] == 'c' else 'b-', linewidth=3 if c[1] == 'c' else 3)
                        if indices:
                            # Adiciona os passos indexados acima da linha da rota
                            centro = line.interpolate(0.5, normalized=True)
                            ax1.text(centro.x, centro.y, str(i+1), fontsize=12, ha='center', va='center', color='black')

        ax1.set_aspect('equal')
        ax1.grid(True)

        # Constroi as regras visuais pra arvore caso a plotagem dela esteja ativa
        pos = layout_hierarquico(arvore, raiz)
        edge_labels = {
            (pai, filho): ''
            for pai, filho in arvore.edges
        }

        # Plota os círculos lógicos de rede, setas e pesos
        if hierarquia:
            nx.draw(arvore, pos, ax=ax2, with_labels=True, node_color=cores(idx), edge_color='gray',
                    node_size=500, font_size=12, font_weight='bold', arrows=True)
            nx.draw_networkx_edge_labels(arvore, pos, edge_labels=edge_labels, ax=ax2, font_color='blue')

        plt.tight_layout()


def reordenar_poligono(pol: Polygon, ponto_referencia: Point) -> Polygon:
    """
    Quando o ponteiro precisa iniciar o corte do polígono, isso ajusta as
    coordenadas baseadas no vértice do polígono mais próximo ao ponto em que estamos,
    evitando que o carrinho corte de trás pra frente ou ande sem necessidade.
    """
    coords = list(pol.exterior.coords[:-1])  # remove ponto duplicado do final

    # Encontra o índice da coordenada que é mais próxima matematicamente da nossa referência
    idx_inicio = min(range(len(coords)), key=lambda i: chebyshev_distance(ponto_referencia, Point(coords[i])))

    # Fatiamento e realocamento: Coloca o novo ínicio pra frente, preserva o restante, e fecha.
    coords_reordenadas = coords[idx_inicio:] + coords[:idx_inicio] + [coords[idx_inicio]]
    return coords_reordenadas


def caminha_arvore(G, raiz, ponto_inicial, pol_cobertos=[]):
    """
    O Coração Funcional do Roteamento: Uma busca recursiva que percorre a árvore topológica
    para criar o Roteiro Final de corte (Caminho).
    """
    caminho = []

    # Estratégia de corte: Filhos/sucessores que englobam mais sub-filhos tem prioridade,
    # garantindo que ramificações densas do mapa se resolvam primeiro
    successors = sorted(G.successors(raiz), key=lambda n: G.nodes[n]['num_filhos'], reverse=True)

    if len(successors) > 0:
        caminho_pol = []
        pol = G.nodes[raiz]['poligono']

        # Puxa os vértices da forma que vamos cortar e realinha o ínicio
        coords = reordenar_poligono(pol, ponto_inicial)

        # Transforma o contorno do polígono atual em pequenos segmentos separadores
        for i in range(len(coords)-1):
            a = Point(coords[i])
            b = Point(coords[i + 1])
            caminho_pol.append(LineString([a, b]))

        for i in caminho_pol:
            # Define o modo 'x' que deve significar algum tipo de corte ou caminho default na aresta atual
            caminho.append((i, 'x'))

            # Testa todos os nós-filhos que faltam cobrir do nosso local atual
            for s in filter(lambda x: x not in pol_cobertos, successors):
                # Se não tem distância (weight 0), o filho toca o pai
                if G.edges[raiz, s]['weight'] == 0:
                    pol_destino = G.nodes[s]['poligono']
                    intersecao = pol.intersection(pol_destino)

                    # Verifica se durante nossa viagem na aresta, nós cruzamos o vizinho
                    if intersecao.distance(Point(i.coords[1])) < 1e-8 or intersecao.covers(Point(i.coords[0])):
                        descendentes = nx.descendants(G, s)
                        descendentes.add(s)
                        if len(descendentes) > 0:
                            pol_cobertos.append(s)
                            # Chama recursão, mergulhando no sub-grafo e gerando caminho interno
                            c, _ = caminha_arvore(G.subgraph(descendentes), s, Point(i.coords[1]), pol_cobertos)
                            caminho.extend(c)
                else:
                    # Se há peso, é uma conexão em vazio (pulo) entre pais e filhos distantes
                    intersecao = G.edges[raiz, s]['connection']
                    if intersecao.distance(Point(i.coords[1])) < 1e-8 or intersecao.covers(Point(i.coords[0])):
                        # Registra que a viagem 'd' (Deslocamento) deve ser feita
                        caminho.append((intersecao, 'd'))

                        descendentes = nx.descendants(G, s)
                        descendentes.add(s)
                        if len(descendentes) > 0:
                            pol_cobertos.append(s)
                            c, _ = caminha_arvore(G.subgraph(descendentes), s, Point(i.coords[1]), pol_cobertos)
                            caminho.extend(c)

    else:
        # Se for uma 'Folha' (último nó sem filhos), simplesmente roda e corta ela toda.
        pol = G.nodes[raiz]['poligono']
        coords = reordenar_poligono(pol, ponto_inicial)
        for i in range(len(coords)-1):
            a = Point(coords[i])
            b = Point(coords[i + 1])
            caminho.append((LineString([a, b]), 'x'))

    return caminho, pol_cobertos


def cortar_arvore(arvore, raiz):
    """
    Função inicializadora/Gatilho. Inicia do Ponto (0,0) (Origem absoluta da máquina),
    descobre como chegar no polígono da raiz da Árvore gerada,
    e invoca o caminhamento recursivo 'caminha_arvore'.
    """
    caminho = []

    # Checa como viajar de x:0 y:0 até encostar no Polígono número "0" ou "Raiz"
    aresta, ponto = distancia_e_aresta_ate_raiz(Point(0, 0), arvore.nodes[raiz]['poligono'])

    # Se existe uma aresta criada (origem tava longe da borda), adicionamos o trecho
    # de deslocamento 'd' (deslocamento vázio/viagem)
    if len(aresta) > 0:
        caminho.append((aresta[0], 'd'))

    # Realiza a extração principal de todo o circuito da imagem/peça
    c, pol_cobertos = caminha_arvore(arvore, raiz, ponto, [raiz])
    caminho += c

    # Ao final, traça o caminho reverso para voltar o braço da máquina
    # de volta para a origem inicial (x0, y0).
    if len(aresta) > 0:
        caminho.append((LineString(list(aresta[0].coords)[::-1]), 'd'))

    return caminho
