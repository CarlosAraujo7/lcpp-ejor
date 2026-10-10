from shapely.geometry import Polygon, LineString, Point


def segment_intersections(polygon1, polygon2):
    """
    Analisa dois polígonos e divide suas arestas (linhas) exatamente nos pontos
    onde eles se cruzam/intersectam. Isso é útil em geometria computacional para
    evitar auto-interseções ou quebrar arestas longas antes de realizar fusões (unions) ou cortes.

    Retorna duas novas instâncias de Polígonos com as arestas subdivididas.
    """

    # Extrai o contorno externo do polígono 1 e cria uma lista de Linhas (Segmentos)
    # conectando cada vértice ao próximo
    edges1 = [LineString([polygon1.exterior.coords[i], polygon1.exterior.coords[i+1]])
              for i in range(len(polygon1.exterior.coords)-1)]

    # Extrai o contorno externo do polígono 2 da mesma maneira
    edges2 = [LineString([polygon2.exterior.coords[i], polygon2.exterior.coords[i+1]])
              for i in range(len(polygon2.exterior.coords)-1)]

    # Listas vazias para guardar as novas arestas subdivididas
    new_edges1 = []
    new_edges2 = []

    # ==========================================
    # PROCESSANDO AS ARESTAS DO POLÍGONO 1
    # ==========================================
    for edge1 in edges1:
        intersections = []

        # Testa a colisão dessa aresta do polígono 1 contra todas as arestas do polígono 2
        for edge2 in edges2:
            if edge1.intersects(edge2):
                # Extrai o objeto geométrico que representa a colisão
                inter = edge1.intersection(edge2)

                # Se for um ponto único de colisão, adiciona à lista
                if inter.geom_type == "Point":
                    intersections.append(inter)
                # Se forem múltiplos pontos, extrai e adiciona cada ponto separadamente
                elif inter.geom_type == "MultiPoint":
                    intersections.extend(inter.geoms)

        # Se houve alguma colisão na aresta 1
        if intersections:
            # Ordena os pontos de interseção de acordo com a distância em relação ao início da linha edge1
            intersections = sorted(intersections, key=edge1.project)

            # Monta uma lista de pontos com: Ponto Inicial + (Pontos de Interseção) + Ponto Final
            points = [Point(edge1.coords[0])] + intersections + [Point(edge1.coords[1])]

            # Cria múltiplos segmentos menores conectando cada ponto ao próximo na lista
            # ignorando pontos que por ventura caírem exatamente no mesmo lugar (points[i] != points[i+1])
            new_edges1.extend([LineString([points[i], points[i+1]]) for i in range(len(points)-1) if points[i] != points[i+1]])
        else:
            # Se não houve nenhuma colisão, a aresta permanece intacta
            new_edges1.append(edge1)


    # ==========================================
    # PROCESSANDO AS ARESTAS DO POLÍGONO 2
    # ==========================================
    # A lógica aplicada aqui é exatamente o espelho do que foi feito acima
    for edge2 in edges2:
        intersections = []
        for edge1 in edges1:
            if edge2.intersects(edge1):
                inter = edge2.intersection(edge1)
                if inter.geom_type == "Point":
                    intersections.append(inter)
                elif inter.geom_type == "MultiPoint":
                    intersections.extend(inter.geoms)

        if intersections:
            # Ordena com relação à distância na aresta 2
            intersections = sorted(intersections, key=edge2.project)
            points = [Point(edge2.coords[0])] + intersections + [Point(edge2.coords[1])]
            new_edges2.extend([LineString([points[i], points[i+1]]) for i in range(len(points)-1) if points[i] != points[i+1]])
        else:
            new_edges2.append(edge2)


    # ==========================================
    # REMONTANDO OS POLÍGONOS
    # ==========================================
    # Pega apenas o primeiro vértice de cada novo segmento gerado, para não duplicar vértices
    # e ao final garante o fechamento adicionando o primeiro vértice novamente no fim.
    return (
        Polygon([p.coords[0] for p in new_edges1] + [new_edges1[0].coords[0]]) if new_edges1 else polygon1,
        Polygon([p.coords[0] for p in new_edges2] + [new_edges2[0].coords[0]]) if new_edges2 else polygon2
    )
