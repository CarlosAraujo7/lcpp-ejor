"""
verificar_proveniencia_esicup.py
=================================
Compara os catalogos brutos ESICUP (formato XML com pecas/quantidades) com as
instancias ja processadas do projeto (instances/original, instances/generated)
que tem o mesmo nome, para saber se as formas poligonais realmente batem (mesma
familia de pecas, a menos de translacao/rotacao/reflexao/escala).

Os catalogos brutos NAO acompanham este repositorio: baixe-os de
https://github.com/ESICUP/datasets (pasta 2d_irregular) e coloque os XML em
instances/external/Set1_ESICUP/.

Assinatura de forma usada (invariante a translacao, rotacao, reflexao e escala
uniforme): (n_vertices, tupla ordenada de (aresta/perimetro), area/perimetro^2).
"""
import xml.etree.ElementTree as ET
from pathlib import Path
import sys
import numpy as np
from collections import defaultdict

PROJECT_ROOT = (Path(__file__).resolve().parent.parent)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from data.load_instances import load_instance  # noqa: E402

RAW_DIR = PROJECT_ROOT / "instances" / "external" / "Set1_ESICUP"
ORIG_PACK = PROJECT_ROOT / "instances" / "original" / "packing"
ORIG_SEP  = PROJECT_ROOT / "instances" / "original" / "separated"
GEN_DIR   = PROJECT_ROOT / "instances" / "generated"

TOL = 3  # casas decimais para comparacao de assinatura


def strip_ns(tag):
    return tag.split('}')[-1] if '}' in tag else tag


def parse_esicup_xml(path):
    """Retorna dict piece_polygon_id -> (quantity, [(x,y),...]), excluindo board/nfp."""
    tree = ET.parse(path)
    root = tree.getroot()
    polygons = {}
    for el in root.iter():
        if strip_ns(el.tag) == 'polygon':
            pid = el.get('id')
            pts = []
            for lines_el in el:
                if strip_ns(lines_el.tag) == 'lines':
                    for seg in lines_el:
                        if strip_ns(seg.tag) == 'segment':
                            pts.append((float(seg.get('x0')), float(seg.get('y0'))))
            polygons[pid] = pts

    board_ids = set()
    piece_qty = {}
    for el in root.iter():
        t = strip_ns(el.tag)
        if t == 'boards':
            for piece_el in el:
                for comp in piece_el:
                    if strip_ns(comp.tag) == 'component':
                        board_ids.add(comp.get('idPolygon'))
        if t == 'lot':
            for piece_el in el:
                if strip_ns(piece_el.tag) != 'piece':
                    continue
                qty = int(piece_el.get('quantity', '1'))
                for comp in piece_el:
                    if strip_ns(comp.tag) == 'component':
                        pid = comp.get('idPolygon')
                        piece_qty[pid] = piece_qty.get(pid, 0) + qty

    pieces = {}
    for pid, qty in piece_qty.items():
        if pid in polygons and pid not in board_ids and not pid.startswith('nfpPolygon'):
            pieces[pid] = (qty, polygons[pid])
    return pieces


def shape_signature(coords):
    pts = np.array(coords, dtype=float)
    if len(pts) < 3:
        return None
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    n = len(pts)
    edges = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    perim = edges.sum()
    if perim < 1e-9:
        return None
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    compactness = area / (perim ** 2)
    norm_edges = tuple(sorted(round(e / perim, TOL) for e in edges))
    return (n, norm_edges, round(compactness, TOL))


def raw_catalog_signatures(xml_path):
    pieces = parse_esicup_xml(xml_path)
    sigs = defaultdict(int)  # signature -> total quantity
    for pid, (qty, coords) in pieces.items():
        sig = shape_signature(coords)
        if sig is not None:
            sigs[sig] += qty
    return sigs, pieces


def our_file_signatures(filepath):
    name, polys = load_instance(filepath)
    sigs = defaultdict(int)
    for p in polys:
        coords = list(p.exterior.coords)
        sig = shape_signature(coords)
        if sig is not None:
            sigs[sig] += 1
    return sigs, len(polys)


def compare(dataset_label, raw_xml, our_files):
    print(f"\n{'='*70}\n{dataset_label}\n{'='*70}")
    if not raw_xml.exists():
        print(f"  [sem catalogo bruto em {raw_xml}]")
        return
    raw_sigs, pieces = raw_catalog_signatures(raw_xml)
    n_piece_types = len(pieces)
    total_qty = sum(q for q, _ in pieces.values())
    print(f"  Catalogo bruto: {raw_xml.relative_to(PROJECT_ROOT)}")
    print(f"    {n_piece_types} tipos de peca distintos, {total_qty} copias no total")
    print(f"    tipos de peca por (n_vertices): {sorted(k[0] for k in raw_sigs)}")

    for label, fp in our_files:
        if not fp.exists():
            print(f"  -- {label}: arquivo nao encontrado ({fp.name})")
            continue
        our_sigs, n_polys = our_file_signatures(fp)
        matched = 0
        matched_qty_ok = 0
        for sig, cnt in our_sigs.items():
            if sig in raw_sigs:
                matched += cnt
        pct = 100.0 * matched / n_polys if n_polys else 0.0
        unmatched_shapes = [s for s in our_sigs if s not in raw_sigs]
        print(f"  -- {label}: {n_polys} poligonos | {matched}/{n_polys} ({pct:.0f}%) "
              f"batem alguma forma do catalogo bruto | {len(unmatched_shapes)} formas sem par")


DATASETS = [
    ("ALBANO", RAW_DIR / "ALBANO" / "albano_2007-05-15" / "albano.xml", [
        ("original/packing/albano.txt", ORIG_PACK / "albano.txt"),
        ("original/separated/albano.txt", ORIG_SEP / "albano.txt"),
        ("generated/albano_pack.txt", GEN_DIR / "albano_pack.txt"),
        ("generated/albano_sep.txt", GEN_DIR / "albano_sep.txt"),
        ("generated/albano_mod_pack.txt", GEN_DIR / "albano_mod_pack.txt"),
        ("generated/albano_mod_sep.txt", GEN_DIR / "albano_mod_sep.txt"),
    ]),
    ("BLAZ (catalogo unico, 7 tipos de peca)", RAW_DIR / "BLAZ" / "blaz.xml", [
        ("original/packing/blaz1.txt", ORIG_PACK / "blaz1.txt"),
        ("original/separated/blaz1.txt", ORIG_SEP / "blaz1.txt"),
        ("original/packing/blaz2.txt", ORIG_PACK / "blaz2.txt"),
        ("original/separated/blaz2.txt", ORIG_SEP / "blaz2.txt"),
        ("original/packing/blaz3.txt", ORIG_PACK / "blaz3.txt"),
        ("original/separated/blaz3.txt", ORIG_SEP / "blaz3.txt"),
        ("generated/blaz_pack.txt", GEN_DIR / "blaz_pack.txt"),
        ("generated/blaz_sep.txt", GEN_DIR / "blaz_sep.txt"),
        ("generated/blaz_mod_sep.txt", GEN_DIR / "blaz_mod_sep.txt"),
    ]),
    ("DAGLI", RAW_DIR / "DAGLI" / "dagli.xml", [
        ("generated/dagli_pack.txt", GEN_DIR / "dagli_pack.txt"),
        ("generated/dagli_sep.txt", GEN_DIR / "dagli_sep.txt"),
        ("generated/dagli_mod_sep.txt", GEN_DIR / "dagli_mod_sep.txt"),
    ]),
    ("FU", RAW_DIR / "FU" / "fu.xml", [
        ("original/packing/fu.txt", ORIG_PACK / "fu.txt"),
        ("original/separated/fu.txt", ORIG_SEP / "fu.txt"),
        ("generated/fu_pack.txt", GEN_DIR / "fu_pack.txt"),
        ("generated/fu_sep.txt", GEN_DIR / "fu_sep.txt"),
        ("generated/fu_mod_sep.txt", GEN_DIR / "fu_mod_sep.txt"),
    ]),
    ("JAKOBS1", RAW_DIR / "JAKOBS" / "jakobs1.xml", [
        ("generated/jakobs1_pack.txt", GEN_DIR / "jakobs1_pack.txt"),
        ("generated/jakobs1_sep.txt", GEN_DIR / "jakobs1_sep.txt"),
        ("generated/jakobs1_mod_sep.txt", GEN_DIR / "jakobs1_mod_sep.txt"),
    ]),
    ("JAKOBS2", RAW_DIR / "JAKOBS" / "jakobs2.xml", [
        ("generated/jakobs2_pack.txt", GEN_DIR / "jakobs2_pack.txt"),
        ("generated/jakobs2_sep.txt", GEN_DIR / "jakobs2_sep.txt"),
    ]),
    ("MAO", RAW_DIR / "MAO" / "mao.xml", [
        ("generated/mao_pack.txt", GEN_DIR / "mao_pack.txt"),
        ("generated/mao_sep.txt", GEN_DIR / "mao_sep.txt"),
        ("generated/mao_mod_sep.txt", GEN_DIR / "mao_mod_sep.txt"),
    ]),
    ("MARQUES", RAW_DIR / "MARQUES" / "marques.xml", [
        ("generated/marques_pack.txt", GEN_DIR / "marques_pack.txt"),
        ("generated/marques_sep.txt", GEN_DIR / "marques_sep.txt"),
        ("generated/marques_mod_sep.txt", GEN_DIR / "marques_mod_sep.txt"),
    ]),
    ("SHAPES0", RAW_DIR / "SHAPES" / "shapes0.xml", [
        ("original/packing/shapes2.txt (mesma familia?)", ORIG_PACK / "shapes2.txt"),
        ("original/separated/shapes2.txt (mesma familia?)", ORIG_SEP / "shapes2.txt"),
        ("original/packing/shapes4.txt (mesma familia?)", ORIG_PACK / "shapes4.txt"),
        ("original/separated/shapes4.txt (mesma familia?)", ORIG_SEP / "shapes4.txt"),
        ("generated/shapes0_pack.txt", GEN_DIR / "shapes0_pack.txt"),
        ("generated/shapes0_sep.txt", GEN_DIR / "shapes0_sep.txt"),
    ]),
    ("SHAPES1", RAW_DIR / "SHAPES" / "shapes1.xml", [
        ("generated/shapes1_pack.txt", GEN_DIR / "shapes1_pack.txt"),
        ("generated/shapes1_sep.txt", GEN_DIR / "shapes1_sep.txt"),
        ("original/packing/shapes2.txt (mesma familia?)", ORIG_PACK / "shapes2.txt"),
        ("original/packing/shapes4.txt (mesma familia?)", ORIG_PACK / "shapes4.txt"),
    ]),
    ("SHIRTS", RAW_DIR / "SHIRTS" / "shirts.xml", [
        ("generated/shirts_pack.txt", GEN_DIR / "shirts_pack.txt"),
        ("generated/shirts_sep.txt", GEN_DIR / "shirts_sep.txt"),
        ("generated/shirts_mod_sep.txt", GEN_DIR / "shirts_mod_sep.txt"),
    ]),
    ("SWIM", RAW_DIR / "SWIM" / "swim.xml", [
        ("generated/swim_pack.txt", GEN_DIR / "swim_pack.txt"),
        ("generated/swim_sep.txt", GEN_DIR / "swim_sep.txt"),
        ("generated/swim_mod_sep.txt", GEN_DIR / "swim_mod_sep.txt"),
    ]),
    ("TROUSERS", RAW_DIR / "TROUSERS" / "trousers.xml", [
        ("original/packing/trousers.txt", ORIG_PACK / "trousers.txt"),
        ("original/separated/trousers.txt", ORIG_SEP / "trousers.txt"),
        ("generated/trousers_pack.txt", GEN_DIR / "trousers_pack.txt"),
        ("generated/trousers_sep.txt", GEN_DIR / "trousers_sep.txt"),
        ("generated/trousers_mod_sep.txt", GEN_DIR / "trousers_mod_sep.txt"),
    ]),
]

if __name__ == "__main__":
    for label, xml_path, files in DATASETS:
        compare(label, xml_path, files)
