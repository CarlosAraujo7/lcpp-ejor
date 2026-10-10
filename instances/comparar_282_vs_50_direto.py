"""
comparar_282_vs_50_direto.py
=============================
As 282 sinteticas proprias sao baseadas nas 50 de literatura (conjunto III)?
verificar_proveniencia_esicup.py compara original/ (50) e generated/ (282)
contra o catalogo BRUTO do ESICUP, separadamente. Este script compara os dois
lados um contra o outro DIRETAMENTE: para cada familia de nome
compartilhada entre os dois conjuntos (albano, blaz, dighe1, dighe2, fu,
shapes, trousers), compara a assinatura de forma (mesma definicao de
verificar_proveniencia_esicup.py: invariante a translacao/rotacao/
reflexao/escala) peca a peca, generated/ contra original/, para medir a
sobreposicao GEOMETRICA real, nao so a coincidencia de nome.
"""
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = (Path(__file__).resolve().parent.parent)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from data.load_instances import load_instance  # noqa: E402

ORIG_PACK = PROJECT_ROOT / "instances" / "original" / "packing"
ORIG_SEP = PROJECT_ROOT / "instances" / "original" / "separated"
GEN_DIR = PROJECT_ROOT / "instances" / "generated"

TOL = 3


def shape_signature(coords):
    import numpy as np
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


def file_signatures(filepath):
    name, polys = load_instance(filepath)
    sigs = defaultdict(int)
    for p in polys:
        coords = list(p.exterior.coords)
        sig = shape_signature(coords)
        if sig is not None:
            sigs[sig] += 1
    return sigs, len(polys)


def compara_par(label_a, fp_a, label_b, fp_b):
    if not fp_a.exists() or not fp_b.exists():
        print(f"  [arquivo faltando: {fp_a if not fp_a.exists() else fp_b}]")
        return
    sig_a, n_a = file_signatures(fp_a)
    sig_b, n_b = file_signatures(fp_b)
    matched_a = sum(c for s, c in sig_a.items() if s in sig_b)
    matched_b = sum(c for s, c in sig_b.items() if s in sig_a)
    pct_a = 100.0 * matched_a / n_a if n_a else 0.0
    pct_b = 100.0 * matched_b / n_b if n_b else 0.0
    identico = (n_a == n_b) and (sig_a == sig_b)
    print(f"  {label_a} ({n_a} pecas) vs {label_b} ({n_b} pecas): "
          f"{matched_a}/{n_a} ({pct_a:.0f}%) das pecas de A tem par em B | "
          f"{matched_b}/{n_b} ({pct_b:.0f}%) das pecas de B tem par em A"
          f"{'  [MULTICONJUNTO DE FORMAS IDENTICO]' if identico else ''}")


PARES = [
    ("albano", [
        (ORIG_SEP / "albano.txt", GEN_DIR / "albano_sep.txt"),
        (ORIG_SEP / "albano.txt", GEN_DIR / "albano_mod_sep.txt"),
        (ORIG_PACK / "albano.txt", GEN_DIR / "albano_pack.txt"),
    ]),
    ("blaz", [
        (ORIG_SEP / "blaz1.txt", GEN_DIR / "blaz_sep.txt"),
        (ORIG_SEP / "blaz2.txt", GEN_DIR / "blaz_sep.txt"),
        (ORIG_SEP / "blaz3.txt", GEN_DIR / "blaz_sep.txt"),
        (ORIG_SEP / "blaz1.txt", GEN_DIR / "blaz_mod_sep.txt"),
    ]),
    ("dighe1", [
        (ORIG_SEP / "dighe1.txt", GEN_DIR / "dighe1_sep.txt"),
        (ORIG_SEP / "dighe1.txt", GEN_DIR / "dighe1_mod_sep.txt"),
        (ORIG_PACK / "dighe1.txt", GEN_DIR / "dighe1_pack.txt"),
    ]),
    ("dighe2", [
        (ORIG_SEP / "dighe2.txt", GEN_DIR / "dighe2_sep.txt"),
        (ORIG_SEP / "dighe2.txt", GEN_DIR / "dighe2_mod_sep.txt"),
        (ORIG_PACK / "dighe2.txt", GEN_DIR / "dighe2_pack.txt"),
    ]),
    ("fu", [
        (ORIG_SEP / "fu.txt", GEN_DIR / "fu_sep.txt"),
        (ORIG_SEP / "fu.txt", GEN_DIR / "fu_mod_sep.txt"),
        (ORIG_PACK / "fu.txt", GEN_DIR / "fu_pack.txt"),
    ]),
    ("shapes (2/4 vs 0/1 -- familias diferentes?)", [
        (ORIG_SEP / "shapes2.txt", GEN_DIR / "shapes0_sep.txt"),
        (ORIG_SEP / "shapes2.txt", GEN_DIR / "shapes1_sep.txt"),
        (ORIG_SEP / "shapes4.txt", GEN_DIR / "shapes0_sep.txt"),
        (ORIG_SEP / "shapes4.txt", GEN_DIR / "shapes1_sep.txt"),
    ]),
    ("trousers", [
        (ORIG_SEP / "trousers.txt", GEN_DIR / "trousers_sep.txt"),
        (ORIG_SEP / "trousers.txt", GEN_DIR / "trousers_mod_sep.txt"),
        (ORIG_PACK / "trousers.txt", GEN_DIR / "trousers_pack.txt"),
    ]),
]

if __name__ == "__main__":
    for familia, pares in PARES:
        print(f"\n{'='*78}\n{familia}\n{'='*78}")
        for fp_a, fp_b in pares:
            compara_par(f"original/{fp_a.parent.name}/{fp_a.name}", fp_a,
                        f"generated/{fp_b.name}", fp_b)
