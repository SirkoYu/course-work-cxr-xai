"""
Виявлення точних і майже-дублікатів зображень.

* Точні дублікати: однаковий SHA-256 вмісту файлу (побайтові копії).
* Майже-дублікати: перцептивний хеш pHash (64 біти) з відстанню Геммінга
  <= порогу. pHash стійкий до зміни розміру, повторного стиснення JPEG
  та незначної зміни яскравості, тому знаходить копії одного знімка,
  що потрапили в об'єднаний датасет з різних джерел.
* Віддзеркалені копії: порівняння pHash зображення з pHash горизонтально
  віддзеркаленого іншого зображення (аугментовані копії в об'єднаних датасетах).

Зображення, пов'язані відношенням «дублікат» (транзитивно), об'єднуються
в групи за допомогою структури «система неперетинних множин» (union-find).
Модуль не залежить від TensorFlow.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageOps


# ---------------------------------------------------------------------------
# Хешування
# ---------------------------------------------------------------------------
def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def to_grayscale_8bit(im: Image.Image) -> Image.Image:
    """Коректно переводить зображення (зокрема 16-бітне) у 8-бітні градації сірого."""
    if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        arr = np.asarray(im, dtype=np.float64)
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(arr)
        return Image.fromarray(arr.astype(np.uint8), mode="L")
    return im.convert("L")


def phash_int(path: Path) -> int:
    """64-бітний pHash зображення як ціле число."""
    with Image.open(path) as im:
        gray = to_grayscale_8bit(im)
        return int(str(imagehash.phash(gray, hash_size=8)), 16)


def phash_pair(path: Path) -> tuple[int, int]:
    """pHash зображення та його горизонтально віддзеркаленої копії."""
    with Image.open(path) as im:
        gray = to_grayscale_8bit(im)
        h = int(str(imagehash.phash(gray, hash_size=8)), 16)
        hm = int(str(imagehash.phash(ImageOps.mirror(gray), hash_size=8)), 16)
        return h, hm


# ---------------------------------------------------------------------------
# Відстань Геммінга
# ---------------------------------------------------------------------------
_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount64(x: np.ndarray) -> np.ndarray:
    """Кількість одиничних бітів у кожному елементі масиву uint64."""
    if hasattr(np, "bitwise_count"):           # numpy >= 2.0
        return np.bitwise_count(x).astype(np.uint8)
    b = x.view(np.uint8).reshape(*x.shape, 8)
    return _POPCOUNT_LUT[b].sum(axis=-1, dtype=np.uint8)


def hamming_neighbors(hashes: np.ndarray, threshold: int, chunk: int = 256):
    """
    Знаходить усі пари (i, j), i < j, з відстанню Геммінга <= threshold,
    а також відстань до найближчого сусіда для кожного елемента.

    Повний попарний перебір виконується блоками, щоб обмежити пам'ять
    (для 30 тис. зображень - кілька секунд).
    Повертає (pairs: list[(i, j, d)], nn_dist: np.ndarray).
    """
    hashes = np.asarray(hashes, dtype=np.uint64)
    n = len(hashes)
    nn_dist = np.full(n, 64, dtype=np.uint8)
    pairs = []
    for start in range(0, n, chunk):
        block = hashes[start:start + chunk]
        d = popcount64(np.bitwise_xor(block[:, None], hashes[None, :]))  # (c, n)
        rows = np.arange(len(block))
        d[rows, rows + start] = 64                         # відстань до самого себе ігноруємо
        nn_dist[start:start + len(block)] = d.min(axis=1)
        ii, jj = np.nonzero(d <= threshold)
        gi = ii + start
        keep = jj > gi                                     # кожну пару лише один раз
        pairs.extend(zip(gi[keep].tolist(), jj[keep].tolist(), d[ii[keep], jj[keep]].tolist(), strict=True))
    return pairs, nn_dist


def mirrored_pairs(hashes: np.ndarray, mirrored: np.ndarray, threshold: int, chunk: int = 256):
    """
    Пари (i, j), i < j, де зображення i майже збігається з віддзеркаленим зображенням j
    (відстань між pHash(i) і pHash(mirror(j)) <= threshold).
    """
    hashes = np.asarray(hashes, dtype=np.uint64)
    mirrored = np.asarray(mirrored, dtype=np.uint64)
    found: dict[tuple[int, int], int] = {}
    for start in range(0, len(hashes), chunk):
        block = hashes[start:start + chunk]
        d = popcount64(np.bitwise_xor(block[:, None], mirrored[None, :]))
        ii, jj = np.nonzero(d <= threshold)
        for i, j, dist in zip((ii + start).tolist(), jj.tolist(), d[ii, jj].tolist(), strict=True):
            if i != j:
                key = (min(i, j), max(i, j))
                found[key] = min(found.get(key, 64), dist)
    return [(i, j, d) for (i, j), d in sorted(found.items())]


# ---------------------------------------------------------------------------
# Union-find
# ---------------------------------------------------------------------------
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]   # стиснення шляху
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def group_duplicates(sha: list[str], phashes: list[int], threshold: int,
                     phashes_mirrored: list[int] | None = None):
    """
    Об'єднує зображення в групи дублікатів (точні копії, майже-копії та, якщо
    передано phashes_mirrored, віддзеркалені копії).
    Повертає (group_id, near_pairs, nn_dist, mirror_pairs); group_id - номери груп 0..G-1.
    """
    n = len(sha)
    uf = UnionFind(n)
    first_by_sha: dict[str, int] = {}
    for i, h in enumerate(sha):
        if h in first_by_sha:
            uf.union(first_by_sha[h], i)
        else:
            first_by_sha[h] = i
    pairs, nn_dist = hamming_neighbors(np.array(phashes, dtype=np.uint64), threshold)
    for i, j, _ in pairs:
        uf.union(i, j)
    mirror = []
    if phashes_mirrored is not None:
        mirror = mirrored_pairs(np.array(phashes, dtype=np.uint64),
                                np.array(phashes_mirrored, dtype=np.uint64), threshold)
        for i, j, _ in mirror:
            uf.union(i, j)
    roots = [uf.find(i) for i in range(n)]
    _, group_id = np.unique(roots, return_inverse=True)
    return group_id, pairs, nn_dist, mirror


def decide_actions(df: pd.DataFrame, policy: str = "keep_one",
                   drop_conflicts: bool = True) -> pd.DataFrame:
    """
    Для кожного зображення визначає дію: keep / drop_duplicate / drop_conflict.

    Очікувані стовпці df: path, label, width, height, group_id.
    * Група з різними мітками (конфлікт) вилучається повністю (drop_conflicts=True),
      бо неможливо встановити, яка мітка правильна.
    * policy="keep_one": залишається зображення з найбільшою роздільністю
      (за рівності - перше за шляхом), решта - drop_duplicate.
    * policy="keep_all": залишаються всі; цілісність групи забезпечує поділ.
    """
    if policy not in ("keep_one", "keep_all"):
        raise ValueError(f"Невідома політика дедуплікації: {policy}")
    df = df.copy()
    df["area"] = df["width"].astype(float) * df["height"].astype(float)
    g = df.groupby("group_id")
    df["group_size"] = g["path"].transform("size")
    df["group_n_labels"] = g["label"].transform("nunique")
    df["group_labels"] = g["label"].transform(lambda s: "|".join(sorted(set(s))))
    df["action"] = "keep"

    conflict = df["group_n_labels"] > 1
    if drop_conflicts:
        df.loc[conflict, "action"] = "drop_conflict"

    if policy == "keep_one":
        candidates = df[(df["group_size"] > 1) & (df["action"] == "keep")]
        order = candidates.sort_values(["group_id", "area", "path"], ascending=[True, False, True])
        representatives = order.drop_duplicates("group_id", keep="first").index
        dup_idx = candidates.index.difference(representatives)
        df.loc[dup_idx, "action"] = "drop_duplicate"
    return df.drop(columns=["area"])
