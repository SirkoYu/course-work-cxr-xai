"""Пошук дублікатів, у тому числі віддзеркалених копій."""
import numpy as np
import pandas as pd
from PIL import Image

from cxr_xai.data import deduplication


def _synthetic_xray(seed: int, size: int = 256) -> Image.Image:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / size
    img = 120 + 60 * np.sin(6 * xx + rng.uniform(0, 6)) * np.cos(4 * yy + rng.uniform(0, 6))
    for _ in range(6):
        cx, cy, r = rng.uniform(0.2, 0.8, 2).tolist() + [rng.uniform(0.05, 0.2)]
        img += 70 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2))
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode="L")


def test_phash_finds_resized_and_recompressed_copies(tmp_path):
    a = _synthetic_xray(1)
    a.save(tmp_path / "a.png")
    a.resize((180, 180)).save(tmp_path / "a_small.png")
    a.save(tmp_path / "a.jpg", quality=70)
    _synthetic_xray(2).save(tmp_path / "b.png")
    h = {p: deduplication.phash_int(tmp_path / p) for p in ["a.png", "a_small.png", "a.jpg", "b.png"]}
    dist = lambda x, y: bin(h[x] ^ h[y]).count("1")  # noqa: E731
    assert dist("a.png", "a_small.png") <= 4
    assert dist("a.png", "a.jpg") <= 4
    assert dist("a.png", "b.png") > 10


def test_hamming_neighbors_matches_bruteforce():
    rng = np.random.default_rng(0)
    hashes = rng.integers(0, 2 ** 63, size=300, dtype=np.int64).astype(np.uint64)
    hashes[10] = hashes[5] ^ np.uint64(0b111)      # відстань 3
    pairs, nn = deduplication.hamming_neighbors(hashes, threshold=3, chunk=64)
    assert (5, 10, 3) in pairs
    assert nn[5] == 3 and nn[10] == 3


def test_grouping_and_actions():
    df = pd.DataFrame({
        "path": ["a1", "a2", "b1", "c1", "c2", "d1"],
        "label": ["Normal", "Normal", "COVID-19", "Viral Pneumonia", "Normal", "Tuberculosis"],
        "width": [512, 1024, 512, 512, 512, 512],
        "height": [512, 1024, 512, 512, 512, 512],
    })
    sha = ["s1", "s2", "s3", "s4", "s4", "s6"]         # c1 і c2 - побайтові копії з різними мітками
    ph = [0b0, 0b1, 0xFFFF0000, 0x0F0F0F0F, 0x0F0F0F0F, 0x12345678ABC]  # a1~a2 (d=1)
    gid, pairs, _, _ = deduplication.group_duplicates(sha, ph, threshold=2)
    df["group_id"] = gid
    out = deduplication.decide_actions(df, "keep_one", drop_conflicts=True).set_index("path")
    assert out.loc["a2", "action"] == "keep"              # більша роздільність
    assert out.loc["a1", "action"] == "drop_duplicate"
    assert out.loc["c1", "action"] == "drop_conflict" and out.loc["c2", "action"] == "drop_conflict"
    assert out.loc["b1", "action"] == "keep" and out.loc["d1", "action"] == "keep"


def test_mirrored_copy_is_detected(tmp_path):
    from PIL import ImageOps
    a = _synthetic_xray(3)
    a.save(tmp_path / "a.png")
    ImageOps.mirror(a).save(tmp_path / "a_mirror.png")
    _synthetic_xray(4).save(tmp_path / "b.png")
    files = ["a.png", "a_mirror.png", "b.png"]
    hp = [deduplication.phash_pair(tmp_path / f) for f in files]
    sha = [deduplication.sha256_file(tmp_path / f) for f in files]
    gid, near, _, mirror = deduplication.group_duplicates(sha, [h for h, _ in hp], 4, [m for _, m in hp])
    assert (0, 1) in {(i, j) for i, j, _ in mirror}
    assert gid[0] == gid[1] != gid[2]
