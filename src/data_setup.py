
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_olivetti_faces

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

ROLL_NUMBER = "M26CSA014"
digits = "".join(c for c in ROLL_NUMBER if c.isdigit())

if not digits:
    raise ValueError("Roll number must contain numeric characters.")

SEED = int(digits)

# Fetch the original dataset
dataset = fetch_olivetti_faces()
images = dataset.images                 # (400, 64, 64)
y = dataset.target.astype(np.int64)     # (400,)

# Raw flattened pixels: no PCA or dimensionality reduction
X = images.reshape(len(images), -1)

assert X.shape == (400, 4096)
assert len(np.unique(y)) == 40

rng = np.random.default_rng(SEED)

train_idx = []
val_idx = []
test_idx = []

# Split independently within each identity
for identity in range(40):
    idx = np.flatnonzero(y == identity)
    assert len(idx) == 10

    shuffled = rng.permutation(idx)

    train_idx.extend(shuffled[:6])
    val_idx.extend(shuffled[6:8])
    test_idx.extend(shuffled[8:10])

train_idx = np.asarray(train_idx, dtype=np.int64)
val_idx = np.asarray(val_idx, dtype=np.int64)
test_idx = np.asarray(test_idx, dtype=np.int64)

# Save the indices once. All experiments must reuse these.
split_path = SPLIT_DIR / "olivetti_split.npz"
np.savez(
    split_path,
    train_idx=train_idx,
    val_idx=val_idx,
    test_idx=test_idx,
)

metadata = {
    "roll_number": ROLL_NUMBER,
    "seed": SEED,
    "dataset": "Olivetti Faces",
    "n_samples": 400,
    "n_features": 4096,
    "n_classes": 40,
    "train_per_identity": 6,
    "validation_per_identity": 2,
    "test_per_identity": 2,
}
with open(SPLIT_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

# Verify each split
for name, idx, expected in [
    ("train", train_idx, 240),
    ("validation", val_idx, 80),
    ("test", test_idx, 80),
]:
    assert len(idx) == expected
    assert len(np.unique(idx)) == expected
    assert all(np.sum(y[idx] == c) == expected // 40
               for c in range(40))
    print(name, X[idx].shape)

print("Saved split:", split_path)
print("Seed:", SEED)
