import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# --- Constants -----------------------------------------------------------

IMAGENET_MEAN    = [0.485, 0.456, 0.406]
IMAGENET_STD     = [0.229, 0.224, 0.225]
DEFAULT_IMG_SIZE = 224

DIAGNOSTIC_TO_IDX: Dict[str, int] = {
    "ACK": 0,   # Actinic Keratosis
    "BCC": 1,   # Basal Cell Carcinoma
    "MEL": 2,   # Melanoma
    "NEV": 3,   # Melanocytic Nevus
    "SCC": 4,   # Squamous Cell Carcinoma
    "SEK": 5,   # Seborrheic Keratosis
}
NUM_CLASSES = len(DIAGNOSTIC_TO_IDX)

DROP_COLS        = ["patient_id", "lesion_id", "biopsed"]
BINARY_COLS      = [
    "smoke", "drink", "pesticide", "gender",
    "skin_cancer_history", "cancer_history",
    "has_piped_water", "has_sewage_system",
    "itch", "grew", "hurt", "changed", "bleed", "elevation",
]
CATEGORICAL_COLS = ["background_father", "background_mother", "fitspatrick", "region"]
NUMERICAL_COLS   = ["age", "diameter_1", "diameter_2"]

IMAGE_SUBDIRS = [
    os.path.join("imgs_part_1", "imgs_part_1"),
    os.path.join("imgs_part_2", "imgs_part_2"),
    os.path.join("imgs_part_3", "imgs_part_3"),
]

# Type alias: {col_name: (mean, std)} for the three numerical columns
NumericalStats = Dict[str, Tuple[float, float]]


# --- Image lookup --------------------------------------------------------

def _build_image_lookup(archive_root: str) -> Dict[str, str]:
    """Scan the three image folders and return {filename -> full_path}."""
    lookup: Dict[str, str] = {}
    for subdir in IMAGE_SUBDIRS:
        folder = os.path.join(archive_root, subdir)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                lookup[fname] = os.path.join(folder, fname)
    return lookup


# --- Tabular encoding ----------------------------------------------------

def _encode_binary(value) -> float:
    """TRUE → 1.0, FALSE → 0.0, unknown/missing → -1.0."""
    if isinstance(value, str):
        v = value.strip().upper()
        if v == "TRUE":  return 1.0
        if v == "FALSE": return 0.0
    return -1.0


def _encode_gender(value) -> float:
    """FEMALE → 0.0, MALE → 1.0, missing → -1.0."""
    if isinstance(value, str):
        v = value.strip().upper()
        if v == "FEMALE": return 0.0
        if v == "MALE":   return 1.0
    return -1.0


def fit_numerical_scaler(
    df: pd.DataFrame, row_indices: Optional[List[int]] = None
) -> NumericalStats:
    """Fit per-column (mean, std) on `df` rows at `row_indices` (or all rows if None)."""
    fit_df = df.iloc[row_indices] if row_indices is not None else df
    stats: NumericalStats = {}
    for col in NUMERICAL_COLS:
        vals  = pd.to_numeric(fit_df[col], errors="coerce").values.astype(np.float32)
        valid = vals[~np.isnan(vals)]
        mean  = float(valid.mean()) if len(valid) > 0 else 0.0
        std   = float(max(valid.std() if len(valid) > 0 else 1.0, 1e-8))
        stats[col] = (mean, std)
    return stats


def _prepare_tabular_data(
    df: pd.DataFrame,
    numerical_stats: Optional[NumericalStats] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Encode the full dataframe into a float32 feature matrix.

    Encoding order: binary (14 cols) → one-hot categoricals → z-scored numericals.
    Running on the full dataframe (not a subset) ensures a consistent feature_dim
    across all clients and the server.
    """
    arrays, names = [], []

    # Binary features (14 columns)
    for col in BINARY_COLS:
        fn  = _encode_gender if col == "gender" else _encode_binary
        arr = df[col].apply(fn).values.astype(np.float32)
        arrays.append(arr.reshape(-1, 1))
        names.append(col)

    # Categorical features → one-hot (unknown/NaN rows become all-zeros)
    for col in CATEGORICAL_COLS:
        series  = df[col].fillna("MISSING").astype(str).str.strip().str.upper()
        dummies = pd.get_dummies(series, prefix=col, dtype=np.float32)
        if f"{col}_MISSING" in dummies.columns:
            dummies = dummies.drop(columns=[f"{col}_MISSING"])
        arrays.append(dummies.values)
        names.extend(dummies.columns.tolist())

    # Numerical features → z-score normalisation
    for col in NUMERICAL_COLS:
        vals = pd.to_numeric(df[col], errors="coerce").values.astype(np.float32)
        if numerical_stats is not None:
            mean, std = numerical_stats[col]
        else:
            valid = vals[~np.isnan(vals)]
            mean  = valid.mean() if len(valid) > 0 else 0.0
            std   = max(valid.std() if len(valid) > 0 else 1.0, 1e-8)
        arrays.append(np.nan_to_num((vals - mean) / std, nan=0.0).reshape(-1, 1))
        names.append(col)

    return np.hstack(arrays), names


# --- Stratified split ----------------------------------------------------

def stratified_split(
    labels: np.ndarray,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Per-class stratified train/val split. Returns (train_indices, val_indices)."""
    rng = np.random.default_rng(seed)
    train_indices: List[int] = []
    val_indices:   List[int] = []

    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n_val = max(1, int(len(cls_idx) * val_fraction))
        val_indices.extend(cls_idx[:n_val].tolist())
        train_indices.extend(cls_idx[n_val:].tolist())

    return train_indices, val_indices


def prepare_train_val_split(
    archive_root: str,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[List[int], List[int], NumericalStats, int]:
    """
    Load the full CSV, perform a stratified split, fit the numerical scaler
    on train rows only, and return everything needed to build datasets.

    Returns:
        train_indices, val_indices, numerical_stats, feature_dim
    """
    full_df = pd.read_csv(os.path.join(archive_root, "metadata.csv"))
    full_df = full_df.drop(columns=DROP_COLS, errors="ignore")
    labels  = full_df["diagnostic"].map(DIAGNOSTIC_TO_IDX).values.astype(np.int64)

    train_indices, val_indices = stratified_split(labels, val_fraction=val_fraction, seed=seed)

    # Fit scaler ONLY on train rows to avoid val leakage
    numerical_stats = fit_numerical_scaler(full_df, row_indices=train_indices)

    # feature_dim: encode full df (consistent column count regardless of subset)
    full_tabular, _ = _prepare_tabular_data(full_df, numerical_stats=numerical_stats)
    feature_dim = full_tabular.shape[1]

    print(f"[Split] total={len(labels)}  train={len(train_indices)}  val={len(val_indices)}"
          f"  (val_fraction={val_fraction:.2f}, seed={seed})")

    return train_indices, val_indices, numerical_stats, feature_dim


# --- Image transforms ----------------------------------------------------

def get_train_transforms(img_size: int = DEFAULT_IMG_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(img_size: int = DEFAULT_IMG_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# --- Dataset class -------------------------------------------------------

class PADUFES20Dataset(Dataset):
    """
    PyTorch Dataset for PAD-UFES-20.

    Returns (image_tensor, tabular_tensor, label_tensor) for each sample.
    Tabular encoding is run on the full dataframe so that every client and
    the server share an identical feature_dim, which is required for
    model state_dict compatibility.

    Args:
        archive_root: Path to the PAD-UFES-20 archive directory.
        indices: Row indices into the full CSV to include (None = all rows).
        train: If True, applies augmentation transforms; otherwise eval transforms.
        img_size: Spatial size to resize images to.
        numerical_stats: Pre-fitted scaler stats. If None, stats are fitted on `indices`.
    """

    def __init__(
        self,
        archive_root: str,
        indices: Optional[List[int]] = None,
        train: bool = True,
        img_size: int = DEFAULT_IMG_SIZE,
        numerical_stats: Optional[NumericalStats] = None,
    ):
        super().__init__()
        self.archive_root = archive_root
        self.img_size     = img_size

        full_df = pd.read_csv(os.path.join(archive_root, "metadata.csv"))
        full_df = full_df.drop(columns=DROP_COLS, errors="ignore")

        # Fit scaler on provided indices only (prevents val-row leakage)
        if numerical_stats is None:
            numerical_stats = fit_numerical_scaler(full_df, row_indices=indices)
        self.numerical_stats = numerical_stats

        # Encode full df for consistent feature_dim across all clients/server
        full_tabular, self.feature_names = _prepare_tabular_data(full_df, numerical_stats=numerical_stats)
        self.feature_dim = full_tabular.shape[1]

        # Subset rows
        if indices is not None:
            idx_arr           = np.asarray(indices, dtype=np.intp)
            self.tabular_data = full_tabular[idx_arr]
            subset_df         = full_df.iloc[indices].reset_index(drop=True)
        else:
            self.tabular_data = full_tabular
            subset_df         = full_df

        self.labels     = subset_df["diagnostic"].map(DIAGNOSTIC_TO_IDX).values.astype(np.int64)
        self.img_ids    = subset_df["img_id"].tolist()
        self.img_lookup = _build_image_lookup(archive_root)
        self.transform  = get_train_transforms(img_size) if train else get_eval_transforms(img_size)

        missing = [i for i in self.img_ids if i not in self.img_lookup]
        if missing:
            print(f"[Dataset] WARNING: {len(missing)} image(s) not found (first 5: {missing[:5]})")
        print(f"[Dataset] {len(self)} samples | tabular_dim={self.feature_dim} | train={train}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path = self.img_lookup.get(self.img_ids[idx])
        if img_path and os.path.isfile(img_path):
            image = Image.open(img_path).convert("RGB")
        else:
            image = Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))
        image = self.transform(image)

        tabular = torch.tensor(self.tabular_data[idx], dtype=torch.float32)
        label   = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, tabular, label

    def get_all_labels(self) -> np.ndarray:
        return self.labels.copy()


# --- Convenience helper --------------------------------------------------

def build_client_datasets(
    archive_root: str,
    client_indices: Dict[int, List[int]],
    numerical_stats: Optional[NumericalStats] = None,
    img_size: int = DEFAULT_IMG_SIZE,
) -> Dict[int, "PADUFES20Dataset"]:
    """Build a {client_id: PADUFES20Dataset} mapping for training."""
    return {
        cid: PADUFES20Dataset(
            archive_root=archive_root, indices=idxs,
            train=True, img_size=img_size, numerical_stats=numerical_stats,
        )
        for cid, idxs in client_indices.items()
    }


# Smoke-test: python dataset.py
if __name__ == "__main__":
    import sys
    ARCHIVE = os.path.join(os.path.dirname(__file__), "archive")
    if not os.path.isdir(ARCHIVE):
        print(f"Archive not found at {ARCHIVE}")
        sys.exit(1)

    train_indices, val_indices, num_stats, feature_dim = prepare_train_val_split(ARCHIVE)
    print(f"feature_dim={feature_dim}")

    train_ds = PADUFES20Dataset(archive_root=ARCHIVE, indices=train_indices, train=True, numerical_stats=num_stats)
    val_ds   = PADUFES20Dataset(archive_root=ARCHIVE, indices=val_indices,   train=False, numerical_stats=num_stats)

    img, tab, lbl = train_ds[0]
    print(f"image: {img.shape}  tabular: {tab.shape}  label: {lbl.item()}")
    print(f"feature names (first 5): {train_ds.feature_names[:5]}")

    from partitioning import dirichlet_partition
    partitions = dirichlet_partition(
        train_ds.get_all_labels(), num_clients=5, alpha=0.5, original_indices=train_indices
    )
    for cid, idxs in partitions.items():
        print(f"  Client {cid}: {len(idxs)} samples")
