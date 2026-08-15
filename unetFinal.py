"""
U-Net para segmentacion de mitocondrias - EPFL EM Hippocampus Dataset
========================================================================

Entrena una U-Net (Dice Loss + Tversky Loss) sobre el dataset de microscopia
electronica de EPFL (hipocampo de raton, CA1). El script se encarga de todo
el pipeline: descarga el dataset si hace falta, arma los DataLoaders,
entrena con early stopping, guarda el mejor checkpoint, grafica el historial
completo y al final genera una visualizacion cualitativa (4 parches del set
de test) tanto en una figura estatica como en Napari.

Metricas usadas: Dice Score e IoU (Jaccard), evaluadas con un threshold fijo
sobre las probabilidades de salida del modelo.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import glob
import time
import random
import urllib.request

import numpy as np
import napari
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# CONFIGURACION DEL EXPERIMENTO
# =============================================================================

# --- Dataset -----------------------------------------------------------------
TARGET_SIZE = (256, 256)
BATCH_SIZE = 8
VAL_FRACTION = 0.15
SEED = 42

# --- Entrenamiento -------------------------------------------------------------
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0

# --- Early stopping ------------------------------------------------------------
# Epocas seguidas sin mejorar el Validation Dice antes de cortar el entrenamiento.
PATIENCE = 10

# --- Threshold para binarizar Dice, IoU y las predicciones ---------------------
THRESHOLD = 0.5

# --- Pesos de la loss combinada -------------------------------------------------
# Loss total = DICE_WEIGHT * DiceLoss + TVERSKY_WEIGHT * TverskyLoss
DICE_WEIGHT = 1.0
TVERSKY_WEIGHT = 1.0

# --- Parametros de Tversky -------------------------------------------------------
# Tversky Index = TP / (TP + alpha*FP + beta*FN)
# beta > alpha castiga mas los falsos negativos, que es lo que nos conviene
# aca porque las mitocondrias son la clase minoritaria (perder una mitocondria
# pesa mas que marcar de mas un poco de fondo).
TVERSKY_ALPHA = 0.3
TVERSKY_BETA = 0.7
TVERSKY_SMOOTH = 1.0


# Consola en UTF-8, por si el sistema anda en otra codificacion (Windows, etc.)
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# =============================================================================
# PARTE 0: DESCARGA Y CARGA DEL DATASET EPFL MITOCHONDRIA
# =============================================================================

EPFL_BASE_URL = (
    "https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/"
    "%20ElectronMicroscopy_Hippocampus/"
)

DATASET_FILES = [
    "training.tif",
    "training_groundtruth.tif",
    "testing.tif",
    "testing_groundtruth.tif",
]


def load_tiff_stack(path: str) -> np.ndarray:
    """Carga un TIFF multipagina como array (n_slices, H, W)."""
    try:
        import tifffile
        return tifffile.imread(path)
    except ImportError:
        from skimage import io
        return io.imread(path)


def download_file(url: str, dest: str) -> None:
    """Descarga un archivo si todavia no existe en disco."""
    if os.path.exists(dest):
        print(f"[INFO] {os.path.basename(dest)} ya esta descargado, se reutiliza.")
        return

    print(f"[INFO] Descargando {os.path.basename(dest)}...")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    with urllib.request.urlopen(request) as response, open(dest, "wb") as f:
        f.write(response.read())

    print(f"[INFO] {os.path.basename(dest)} listo.")


def download_epfl_dataset(data_dir: str) -> dict:
    """
    Se asegura de que los 4 volumenes del dataset EPFL Mitochondria esten
    en disco (los descarga directo del sitio de EPFL si hace falta) y
    devuelve las rutas locales listas para usar con load_tiff_stack.
    """
    os.makedirs(data_dir, exist_ok=True)

    local_paths = {}
    for fname in DATASET_FILES:
        dest = os.path.join(data_dir, fname)
        download_file(EPFL_BASE_URL + fname, dest)
        local_paths[fname] = dest

    return {
        "train_vol": local_paths["training.tif"],
        "train_gt": local_paths["training_groundtruth.tif"],
        "test_vol": local_paths["testing.tif"],
        "test_gt": local_paths["testing_groundtruth.tif"],
    }


# =============================================================================
# PARTE 1: DATASET DE PYTORCH
# =============================================================================

class MitochondriaEMDataset(Dataset):
    """
    Cada muestra es un slice 2D sacado de un volumen 3D de microscopia
    electronica, reescalado a target_size.

    Imagen: TEM en escala de grises, normalizada min-max.
    Mascara: 0 = fondo, 1 = mitocondria.
    """

    def __init__(self, volume, mask_volume, indices, target_size=(256, 256), transform=True):
        assert volume.shape == mask_volume.shape, (
            "El volumen de imagen y el de mascara deben tener la misma forma."
        )
        self.volume = volume
        self.mask_volume = mask_volume
        self.indices = list(indices)
        self.target_size = target_size
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def _apply_synchronous_transforms(self, img_np, mask_np):
        # Flip horizontal
        if random.random() > 0.5:
            img_np = np.fliplr(img_np)
            mask_np = np.fliplr(mask_np)

        # Flip vertical
        if random.random() > 0.5:
            img_np = np.flipud(img_np)
            mask_np = np.flipud(mask_np)

        # Rotacion en multiplos de 90
        if random.random() > 0.5:
            k = random.choice([1, 2, 3])
            img_np = np.rot90(img_np, k, axes=(0, 1))
            mask_np = np.rot90(mask_np, k, axes=(0, 1))

        return img_np.copy(), mask_np.copy()

    def __getitem__(self, idx):
        slice_idx = self.indices[idx]

        img_slice = self.volume[slice_idx].astype(np.float32)
        mask_slice = self.mask_volume[slice_idx]

        # Normalizacion min-max de la imagen
        img_min, img_max = float(img_slice.min()), float(img_slice.max())
        img_norm = (img_slice - img_min) / (img_max - img_min + 1e-8)

        # Mascara binaria
        mask_bin = (mask_slice > 0).astype(np.float32)

        # Reescalado al tamano objetivo (bilinear para la imagen, nearest para la mascara)
        img_pil = Image.fromarray((img_norm * 255).astype(np.uint8)).resize(
            self.target_size, Image.BILINEAR
        )
        mask_pil = Image.fromarray((mask_bin * 255).astype(np.uint8)).resize(
            self.target_size, Image.NEAREST
        )

        img_np = np.array(img_pil, dtype=np.float32) / 255.0
        mask_np = (np.array(mask_pil, dtype=np.float32) > 0).astype(np.float32)

        if self.transform:
            img_np, mask_np = self._apply_synchronous_transforms(img_np, mask_np)

        img_tensor = torch.from_numpy(img_np).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()

        return img_tensor, mask_tensor


# =============================================================================
# PARTE 2: DATALOADERS
# =============================================================================

def build_dataloaders(data_dir, target_size=(256, 256), batch_size=8, val_fraction=0.15, seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Descargamos (o reutilizamos) el dataset y cargamos los 4 volumenes
    paths = download_epfl_dataset(data_dir)

    train_volume = load_tiff_stack(paths["train_vol"])
    train_gt = load_tiff_stack(paths["train_gt"])
    test_volume = load_tiff_stack(paths["test_vol"])
    test_gt = load_tiff_stack(paths["test_gt"])

    print(f"\n[INFO] Volumen de entrenamiento: {train_volume.shape} slices")
    print(f"[INFO] Volumen de testing: {test_volume.shape} slices")

    # Split train / validation por slices Z (evita fuga de info entre cortes vecinos)
    n_train_slices = train_volume.shape[0]
    all_train_indices = list(range(n_train_slices))
    random.shuffle(all_train_indices)

    n_val = max(1, int(val_fraction * n_train_slices))
    val_indices = all_train_indices[:n_val]
    train_indices = all_train_indices[n_val:]
    test_indices = list(range(test_volume.shape[0]))

    print("\n Split del dataset:")
    print(f"  - Train      : {len(train_indices)} slices (de training.tif)")
    print(f"  - Validation : {len(val_indices)} slices (de training.tif)")
    print(f"  - Test       : {len(test_indices)} slices (de testing.tif, reservado completo)")

    train_ds = MitochondriaEMDataset(train_volume, train_gt, train_indices, target_size, transform=True)
    val_ds = MitochondriaEMDataset(train_volume, train_gt, val_indices, target_size, transform=False)
    test_ds = MitochondriaEMDataset(test_volume, test_gt, test_indices, target_size, transform=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# =============================================================================
# PARTE 3: BLOQUES DEL U-NET
# =============================================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Por si el tamano no calza exacto (p.ej. dimensiones impares)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        self.down4 = DownBlock(512, 1024)
        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)


# =============================================================================
# PARTE 4: DICE LOSS
# =============================================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )

        return 1.0 - dice_coeff


# =============================================================================
# PARTE 5: TVERSKY LOSS
# =============================================================================

class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        if alpha < 0 or beta < 0:
            raise ValueError("alpha y beta deben ser >= 0.")
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        true_positive = (probs_flat * targets_flat).sum()
        false_positive = (probs_flat * (1.0 - targets_flat)).sum()
        false_negative = ((1.0 - probs_flat) * targets_flat).sum()

        tversky_index = (true_positive + self.smooth) / (
            true_positive + self.alpha * false_positive + self.beta * false_negative + self.smooth
        )

        return 1.0 - tversky_index


# =============================================================================
# PARTE 6: DICE SCORE
# =============================================================================

def dice_score(logits, targets, threshold: float = 0.5, smooth: float = 1.0) -> float:
    """Dice score para evaluar el modelo (no se usa para backprop)."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    intersection = (preds_flat * targets_flat).sum()
    dice = (2.0 * intersection + smooth) / (preds_flat.sum() + targets_flat.sum() + smooth)

    return dice.item()


# =============================================================================
# PARTE 7: IoU SCORE
# =============================================================================

def iou_score(logits, targets, threshold: float = 0.5, smooth: float = 1.0) -> float:
    """
    Intersection over Union (IoU) = TP / (TP + FP + FN).
    Los True Negatives no entran en la cuenta.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    intersection = (preds_flat * targets_flat).sum()
    union = (preds_flat + targets_flat - preds_flat * targets_flat).sum()

    iou = (intersection + smooth) / (union + smooth)

    return iou.item()


# =============================================================================
# PARTE 8: MAPA DIAGNOSTICO DE ERROR
# =============================================================================

def create_error_map(ground_truth, prediction):
    """
    Mapa RGB de errores: verde = TP, amarillo = FP, rojo = FN, negro = TN.
    """
    gt = ground_truth > 0
    pred = prediction > 0

    error_map = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)

    error_map[gt & pred] = [0, 255, 0]        # TP -> verde
    error_map[~gt & pred] = [255, 255, 0]     # FP -> amarillo
    error_map[gt & ~pred] = [255, 0, 0]       # FN -> rojo
    # lo que queda en negro son los TN

    return error_map


# =============================================================================
# PARTE 9: GROUND TRUTH OVERLAY
# =============================================================================

def create_ground_truth_overlay(image, ground_truth, alpha=0.45):
    """Superpone la mascara ground truth (en cian) sobre la imagen TEM en gris."""
    image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([image_uint8] * 3, axis=-1).astype(np.float32)

    mask = ground_truth > 0
    overlay_color = np.array([0, 255, 255], dtype=np.float32)  # cian

    rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * overlay_color

    return np.clip(rgb, 0, 255).astype(np.uint8)


# =============================================================================
# PARTE 10: GRAFICAS DEL ENTRENAMIENTO
# =============================================================================

def _plot_metric_curve(epochs, train_values, val_values, train_label, val_label,
                        title, ylabel, output_dir, filename, best_epoch=None, ylim=None):
    """Grafica una curva train/val y la guarda como PNG. Chiquito helper para no
    repetir el mismo bloque de matplotlib seis veces."""
    plt.figure(figsize=(9, 6))
    plt.plot(epochs, train_values, label=train_label, linewidth=2)
    plt.plot(epochs, val_values, label=val_label, linewidth=2)

    if best_epoch is not None:
        plt.axvline(best_epoch, linestyle="--", label=f"Best Epoch ({best_epoch})")

    plt.xlabel("Epoca")
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[GRAFICA] Guardada: {path}")


def plot_training_history(history, output_dir):

    epochs = range(1, len(history["train_loss"]) + 1)
    best_epoch = history["best_epoch"]

    _plot_metric_curve(
        epochs, history["train_loss"], history["val_loss"],
        "Train Loss", "Validation Loss", "Loss Total vs Epocas", "Loss",
        output_dir, "training_validation_loss.png", best_epoch,
    )

    _plot_metric_curve(
        epochs, history["train_tversky_loss"], history["val_tversky_loss"],
        "Train Tversky Loss", "Validation Tversky Loss", "Tversky Loss vs Epocas", "Tversky Loss",
        output_dir, "tversky_loss.png", best_epoch,
    )

    _plot_metric_curve(
        epochs, history["train_dice_loss"], history["val_dice_loss"],
        "Train Dice Loss", "Validation Dice Loss", "Dice Loss vs Epocas", "Dice Loss",
        output_dir, "dice_loss.png", best_epoch,
    )

    _plot_metric_curve(
        epochs, history["train_dice"], history["val_dice"],
        "Train Dice", "Validation Dice", "Dice Score vs Epocas", "Dice",
        output_dir, "dice_score.png", best_epoch, ylim=(0, 1),
    )

    # Learning rate: no tiene curva de validation, va sola
    plt.figure(figsize=(9, 6))
    plt.plot(epochs, history["learning_rate"], linewidth=2, color="tab:green")
    plt.xlabel("Epoca")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate vs Epocas")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    lr_path = os.path.join(output_dir, "learning_rate.png")
    plt.savefig(lr_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[GRAFICA] Guardada: {lr_path}")

    # Resumen 2x2 con las cuatro curvas principales juntas
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0, 0].plot(epochs, history["val_loss"], label="Validation Loss")
    axes[0, 0].set_title("Loss Total")
    axes[0, 0].set_xlabel("Epoca")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history["train_tversky_loss"], label="Train Tversky")
    axes[0, 1].plot(epochs, history["val_tversky_loss"], label="Validation Tversky")
    axes[0, 1].set_title("Tversky Loss")
    axes[0, 1].set_xlabel("Epoca")
    axes[0, 1].set_ylabel("Tversky Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, history["train_dice_loss"], label="Train Dice Loss")
    axes[1, 0].plot(epochs, history["val_dice_loss"], label="Validation Dice Loss")
    axes[1, 0].set_title("Dice Loss")
    axes[1, 0].set_xlabel("Epoca")
    axes[1, 0].set_ylabel("Dice Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, history["train_dice"], label="Train Dice")
    axes[1, 1].plot(epochs, history["val_dice"], label="Validation Dice")
    axes[1, 1].set_title("Dice Score")
    axes[1, 1].set_xlabel("Epoca")
    axes[1, 1].set_ylabel("Dice")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Historial de entrenamiento U-Net", fontsize=16)
    plt.tight_layout()

    summary_path = os.path.join(output_dir, "training_history_summary.png")
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[GRAFICA] Guardada: {summary_path}")


# =============================================================================
# PARTE 11: ENTRENAMIENTO PRINCIPAL
# =============================================================================

def train_unet_mitochondria():

    print("==========================================================")
    print(" U-NET - SEGMENTACION DE MITOCONDRIAS (EPFL EM Hippocampus)")
    print("==========================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" -> Dispositivo: {device}")

    # El script y sus outputs (checkpoints, graficas, figuras) viven en la misma
    # carpeta desde donde se corre. El dataset se guarda aparte, en data/.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = base_dir
    dataset_dir = os.path.join(base_dir, "data", "epfl_mitochondria")

    # -------------------------------------------------------------------------
    # Resumen de configuracion
    # -------------------------------------------------------------------------
    print("\n==========================================================")
    print(" CONFIGURACION DEL EXPERIMENTO")
    print("==========================================================")
    print(f" Target size       : {TARGET_SIZE}")
    print(f" Batch size        : {BATCH_SIZE}")
    print(f" Epocas maximas    : {EPOCHS}")
    print(f" Learning rate     : {LEARNING_RATE}")
    print(f" Weight decay      : {WEIGHT_DECAY}")
    print(f" Validation        : {VAL_FRACTION * 100:.1f}%")
    print(f" Early stopping    : patience={PATIENCE}")
    print(f" Threshold         : {THRESHOLD}")
    print(f" Dice weight       : {DICE_WEIGHT}")
    print(f" Tversky weight    : {TVERSKY_WEIGHT}")
    print(f" Tversky alpha     : {TVERSKY_ALPHA}")
    print(f" Tversky beta      : {TVERSKY_BETA}")
    print("==========================================================")

    # -------------------------------------------------------------------------
    # Paso 1: dataset (se descarga solo si hace falta) y DataLoaders
    # -------------------------------------------------------------------------
    print("\n[Paso 1] Descargando/cargando el dataset EPFL y armando los DataLoaders...")

    train_loader, val_loader, test_loader = build_dataloaders(
        data_dir=dataset_dir,
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        val_fraction=VAL_FRACTION,
        seed=SEED,
    )

    # -------------------------------------------------------------------------
    # Paso 2: modelo
    # -------------------------------------------------------------------------
    print("\n[Paso 2] Construyendo la U-Net (in_channels=1, escala de grises)...")

    model = UNet(in_channels=1, out_channels=1).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" -> Parametros entrenables: {total_params:,}")

    criterion_dice = DiceLoss()
    criterion_tversky = TverskyLoss(alpha=TVERSKY_ALPHA, beta=TVERSKY_BETA, smooth=TVERSKY_SMOOTH)
    print("\n -> Funcion de perdida: Loss Total = Dice Loss + Tversky Loss")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    checkpoint_path = os.path.join(output_dir, "best_unet_mitochondria.pt")

    history = {
        "train_loss": [], "val_loss": [],
        "train_tversky_loss": [], "val_tversky_loss": [],
        "train_dice_loss": [], "val_dice_loss": [],
        "train_dice": [], "val_dice": [],
        "learning_rate": [],
        "best_epoch": None,
    }

    best_val_dice = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    # -------------------------------------------------------------------------
    # Paso 3: loop de entrenamiento con early stopping
    # -------------------------------------------------------------------------
    print("\n[Paso 3] Entrenamiento.")
    print(f"Epocas maximas: {EPOCHS}")
    print(f"Early Stopping patience: {PATIENCE}")
    print("==========================================================")

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):

        # ---- Train --------------------------------------------------------
        model.train()
        running_total_loss = 0.0
        running_tversky_loss = 0.0
        running_dice_loss = 0.0
        running_dice_score = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            logits = model(images)

            dice_loss = criterion_dice(logits, masks)
            tversky_loss = criterion_tversky(logits, masks)
            loss = DICE_WEIGHT * dice_loss + TVERSKY_WEIGHT * tversky_loss

            loss.backward()
            optimizer.step()

            running_total_loss += loss.item()
            running_tversky_loss += tversky_loss.item()
            running_dice_loss += dice_loss.item()
            running_dice_score += dice_score(logits, masks, threshold=THRESHOLD)

        train_loss = running_total_loss / len(train_loader)
        train_tversky_loss = running_tversky_loss / len(train_loader)
        train_dice_loss = running_dice_loss / len(train_loader)
        train_dice = running_dice_score / len(train_loader)

        # ---- Validation -----------------------------------------------------
        model.eval()
        val_total_loss = 0.0
        val_tversky_loss_total = 0.0
        val_dice_loss_total = 0.0
        val_dice_total = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)

                dice_loss = criterion_dice(logits, masks)
                tversky_loss = criterion_tversky(logits, masks)
                loss = DICE_WEIGHT * dice_loss + TVERSKY_WEIGHT * tversky_loss
                current_dice = dice_score(logits, masks, threshold=THRESHOLD)

                val_total_loss += loss.item()
                val_tversky_loss_total += tversky_loss.item()
                val_dice_loss_total += dice_loss.item()
                val_dice_total += current_dice

        val_loss = val_total_loss / len(val_loader)
        val_tversky_loss = val_tversky_loss_total / len(val_loader)
        val_dice_loss = val_dice_loss_total / len(val_loader)
        val_dice = val_dice_total / len(val_loader)

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_tversky_loss"].append(train_tversky_loss)
        history["val_tversky_loss"].append(val_tversky_loss)
        history["train_dice_loss"].append(train_dice_loss)
        history["val_dice_loss"].append(val_dice_loss)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)
        history["learning_rate"].append(current_lr)

        print(f"\nEpoca [{epoch}/{EPOCHS}]")
        print(f"  Train Loss       : {train_loss:.4f}")
        print(f"  Train Tversky    : {train_tversky_loss:.4f}")
        print(f"  Train Dice Loss  : {train_dice_loss:.4f}")
        print(f"  Train Dice       : {train_dice:.4f}")
        print(f"  Val Loss         : {val_loss:.4f}")
        print(f"  Val Tversky      : {val_tversky_loss:.4f}")
        print(f"  Val Dice Loss    : {val_dice_loss:.4f}")
        print(f"  Val Dice         : {val_dice:.4f}")
        print(f"  Learning Rate    : {current_lr:.8f}")

        # ---- Early stopping / checkpoint -------------------------------------
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            epochs_without_improvement = 0
            history["best_epoch"] = best_epoch

            torch.save(model.state_dict(), checkpoint_path)

            print("  *** NUEVO MEJOR MODELO ***")
            print(f"  Mejor Val Dice: {best_val_dice:.4f}")
            print(f"  Checkpoint: {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            print(f"  Sin mejora de Val Dice: {epochs_without_improvement}/{PATIENCE}")

        if epochs_without_improvement >= PATIENCE:
            print("\n==========================================================")
            print(" EARLY STOPPING ACTIVADO")
            print("==========================================================")
            print(f"No hubo mejora del Validation Dice durante {PATIENCE} epocas consecutivas.")
            print(f"Mejor epoca: {best_epoch}")
            print(f"Mejor Validation Dice: {best_val_dice:.4f}")
            print(f"Entrenamiento detenido en la epoca: {epoch}")
            break

    trained_epochs = len(history["train_loss"])
    total_minutes = (time.time() - start_time) / 60.0

    print("\n==========================================================")
    print(" ENTRENAMIENTO FINALIZADO")
    print("==========================================================")
    print(f"Epocas ejecutadas : {trained_epochs}")
    print(f"Mejor epoca       : {best_epoch}")
    print(f"Mejor Val Dice    : {best_val_dice:.4f}")
    print(f"Tiempo total      : {total_minutes:.2f} minutos")
    print(f"Checkpoint        : {checkpoint_path}")

    # -------------------------------------------------------------------------
    # Paso 4: graficas del entrenamiento
    # -------------------------------------------------------------------------
    print("\n[Paso 4] Generando graficas del entrenamiento...")
    plot_training_history(history, output_dir)

    # -------------------------------------------------------------------------
    # Paso 5: evaluacion final en el test set
    # -------------------------------------------------------------------------
    print("\n[Paso 5] Evaluando en testing.tif (reservado, nunca visto en entrenamiento)...")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    test_dice_total = 0.0
    test_iou_total = 0.0

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            test_dice_total += dice_score(logits, masks, threshold=THRESHOLD)
            test_iou_total += iou_score(logits, masks, threshold=THRESHOLD)

    test_dice = test_dice_total / len(test_loader)
    test_iou = test_iou_total / len(test_loader)

    print("\n==========================================================")
    print(" RESULTADOS FINALES EN TEST")
    print("==========================================================")
    print(f" Test Dice : {test_dice:.4f}")
    print(f" Test IoU  : {test_iou:.4f}")
    print("==========================================================")

    # -------------------------------------------------------------------------
    # Paso 6: visualizacion cualitativa - 4 parches de test, 5 paneles cada uno
    # -------------------------------------------------------------------------
    print("\n[Paso 6] Generando visualizacion cualitativa (4 parches de test)...")

    test_dataset = test_loader.dataset

    # Elegimos parches que tengan una cantidad razonable de mitocondria, para
    # que las imagenes de ejemplo no salgan practicamente vacias.
    sample_indices = []
    for idx in range(len(test_dataset)):
        _, mask_t = test_dataset[idx]
        if (mask_t > 0.5).sum() > 300:
            sample_indices.append(idx)
        if len(sample_indices) == 4:
            break

    # Si no encontramos 4 parches con suficiente mitocondria, completamos con
    # los que haya para no dejar la figura a medias.
    if len(sample_indices) < 4:
        for idx in range(len(test_dataset)):
            if idx not in sample_indices:
                sample_indices.append(idx)
            if len(sample_indices) == 4:
                break

    col_titles = [
        "TEM Grayscale",
        "Ground Truth Overlay",
        "Probability Heatmap",
        f"Prediccion Binaria (t={THRESHOLD})",
        "Mapa Diagnostico de Error",
    ]

    fig, axes = plt.subplots(4, 5, figsize=(22, 18))

    # Vamos guardando todo tambien para poder mandarlo a Napari despues
    napari_imgs, napari_masks = [], []
    napari_probs, napari_preds, napari_errors = [], [], []

    with torch.no_grad():
        for row, idx in enumerate(sample_indices):

            img_t, mask_t = test_dataset[idx]

            pred_logit = model(img_t.unsqueeze(0).to(device))
            pred_prob = torch.sigmoid(pred_logit).squeeze().cpu().numpy()
            binary_pred = (pred_prob > THRESHOLD).astype(np.uint8)

            display_img = img_t.squeeze().numpy()
            display_mask = mask_t.squeeze().numpy().astype(np.uint8)

            gt_overlay = create_ground_truth_overlay(display_img, display_mask, alpha=0.45)
            error_map = create_error_map(display_mask, binary_pred)

            axes[row, 0].imshow(display_img, cmap="gray")
            axes[row, 1].imshow(gt_overlay)
            im = axes[row, 2].imshow(pred_prob, cmap="magma", vmin=0, vmax=1)
            axes[row, 3].imshow(binary_pred, cmap="gray")
            axes[row, 4].imshow(error_map)

            for col in range(5):
                axes[row, col].axis("off")
                if row == 0:
                    axes[row, col].set_title(col_titles[col])

            axes[row, 0].set_ylabel(f"Parche {row + 1}", fontsize=10)

            napari_imgs.append(display_img)
            napari_masks.append(display_mask)
            napari_probs.append(pred_prob)
            napari_preds.append(binary_pred)
            napari_errors.append(error_map)

    # Barra de color para la columna de probabilidad y leyenda para la de errores
    fig.colorbar(im, ax=axes[:, 2].tolist(), fraction=0.025, pad=0.02, label="Probabilidad")

    legend_elements = [
        Patch(facecolor="green", label="TP"),
        Patch(facecolor="yellow", label="FP"),
        Patch(facecolor="red", label="FN"),
        Patch(facecolor="black", label="TN"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.01))

    plt.suptitle("U-Net - Segmentacion de Mitocondrias (4 parches de test)", fontsize=16)
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])

    output_png = os.path.join(output_dir, "unet_mitochondria_qualitative_results.png")
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f" -> Figura cualitativa guardada en: {output_png}")

    # -------------------------------------------------------------------------
    # Napari, para explorar los 4 parches de forma interactiva
    # -------------------------------------------------------------------------
    try:
        viewer = napari.Viewer(title="U-Net Mitocondrias - EPFL EM Hippocampus")

        viewer.add_image(np.stack(napari_imgs), name="1. TEM Grayscale", colormap="gray")
        viewer.add_labels(np.stack(napari_masks), name="2. Ground Truth", opacity=0.5)
        viewer.add_image(
            np.stack(napari_probs), name="3. Probability Heatmap",
            colormap="magma", opacity=0.7, visible=False,
        )
        viewer.add_labels(np.stack(napari_preds), name="4. Prediccion Binaria", opacity=0.5)
        viewer.add_image(
            np.stack(napari_errors), name="5. Mapa Diagnostico TP-FP-FN",
            rgb=True, visible=False,
        )

        print(" -> Napari activo (usa el slider para pasar entre los 4 parches).")
        print(" -> Cierra la ventana para terminar.")
        napari.run()

    except Exception as err:
        print(f" -> Napari no disponible en este entorno ({err}). Se usa solamente la figura estatica.")

    # -------------------------------------------------------------------------
    # Cierre
    # -------------------------------------------------------------------------
    print("\n==========================================================")
    print(" ENTRENAMIENTO Y EVALUACION COMPLETADOS")
    print("==========================================================")
    print(f" Mejor Validation Dice : {best_val_dice:.4f}")
    print(f" Test Dice             : {test_dice:.4f}")
    print(f" Test IoU              : {test_iou:.4f}")
    print(f" Mejor epoca           : {best_epoch}")
    print(f" Epocas ejecutadas     : {trained_epochs}")
    print("==========================================================")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    train_unet_mitochondria()