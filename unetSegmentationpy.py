"""
===============================================================================
U-NET PARA SEGMENTACION DE MITOCONDRIAS (EPFL EM Hippocampus Dataset)
Adaptado de 04_unet_segmentation.py
===============================================================================
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import glob
import random
import numpy as np
import napari
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


# =============================================================================
# PART 0: CARGA DE VOLUMENES TIFF MULTIPAGINA
# =============================================================================

def load_tiff_stack(path: str) -> np.ndarray:
    """Carga un TIFF multipagina como array (n_slices, H, W)."""
    try:
        import tifffile
        return tifffile.imread(path)
    except ImportError:
        from skimage import io
        return io.imread(path)


def find_tiff_file(data_dir: str, must_include, must_exclude=None):
    """Busca un .tif en data_dir cuyo nombre contenga todas las palabras de must_include
    y ninguna de must_exclude. Evita depender de que el nombre exacto coincida al 100%."""
    must_exclude = must_exclude or []
    candidates = glob.glob(os.path.join(data_dir, "*.tif")) + glob.glob(os.path.join(data_dir, "*.tiff"))
    for path in candidates:
        name = os.path.basename(path).lower()
        if all(k in name for k in must_include) and not any(k in name for k in must_exclude):
            return path
    return None


def resolve_dataset_paths(data_dir: str):
    """Ubica los 4 archivos: training.tif, training_groundtruth.tif, testing.tif, testing_groundtruth.tif"""
    paths = {
        "train_vol": find_tiff_file(data_dir, ["train"], ["ground"]),
        "train_gt":  find_tiff_file(data_dir, ["train", "ground"]),
        "test_vol":  find_tiff_file(data_dir, ["test"], ["ground"]),
        "test_gt":   find_tiff_file(data_dir, ["test", "ground"]),
    }
    missing = [k for k, v in paths.items() if v is None]
    if missing:
        raise FileNotFoundError(
            f"No se encontraron estos archivos en '{data_dir}': {missing}. "
            f"Verifica que training.tif, training_groundtruth.tif, testing.tif y "
            f"testing_groundtruth.tif estén en esa carpeta."
        )
    return paths


# =============================================================================
# PART 1: DATASET PYTORCH PARA SLICES DE MICROSCOPIA ELECTRONICA
# =============================================================================

class MitochondriaEMDataset(Dataset):
    """
    Cada muestra es un slice 2D extraído de un volumen 3D de microscopía electrónica,
    junto con su máscara binaria de mitocondrias (0 = fondo, 1 = mitocondria).
    """

    def __init__(self, volume: np.ndarray, mask_volume: np.ndarray, indices,
                 target_size=(256, 256), transform: bool = True):
        assert volume.shape == mask_volume.shape, "El volumen de imagen y el de máscara deben tener la misma forma."
        self.volume = volume
        self.mask_volume = mask_volume
        self.indices = list(indices)
        self.target_size = target_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def _apply_synchronous_transforms(self, img_np, mask_np):
        if random.random() > 0.5:
            img_np = np.fliplr(img_np); mask_np = np.fliplr(mask_np)
        if random.random() > 0.5:
            img_np = np.flipud(img_np); mask_np = np.flipud(mask_np)
        if random.random() > 0.5:
            k = random.choice([1, 2, 3])
            img_np = np.rot90(img_np, k, axes=(0, 1))
            mask_np = np.rot90(mask_np, k, axes=(0, 1))
        return img_np.copy(), mask_np.copy()

    def __getitem__(self, idx: int):
        slice_idx = self.indices[idx]
        img_slice = self.volume[slice_idx].astype(np.float32)
        mask_slice = self.mask_volume[slice_idx]

        # Normalizacion min-max del slice a [0, 1]
        img_min, img_max = float(img_slice.min()), float(img_slice.max())
        img_norm = (img_slice - img_min) / (img_max - img_min + 1e-8)

        # Mascara binaria (cualquier valor > 0 se considera mitocondria)
        mask_bin = (mask_slice > 0).astype(np.float32)

        # Resize al tamaño objetivo del U-Net
        img_pil = Image.fromarray((img_norm * 255).astype(np.uint8)).resize(self.target_size, Image.BILINEAR)
        mask_pil = Image.fromarray((mask_bin * 255).astype(np.uint8)).resize(self.target_size, Image.NEAREST)

        img_np = np.array(img_pil, dtype=np.float32) / 255.0
        mask_np = (np.array(mask_pil, dtype=np.float32) > 0).astype(np.float32)

        if self.transform:
            img_np, mask_np = self._apply_synchronous_transforms(img_np, mask_np)

        img_tensor = torch.from_numpy(img_np).unsqueeze(0).float()    # [1, H, W] (escala de grises)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()  # [1, H, W]
        return img_tensor, mask_tensor


def build_dataloaders(data_dir: str, target_size=(256, 256), batch_size: int = 8,
                       val_fraction: float = 0.15, seed: int = 42):
    """
    Construye train/val/test loaders respetando la separación física del dataset:
      - training.tif -> se divide en train / val (por slice, sin mezclar con testing)
      - testing.tif  -> se usa completo y únicamente como test
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    paths = resolve_dataset_paths(data_dir)
    print(f"[INFO] training.tif           -> {paths['train_vol']}")
    print(f"[INFO] training_groundtruth   -> {paths['train_gt']}")
    print(f"[INFO] testing.tif            -> {paths['test_vol']}")
    print(f"[INFO] testing_groundtruth    -> {paths['test_gt']}")

    train_volume = load_tiff_stack(paths["train_vol"])
    train_gt = load_tiff_stack(paths["train_gt"])
    test_volume = load_tiff_stack(paths["test_vol"])
    test_gt = load_tiff_stack(paths["test_gt"])

    print(f"\n[INFO] Volumen de entrenamiento: {train_volume.shape} slices")
    print(f"[INFO] Volumen de testing:       {test_volume.shape} slices")

    # Division de indices SOLO dentro del volumen de training
    n_train_slices = train_volume.shape[0]
    all_train_indices = list(range(n_train_slices))
    random.shuffle(all_train_indices)
    n_val = max(1, int(val_fraction * n_train_slices))
    val_indices = all_train_indices[:n_val]
    train_indices = all_train_indices[n_val:]

    test_indices = list(range(test_volume.shape[0]))

    print(f"\n Split del dataset:")
    print(f"  • Train      : {len(train_indices)} slices (de training.tif)")
    print(f"  • Validation : {len(val_indices)} slices (de training.tif)")
    print(f"  • Test       : {len(test_indices)} slices (de testing.tif, reservado completo)")

    train_ds = MitochondriaEMDataset(train_volume, train_gt, train_indices, target_size, transform=True)
    val_ds = MitochondriaEMDataset(train_volume, train_gt, val_indices, target_size, transform=False)
    test_ds = MitochondriaEMDataset(test_volume, test_gt, test_indices, target_size, transform=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# =============================================================================
# PART 2: BLOQUES DEL U-NET (identico a 04_unet_segmentation.py)
# =============================================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
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


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice_coeff


def dice_score(logits, targets, threshold: float = 0.5, smooth: float = 1.0) -> float:
    """Dice score (no loss) para reportar métricas, no para backprop."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)
    intersection = (preds_flat * targets_flat).sum()
    dice = (2.0 * intersection + smooth) / (preds_flat.sum() + targets_flat.sum() + smooth)
    return dice.item()


# =============================================================================
# PART 3: ENTRENAMIENTO
# =============================================================================

def train_unet_mitochondria():
    print("==========================================================")
    print(" U-NET - SEGMENTACION DE MITOCONDRIAS (EPFL EM Hippocampus)")
    print("==========================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" -> Dispositivo: {device}")

    data_dir = os.path.dirname(os.path.abspath(__file__))  # misma carpeta que el script

    print("\n[Paso 1] Cargando volumenes TIFF y construyendo DataLoaders...")
    train_loader, val_loader, test_loader = build_dataloaders(
        data_dir=data_dir,
        target_size=(256, 256),
        batch_size=8,
        val_fraction=0.15,
        seed=42
    )

    print("\n[Paso 2] Construyendo U-Net (in_channels=1, escala de grises)...")
    model = UNet(in_channels=1, out_channels=1).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" -> Parametros entrenables: {total_params:,}")

    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 15
    best_val_dice = 0.0
    checkpoint_path = os.path.join(data_dir, "best_unet_mitochondria.pt")

    print(f"\n[Paso 3] Entrenando {epochs} epocas...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion_bce(logits, masks) + criterion_dice(logits, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)

        model.eval()
        val_dice_total = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                val_dice_total += dice_score(logits, masks)
        val_dice = val_dice_total / len(val_loader)

        print(f" -> Epoca [{epoch}/{epochs}] | Train Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), checkpoint_path)

    print(f"\n[OK] Mejor Val Dice: {best_val_dice:.4f} | Checkpoint guardado en '{checkpoint_path}'")

    # ---- Evaluacion final en test set (volumen reservado por completo) ----
    print("\n[Paso 4] Evaluando en testing.tif (reservado, nunca visto en entrenamiento)...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    test_dice_total = 0.0
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            test_dice_total += dice_score(logits, masks)
    test_dice = test_dice_total / len(test_loader)
    print(f" -> Test Dice final: {test_dice:.4f}")

    # ---- Visualizacion cualitativa ----
    print("\n[Paso 5] Visualizando un ejemplo del test set...")
    sample_img, sample_mask = test_loader.dataset[0]
    with torch.no_grad():
        input_tensor = sample_img.unsqueeze(0).to(device)
        pred_logit = model(input_tensor)
        pred_prob = torch.sigmoid(pred_logit).squeeze().cpu().numpy()

    display_img = sample_img.squeeze().numpy()
    display_mask = sample_mask.squeeze().numpy().astype(np.uint32)
    binary_pred = (pred_prob > 0.5).astype(np.uint32)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(display_img, cmap='gray'); axes[0].set_title("1. Slice EM (Input)"); axes[0].axis('off')
    axes[1].imshow(display_mask, cmap='gray'); axes[1].set_title("2. Mascara Ground-Truth"); axes[1].axis('off')
    im2 = axes[2].imshow(pred_prob, cmap='magma'); axes[2].set_title("3. Prediccion U-Net"); axes[2].axis('off')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    output_png = os.path.join(data_dir, "unet_mitochondria_output.png")
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f" -> Figura guardada en '{output_png}'.")

    try:
        viewer = napari.Viewer(title="U-Net Mitocondrias - EPFL EM Hippocampus")
        viewer.add_image(display_img, name="1. Slice EM (Input)", colormap="gray")
        viewer.add_labels(display_mask, name="2. Mascara Ground-Truth", opacity=0.5)
        viewer.add_image(pred_prob, name="3. Prediccion (Probabilidad)", colormap="magma", opacity=0.7, visible=False)
        viewer.add_labels(binary_pred, name="4. Prediccion Binaria (>0.5)", opacity=0.5)
        print(" -> Napari activo. Cierra la ventana para terminar.")
        napari.run()
    except Exception as err:
        print(f" -> Napari no disponible en este entorno ({err}). Se usa solo la figura estatica.")

    print("\nEntrenamiento y evaluacion completados!")


if __name__ == "__main__":
    train_unet_mitochondria()