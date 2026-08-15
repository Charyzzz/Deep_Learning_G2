"""
===============================================================================
U-NET PARA SEGMENTACION DE MITOCONDRIAS (EPFL EM Hippocampus Dataset)
===============================================================================

FUNCIONES IMPLEMENTADAS:

    - Entrenamiento U-Net
    - Dice Loss
    - Tversky Loss
    - Loss total = Dice Loss + Tversky Loss
    - Train Dice
    - Validation Dice
    - Validation Loss
    - Early Stopping
    - Guardado del mejor modelo
    - Historial completo por epoca
    - Graficas de:
        * Train Loss
        * Validation Loss
        * Dice Loss
        * Tversky Loss
        * Dice Score
        * Learning Rate
    - Evaluacion final sobre test set
    - Test Dice
    - Test IoU
    - Visualizacion cualitativa:
        * TEM Grayscale
        * Ground Truth Overlay
        * Probability Heatmap
        * Prediccion Binaria
        * Mapa Diagnostico de Error
    - Visualizacion con Napari
===============================================================================
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import glob
import random
import numpy as np
# napari requiere entorno grafico (GUI), no disponible en Colab.
# Se importa de forma perezosa (try/except) mas abajo, solo si
# realmente se va a usar. Si corres esto en tu PC con GUI, puedes
# descomentar la siguiente linea:
# import napari
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# CONFIGURACION DEL EXPERIMENTO
# =============================================================================

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

TARGET_SIZE = (256, 256)

BATCH_SIZE = 8

VAL_FRACTION = 0.15

SEED = 42


# -----------------------------------------------------------------------------
# Entrenamiento
# -----------------------------------------------------------------------------

EPOCHS = 100

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4  # regularizacion L2 real (antes estaba en 0.0, sin efecto)


# -----------------------------------------------------------------------------
# Early Stopping
# -----------------------------------------------------------------------------

# Numero de epocas consecutivas sin mejora de Validation Dice
# antes de detener el entrenamiento.

PATIENCE = 10


# -----------------------------------------------------------------------------
# Threshold para Dice, IoU y prediccion
# -----------------------------------------------------------------------------

THRESHOLD = 0.5


# -----------------------------------------------------------------------------
# Pesos de las funciones de perdida
# -----------------------------------------------------------------------------

# Loss total:
#
# Loss = DICE_WEIGHT * DiceLoss
#        + TVERSKY_WEIGHT * TverskyLoss

DICE_WEIGHT = 1.0

TVERSKY_WEIGHT = 1.0


# -----------------------------------------------------------------------------
# Parametros de Tversky Loss
# -----------------------------------------------------------------------------

# Formula:
#
# Tversky Index =
#
#       TP
# -------------------------
# TP + alpha*FP + beta*FN
#
# Tversky Loss = 1 - Tversky Index
#
# Si beta > alpha:
#       se penalizan mas los FN
#
# Si alpha > beta:
#       se penalizan mas los FP
#
# Valores comunes:
#       alpha = 0.3
#       beta  = 0.7
#
# Aqui se prioriza ligeramente reducir falsos negativos.

TVERSKY_ALPHA = 0.3

TVERSKY_BETA = 0.7

TVERSKY_SMOOTH = 1.0


# =============================================================================
# CONFIGURACION DE CODIFICACION DE CONSOLA
# =============================================================================

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):

    try:

        sys.stdout.reconfigure(
            encoding='utf-8'
        )

    except Exception:

        pass


# =============================================================================
# PART 0: CARGA DE VOLUMENES TIFF MULTIPAGINA
# =============================================================================

def load_tiff_stack(path: str) -> np.ndarray:

    """
    Carga un TIFF multipagina como array:

        (n_slices, H, W)
    """

    try:

        import tifffile

        return tifffile.imread(path)

    except ImportError:

        from skimage import io

        return io.imread(path)


def find_tiff_file(
    data_dir: str,
    must_include,
    must_exclude=None
):

    """
    Busca un .tif/.tiff cuyo nombre contenga todas las palabras
    de must_include y ninguna de must_exclude.
    """

    must_exclude = must_exclude or []

    candidates = (
        glob.glob(
            os.path.join(
                data_dir,
                "*.tif"
            )
        )
        +
        glob.glob(
            os.path.join(
                data_dir,
                "*.tiff"
            )
        )
    )

    for path in candidates:

        name = os.path.basename(
            path
        ).lower()

        if (
            all(
                k in name
                for k in must_include
            )
            and
            not any(
                k in name
                for k in must_exclude
            )
        ):

            return path

    return None


def resolve_dataset_paths(
    data_dir: str
):

    """
    Ubica los cuatro archivos:

        training.tif
        training_groundtruth.tif
        testing.tif
        testing_groundtruth.tif
    """

    paths = {

        "train_vol":
        find_tiff_file(
            data_dir,
            ["train"],
            ["ground"]
        ),

        "train_gt":
        find_tiff_file(
            data_dir,
            ["train", "ground"]
        ),

        "test_vol":
        find_tiff_file(
            data_dir,
            ["test"],
            ["ground"]
        ),

        "test_gt":
        find_tiff_file(
            data_dir,
            ["test", "ground"]
        ),
    }

    missing = [
        key
        for key, value in paths.items()
        if value is None
    ]

    if missing:

        raise FileNotFoundError(

            f"No se encontraron estos archivos "
            f"en '{data_dir}': {missing}. "

            f"Verifica que training.tif, "
            f"training_groundtruth.tif, "
            f"testing.tif y "
            f"testing_groundtruth.tif "
            f"esten en esa carpeta."

        )

    return paths


# =============================================================================
# PART 1: DATASET PYTORCH
# =============================================================================

class MitochondriaEMDataset(Dataset):

    """
    Cada muestra es un slice 2D extraido de un volumen 3D.

    Imagen:
        Microscopía electronica

    Mascara:
        0 = fondo
        1 = mitocondria
    """

    def __init__(
        self,
        volume: np.ndarray,
        mask_volume: np.ndarray,
        indices,
        target_size=(256, 256),
        transform: bool = True
    ):

        assert volume.shape == mask_volume.shape, (
            "El volumen de imagen y el de mascara "
            "deben tener la misma forma."
        )

        self.volume = volume

        self.mask_volume = mask_volume

        self.indices = list(indices)

        self.target_size = target_size

        self.transform = transform

    def __len__(self) -> int:

        return len(
            self.indices
        )

    def _apply_synchronous_transforms(
        self,
        img_np,
        mask_np
    ):

        # ---------------------------------------------------------
        # Flip horizontal
        # ---------------------------------------------------------

        if random.random() > 0.5:

            img_np = np.fliplr(
                img_np
            )

            mask_np = np.fliplr(
                mask_np
            )

        # ---------------------------------------------------------
        # Flip vertical
        # ---------------------------------------------------------

        if random.random() > 0.5:

            img_np = np.flipud(
                img_np
            )

            mask_np = np.flipud(
                mask_np
            )

        # ---------------------------------------------------------
        # Rotacion
        # ---------------------------------------------------------

        if random.random() > 0.5:

            k = random.choice(
                [1, 2, 3]
            )

            img_np = np.rot90(
                img_np,
                k,
                axes=(0, 1)
            )

            mask_np = np.rot90(
                mask_np,
                k,
                axes=(0, 1)
            )

        return (
            img_np.copy(),
            mask_np.copy()
        )

    def __getitem__(
        self,
        idx: int
    ):

        slice_idx = self.indices[idx]

        # ---------------------------------------------------------
        # Imagen
        # ---------------------------------------------------------

        img_slice = self.volume[
            slice_idx
        ].astype(
            np.float32
        )

        # ---------------------------------------------------------
        # Mascara
        # ---------------------------------------------------------

        mask_slice = self.mask_volume[
            slice_idx
        ]

        # ---------------------------------------------------------
        # Normalizacion min-max
        # ---------------------------------------------------------

        img_min = float(
            img_slice.min()
        )

        img_max = float(
            img_slice.max()
        )

        img_norm = (
            (img_slice - img_min)
            /
            (img_max - img_min + 1e-8)
        )

        # ---------------------------------------------------------
        # Mascara binaria
        # ---------------------------------------------------------

        mask_bin = (
            mask_slice > 0
        ).astype(
            np.float32
        )

        # ---------------------------------------------------------
        # Resize
        # ---------------------------------------------------------

        img_pil = Image.fromarray(
            (
                img_norm * 255
            ).astype(
                np.uint8
            )
        ).resize(
            self.target_size,
            Image.BILINEAR
        )

        mask_pil = Image.fromarray(
            (
                mask_bin * 255
            ).astype(
                np.uint8
            )
        ).resize(
            self.target_size,
            Image.NEAREST
        )

        img_np = np.array(
            img_pil,
            dtype=np.float32
        ) / 255.0

        mask_np = (
            np.array(
                mask_pil,
                dtype=np.float32
            ) > 0
        ).astype(
            np.float32
        )

        # ---------------------------------------------------------
        # Data augmentation
        # ---------------------------------------------------------

        if self.transform:

            img_np, mask_np = (
                self._apply_synchronous_transforms(
                    img_np,
                    mask_np
                )
            )

        # ---------------------------------------------------------
        # Convertir a Tensor
        # ---------------------------------------------------------

        img_tensor = (
            torch.from_numpy(
                img_np
            )
            .unsqueeze(0)
            .float()
        )

        mask_tensor = (
            torch.from_numpy(
                mask_np
            )
            .unsqueeze(0)
            .float()
        )

        return (
            img_tensor,
            mask_tensor
        )


# =============================================================================
# PART 2: DATALOADERS
# =============================================================================

def build_dataloaders(
    data_dir: str,
    target_size=(256, 256),
    batch_size: int = 8,
    val_fraction: float = 0.15,
    seed: int = 42
):

    # ---------------------------------------------------------
    # Seeds
    # ---------------------------------------------------------

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )

    # ---------------------------------------------------------
    # Localizar archivos
    # ---------------------------------------------------------

    paths = resolve_dataset_paths(
        data_dir
    )

    print(
        f"[INFO] training.tif           -> "
        f"{paths['train_vol']}"
    )

    print(
        f"[INFO] training_groundtruth   -> "
        f"{paths['train_gt']}"
    )

    print(
        f"[INFO] testing.tif            -> "
        f"{paths['test_vol']}"
    )

    print(
        f"[INFO] testing_groundtruth    -> "
        f"{paths['test_gt']}"
    )

    # ---------------------------------------------------------
    # Cargar volumenes
    # ---------------------------------------------------------

    train_volume = load_tiff_stack(
        paths["train_vol"]
    )

    train_gt = load_tiff_stack(
        paths["train_gt"]
    )

    test_volume = load_tiff_stack(
        paths["test_vol"]
    )

    test_gt = load_tiff_stack(
        paths["test_gt"]
    )

    print(
        f"\n[INFO] Volumen de entrenamiento: "
        f"{train_volume.shape} slices"
    )

    print(
        f"[INFO] Volumen de testing: "
        f"{test_volume.shape} slices"
    )

    # ---------------------------------------------------------
    # Split train / validation
    # ---------------------------------------------------------

    n_train_slices = (
        train_volume.shape[0]
    )

    all_train_indices = list(
        range(
            n_train_slices
        )
    )

    random.shuffle(
        all_train_indices
    )

    n_val = max(
        1,
        int(
            val_fraction *
            n_train_slices
        )
    )

    val_indices = (
        all_train_indices[
            :n_val
        ]
    )

    train_indices = (
        all_train_indices[
            n_val:
        ]
    )

    test_indices = list(
        range(
            test_volume.shape[0]
        )
    )

    print(
        "\n Split del dataset:"
    )

    print(
        f"  • Train      : "
        f"{len(train_indices)} slices "
        f"(de training.tif)"
    )

    print(
        f"  • Validation : "
        f"{len(val_indices)} slices "
        f"(de training.tif)"
    )

    print(
        f"  • Test       : "
        f"{len(test_indices)} slices "
        f"(de testing.tif, reservado completo)"
    )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    train_ds = MitochondriaEMDataset(
        train_volume,
        train_gt,
        train_indices,
        target_size,
        transform=True
    )

    val_ds = MitochondriaEMDataset(
        train_volume,
        train_gt,
        val_indices,
        target_size,
        transform=False
    )

    test_ds = MitochondriaEMDataset(
        test_volume,
        test_gt,
        test_indices,
        target_size,
        transform=False
    )

    # ---------------------------------------------------------
    # DataLoaders
    # ---------------------------------------------------------

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )


# =============================================================================
# PART 3: BLOQUES DEL U-NET
# =============================================================================

class DoubleConv(nn.Module):

    """
    Bloque Conv->BN->LeakyReLU x2 con conexion residual:

        H(x) = F(x) + x

    Si in_channels != out_channels, el shortcut usa una
    conv 1x1 para igualar canales antes de sumar (idea de
    Clase 3: Residual Connections & Gradient Flow).

    dropout_p > 0 solo se usa en el bottleneck (down4) para
    regularizar sin afectar la capacidad de las capas tempranas.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_p: float = 0.0
    ):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(
            out_channels
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(
            out_channels
        )

        self.act = nn.LeakyReLU(
            negative_slope=0.01,
            inplace=True
        )

        # Shortcut: identidad si los canales coinciden,
        # o conv 1x1 + BN si hay que igualar canales.
        if in_channels != out_channels:

            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm2d(
                    out_channels
                )
            )

        else:

            self.shortcut = nn.Identity()

        self.dropout = (
            nn.Dropout2d(p=dropout_p)
            if dropout_p > 0.0
            else nn.Identity()
        )

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity  # conexion residual

        out = self.act(out)
        out = self.dropout(out)

        return out


class DownBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_p: float = 0.0
    ):

        super().__init__()

        self.maxpool_conv = nn.Sequential(

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            DoubleConv(
                in_channels,
                out_channels,
                dropout_p=dropout_p
            )
        )

    def forward(self, x):

        return self.maxpool_conv(x)


class UpBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ):

        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2
        )

        self.conv = DoubleConv(
            in_channels,
            out_channels
        )

    def forward(
        self,
        x1,
        x2
    ):

        x1 = self.up(
            x1
        )

        diff_y = (
            x2.size()[2]
            -
            x1.size()[2]
        )

        diff_x = (
            x2.size()[3]
            -
            x1.size()[3]
        )

        x1 = F.pad(
            x1,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2
            ]
        )

        x = torch.cat(
            [x2, x1],
            dim=1
        )

        return self.conv(
            x
        )


def init_weights(m):

    """
    Inicializacion Kaiming/He para las convoluciones,
    ajustada a la pendiente de LeakyReLU (a=0.01).

    BatchNorm se inicializa con weight=1, bias=0 (estandar).

    (Clase 3: "los valores iniciales de los pesos deben
    modificarse porque el objetivo es hallar el minimo
    de la funcion").
    """

    if isinstance(m, nn.Conv2d):

        nn.init.kaiming_normal_(
            m.weight,
            mode="fan_out",
            nonlinearity="leaky_relu",
            a=0.01
        )

        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.BatchNorm2d):

        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class UNet(nn.Module):

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1
    ):

        super().__init__()

        self.inc = DoubleConv(
            in_channels,
            64
        )

        self.down1 = DownBlock(
            64,
            128
        )

        self.down2 = DownBlock(
            128,
            256
        )

        self.down3 = DownBlock(
            256,
            512
        )

        self.down4 = DownBlock(
            512,
            1024,
            dropout_p=0.3  # dropout solo en el bottleneck (Clase 3: overfitting -> dropout p=0.3)
        )

        self.up1 = UpBlock(
            1024,
            512
        )

        self.up2 = UpBlock(
            512,
            256
        )

        self.up3 = UpBlock(
            256,
            128
        )

        self.up4 = UpBlock(
            128,
            64
        )

        self.outc = nn.Conv2d(
            64,
            out_channels,
            kernel_size=1
        )

        # Inicializacion Kaiming/He, ajustada a LeakyReLU
        # (Clase 3: "los valores iniciales de los pesos deben modificarse")
        self.apply(init_weights)

    def forward(self, x):

        x1 = self.inc(
            x
        )

        x2 = self.down1(
            x1
        )

        x3 = self.down2(
            x2
        )

        x4 = self.down3(
            x3
        )

        x5 = self.down4(
            x4
        )

        x = self.up1(
            x5,
            x4
        )

        x = self.up2(
            x,
            x3
        )

        x = self.up3(
            x,
            x2
        )

        x = self.up4(
            x,
            x1
        )

        return self.outc(
            x
        )


# =============================================================================
# PART 4: DICE LOSS
# =============================================================================

class DiceLoss(nn.Module):

    def __init__(
        self,
        smooth: float = 1.0
    ):

        super().__init__()

        self.smooth = smooth

    def forward(
        self,
        logits,
        targets
    ):

        probs = torch.sigmoid(
            logits
        )

        probs_flat = probs.view(
            -1
        )

        targets_flat = targets.view(
            -1
        )

        intersection = (
            probs_flat *
            targets_flat
        ).sum()

        dice_coeff = (
            2.0 * intersection
            +
            self.smooth
        ) / (
            probs_flat.sum()
            +
            targets_flat.sum()
            +
            self.smooth
        )

        return 1.0 - dice_coeff


# =============================================================================
# PART 5: TVERSKY LOSS
# =============================================================================

class TverskyLoss(nn.Module):

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1.0
    ):

        super().__init__()

        if alpha < 0 or beta < 0:

            raise ValueError(
                "alpha y beta deben ser >= 0."
            )

        self.alpha = alpha

        self.beta = beta

        self.smooth = smooth

    def forward(
        self,
        logits,
        targets
    ):

        # ---------------------------------------------------------
        # Probabilidades
        # ---------------------------------------------------------

        probs = torch.sigmoid(
            logits
        )

        # ---------------------------------------------------------
        # Flatten
        # ---------------------------------------------------------

        probs_flat = probs.view(
            -1
        )

        targets_flat = targets.view(
            -1
        )

        # ---------------------------------------------------------
        # True Positives
        # ---------------------------------------------------------

        true_positive = (
            probs_flat *
            targets_flat
        ).sum()

        # ---------------------------------------------------------
        # False Positives
        # ---------------------------------------------------------

        false_positive = (
            probs_flat *
            (1.0 - targets_flat)
        ).sum()

        # ---------------------------------------------------------
        # False Negatives
        # ---------------------------------------------------------

        false_negative = (
            (1.0 - probs_flat) *
            targets_flat
        ).sum()

        # ---------------------------------------------------------
        # Tversky Index
        # ---------------------------------------------------------

        tversky_index = (

            true_positive
            +
            self.smooth

        ) / (

            true_positive
            +
            self.alpha *
            false_positive
            +
            self.beta *
            false_negative
            +
            self.smooth

        )

        # ---------------------------------------------------------
        # Tversky Loss
        # ---------------------------------------------------------

        return 1.0 - tversky_index


# =============================================================================
# PART 6: DICE SCORE
# =============================================================================

def dice_score(
    logits,
    targets,
    threshold: float = 0.5,
    smooth: float = 1.0
) -> float:

    """
    Dice score para evaluar el modelo.

    No se utiliza para backpropagation.
    """

    probs = torch.sigmoid(
        logits
    )

    preds = (
        probs > threshold
    ).float()

    preds_flat = preds.view(
        -1
    )

    targets_flat = targets.view(
        -1
    )

    intersection = (
        preds_flat *
        targets_flat
    ).sum()

    dice = (
        2.0 * intersection
        +
        smooth
    ) / (
        preds_flat.sum()
        +
        targets_flat.sum()
        +
        smooth
    )

    return dice.item()


# =============================================================================
# PART 7: IoU SCORE
# =============================================================================

def iou_score(
    logits,
    targets,
    threshold: float = 0.5,
    smooth: float = 1.0
) -> float:

    """
    Intersection over Union (IoU).

    Formula:

                    TP
        IoU = --------------
                TP + FP + FN

    TN no afecta directamente al IoU.
    """

    probs = torch.sigmoid(
        logits
    )

    preds = (
        probs > threshold
    ).float()

    preds_flat = preds.view(
        -1
    )

    targets_flat = targets.view(
        -1
    )

    intersection = (
        preds_flat *
        targets_flat
    ).sum()

    union = (
        preds_flat
        +
        targets_flat
        -
        preds_flat *
        targets_flat
    ).sum()

    iou = (
        intersection
        +
        smooth
    ) / (
        union
        +
        smooth
    )

    return iou.item()


# =============================================================================
# PART 8: MAPA DIAGNOSTICO DE ERROR
# =============================================================================

def create_error_map(
    ground_truth,
    prediction
):

    """
    Crea un mapa RGB de errores:

        Verde  = True Positive (TP)
        Amarillo = False Positive (FP)
        Rojo   = False Negative (FN)
        Negro  = True Negative (TN)

    Entrada:
        ground_truth -> mascara binaria
        prediction   -> prediccion binaria

    Salida:
        RGB uint8
    """

    gt = (
        ground_truth > 0
    )

    pred = (
        prediction > 0
    )

    # ---------------------------------------------------------
    # Inicialmente todo es negro = TN
    # ---------------------------------------------------------

    error_map = np.zeros(
        (
            gt.shape[0],
            gt.shape[1],
            3
        ),
        dtype=np.uint8
    )

    # ---------------------------------------------------------
    # True Positive
    #
    # GT = 1
    # Pred = 1
    #
    # Verde
    # ---------------------------------------------------------

    tp = (
        gt &
        pred
    )

    error_map[tp] = [
        0,
        255,
        0
    ]

    # ---------------------------------------------------------
    # False Positive
    #
    # GT = 0
    # Pred = 1
    #
    # Amarillo
    # ---------------------------------------------------------

    fp = (
        ~gt &
        pred
    )

    error_map[fp] = [
        255,
        255,
        0
    ]

    # ---------------------------------------------------------
    # False Negative
    #
    # GT = 1
    # Pred = 0
    #
    # Rojo
    # ---------------------------------------------------------

    fn = (
        gt &
        ~pred
    )

    error_map[fn] = [
        255,
        0,
        0
    ]

    return error_map


# =============================================================================
# PART 9: GROUND TRUTH OVERLAY
# =============================================================================

def create_ground_truth_overlay(
    image,
    ground_truth,
    alpha=0.45
):

    """
    Superpone la mascara Ground Truth sobre la imagen TEM.

    La imagen base permanece en escala de grises.

    Las regiones de mitocondria Ground Truth se muestran
    en color cian.
    """

    # ---------------------------------------------------------
    # Convertir imagen grayscale a RGB
    # ---------------------------------------------------------

    image_uint8 = (
        np.clip(
            image,
            0,
            1
        ) * 255
    ).astype(
        np.uint8
    )

    rgb = np.stack(
        [
            image_uint8,
            image_uint8,
            image_uint8
        ],
        axis=-1
    ).astype(
        np.float32
    )

    # ---------------------------------------------------------
    # Mascara
    # ---------------------------------------------------------

    mask = (
        ground_truth > 0
    )

    # ---------------------------------------------------------
    # Color cyan
    # RGB = 0, 255, 255
    # ---------------------------------------------------------

    overlay_color = np.array(
        [
            0,
            255,
            255
        ],
        dtype=np.float32
    )

    rgb[mask] = (
        (1.0 - alpha) *
        rgb[mask]
        +
        alpha *
        overlay_color
    )

    return np.clip(
        rgb,
        0,
        255
    ).astype(
        np.uint8
    )


# =============================================================================
# PART 10: GRAFICAS DEL ENTRENAMIENTO
# =============================================================================

def plot_training_history(
    history,
    output_dir
):

    epochs = range(
        1,
        len(
            history["train_loss"]
        ) + 1
    )

    # =========================================================================
    # GRAFICA 1: LOSS TOTAL
    # =========================================================================

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train Loss",
        linewidth=2
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss",
        linewidth=2
    )

    if history["best_epoch"] is not None:

        best_epoch = (
            history["best_epoch"]
        )

        plt.axvline(
            best_epoch,
            linestyle="--",
            label=f"Best Epoch ({best_epoch})"
        )

    plt.xlabel(
        "Epoca"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Loss Total vs Epocas"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    path = os.path.join(
        output_dir,
        "training_validation_loss.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"[GRAFICA] Guardada: {path}"
    )

    # =========================================================================
    # GRAFICA 2: TVERSKY LOSS
    # =========================================================================

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        epochs,
        history["train_tversky_loss"],
        label="Train Tversky Loss",
        linewidth=2
    )

    plt.plot(
        epochs,
        history["val_tversky_loss"],
        label="Validation Tversky Loss",
        linewidth=2
    )

    plt.xlabel(
        "Epoca"
    )

    plt.ylabel(
        "Tversky Loss"
    )

    plt.title(
        "Tversky Loss vs Epocas"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    path = os.path.join(
        output_dir,
        "tversky_loss.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"[GRAFICA] Guardada: {path}"
    )

    # =========================================================================
    # GRAFICA 3: DICE LOSS
    # =========================================================================

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        epochs,
        history["train_dice_loss"],
        label="Train Dice Loss",
        linewidth=2
    )

    plt.plot(
        epochs,
        history["val_dice_loss"],
        label="Validation Dice Loss",
        linewidth=2
    )

    plt.xlabel(
        "Epoca"
    )

    plt.ylabel(
        "Dice Loss"
    )

    plt.title(
        "Dice Loss vs Epocas"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    path = os.path.join(
        output_dir,
        "dice_loss.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"[GRAFICA] Guardada: {path}"
    )

    # =========================================================================
    # GRAFICA 4: DICE SCORE
    # =========================================================================

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        epochs,
        history["train_dice"],
        label="Train Dice",
        linewidth=2
    )

    plt.plot(
        epochs,
        history["val_dice"],
        label="Validation Dice",
        linewidth=2
    )

    if history["best_epoch"] is not None:

        best_epoch = (
            history["best_epoch"]
        )

        plt.axvline(
            best_epoch,
            linestyle="--",
            label=f"Best Epoch ({best_epoch})"
        )

    plt.xlabel(
        "Epoca"
    )

    plt.ylabel(
        "Dice Score"
    )

    plt.title(
        "Dice Score vs Epocas"
    )

    plt.ylim(
        0,
        1.0
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    path = os.path.join(
        output_dir,
        "dice_score.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"[GRAFICA] Guardada: {path}"
    )

    # =========================================================================
    # GRAFICA 5: LEARNING RATE
    # =========================================================================

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        epochs,
        history["learning_rate"],
        linewidth=2
    )

    plt.xlabel(
        "Epoca"
    )

    plt.ylabel(
        "Learning Rate"
    )

    plt.title(
        "Learning Rate vs Epocas"
    )

    plt.yscale(
        "log"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    path = os.path.join(
        output_dir,
        "learning_rate.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"[GRAFICA] Guardada: {path}"
    )

    # =========================================================================
    # GRAFICA 6: RESUMEN
    # =========================================================================

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10)
    )

    # -------------------------------------------------------------------------
    # Loss total
    # -------------------------------------------------------------------------

    axes[0, 0].plot(
        epochs,
        history["train_loss"],
        label="Train Loss"
    )

    axes[0, 0].plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    axes[0, 0].set_title(
        "Loss Total"
    )

    axes[0, 0].set_xlabel(
        "Epoca"
    )

    axes[0, 0].set_ylabel(
        "Loss"
    )

    axes[0, 0].legend()

    axes[0, 0].grid(
        True,
        alpha=0.3
    )

    # -------------------------------------------------------------------------
    # Tversky
    # -------------------------------------------------------------------------

    axes[0, 1].plot(
        epochs,
        history["train_tversky_loss"],
        label="Train Tversky"
    )

    axes[0, 1].plot(
        epochs,
        history["val_tversky_loss"],
        label="Validation Tversky"
    )

    axes[0, 1].set_title(
        "Tversky Loss"
    )

    axes[0, 1].set_xlabel(
        "Epoca"
    )

    axes[0, 1].set_ylabel(
        "Tversky Loss"
    )

    axes[0, 1].legend()

    axes[0, 1].grid(
        True,
        alpha=0.3
    )

    # -------------------------------------------------------------------------
    # Dice Loss
    # -------------------------------------------------------------------------

    axes[1, 0].plot(
        epochs,
        history["train_dice_loss"],
        label="Train Dice Loss"
    )

    axes[1, 0].plot(
        epochs,
        history["val_dice_loss"],
        label="Validation Dice"
    )

    axes[1, 0].set_title(
        "Dice Loss"
    )

    axes[1, 0].set_xlabel(
        "Epoca"
    )

    axes[1, 0].set_ylabel(
        "Dice Loss"
    )

    axes[1, 0].legend()

    axes[1, 0].grid(
        True,
        alpha=0.3
    )

    # -------------------------------------------------------------------------
    # Dice Score
    # -------------------------------------------------------------------------

    axes[1, 1].plot(
        epochs,
        history["train_dice"],
        label="Train Dice"
    )

    axes[1, 1].plot(
        epochs,
        history["val_dice"],
        label="Validation Dice"
    )

    axes[1, 1].set_title(
        "Dice Score"
    )

    axes[1, 1].set_xlabel(
        "Epoca"
    )

    axes[1, 1].set_ylabel(
        "Dice"
    )

    axes[1, 1].set_ylim(
        0,
        1
    )

    axes[1, 1].legend()

    axes[1, 1].grid(
        True,
        alpha=0.3
    )

    plt.suptitle(
        "Historial de entrenamiento U-Net",
        fontsize=16
    )

    plt.tight_layout()

    path = os.path.join(
        output_dir,
        "training_history_summary.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"[GRAFICA] Guardada: {path}"
    )


# =============================================================================
# PART 11: ENTRENAMIENTO
# =============================================================================

def train_unet_mitochondria():

    print(
        "=========================================================="
    )

    print(
        " U-NET - SEGMENTACION DE MITOCONDRIAS "
        "(EPFL EM Hippocampus)"
    )

    print(
        "=========================================================="
    )

    # =========================================================================
    # DISPOSITIVO
    # =========================================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f" -> Dispositivo: {device}"
    )

    # =========================================================================
    # DIRECTORIO
    # =========================================================================

    # En Colab, los 4 archivos .tif viven en tu Google Drive, no
    # junto al script. Ajusta esta ruta a tu carpeta (la viste en
    # la captura: Mi unidad > DBLA).
    data_dir = "/content/drive/MyDrive/DBLA"

    # =========================================================================
    # CONFIGURACION
    # =========================================================================

    print(
        "\n=========================================================="
    )

    print(
        " CONFIGURACION DEL EXPERIMENTO"
    )

    print(
        "=========================================================="
    )

    print(
        f" Target size       : {TARGET_SIZE}"
    )

    print(
        f" Batch size        : {BATCH_SIZE}"
    )

    print(
        f" Epocas maximas    : {EPOCHS}"
    )

    print(
        f" Learning rate     : {LEARNING_RATE}"
    )

    print(
        f" Weight decay      : {WEIGHT_DECAY}"
    )

    print(
        f" Validation        : "
        f"{VAL_FRACTION * 100:.1f}%"
    )

    print(
        f" Early stopping    : "
        f"patience={PATIENCE}"
    )

    print(
        f" Threshold         : {THRESHOLD}"
    )

    print(
        f" Dice weight       : {DICE_WEIGHT}"
    )

    print(
        f" Tversky weight    : "
        f"{TVERSKY_WEIGHT}"
    )

    print(
        f" Tversky alpha     : "
        f"{TVERSKY_ALPHA}"
    )

    print(
        f" Tversky beta      : "
        f"{TVERSKY_BETA}"
    )

    print(
        "=========================================================="
    )

    # =========================================================================
    # DATA LOADERS
    # =========================================================================

    print(
        "\n[Paso 1] Cargando volumenes TIFF "
        "y construyendo DataLoaders..."
    )

    (
        train_loader,
        val_loader,
        test_loader
    ) = build_dataloaders(

        data_dir=data_dir,

        target_size=TARGET_SIZE,

        batch_size=BATCH_SIZE,

        val_fraction=VAL_FRACTION,

        seed=SEED
    )

    # =========================================================================
    # MODELO
    # =========================================================================

    print(
        "\n[Paso 2] Construyendo U-Net "
        "(in_channels=1, escala de grises)..."
    )

    model = UNet(
        in_channels=1,
        out_channels=1
    ).to(
        device
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f" -> Parametros entrenables: "
        f"{total_params:,}"
    )

    # =========================================================================
    # FUNCIONES DE PERDIDA
    # =========================================================================

    criterion_dice = DiceLoss()

    criterion_tversky = TverskyLoss(
        alpha=TVERSKY_ALPHA,
        beta=TVERSKY_BETA,
        smooth=TVERSKY_SMOOTH
    )

    print(
        "\n -> Funcion de perdida:"
    )

    print(
        "    Loss Total = "
        "Dice Loss + Tversky Loss"
    )

    # =========================================================================
    # OPTIMIZADOR
    # =========================================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Cosine Annealing: el LR baja suavemente de LEARNING_RATE a ~0
    # a lo largo de EPOCHS (Clase 3: "Loss Oscillations -> AdamW con
    # cosine annealing scheduler"). Asi el grafico de "Learning Rate"
    # que ya armaste deja de ser una linea plana.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6
    )

    # =========================================================================
    # CHECKPOINT
    # =========================================================================

    checkpoint_path = os.path.join(
        data_dir,
        "best_unet_mitochondria.pt"
    )

    # =========================================================================
    # HISTORIAL
    # =========================================================================

    history = {

        # Loss total
        "train_loss": [],
        "val_loss": [],

        # Tversky Loss
        "train_tversky_loss": [],
        "val_tversky_loss": [],

        # Dice Loss
        "train_dice_loss": [],
        "val_dice_loss": [],

        # Dice Score
        "train_dice": [],
        "val_dice": [],

        # Learning rate
        "learning_rate": [],

        # Mejor epoca
        "best_epoch": None
    }

    # =========================================================================
    # EARLY STOPPING
    # =========================================================================

    best_val_dice = 0.0

    best_epoch = 0

    epochs_without_improvement = 0

    # =========================================================================
    # ENTRENAMIENTO
    # =========================================================================

    print(
        "\n[Paso 3] Entrenamiento."
    )

    print(
        f"Epocas maximas: {EPOCHS}"
    )

    print(
        f"Early Stopping patience: {PATIENCE}"
    )

    print(
        "=========================================================="
    )

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        # =====================================================================
        # TRAIN
        # =====================================================================

        model.train()

        running_total_loss = 0.0

        running_tversky_loss = 0.0

        running_dice_loss = 0.0

        running_dice_score = 0.0

        for images, masks in train_loader:

            images = images.to(
                device
            )

            masks = masks.to(
                device
            )

            # -------------------------------------------------------------
            # Reset gradients
            # -------------------------------------------------------------

            optimizer.zero_grad()

            # -------------------------------------------------------------
            # Forward
            # -------------------------------------------------------------

            logits = model(
                images
            )

            # -------------------------------------------------------------
            # Dice Loss
            # -------------------------------------------------------------

            dice_loss = criterion_dice(
                logits,
                masks
            )

            # -------------------------------------------------------------
            # Tversky Loss
            # -------------------------------------------------------------

            tversky_loss = criterion_tversky(
                logits,
                masks
            )

            # -------------------------------------------------------------
            # Loss total
            # -------------------------------------------------------------

            loss = (

                DICE_WEIGHT *
                dice_loss

                +

                TVERSKY_WEIGHT *
                tversky_loss

            )

            # -------------------------------------------------------------
            # Backpropagation
            # -------------------------------------------------------------

            loss.backward()

            # -------------------------------------------------------------
            # Actualizacion de pesos
            # -------------------------------------------------------------

            optimizer.step()

            # -------------------------------------------------------------
            # Acumular metricas
            # -------------------------------------------------------------

            running_total_loss += (
                loss.item()
            )

            running_tversky_loss += (
                tversky_loss.item()
            )

            running_dice_loss += (
                dice_loss.item()
            )

            running_dice_score += dice_score(
                logits,
                masks,
                threshold=THRESHOLD
            )

        # =====================================================================
        # PROMEDIOS TRAIN
        # =====================================================================

        train_loss = (
            running_total_loss
            /
            len(train_loader)
        )

        train_tversky_loss = (
            running_tversky_loss
            /
            len(train_loader)
        )

        train_dice_loss = (
            running_dice_loss
            /
            len(train_loader)
        )

        train_dice = (
            running_dice_score
            /
            len(train_loader)
        )

        # =====================================================================
        # VALIDATION
        # =====================================================================

        model.eval()

        val_total_loss = 0.0

        val_tversky_loss_total = 0.0

        val_dice_loss_total = 0.0

        val_dice_total = 0.0

        with torch.no_grad():

            for images, masks in val_loader:

                images = images.to(
                    device
                )

                masks = masks.to(
                    device
                )

                logits = model(
                    images
                )

                # ---------------------------------------------------------
                # Validation Dice Loss
                # ---------------------------------------------------------

                dice_loss = criterion_dice(
                    logits,
                    masks
                )

                # ---------------------------------------------------------
                # Validation Tversky Loss
                # ---------------------------------------------------------

                tversky_loss = criterion_tversky(
                    logits,
                    masks
                )

                # ---------------------------------------------------------
                # Validation total loss
                # ---------------------------------------------------------

                loss = (

                    DICE_WEIGHT *
                    dice_loss

                    +

                    TVERSKY_WEIGHT *
                    tversky_loss

                )

                # ---------------------------------------------------------
                # Validation Dice
                # ---------------------------------------------------------

                current_dice = dice_score(
                    logits,
                    masks,
                    threshold=THRESHOLD
                )

                val_total_loss += (
                    loss.item()
                )

                val_tversky_loss_total += (
                    tversky_loss.item()
                )

                val_dice_loss_total += (
                    dice_loss.item()
                )

                val_dice_total += (
                    current_dice
                )

        # =====================================================================
        # PROMEDIOS VALIDATION
        # =====================================================================

        val_loss = (
            val_total_loss
            /
            len(val_loader)
        )

        val_tversky_loss = (
            val_tversky_loss_total
            /
            len(val_loader)
        )

        val_dice_loss = (
            val_dice_loss_total
            /
            len(val_loader)
        )

        val_dice = (
            val_dice_total
            /
            len(val_loader)
        )

        # =====================================================================
        # LEARNING RATE
        # =====================================================================

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        # Avanza el cosine annealing un paso por epoca
        # (se hace despues de guardar current_lr, para que el
        # historial refleje el LR realmente usado en esta epoca)
        scheduler.step()

        # =====================================================================
        # GUARDAR HISTORIAL
        # =====================================================================

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        history["train_tversky_loss"].append(
            train_tversky_loss
        )

        history["val_tversky_loss"].append(
            val_tversky_loss
        )

        history["train_dice_loss"].append(
            train_dice_loss
        )

        history["val_dice_loss"].append(
            val_dice_loss
        )

        history["train_dice"].append(
            train_dice
        )

        history["val_dice"].append(
            val_dice
        )

        history["learning_rate"].append(
            current_lr
        )

        # =====================================================================
        # MOSTRAR RESULTADOS
        # =====================================================================

        print(
            f"\nEpoca [{epoch}/{EPOCHS}]"
        )

        print(
            f"  Train Loss       : "
            f"{train_loss:.4f}"
        )

        print(
            f"  Train Tversky    : "
            f"{train_tversky_loss:.4f}"
        )

        print(
            f"  Train Dice Loss  : "
            f"{train_dice_loss:.4f}"
        )

        print(
            f"  Train Dice       : "
            f"{train_dice:.4f}"
        )

        print(
            f"  Val Loss         : "
            f"{val_loss:.4f}"
        )

        print(
            f"  Val Tversky      : "
            f"{val_tversky_loss:.4f}"
        )

        print(
            f"  Val Dice Loss    : "
            f"{val_dice_loss:.4f}"
        )

        print(
            f"  Val Dice         : "
            f"{val_dice:.4f}"
        )

        print(
            f"  Learning Rate    : "
            f"{current_lr:.8f}"
        )

        # =====================================================================
        # EARLY STOPPING
        # =====================================================================

        if val_dice > best_val_dice:

            best_val_dice = (
                val_dice
            )

            best_epoch = (
                epoch
            )

            epochs_without_improvement = 0

            history["best_epoch"] = (
                best_epoch
            )

            # -------------------------------------------------------------
            # Guardar checkpoint
            # -------------------------------------------------------------

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

            print(
                "  *** NUEVO MEJOR MODELO ***"
            )

            print(
                f"  Mejor Val Dice: "
                f"{best_val_dice:.4f}"
            )

            print(
                f"  Checkpoint: "
                f"{checkpoint_path}"
            )

        else:

            epochs_without_improvement += 1

            print(
                f"  Sin mejora de Val Dice: "
                f"{epochs_without_improvement}/"
                f"{PATIENCE}"
            )

        # =====================================================================
        # COMPROBAR EARLY STOPPING
        # =====================================================================

        if (
            epochs_without_improvement
            >=
            PATIENCE
        ):

            print(
                "\n=========================================================="
            )

            print(
                " EARLY STOPPING ACTIVADO"
            )

            print(
                "=========================================================="
            )

            print(
                f"No hubo mejora del Validation Dice "
                f"durante {PATIENCE} epocas consecutivas."
            )

            print(
                f"Mejor epoca: "
                f"{best_epoch}"
            )

            print(
                f"Mejor Validation Dice: "
                f"{best_val_dice:.4f}"
            )

            print(
                f"Entrenamiento detenido en la epoca: "
                f"{epoch}"
            )

            break

    # =========================================================================
    # INFORMACION FINAL DEL ENTRENAMIENTO
    # =========================================================================

    trained_epochs = len(
        history["train_loss"]
    )

    print(
        "\n=========================================================="
    )

    print(
        " ENTRENAMIENTO FINALIZADO"
    )

    print(
        "=========================================================="
    )

    print(
        f"Epocas ejecutadas : "
        f"{trained_epochs}"
    )

    print(
        f"Mejor epoca       : "
        f"{best_epoch}"
    )

    print(
        f"Mejor Val Dice    : "
        f"{best_val_dice:.4f}"
    )

    print(
        f"Checkpoint        : "
        f"{checkpoint_path}"
    )

    # =========================================================================
    # GRAFICAS
    # =========================================================================

    print(
        "\n[Paso 4] Generando graficas "
        "del entrenamiento..."
    )

    plot_training_history(
        history,
        data_dir
    )

    # =========================================================================
    # EVALUACION FINAL TEST
    # =========================================================================

    print(
        "\n[Paso 5] Evaluando en testing.tif "
        "(reservado, nunca visto en entrenamiento)..."
    )

    # -------------------------------------------------------------------------
    # Cargar mejor modelo
    # -------------------------------------------------------------------------

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device
        )
    )

    model.eval()

    test_dice_total = 0.0

    test_iou_total = 0.0

    # =========================================================================
    # TEST
    # =========================================================================

    with torch.no_grad():

        for images, masks in test_loader:

            images = images.to(
                device
            )

            masks = masks.to(
                device
            )

            logits = model(
                images
            )

            # -------------------------------------------------------------
            # Test Dice
            # -------------------------------------------------------------

            test_dice_total += dice_score(
                logits,
                masks,
                threshold=THRESHOLD
            )

            # -------------------------------------------------------------
            # Test IoU
            # -------------------------------------------------------------

            test_iou_total += iou_score(
                logits,
                masks,
                threshold=THRESHOLD
            )

    test_dice = (
        test_dice_total
        /
        len(test_loader)
    )

    test_iou = (
        test_iou_total
        /
        len(test_loader)
    )

    # =========================================================================
    # RESULTADOS TEST
    # =========================================================================

    print(
        "\n=========================================================="
    )

    print(
        " RESULTADOS FINALES EN TEST"
    )

    print(
        "=========================================================="
    )

    print(
        f" Test Dice : "
        f"{test_dice:.4f}"
    )

    print(
        f" Test IoU  : "
        f"{test_iou:.4f}"
    )

    print(
        "=========================================================="
    )

    # =========================================================================
    # VISUALIZACION CUALITATIVA
    # =========================================================================

    print(
        "\n[Paso 6] Generando visualizacion cualitativa..."
    )

    # -------------------------------------------------------------------------
    # Obtener muestra
    # -------------------------------------------------------------------------

    sample_img, sample_mask = (
        test_loader.dataset[0]
    )

    # -------------------------------------------------------------------------
    # Prediccion
    # -------------------------------------------------------------------------

    with torch.no_grad():

        input_tensor = (
            sample_img
            .unsqueeze(0)
            .to(device)
        )

        pred_logit = model(
            input_tensor
        )

        pred_prob = (
            torch.sigmoid(
                pred_logit
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    # -------------------------------------------------------------------------
    # Convertir a NumPy
    # -------------------------------------------------------------------------

    display_img = (
        sample_img
        .squeeze()
        .numpy()
    )

    display_mask = (
        sample_mask
        .squeeze()
        .numpy()
        .astype(
            np.uint8
        )
    )

    binary_pred = (
        pred_prob > THRESHOLD
    ).astype(
        np.uint8
    )

    # =========================================================================
    # GROUND TRUTH OVERLAY
    # =========================================================================

    gt_overlay = create_ground_truth_overlay(
        display_img,
        display_mask,
        alpha=0.45
    )

    # =========================================================================
    # MAPA DIAGNOSTICO
    # =========================================================================

    error_map = create_error_map(
        display_mask,
        binary_pred
    )

    # =========================================================================
    # FIGURA FINAL DE 5 PANELES
    # =========================================================================

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(22, 5)
    )

    # -------------------------------------------------------------------------
    # 1. TEM GRAYSCALE
    # -------------------------------------------------------------------------

    axes[0].imshow(
        display_img,
        cmap="gray"
    )

    axes[0].set_title(
        "1. TEM Grayscale"
    )

    axes[0].axis(
        "off"
    )

    # -------------------------------------------------------------------------
    # 2. GROUND TRUTH OVERLAY
    # -------------------------------------------------------------------------

    axes[1].imshow(
        gt_overlay
    )

    axes[1].set_title(
        "2. Ground Truth Overlay"
    )

    axes[1].axis(
        "off"
    )

    # -------------------------------------------------------------------------
    # 3. PROBABILITY HEATMAP
    # -------------------------------------------------------------------------

    im = axes[2].imshow(
        pred_prob,
        cmap="magma",
        vmin=0,
        vmax=1
    )

    axes[2].set_title(
        "3. Probability Heatmap"
    )

    axes[2].axis(
        "off"
    )

    cbar = fig.colorbar(
        im,
        ax=axes[2],
        fraction=0.046,
        pad=0.04
    )

    cbar.set_label(
        "Probabilidad"
    )

    # -------------------------------------------------------------------------
    # 4. PREDICCION BINARIA
    # -------------------------------------------------------------------------

    axes[3].imshow(
        binary_pred,
        cmap="gray"
    )

    axes[3].set_title(
        f"4. Predicción Binaria\n"
        f"(Threshold = {THRESHOLD})"
    )

    axes[3].axis(
        "off"
    )

    # -------------------------------------------------------------------------
    # 5. MAPA DIAGNOSTICO
    # -------------------------------------------------------------------------

    axes[4].imshow(
        error_map
    )

    axes[4].set_title(
        "5. Mapa Diagnóstico de Error"
    )

    axes[4].axis(
        "off"
    )

    # -------------------------------------------------------------------------
    # Leyenda del mapa diagnostico
    # -------------------------------------------------------------------------

    from matplotlib.patches import Patch

    legend_elements = [

        Patch(
            facecolor="green",
            label="TP"
        ),

        Patch(
            facecolor="yellow",
            label="FP"
        ),

        Patch(
            facecolor="red",
            label="FN"
        ),

        Patch(
            facecolor="black",
            label="TN"
        )

    ]

    axes[4].legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        fontsize=8
    )

    # -------------------------------------------------------------------------
    # Titulo general
    # -------------------------------------------------------------------------

    plt.suptitle(
        "U-Net - Segmentación de Mitocondrias",
        fontsize=16
    )

    plt.tight_layout()

    # =========================================================================
    # GUARDAR FIGURA
    # =========================================================================

    output_png = os.path.join(
        data_dir,
        "unet_mitochondria_qualitative_results.png"
    )

    plt.savefig(
        output_png,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    print(
        f" -> Figura cualitativa guardada en: "
        f"{output_png}"
    )

    # =========================================================================
    # NAPARI
    # =========================================================================

    try:

        import napari  # import perezoso: solo si hay entorno grafico

        viewer = napari.Viewer(
            title=(
                "U-Net Mitocondrias - "
                "EPFL EM Hippocampus"
            )
        )

        # ---------------------------------------------------------------------
        # Imagen TEM
        # ---------------------------------------------------------------------

        viewer.add_image(
            display_img,
            name="1. TEM Grayscale",
            colormap="gray"
        )

        # ---------------------------------------------------------------------
        # Ground Truth
        # ---------------------------------------------------------------------

        viewer.add_labels(
            display_mask,
            name="2. Ground Truth",
            opacity=0.5
        )

        # ---------------------------------------------------------------------
        # Probability
        # ---------------------------------------------------------------------

        viewer.add_image(
            pred_prob,
            name="3. Probability Heatmap",
            colormap="magma",
            opacity=0.7,
            visible=False
        )

        # ---------------------------------------------------------------------
        # Prediccion binaria
        # ---------------------------------------------------------------------

        viewer.add_labels(
            binary_pred,
            name="4. Prediccion Binaria",
            opacity=0.5
        )

        # ---------------------------------------------------------------------
        # Mapa diagnostico
        # ---------------------------------------------------------------------

        viewer.add_image(
            error_map,
            name="5. Mapa Diagnostico TP-FP-FN",
            rgb=True,
            visible=False
        )

        print(
            " -> Napari activo. "
            "Cierra la ventana para terminar."
        )

        napari.run()

    except Exception as err:

        print(
            " -> Napari no disponible en este "
            f"entorno ({err}). "
            "Se usa solamente la figura estatica."
        )

    # =========================================================================
    # FINAL
    # =========================================================================

    print(
        "\n=========================================================="
    )

    print(
        " ENTRENAMIENTO Y EVALUACION COMPLETADOS"
    )

    print(
        "=========================================================="
    )

    print(
        f" Mejor Validation Dice : "
        f"{best_val_dice:.4f}"
    )

    print(
        f" Test Dice             : "
        f"{test_dice:.4f}"
    )

    print(
        f" Test IoU              : "
        f"{test_iou:.4f}"
    )

    print(
        f" Mejor epoca           : "
        f"{best_epoch}"
    )

    print(
        f" Epocas ejecutadas     : "
        f"{trained_epochs}"
    )

    print(
        "=========================================================="
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    train_unet_mitochondria()
