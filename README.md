# Deep_Learning_G2
Repositorio destinado al aprendizaje del uso de modelos de Deep Learning aplicado al análisis de bioimágenes.
Se incluyen los siguientes trabajos:

- Pipeline de Deep Learning para exploración, preparación, segmentación y entrenamiento de modelos U-Net utilizando imágenes biomédicas en formato `.tif`.
El proyecto utiliza PyTorch para el entrenamiento del modelo y Napari para la exploración y visualización interactiva de imágenes y máscaras.

## Integrantes G2
- Mamani Casas, Lucero
- Samillán García, Leonardo
- Mejía Barreto, Astrid
- Bazalar Gutierrez, Renzo
- Vega Jauregui, Enmanuel

## Configuración

### 1. Clonar el repositorio
```bash
git clone https://github.com/Charyzzz/Deep_Learning_G2.git

# Entrar en la carpeta
cd dlba-pucp
```

### 2. Crear el entorno mediante Conda
```bash
# Desde el archivo environment.yml
conda env create -f environment.yml

# Activar el entorno
conda activate dlba
```

### 3. Ejecutar códigos
```bash
# Reemplaza el nombre por el archivo a ejecutar:
python unetSegmentationpy.py
```
