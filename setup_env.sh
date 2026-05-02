#!/bin/bash
# =============================================================================
# NCT Depth Motif - Setup del Entorno Virtual
# =============================================================================
#
# Este script configura el entorno Python virtual e instala las dependencias
# necesarias para ejecutar los experimentos NCT.
#
# QUÉ HACE:
#   - Crea entorno virtual en ./.venv (si no existe)
#   - Activa el entorno virtual
#   - Actualiza pip a la última versión
#   - Instala dependencias desde requirements.txt
#
# USO:
#   bash setup_env.sh
#   source .venv/bin/activate
#
# REQUISITOS PREVIOS:
#   - Python 3.9 o superior (3.10+ recomendado)
#   - pip disponible
#
# NOTAS:
#   - Para usar GPU, instalar PyTorch con CUDA manualmente:
#     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
#   - El entorno se crea solo una vez. Ejecutar de nuevo actualiza dependencias.
#
# =============================================================================

set -e

echo "=========================================="
echo "NCT Depth Motif - Setup de Entorno"
echo "=========================================="
echo ""

# Detectar versión de Python
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python detectado: $PYTHON_VERSION"

# Crear entorno virtual si no existe
if [ ! -d ".venv" ]; then
    echo "→ Creando entorno virtual en ./.venv..."
    python3 -m venv .venv
    echo "  ✓ Entorno creado"
else
    echo "→ Entorno virtual ya existe en ./.venv"
fi

# Activar entorno
echo "→ Activando entorno virtual..."
source .venv/bin/activate

# Actualizar pip
echo "→ Actualizando pip..."
pip install --upgrade pip -q

# Instalar dependencias
echo "→ Instalando dependencias desde requirements.txt..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "✓ Setup completado exitosamente"
echo "=========================================="
echo ""
echo "Para activar el entorno en el futuro, ejecutar:"
echo "  source .venv/bin/activate"
echo ""
echo "Verificar instalación:"
echo "  python3 -c \"import torch; print(f'PyTorch: {torch.__version__}')\""
echo ""
echo "Próximos pasos:"
echo "  1. Descargar NYU Depth V2 (ver dataset/README.md)"
echo "  2. Ejecutar: bash examples/run_grouped_split.sh"
