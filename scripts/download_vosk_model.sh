#!/bin/bash
set -e

# Vosk Hindi Model Download Script
# Downloads vosk-model-hi-0.22 if not present

MODEL_DIR="models"
MODEL_NAME="vosk-model-hi-0.22"
MODEL_ZIP="${MODEL_NAME}.zip"
MODEL_URL="https://alphacephei.com/vosk/models/${MODEL_ZIP}"
MODEL_PATH="${MODEL_DIR}/${MODEL_NAME}"

echo "🔍 Checking Vosk Hindi model..."

# Create models directory if it doesn't exist
mkdir -p "${MODEL_DIR}"

# Check if model already exists
if [ -d "${MODEL_PATH}" ]; then
    echo "✅ Vosk Hindi model already exists at ${MODEL_PATH}"
    
    # Verify model has required files
    if [ -d "${MODEL_PATH}/am" ] && [ -f "${MODEL_PATH}/conf/model.conf" ]; then
        echo "✅ Model validation passed"
        exit 0
    else
        echo "⚠️ Model directory exists but appears corrupted. Re-downloading..."
        rm -rf "${MODEL_PATH}"
    fi
fi

echo "📥 Downloading Vosk Hindi model (1.4GB)..."
echo "   URL: ${MODEL_URL}"
echo "   This may take a few minutes..."

# Download with progress bar
if command -v wget &> /dev/null; then
    wget -O "${MODEL_DIR}/${MODEL_ZIP}" "${MODEL_URL}" --progress=bar:force
elif command -v curl &> /dev/null; then
    curl -L "${MODEL_URL}" -o "${MODEL_DIR}/${MODEL_ZIP}" --progress-bar
else
    echo "❌ Neither wget nor curl found. Please install one."
    exit 1
fi

# Check download success
if [ ! -f "${MODEL_DIR}/${MODEL_ZIP}" ]; then
    echo "❌ Download failed"
    exit 1
fi

echo "📦 Extracting model..."
cd "${MODEL_DIR}"

# Extract based on available tool
if command -v unzip &> /dev/null; then
    unzip -q "${MODEL_ZIP}"
elif command -v python3 &> /dev/null; then
    python3 -m zipfile -e "${MODEL_ZIP}" .
else
    echo "❌ No unzip tool found"
    exit 1
fi

# Verify extraction
if [ ! -d "${MODEL_NAME}" ]; then
    echo "❌ Extraction failed"
    exit 1
fi

# 🔧 FIX: Handle nested folder issue (very common in Vosk zip)
if [ -d "${MODEL_NAME}/${MODEL_NAME}" ]; then
    echo "🔧 Fixing nested folder structure..."
    mv "${MODEL_NAME}/${MODEL_NAME}"/* "${MODEL_NAME}/"
    rm -rf "${MODEL_NAME}/${MODEL_NAME}"
fi

# Clean up zip file
rm -f "${MODEL_ZIP}"

# Return to root directory
cd ..

echo "✅ Vosk Hindi model downloaded and extracted successfully"
echo "   Location: ${MODEL_PATH}"

# ✅ FINAL verification (production-safe)
if [ -d "${MODEL_PATH}/am" ] && \
   [ -d "${MODEL_PATH}/graph" ] && \
   [ -f "${MODEL_PATH}/conf/model.conf" ];
then
    
    echo "✅ Model ready for use"
    
else
    echo "❌ Model verification failed"
    echo "Contents of ${MODEL_PATH}:"
    ls -lah "${MODEL_PATH}" || true
    exit 1
fi
