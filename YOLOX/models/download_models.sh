#!/bin/bash

# Default values
MODELS_URL="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0"
MODELS_DIR="models"
MODEL=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -d|--models_dir)
            MODELS_DIR="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./download_models.sh [-d|--models_dir <dir>] [-m|--model <s|m|l|x|darknet|nano|tiny|all>] [-h|--help]"
            exit 0
            ;;
        *)
            echo "Invalid argument: $1. Use -h or --help to see available options."
            exit 1
            ;;
    esac
done

# Download and extract dataset
if [[ $MODEL == "s" || $MODEL == "all" ]]; then
    echo "Downloading YOLOX-s model ..."
    wget -c "$MODELS_URL/yolox_s.pth" -P "$MODELS_DIR"
fi

if [[ $MODEL == "m" || $MODEL == "all" ]]; then
    echo "Downloading YOLOX-m model ..."
    wget -c "$MODELS_URL/yolox_m.pth" -P "$MODELS_DIR"
fi

if [[ $MODEL == "l" || $MODEL == "all" ]]; then
    echo "Downloading YOLOX-l model ..."
    wget -c "$MODELS_URL/yolox_l.pth" -P "$MODELS_DIR"
fi

if [[ $MODEL == "x" || $MODEL == "all" ]]; then
    echo "Downloading YOLOX-x model ..."
    wget -c "$MODELS_URL/yolox_x.pth" -P "$MODELS_DIR"
fi

if [[ $MODEL == "darknet" || $MODEL == "all" ]]; then
    echo "Downloading YOLOX-darknet model ..."
    wget -c "$MODELS_URL/yolox_darknet.pth" -P "$MODELS_DIR"
fi

if [[ $MODEL == "nano" || $MODEL == "all" ]]; then
    echo "Downloading YOLOX-nano model ..."
    wget -c "$MODELS_URL/yolox_nano.pth" -P "$MODELS_DIR"
fi

if [[ $MODEL == "tiny" || $MODEL == "all" ]]; then
    echo "Downloading YOLOX-tiny model ..."
    wget -c "$MODELS_URL/yolox_tiny.pth" -P "$MODELS_DIR"
fi

echo "Done downloading."