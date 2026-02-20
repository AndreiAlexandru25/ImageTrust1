#!/bin/bash
# ============================================================================
# ImageTrust v2.0 - Phase 1 Pipeline Launcher for Linux/macOS
# ============================================================================
#
# This script launches the Phase 1 pipeline with nice priority control
# to allow concurrent usage while processing runs in the background.
#
# Priority Levels (nice values):
#   - low: nice 15 (lowest priority, minimal CPU impact)
#   - belownormal: nice 10 (reduced priority)
#   - normal: nice 0 (default priority)
#
# Usage:
#   ./launch_phase1_linux.sh                    # Run with defaults
#   ./launch_phase1_linux.sh --priority low     # Run at low priority
#   ./launch_phase1_linux.sh --help             # Show all options
#
# ============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
PIPELINE_SCRIPT="${SCRIPT_DIR}/run_phase1_pipeline.py"

# Default configuration
INPUT_DIR="${PROJECT_ROOT}/data/train"
OUTPUT_BASE="${PROJECT_ROOT}/data/phase1"
PRIORITY="belownormal"
NUM_WORKERS=6
BATCH_SIZE=256
GPU_MEMORY=0.85
SKIP_SYNTHETIC=""
SKIP_EMBEDDING=""
NO_NIQE=""
WEBHOOK=""
BACKGROUND=""

# Show help
show_help() {
    echo ""
    echo -e "${CYAN}ImageTrust v2.0 - Phase 1 Pipeline Launcher${NC}"
    echo "============================================"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --priority LEVEL    Process priority (low, belownormal, normal)"
    echo "                      Default: belownormal"
    echo "  --input DIR         Input directory with training images"
    echo "                      Default: data/train"
    echo "  --output DIR        Output base directory"
    echo "                      Default: data/phase1"
    echo "  --workers N         Number of CPU workers (1-8)"
    echo "                      Default: 6"
    echo "  --batch_size N      GPU batch size"
    echo "                      Default: 256"
    echo "  --gpu_memory FRAC   GPU memory fraction (0.5-0.95)"
    echo "                      Default: 0.85"
    echo "  --skip_synthetic    Skip synthetic data generation"
    echo "  --skip_embedding    Skip embedding extraction"
    echo "  --no_niqe           Skip NIQE quality scoring"
    echo "  --webhook URL       Webhook URL for notifications"
    echo "  --background        Run in background with nohup"
    echo "  --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0"
    echo "  $0 --priority low"
    echo "  $0 --priority low --workers 4 --batch_size 128"
    echo "  $0 --priority low --background  # Run overnight"
    echo ""
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --priority)
            PRIORITY="$2"
            shift 2
            ;;
        --input)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --gpu_memory)
            GPU_MEMORY="$2"
            shift 2
            ;;
        --skip_synthetic)
            SKIP_SYNTHETIC="--skip_synthetic"
            shift
            ;;
        --skip_embedding)
            SKIP_EMBEDDING="--skip_embedding"
            shift
            ;;
        --no_niqe)
            NO_NIQE="--no_niqe"
            shift
            ;;
        --webhook)
            WEBHOOK="--webhook $2"
            shift 2
            ;;
        --background)
            BACKGROUND="yes"
            shift
            ;;
        --help)
            show_help
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            ;;
    esac
done

# Map priority to nice value
case $PRIORITY in
    low)
        NICE_VALUE=15
        ;;
    belownormal)
        NICE_VALUE=10
        ;;
    normal)
        NICE_VALUE=0
        ;;
    *)
        echo -e "${RED}Invalid priority: $PRIORITY${NC}"
        echo "Valid options: low, belownormal, normal"
        exit 1
        ;;
esac

# Banner
echo ""
echo -e "${CYAN}============================================================================${NC}"
echo -e "${CYAN}  IMAGETRUST v2.0 - PHASE 1 PIPELINE (LINUX/MACOS)${NC}"
echo -e "${CYAN}============================================================================${NC}"
echo ""

# Display configuration
echo -e "${YELLOW}Configuration:${NC}"
echo "  Priority:     $PRIORITY (nice $NICE_VALUE)"
echo "  Input:        $INPUT_DIR"
echo "  Output:       $OUTPUT_BASE"
echo "  CPU Workers:  $NUM_WORKERS"
echo "  GPU Batch:    $BATCH_SIZE"
echo "  GPU Memory:   ${GPU_MEMORY}"
echo "  Background:   ${BACKGROUND:-no}"
echo ""

# Validate input directory
if [[ ! -d "$INPUT_DIR" ]]; then
    echo -e "${RED}ERROR: Input directory not found: $INPUT_DIR${NC}"
    exit 1
fi

# Check Python availability
if ! command -v python &> /dev/null; then
    echo -e "${RED}ERROR: Python not found. Please ensure Python is installed.${NC}"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1)
echo -e "${GREEN}Python: $PYTHON_VERSION${NC}"

# Check CUDA availability
CUDA_INFO=$(python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "CUDA check failed")
echo -e "${GREEN}$CUDA_INFO${NC}"
echo ""

# Build command
CMD="python \"$PIPELINE_SCRIPT\" \
    --input_dir \"$INPUT_DIR\" \
    --output_base \"$OUTPUT_BASE\" \
    --priority $PRIORITY \
    --num_workers $NUM_WORKERS \
    --batch_size $BATCH_SIZE \
    --gpu_memory $GPU_MEMORY \
    $SKIP_SYNTHETIC \
    $SKIP_EMBEDDING \
    $NO_NIQE \
    $WEBHOOK"

# Remove extra spaces
CMD=$(echo $CMD | tr -s ' ')

echo -e "${GREEN}Starting pipeline at $PRIORITY priority (nice $NICE_VALUE)...${NC}"
echo ""

if [[ -n "$BACKGROUND" ]]; then
    # Run in background with nohup
    LOG_FILE="${OUTPUT_BASE}/phase1_pipeline.log"
    mkdir -p "$(dirname "$LOG_FILE")"

    echo -e "${CYAN}Running in background mode${NC}"
    echo "Log file: $LOG_FILE"
    echo ""

    nohup nice -n $NICE_VALUE bash -c "$CMD" > "$LOG_FILE" 2>&1 &
    PID=$!

    echo -e "${GREEN}Pipeline started with PID: $PID${NC}"
    echo ""
    echo "To monitor progress:"
    echo "  tail -f $LOG_FILE"
    echo ""
    echo "To stop the pipeline:"
    echo "  kill $PID"
    echo ""

    # Save PID to file
    echo $PID > "${OUTPUT_BASE}/pipeline.pid"

else
    # Run in foreground
    echo "Press Ctrl+C to interrupt. Progress will be saved and can be resumed."
    echo ""

    if [[ $NICE_VALUE -gt 0 ]]; then
        nice -n $NICE_VALUE bash -c "$CMD"
    else
        bash -c "$CMD"
    fi

    EXIT_CODE=$?

    if [[ $EXIT_CODE -eq 0 ]]; then
        echo ""
        echo -e "${GREEN}Pipeline completed successfully!${NC}"
    else
        echo ""
        echo -e "${YELLOW}Pipeline completed with errors (exit code: $EXIT_CODE)${NC}"
    fi

    exit $EXIT_CODE
fi
