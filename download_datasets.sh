#!/bin/bash
#############################################################################
# OpenUni Dataset Download Script
#
# Purpose: Download and prepare all datasets for OpenUni/OpenUni2 training
# Author: Auto-generated based on OpenUni documentation
# Date: 2026-03-22
#
# Usage:
#   bash download_datasets.sh [options]
#
# Options:
#   --stage1    Download priority datasets (BLIP3o-60k, text-to-image-2M, megalith-10m)
#   --stage2    Download extended datasets (cc12m-wds)
#   --laion     Download laion6m (requires img2dataset tool)
#   --all       Download all datasets
#   --extract   Extract all tar files after download
#
#############################################################################

set -e  # Exit on error

# Configuration
BASE_DIR="/gemini/code/OpenUni"
DATA_DIR="${BASE_DIR}/data"
NUM_EXTRACT_PROCESSES=8  # Parallel processes for extraction

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_disk_space() {
    local required_gb=$1
    local available_gb=$(df -BG "${DATA_DIR}" | awk 'NR==2 {print $4}' | sed 's/G//')

    if [ "$available_gb" -lt "$required_gb" ]; then
        log_error "Insufficient disk space. Required: ${required_gb}GB, Available: ${available_gb}GB"
        return 1
    fi
    log_info "Disk space check passed. Available: ${available_gb}GB"
    return 0
}

create_extract_script() {
    local target_dir=$1
    local script_name="${target_dir}/extract.py"

    cat > "$script_name" << 'EOF'
import multiprocessing as mp
import argparse
import os
from tqdm import tqdm
from glob import glob
import subprocess


def single_process(tar_list,):
    for tar_file in tqdm(tar_list):
        folder = tar_file[:-4]
        os.makedirs(folder, exist_ok=True)
        subprocess.run(["tar", "-xf", tar_file, "-C", folder, "--no-same-owner"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=-1, type=int)
    parser.add_argument('--num-processes', default=8, type=int)
    args = parser.parse_args()

    tar_files = sorted(glob(f'*.tar'))

    if args.end == -1:
        args.end = len(tar_files)

    tar_files = tar_files[args.start:args.end]

    num_tars = len(tar_files)
    num_processes = args.num_processes
    num_tars_per_process = num_tars // num_processes
    res = num_tars % num_processes
    if res > 0:
        num_processes += 1

    processes = [mp.Process(target=single_process,
                            args=(tar_files[process_id * num_tars_per_process:
                                            (process_id + 1) * num_tars_per_process]
                                  if process_id < num_processes - 1
                                  else tar_files[process_id * num_tars_per_process:],
                                  ))
                 for process_id in range(num_processes)]

    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()
EOF

    log_info "Created extraction script: $script_name"
}

create_cc12m_extract_script() {
    local target_dir=$1
    local script_name="${target_dir}/extract.py"

    cat > "$script_name" << 'EOF'
import multiprocessing as mp
import argparse
import os
from tqdm import tqdm
from glob import glob
import subprocess


def single_process(tar_list,):
    for tar_file in tqdm(tar_list):
        folder = tar_file[:-4]
        folder = folder.split('-')[-1]  # CC12M specific
        os.makedirs(folder, exist_ok=True)
        subprocess.run(["tar", "-xf", tar_file, "-C", folder, "--no-same-owner"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=-1, type=int)
    parser.add_argument('--num-processes', default=8, type=int)
    args = parser.parse_args()

    tar_files = sorted(glob(f'*.tar'))

    if args.end == -1:
        args.end = len(tar_files)

    tar_files = tar_files[args.start:args.end]

    num_tars = len(tar_files)
    num_processes = args.num_processes
    num_tars_per_process = num_tars // num_processes
    res = num_tars % num_processes
    if res > 0:
        num_processes += 1

    processes = [mp.Process(target=single_process,
                            args=(tar_files[process_id * num_tars_per_process:
                                            (process_id + 1) * num_tars_per_process]
                                  if process_id < num_processes - 1
                                  else tar_files[process_id * num_tars_per_process:],
                                  ))
                 for process_id in range(num_processes)]

    for p in processes:
        p.start()

    for p in processes:
        p.join()
EOF

    log_info "Created CC12M-specific extraction script: $script_name"
}

#############################################################################
# Dataset Download Functions
#############################################################################

download_blip3o_60k() {
    log_info "=========================================="
    log_info "Downloading BLIP3o-60k (Finetune dataset)"
    log_info "=========================================="

    mkdir -p "${DATA_DIR}/BLIP3o-60k"

    # Download images and captions
    log_info "Downloading raw images and annotations..."
    huggingface-cli download BLIP3o/BLIP3o-60k \
        --local-dir "${DATA_DIR}/BLIP3o-60k/raw" \
        --repo-type dataset

    # Download metadata json
    log_info "Downloading metadata json files..."
    huggingface-cli download wusize/BLIP3o-60k \
        --local-dir "${DATA_DIR}/BLIP3o-60k" \
        --repo-type dataset \
        --include "*.json"

    log_info "✓ BLIP3o-60k download completed"
}

extract_blip3o_60k() {
    log_info "Extracting BLIP3o-60k tar files..."

    cd "${DATA_DIR}/BLIP3o-60k/raw"
    create_extract_script "."

    python extract.py --num-processes ${NUM_EXTRACT_PROCESSES}

    log_info "✓ BLIP3o-60k extraction completed"
}

download_text2image_2m() {
    log_info "=========================================="
    log_info "Downloading text-to-image-2M"
    log_info "=========================================="

    mkdir -p "${DATA_DIR}/text-to-image-2M"

    # Download images and captions
    log_info "Downloading raw images (this may take several hours)..."
    huggingface-cli download jackyhate/text-to-image-2M \
        --local-dir "${DATA_DIR}/text-to-image-2M/raw" \
        --repo-type dataset

    # Download metadata
    log_info "Downloading metadata json files..."
    huggingface-cli download wusize/text-to-image-2M \
        --local-dir "${DATA_DIR}/text-to-image-2M/data" \
        --repo-type dataset

    log_info "✓ text-to-image-2M download completed"
}

extract_text2image_2m() {
    log_info "Extracting text-to-image-2M tar files..."

    # Extract 1024px data (single tar)
    log_info "Extracting 1024px dataset..."
    cd "${DATA_DIR}/text-to-image-2M/raw/data_1024_10K"
    mkdir -p data_000000
    tar -xf data_000000.tar -C data_000000 --no-same-owner

    # Extract 512px data (multiple tars)
    log_info "Extracting 512px dataset..."
    cd "${DATA_DIR}/text-to-image-2M/raw/data_512_2M"
    create_extract_script "."
    python extract.py --num-processes ${NUM_EXTRACT_PROCESSES}

    log_info "✓ text-to-image-2M extraction completed"
}

download_megalith_10m() {
    log_info "=========================================="
    log_info "Downloading megalith-10m"
    log_info "=========================================="

    mkdir -p "${DATA_DIR}/megalith-10m"

    # Download images
    log_info "Downloading raw images (this may take many hours)..."
    huggingface-cli download drawthingsai/megalith-10m \
        --local-dir "${DATA_DIR}/megalith-10m/raw" \
        --repo-type dataset

    # Download metadata and captions
    log_info "Downloading metadata and captions..."
    huggingface-cli download wusize/megalith-10m \
        --local-dir "${DATA_DIR}/megalith-10m" \
        --repo-type dataset

    log_info "✓ megalith-10m download completed"
}

extract_megalith_10m() {
    log_info "Extracting megalith-10m tar files..."

    # Extract images
    log_info "Extracting images..."
    cd "${DATA_DIR}/megalith-10m/raw"
    create_extract_script "."
    python extract.py --num-processes ${NUM_EXTRACT_PROCESSES}

    # Extract captions
    log_info "Extracting captions..."
    cd "${DATA_DIR}/megalith-10m/captions"
    create_extract_script "."
    python extract.py --num-processes ${NUM_EXTRACT_PROCESSES}

    log_info "✓ megalith-10m extraction completed"
}

download_cc12m_wds() {
    log_info "=========================================="
    log_info "Downloading CC12M-WDS"
    log_info "=========================================="

    mkdir -p "${DATA_DIR}/cc12m"

    # Download images
    log_info "Downloading raw images (this will take a very long time)..."
    huggingface-cli download pixparse/cc12m-wds \
        --local-dir "${DATA_DIR}/cc12m/raw" \
        --repo-type dataset

    # Download captions
    log_info "Downloading re-captioned annotations..."
    huggingface-cli download wusize/cc12m_recap \
        --local-dir "${DATA_DIR}/cc12m" \
        --repo-type dataset

    log_info "✓ CC12M-WDS download completed"
}

extract_cc12m_wds() {
    log_info "Extracting CC12M-WDS tar files..."

    # Extract images (using CC12M-specific script)
    log_info "Extracting images..."
    cd "${DATA_DIR}/cc12m/raw"
    create_cc12m_extract_script "."
    python extract.py --num-processes ${NUM_EXTRACT_PROCESSES}

    # Extract captions
    log_info "Extracting captions..."
    cd "${DATA_DIR}/cc12m/captions"
    create_extract_script "."
    python extract.py --num-processes ${NUM_EXTRACT_PROCESSES}

    log_info "✓ CC12M-WDS extraction completed"
}

download_laion6m() {
    log_info "=========================================="
    log_info "Downloading LAION6M"
    log_info "=========================================="

    log_warn "LAION6M requires img2dataset tool to download images from URLs"
    log_warn "This process may take 24+ hours and some URLs may fail"

    mkdir -p "${DATA_DIR}/laion6m"

    # Download URLs and captions (parquet files)
    log_info "Downloading URL list and captions..."
    huggingface-cli download wusize/laion6m_recap \
        --local-dir "${DATA_DIR}/laion6m/parquets" \
        --repo-type dataset

    log_info "✓ LAION6M parquet files downloaded"

    # Check if img2dataset is installed
    if ! command -v img2dataset &> /dev/null; then
        log_warn "img2dataset not found. Please install it:"
        log_warn "  pip install img2dataset"
        log_warn "Then run the following command manually:"
        echo ""
        echo "img2dataset --url_list ${DATA_DIR}/laion6m/parquets \\"
        echo "            --input_format 'parquet' \\"
        echo "            --output_folder ${DATA_DIR}/laion6m/raw \\"
        echo "            --processes_count 16 \\"
        echo "            --thread_count 64 \\"
        echo "            --image_size 512 \\"
        echo "            --resize_mode keep_ratio \\"
        echo "            --min_image_size 256"
        echo ""
        return 0
    fi

    # Download images using img2dataset
    log_info "Downloading images from URLs using img2dataset..."
    img2dataset --url_list "${DATA_DIR}/laion6m/parquets" \
                --input_format "parquet" \
                --output_folder "${DATA_DIR}/laion6m/raw" \
                --processes_count 16 \
                --thread_count 64 \
                --image_size 512 \
                --resize_mode keep_ratio \
                --min_image_size 256

    log_info "✓ LAION6M download completed"
}

#############################################################################
# Main execution logic
#############################################################################

print_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --stage1    Download priority datasets (BLIP3o-60k, text-to-image-2M, megalith-10m)"
    echo "  --stage2    Download extended datasets (cc12m-wds)"
    echo "  --laion     Download laion6m (requires img2dataset tool)"
    echo "  --all       Download all datasets"
    echo "  --extract   Extract all tar files after download"
    echo "  --help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --stage1                # Download priority datasets only"
    echo "  $0 --stage1 --extract      # Download and extract priority datasets"
    echo "  $0 --all --extract         # Download and extract everything"
}

# Parse command line arguments
STAGE1=false
STAGE2=false
LAION=false
ALL=false
EXTRACT=false

if [ $# -eq 0 ]; then
    print_usage
    exit 0
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --stage1)
            STAGE1=true
            shift
            ;;
        --stage2)
            STAGE2=true
            shift
            ;;
        --laion)
            LAION=true
            shift
            ;;
        --all)
            ALL=true
            shift
            ;;
        --extract)
            EXTRACT=true
            shift
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Set flags based on --all option
if [ "$ALL" = true ]; then
    STAGE1=true
    STAGE2=true
    LAION=true
fi

# Create base data directory
mkdir -p "${DATA_DIR}"
cd "${BASE_DIR}"

# Check HuggingFace authentication
log_info "Checking HuggingFace authentication..."
if ! huggingface-cli whoami &> /dev/null; then
    log_warn "Not logged in to HuggingFace. Some datasets may require authentication."
    log_warn "Run: huggingface-cli login"
fi

log_info "Starting dataset download process..."
log_info "Target directory: ${DATA_DIR}"

# Stage 1: Priority datasets
if [ "$STAGE1" = true ]; then
    log_info "=========================================="
    log_info "STAGE 1: Priority Datasets"
    log_info "=========================================="

    # Check disk space (approximately 1.5 TB needed for stage 1)
    check_disk_space 1500 || exit 1

    # Download datasets
    download_blip3o_60k
    download_text2image_2m
    download_megalith_10m

    # Extract if requested
    if [ "$EXTRACT" = true ]; then
        log_info "=========================================="
        log_info "Extracting Stage 1 datasets..."
        log_info "=========================================="

        extract_blip3o_60k
        extract_text2image_2m
        extract_megalith_10m
    fi
fi

# Stage 2: Extended datasets
if [ "$STAGE2" = true ]; then
    log_info "=========================================="
    log_info "STAGE 2: Extended Datasets"
    log_info "=========================================="

    # Check disk space (approximately 1 TB more needed)
    check_disk_space 1000 || exit 1

    download_cc12m_wds

    if [ "$EXTRACT" = true ]; then
        extract_cc12m_wds
    fi
fi

# LAION6M (special handling)
if [ "$LAION" = true ]; then
    log_info "=========================================="
    log_info "LAION6M Dataset"
    log_info "=========================================="

    # Check disk space (approximately 600 GB needed)
    check_disk_space 600 || exit 1

    download_laion6m
fi

# Final summary
log_info "=========================================="
log_info "Download process completed!"
log_info "=========================================="

log_info "Summary:"
log_info "  Data directory: ${DATA_DIR}"
log_info "  Stage 1 completed: $STAGE1"
log_info "  Stage 2 completed: $STAGE2"
log_info "  LAION6M completed: $LAION"
log_info "  Extraction performed: $EXTRACT"

log_info ""
log_info "Next steps:"
log_info "  1. Verify data integrity"
log_info "  2. Check dataset statistics"
log_info "  3. Configure training config files"
log_info "  4. Start training!"

exit 0
