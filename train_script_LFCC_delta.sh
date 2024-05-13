#!/bin/bash

# Define the paths to the Python training scripts
SCRIPTS=(
    "/Users/zainhazzouri/projects/Bachelor_Thesis/Models_LFCC_delta/UNet/Code/train.py"
    "/Users/zainhazzouri/projects/Bachelor_Thesis/Models_LFCC_delta/AttentionUNet/Code/train.py"
    "/Users/zainhazzouri/projects/Bachelor_Thesis/Models_LFCC_delta/R2UNet/Code/train.py"
    "/Users/zainhazzouri/projects/Bachelor_Thesis/Models_LFCC_delta/AttentionR2UNet/Code/train.py"
)

# Function to clear Python cache
clear_cache() {
    find . -type d -name "__pycache__" -exec rm -r {} +
    echo "Cache cleared."
}

# Run each script and clear cache afterwards
for script in "${SCRIPTS[@]}"; do
    echo "Running script: $script"
    python3 "$script" && echo "Script $script completed successfully."
    clear_cache
done

echo "All scripts have been executed."
