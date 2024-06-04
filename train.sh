#!/bin/bash

# Define the paths to the Python training scripts
SCRIPTS=(
    "/Users/zainhazzouri/projects/Bachelor_Thesis/Models/UNet/Code/train.py"
    "/Users/zainhazzouri/projects/Bachelor_Thesis/Models/AttentionUNet/Code/train.py"
    "/Users/zainhazzouri/projects/Bachelor_Thesis/Models/R2UNet/Code/train.py"
    "/Users/zainhazzouri/projects/Bachelor_Thesis/Models/AttentionR2UNet/Code/train.py"
    "/Users/zainhazzouri/projects/Bachelor_Thesis/Models_delta/UNet/Code/train.py"
    "/Users/zainhazzouri/projects/Bachelor_Thesis/Models_delta/AttentionUNet/Code/train.py"
    "/Users/zainhazzouri/projects/Bachelor_Thesis/Models_delta/R2UNet/Code/train.py"
    "/Users/zainhazzouri/projects/Bachelor_Thesis/Models_delta/AttentionR2UNet/Code/train.py"
    "/Users/zainhazzouri/projects/Bachelor_Thesis/Models_delta_delta/UNet/Code/train.py"
    "/Users/zainhazzouri/projects/Bachelor_Thesis/Models_delta_delta/AttentionUNet/Code/train.py"
    "/Users/zainhazzouri/projects/Bachelor_Thesis/Models_delta_delta/R2UNet/Code/train.py"
    "/Users/zainhazzouri/projects/Bachelor_Thesis/Models_delta_delta/AttentionR2UNet/Code/train.py"
    "/Users/zainhazzouri/projects/Bachelor_Thesis/Models_LFCC/UNet/Code/train.py"
    "/Users/zainhazzouri/projects/Bachelor_Thesis/Models_LFCC/AttentionUNet/Code/train.py"
    "/Users/zainhazzouri/projects/Bachelor_Thesis/Models_LFCC/R2UNet/Code/train.py"
    "/Users/zainhazzouri/projects/Bachelor_Thesis/Models_LFCC/AttentionR2UNet/Code/train.py"
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
