## Parameter Grid for Speech-Music Classification Experiments

### Model Architectures
- U_Net
- AttentionUNet
- R2AttU_Net
- R2U_Net

### Input Parameters
- n_mfcc: [32, 80]
- length_in_seconds: [0.5, 1, 2, 3, 5, 10, 15, 30]
- batch_size: [32, 64]

### Feature Transformations
- MFCC
- delta
- delta-delta
- LFCC
- lfcc-delta
- lfcc-delta-delta

### Total Configurations Summary
- Total Models: 4
- MFCC variants: 2
- Time lengths: 8
- Feature types: 6
- Batch sizes: 2

Total unique combinations: 384 experiments
(4 models × 2 MFCC variants × 8 time lengths × 6 feature types × 2 batch sizes)

### Parameter Groups
1. Base UNet variations (Group 1-3)
   - Model: U_Net
   - n_mfcc: [32, 80]
   - length_in_seconds: 5
   - batch_size: 64
   - All transformations

2. Time Length Variations (Group 7-12)
   - Model: U_Net
   - n_mfcc: 32
   - length_in_seconds: [0.5, 1, 2, 3, 10, 15, 30]
   - batch_size: [32, 64]
   - All transformations

3. Architecture Variations (Group 4-6)
   - Model: [AttentionUNet, R2AttU_Net, R2U_Net]
   - n_mfcc: 32
   - length_in_seconds: 5
   - batch_size: 64
   - All transformations
