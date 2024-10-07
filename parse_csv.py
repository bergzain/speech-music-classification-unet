import pandas as pd

# Data for all 11 papers with additional details
data = {
    "Paper": [
        "Comparative analysis of audio classification with MFCC and STFT features (2024)",
        "Enhancing Audio Classification Through MFCC Feature Extraction (2024)",
        "Speech/Music Classification Using Block Based MFCC Features (2012)",
        "Music Classification based on MFCC Variants (2012)",
        "Speech/Music Classification Using MFCC and KNN (2017)",
        "Knowledge Transfer from Neural Networks for Speech-Music Classification (2021)",
        "Audio Classification in Speech and Music (1999)",
        "MFCC Based Audio Classification Using Machine Learning (2021)",
        "Speech/Music Classification Using Phase-Based and Magnitude-Based Features (2022)",
        "Speech/Music Classification using MFCC and KNN (2017)"
    ],
    "Dataset": [
        "UrbanSound8K (8732 samples), Sound Event Audio Dataset (1288 samples)",
        "FSD-Kaggle (300 files), ESC-50 (2000 files, 50 classes)",
        "GTZAN Music/Speech Dataset (128 audio files, 64 music, 64 speech)",
        "Custom music database (837 audio files, 30-45s each)",
        "Custom dataset of 600 audio clips (300 music, 300 speech, 1-10s each)",
        "MUSAN (660 music files, 426 speech files), GTZAN, Marolt19, ACMusYT",
        "FM radio station dataset (30 minutes of audio, alternating 5-min speech/music segments)",
        "RAVDESS (576 training samples, 192 testing samples, 24 actors)",
        "Movie-MUSNOMIX, MUSAN, GTZAN, Scheirer-Slaney (various durations and sampling rates)",
        "Custom dataset of 600 audio clips (1-10s each)"
    ],
    "Models": [
        "ANN, Logistic Regression, KNN",
        "CNN (15 layers), RNN (9 layers) with LSTM",
        "SVM with Radial Basis Function (RBF) kernel",
        "RANSAC, Multilayer Perceptron (MLP), SVM",
        "KNN (K=5, Euclidean distance)",
        "INA (CNN-based), SwishNet (residual CNN), VGG-like CNN, OpenL3 embeddings",
        "MLP with 15 hidden neurons, Gaussian classifier for ZCR features",
        "Random Forest (100 trees), SVM (RBF kernel), Decision Tree",
        "CNN (4 convolutional layers, max pooling), DNN (Adam optimizer), SVM",
        "KNN (K=3, cosine distance)"
    ],
    "Features": [
        "40 MFCCs, 2D STFT for time-frequency representation",
        "13 MFCCs, Mel-Spectrogram with 128 Mel bands, data augmentation (pitch shifting, noise)",
        "Block-based MFCC (40 filter banks, 3 frequency bands), delta and delta-delta features",
        "MFCC (13 coefficients), Wavelet-based features (Haar decomposition, STE)",
        "13 MFCC features, frame size 20 ms, frame shift 10 ms",
        "21-22 MFCCs, Mel-Spectrogram, OpenL3 embeddings from pre-trained models",
        "Zero-Crossing Rate (ZCR), Spectral flux, Short-Time Energy (STE), Cepstral coefficients",
        "13 MFCC features, Periodogram, Discrete Cosine Transform (DCT)",
        "MFCC from Hilbert Envelope of Group Delay, Modified Group Delay Cepstral Coefficient, Instantaneous Frequency Cosine Coefficient",
        "13 MFCC features, 20 ms frame size"
    ],
    "Classification Method": [
        "Artificial Neural Network (ANN), Logistic Regression, K-Nearest Neighbors (KNN)",
        "Convolutional Neural Network (CNN), Recurrent Neural Network (RNN) with LSTM",
        "Support Vector Machine (SVM) with cross-validation",
        "RANSAC (Robust Estimation), MLP, SVM for hierarchical classification",
        "K-Nearest Neighbors (KNN), Euclidean distance metric",
        "Transfer learning with CNN, RNN, OpenL3 embeddings, fine-tuning on specific tasks",
        "Multi-Layer Perceptron (MLP) for 5-class audio categorization, Gaussian classifier for ZCR",
        "Random Forest, SVM, Decision Tree classifiers",
        "Deep Neural Network (DNN) and CNN with combined features, Naive Bayes, SVM",
        "K-Nearest Neighbors (KNN) for speech/music classification"
    ],
    "Accuracy / Results": [
        "91.27% accuracy with ANN, KNN performed poorly on noisy samples",
        "96.52% for CNN, 92.89% for RNN on ESC-50 dataset",
        "98.43% accuracy using block-based MFCC with SVM",
        "92.71% accuracy using RANSAC, SVM achieved 83.86%",
        "91% accuracy using KNN, performance consistent across different audio clip lengths",
        "99.2% accuracy with OpenL3 embeddings for speech/music classification",
        "MLP achieved 6% Total Error Rate, significantly better than ZCR-based methods",
        "88.54% accuracy for Random Forest on RAVDESS dataset, SVM and Decision Tree performed lower",
        "99.81% accuracy combining magnitude and phase-based features with CNN",
        "91% accuracy using KNN for speech/music classification"
    ],
    "Key Contributions / Findings": [
        "Comparison of MFCC and STFT features, ANN achieved highest performance for noisy environment sound classification",
        "CNN outperformed RNN on smaller datasets, while RNN performed better on larger datasets (e.g., ESC-50)",
        "Block-based MFCC achieved superior performance with SVM compared to traditional MFCC",
        "Hierarchical classification with MFCC variants and wavelet-based features significantly improved accuracy for instrumental and genre classification",
        "KNN with MFCC features is a lightweight, effective model for speech/music classification",
        "OpenL3 embeddings consistently outperformed CNN and RNN models for various tasks, demonstrating the effectiveness of transfer learning",
        "MLP classifier performed significantly better than ZCR-based Gaussian classifiers for speech/music discrimination",
        "Random Forest was the best performer for emotion classification on the RAVDESS dataset, while SVM struggled with multi-class emotion data",
        "Combining phase-based and magnitude-based features enhanced the accuracy of CNN for speech/music/noise classification",
        "KNN with simple MFCC features proved to be an efficient approach for distinguishing speech from music in short audio clips"
    ]
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Convert to Excel
df.to_excel("papers_Review.xlsx", index=False)

print("Excel file 'papers_Review.xlsx' created successfully.")