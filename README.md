# Hidden Markov Models for Human Activity Recognition

**Course**: Formative Assignment 2

**Contributors**:

- Erneste Ntezirizaza (Jumping & Still data, Viterbi algorithm)
- Noella Umwali (Standing & Walking data, Baum-Welch algorithm)

## Project Overview

This project implements a complete Hidden Markov Model (HMM) pipeline to recognize human activities from smartphone sensor data (accelerometer and gyroscope signals). The system classifies four activities: **Standing**, **Walking**, **Jumping**, and **Still** (no movement).

## Key Components

### 1. Data Collection & Preprocessing

- **50 total samples** combining train (~40) and test (~10) recordings
- **4 activities** with 5-10 second duration recordings
- **2 sensor types**: Accelerometer (x, y, z) and Gyroscope (x, y, z)
- **Sliding window approach**: 1-second windows with 50% overlap at ~100 Hz sampling rate
- **Harmonized sampling rates** across both team members' devices

### 2. Feature Extraction

- **Time/Impact Features**:
  - 26 features from Accelerometer + 26 from Gyroscope
  - Includes mean/std/variance/RMS/SMA/correlation and impact-sensitive descriptors (jerk, peak-to-peak, magnitude max/range)
- **Frequency-Domain Features**:
  - 15 FFT-based features from Accelerometer + 15 from Gyroscope
  - Dominant frequency, spectral energy, low/high band power
- **Total fused features**: **82 features per window**

- **Normalization**: Z-score standardization for scale independence

### 3. HMM Implementation

- **Viterbi Algorithm** (Erneste): Dynamic programming for optimal sequence decoding
- **Baum-Welch Algorithm** (Noella): EM-based parameter learning with convergence checking
- **Class-specific Gaussian HMMs**: One tuned model per activity (Standing, Walking, Jumping, Still)
- **Numerically stable training**: Scaled forward/backward recursions and regularized covariance estimates
- **Learned Parameters**:
  - Transition probabilities (sub-state transition matrices per activity model)
  - Initial state probabilities
  - Emission means and covariances

### 4. Model Evaluation

- Testing on unseen data (**162 test windows**)
- **Per-activity metrics**: Sensitivity, Specificity
- **Overall Accuracy**: **98.77%**
- **Confusion Matrix**: Visualization of classification patterns
- **Transition Analysis**: Understanding learned behavior patterns

## Files & Structure

```
Formative-2_Hidden_Markov_Models/
├── README.md                         # This file
├── Formative-2_HMM_Report_Final.pdf  # Final HMM report document
├── Team Task Sheet_[Machine_Learning_Techniques_II_Hidden Markov Models_Cohort 1_Team10].pdf
│                                     # Team task allocation sheet
├── Notebook/
│   ├── HMM_Activity_Recognition.ipynb    # Main notebook (all code & analysis)
│   ├── Figures/
│   │   ├── raw_sensor_signals.png
│   │   ├── feature_distributions.png
│   │   ├── baum_welch_convergence.png
│   │   ├── confusion_matrix.png
│   │   ├── transition_probabilities.png
│   │   └── emission_parameters.png
│   └── Results/
│       └── evaluation_metrics.csv
├── Dataset/
│   ├── Train/
│   │   ├── Accelerometer/
│   │   │   ├── Standing/       (10 CSV files)
│   │   │   ├── Walking/        (10 CSV files)
│   │   │   ├── Jumping/        (10 CSV files)
│   │   │   └── Still/          (10 CSV files)
│   │   └── Gyroscope/          (40 CSV files)
│   ├── Test/
│   │   ├── Accelerometer/
│   │   │   ├── Standing/       (3 CSV files)
│   │   │   ├── Walking/        (2 CSV files)
│   │   │   ├── Jumping/        (2 CSV files)
│   │   │   └── Still/          (3 CSV files)
│   │   └── Gyroscope/          (10 CSV files)
└── .git/                             # Git metadata
```

## Algorithm Summary

### Viterbi Algorithm (Sequence Decoding)

```
Goal: Find most likely sequence of hidden states given observations
Time Complexity: O(T × N²) where T = sequence length, N = num states

Steps:
1. Initialize: viterbi[0] = π × B(O₀)
2. Recursion: viterbi[t,j] = max_i(viterbi[t-1,i] × A[i,j]) × B(O_t)
3. Backtrack: Trace maximum probability path
```

### Baum-Welch Algorithm (Parameter Learning)

```
Goal: Learn HMM parameters (A, B, π) from training data using EM
Convergence: ΔLog-Likelihood < 1e-3

E-Step:
  - Forward pass: Compute P(O₁:t | θ)
  - Backward pass: Compute P(O_{t+1:T} | θ)
  - Compute posterior: γ[t,i] = P(Z_t=i | O)

M-Step:
  - Update π[i] ← γ[0,i]
  - Update A[i,j] ← Σ ξ[t,i,j] / Σ γ[t,i]
  - Update μ[i] ← Σ γ[t,i]·O_t / Σ γ[t,i]
  - Update Σ[i] ← Σ γ[t,i]·(O_t - μ[i])² / Σ γ[t,i]
```

## Key Findings

### Model Performance

- Near-perfect test performance on all four activities
- Only minor confusion between Standing and Still (stationary-state overlap)
- Transition probabilities capture realistic human behavior patterns

### What Activities Were Easiest to Distinguish?

1. **Walking**: Clear 1-2 Hz periodic signature from stride
2. **Jumping**: Broad high-frequency energy content from impacts

### What Activities Were Hardest to Distinguish?

1. **Standing vs. Still**: Both represent low-acceleration states; differ mainly by context

### Impact of Design Choices

- **1-second windows**: Better temporal responsiveness while preserving enough activity signal
- **Z-score normalization**: Essential for fair feature comparison across scales
- **Impact + frequency features**: Strongly improved Jumping/Walking separation
- **100 Hz sampling**: Sufficient for capturing activity-specific patterns

## How to Run the Notebook

### Prerequisites

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn
```

### Execution

```python
jupyter notebook Notebook/HMM_Activity_Recognition.ipynb
```

Then run cells **top-to-bottom**:

1. Imports and configuration
2. Data loading
3. Feature extraction (time-domain)
4. Feature extraction (frequency-domain)
5. Normalization and visualization
6. Model initialization and training
7. Evaluation on test data
8. Results analysis

Expected runtime: ~5-10 minutes

## Task Allocation (GitHub Evidence)

### Erneste Ntezirizaza (50%)

- Jumping & Still data collection and preprocessing
- **Viterbi algorithm implementation** in GaussianHMM class
- Report sections: Background, Data collection, Results discussion
- Commits:
  - "Implement Viterbi algorithm for HMM state decoding..."
  - "Add Jumping and Still activity analysis..."

### Noella Umwali (50%)

- Standing & Walking data collection and preprocessing
- **Baum-Welch algorithm implementation** in GaussianHMM class
- Report sections: HMM setup, Implementation details, Results discussion
- Commits:
  - "Implement Baum-Welch EM algorithm with convergence checking..."
  - "Add Standing and Walking activity analysis..."

Both contributed to:

- Feature extraction functions
- Model training and evaluation
- Visualization and analysis
- Report integration and final review

## Evaluation Metrics

See `Notebook/Results/evaluation_metrics.csv` for detailed per-activity metrics:

```
Activity   | # Samples | Sensitivity | Specificity | Accuracy
-----------|-----------|-------------|-------------|----------
Standing   | 47        | 100.00%     | 98.26%      | 98.77%
Walking    | 32        | 100.00%     | 100.00%     | 98.77%
Jumping    | 34        | 100.00%     | 100.00%     | 98.77%
Still      | 49        | 95.92%      | 100.00%     | 98.77%
Overall    | 162       | -           | -           | 98.77%
```

## Future Improvements

1. **More training data**: 100+ samples per activity (we used minimal ~10/activity)
2. **Richer features**: Entropy, wavelet transforms, and higher-order spectral descriptors
3. **Extended fusion**: Add magnetometer and orientation-aware feature alignment
4. **Person-specific models**: Individual calibration for different body types/movement styles
5. **Temporal models**: Hidden Semi-Markov Model for explicit state duration modeling
6. **Deep learning**: LSTM/CNN models for learning temporal patterns automatically
7. **Multi-modal context**: Time of day, location, device position, prior activity

## References & Implementation Notes

- **Viterbi Algorithm**: Classic dynamic programming approach for HMM inference
- **Baum-Welch EM**: Standard algorithm for unsupervised HMM parameter learning
- **Gaussian Emissions**: Continuous observation modeling common in activity recognition
- **Feature Engineering**: Based on standard accelerometer signal processing in literature
- **Convergence Checking**: Epsilon-based method more robust than fixed iteration limits
