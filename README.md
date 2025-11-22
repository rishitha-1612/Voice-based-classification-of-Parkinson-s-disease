# Parkinson’s Disease Detection using Voice

This project presents a **machine-learning-based diagnostic tool** for detecting Parkinson’s Disease from voice recordings.  
It extracts acoustic and frequency-based features such as *jitter*, *shimmer*, and *pitch variation*, and trains a **Gradient Boosting Classifier** for classification.  
A **Streamlit** interface enables real-time voice testing and instant visualization of results.

---

## Features
- **Voice-Based Analysis**: Record voice samples or upload audio files for analysis
- **Multiple ML Models**: Support for 11+ classification algorithms
- **Advanced Preprocessing**:
  - Multiple scaler options (Standard, MinMax, Robust, Normalizer, MaxAbs, QuantileTransformer)
  - PCA (Principal Component Analysis) for dimensionality reduction
  - SMOTE for class balancing
    
- **Comprehensive Evaluation**:
  - Training and testing accuracy metrics
  - Overfitting detection
  - Confusion matrix
  - Classification reports
  - Model comparison dashboard
    
- **Feature Extraction**: Extracts 22 acoustic features including:
  - Fundamental frequency measures (Fo, Fhi, Flo)
  - Jitter measurements
  - Shimmer measurements
  - Harmonics-to-Noise Ratio (HNR)
  - Noise-to-Harmonics Ratio (NHR)
  - And other advanced voice analysis features

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Audio recording capability (microphone)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd parkinsons-detection
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Ensure you have the training data file:
   - Place `pd_voice_data.csv` in the project root directory
   - The file should contain voice features with columns matching the 22 extracted features plus a 'status' column

## Usage

### Running the Application

Start the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Workflow

#### 1. Model Training Tab

1. **Select a Scaler**: Choose from available scaling methods
2. **Configure Class Balancing**: Enable SMOTE if needed
3. **Configure PCA**: 
   - Enable PCA for dimensionality reduction
   - Select number of components (2-22)
4. **Train Models**: Click on any model button to train with current configuration
5. **View Performance**: Expand model reports to see:
   - Training and testing accuracy
   - Overfitting metrics
   - Confusion matrix
   - Classification report
   - Best hyperparameters
6. **Download Models**: Save trained models, scalers, and PCA transformers

#### 2. Voice Prediction Tab

1. **Record or Upload**:
   - Record a 3-second voice sample (sustain "Aaaah")
   - Or upload an existing .wav file
2. **Analyze**: Click "Analyze Voice Sample"
3. **View Results**:
   - Prediction (Healthy/Parkinson's)
   - Confidence scores
   - Extracted acoustic features

#### 3. Model Comparison Tab

- View all trained models side-by-side
- Compare accuracy metrics
- Identify best performing configurations
- Export comparison data as CSV

## Data Format

The training data file (`pd_voice_data.csv`) should contain:

- 22 feature columns:
  - MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz)
  - MDVP:Jitter(%), MDVP:Jitter(Abs), MDVP:RAP, MDVP:PPQ, Jitter:DDP
  - MDVP:Shimmer, MDVP:Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5, MDVP:APQ, Shimmer:DDA
  - NHR, HNR
  - RPDE, DFA, spread1, spread2, D2, PPE
- 1 target column: `status` (0 = Healthy, 1 = Parkinson's)
- Optional: `name` column (will be ignored during training)
  
---

## Model Configuration

All models use GridSearchCV with 5-fold cross-validation and anti-overfitting parameters:

- Regularization constraints
- Tree depth limits
- Minimum sample requirements
- Learning rate controls
- Feature sampling restrictions

## Technical Details

### Audio Processing
- Sample rate: 44,100 Hz
- Recording duration: 3 seconds
- Normalization and pre-emphasis filtering applied
- Resampling for consistency

### Feature Extraction
Uses Praat-Parselmouth library for acoustic analysis:
- Pitch analysis (75-500 Hz range)
- Jitter and shimmer calculations
- Harmonicity measurements
- Point process analysis

### Model Pipeline
1. Data Loading
2. Train-Test Split (70-30, stratified)
3. Scaling (selected method)
4. SMOTE (optional)
5. PCA (optional)
6. Model Training with GridSearchCV
7. Evaluation and Metrics

## Performance Considerations

- **Overfitting Detection**: System flags models with training accuracy > 95% or overfitting gap > 10%
- **PCA Benefits**: Reduces computational complexity and may improve generalization
- **SMOTE**: Helps with imbalanced datasets but may increase overfitting risk

---

## Dataset
- For training the model: [UCI Parkinson’s Dataset](https://archive.ics.uci.edu/dataset/174/parkinsons) (file: `pd_voice_data.csv`)  
- Contents: 22 extracted acoustic features + binary label `status` (1 = Parkinson’s, 0 = Healthy)
- [Voice Samples Dataset – Figshare](https://figshare.com/articles/dataset/Voice_Samples_for_Patients_with_Parkinson_s_Disease_and_Healthy_Controls/23849127)
 can be used to test and validate the trained Parkinson’s Disease detection model. (file: `PD Voices.zip`)  
- Included in repository for reproducibility.

---

## Limitations

- This is a **screening tool only**, not a diagnostic device
- Results should be interpreted by qualified medical professionals
- Voice quality and recording conditions affect accuracy
- Model performance depends on training data quality and diversity

## Future Scope
Incorporate deep learning models (e.g., CNN, RNN) to improve classification accuracy.
Expand dataset with diverse voice samples (multiple languages, varied demographics).
Add medical report generation, patient history tracking and integration with healthcare systems.

---

### Authors
Project developed by a team of four Computer Science Engineering students from RV Institute of Technology and Management, working collaboratively on all modules including AI development, OCR processing, and financial data analysis.

| Name              | GitHub                              | Email ID                                               |
|-------------------|-------------------------------------|--------------------------------------------------------|
| Sowmya P R        | https://github.com/2406-Sowmya      | srsb2406@gmail.com                                     |
| Aishwarya R       | https://github.com/AISHWARYA251166  | ar2573564@gmail.com                                    |
| M N Monisha       | https://github.com/rvitmonisha      | mnmonisha@gmail.com                                    |
| Rishitha Rasineni | https://github.com/rishitha-1612    | rishitharasineni@gmail.com                             |
