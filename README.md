# Parkinson’s Disease Detection using Voice and Machine Learning

This project develops a **machine learning–based diagnostic tool** for detecting Parkinson’s Disease from voice recordings.  
It extracts acoustic and frequency-based features such as **jitter, shimmer, and pitch variation**, and trains a **Gradient Boosting Classifier** for accurate classification.  
A **Streamlit interface** enables real-time testing and instant result visualization.

---

## 🚀 Features
1. Extracts 22 key acoustic features from voice recordings.  
2. Uses **Gradient Boosting** with **Grid Search** for high-accuracy classification.  
3. Records or uploads voice samples directly through Streamlit.  
4. Displays prediction results and confidence levels in real time.  
5. Supports both offline training and live voice testing.

---

## 🧠 Technologies Used
- **Python**, **Streamlit**  
- **Pandas**, **NumPy**, **scikit-learn**, **imblearn**  
- **Parselmouth (Praat API)** for feature extraction  
- **Sounddevice** and **SciPy** for voice recording and processing  
- **Joblib** for model storage and retrieval  

---

## 📊 Dataset
- **Source:** UCI Parkinson’s Disease Dataset (`pd_voice_data.csv`)  
- **Contents:** 22 extracted acoustic features and a binary `status` (1 = Parkinson’s, 0 = Healthy).  

---

## 🧩 How It Works
1. Loads and preprocesses the Parkinson’s dataset.  
2. Scales features using **StandardScaler** and balances data with **SMOTE**.  
3. Trains a **Gradient Boosting Classifier** optimized through **Grid Search**.  
4. Records or uploads voice samples via the Streamlit interface.  
5. Extracts acoustic features and predicts Parkinson’s Disease in real time.  
6. Displays the result (Healthy / Affected) with model confidence.

---

## 🖥️ Running the Project
### 1. Clone the repository
```bash
git clone https://github.com/yourusername/parkinsons-voice-detection.git
cd parkinsons-voice-detection
