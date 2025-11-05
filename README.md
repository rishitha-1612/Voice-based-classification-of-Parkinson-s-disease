# Parkinson’s Disease Detection using Voice

This project presents a **machine-learning-based diagnostic tool** for detecting Parkinson’s Disease from voice recordings.  
It extracts acoustic and frequency-based features such as *jitter*, *shimmer*, and *pitch variation*, and trains a **Gradient Boosting Classifier** for classification.  
A **Streamlit** interface enables real-time voice testing and instant visualization of results.

---

## Features
1. Extracts 22 key acoustic features from voice recordings.  
2. Implements preprocessing (scaling, balancing) and uses **Gradient Boosting** with grid‐search for high accuracy.  
3. Supports voice sample **upload or live recording** through the Streamlit web interface.  
4. Displays prediction results with **confidence scores** in real time.  
5. Offers both offline model training and live voice-based testing.

---

## Technologies Used
- **Python**, **Streamlit**  
- Libraries: **Pandas**, **NumPy**, **scikit-learn**, **imblearn**  
- **Parselmouth (Praat API)** for acoustic feature extraction  
- **sounddevice**, **SciPy** for live audio capture and signal processing  
- **joblib** for model serialization and loading

---

## Dataset
- For training the model: [UCI Parkinson’s Dataset](https://archive.ics.uci.edu/dataset/174/parkinsons) (file: `pd_voice_data.csv`)  
- Contents: 22 extracted acoustic features + binary label `status` (1 = Parkinson’s, 0 = Healthy)
- [Voice Samples Dataset – Figshare](https://figshare.com/articles/dataset/Voice_Samples_for_Patients_with_Parkinson_s_Disease_and_Healthy_Controls/23849127)
 can be used to test and validate the trained Parkinson’s Disease detection model. (file: `PD Voices.zip`)  
- Included in repository for reproducibility.

---

## How It Works
1. Load and preprocess the dataset (feature scaling, balancing with SMOTE).  
2. Train and tune a Gradient Boosting Classifier to identify reliable performance.  
3. In the user interface:  
   - Record or upload a voice sample.  
   - Extract acoustic features in real time.  
   - Predict Parkinson’s status and display results with confidence.  
4. Integrate predictions with a user-friendly Streamlit dashboard for easy access.

---

## Running the Project
Clone the repository
```bash
git clone https://github.com/rishitha-1612/Voice-based-classification-of-Parkinson-s-disease.git
cd Voice-based-classification-of-Parkinson-s-disease
```
Install dependencies
```
pip install -r requirements.txt
```

Train the model (optional if using provided model)
```
python mic_test.py   # or your training script if available
```

Launch the Streamlit interface

```
streamlit run parkinsons_streamlit_app.py
```

## Future Scope
Incorporate deep learning models (e.g., CNN, RNN) to improve classification accuracy.
Expand dataset with diverse voice samples (multiple languages, varied demographics).
Add medical report generation, patient history tracking and integration with healthcare systems.

### Authors
Project developed by a team of three Computer Science Engineering students from RV Institute of Technology and Management, working collaboratively on all modules including AI development, OCR processing, and financial data analysis.

| Name              | GitHub                              | Email ID                                               |
|-------------------|-------------------------------------|--------------------------------------------------------|
| Sowmya P R        | https://github.com/2406-Sowmya      | srsb2406@gmail.com                                     |
| Aishwarya R       | https://github.com/AISHWARYA251166  | ar2573564@gmail.com                                    |
| M N Monisha       |
| Rishitha Rasineni | https://github.com/rishitha-1612    | rishitharasineni@gmail.com                             |
