import os
import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import pandas as pd
import joblib
import parselmouth 
from parselmouth.praat import call
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# -------------------------
# Config
# -------------------------
DATA_FILE = 'pd_voice_data.csv'
MODEL_FILE = 'best_gb_parkinsons_model.pkl'
SCALER_FILE = 'data_scaler.pkl'
LIVE_WAV_FILE = 'live_voice_sample.wav'
SAMPLE_RATE = 44100
RECORD_DURATION = 5
FEATURE_NAMES = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 
    'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 
    'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 
    'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
]

# -------------------------
# Safe feature extraction
# -------------------------
def extract_acoustic_features(audio_file_path):
    try:
        sound = parselmouth.Sound(audio_file_path)
        pitch = call(sound, "To Pitch", 0.0, 50, 600)
        point_process = call(sound, "To PointProcess (periodic, cc)", 50, 600)

        def safe_call(obj, command, *args):
            try:
                return call(obj, command, *args)
            except:
                return 0.0

        features_dict = {
            'MDVP:Fo(Hz)': safe_call(pitch, "Get mean", 0, 0, "Hertz"),
            'MDVP:Fhi(Hz)': safe_call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic"),
            'MDVP:Flo(Hz)': safe_call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic"),
            'MDVP:Jitter(%)': safe_call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3),
            'MDVP:Jitter(Abs)': safe_call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3) * 1000,
            'MDVP:RAP': safe_call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3),
            'MDVP:PPQ': safe_call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3),
            'Jitter:DDP': safe_call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3),
            'MDVP:Shimmer': safe_call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            'MDVP:Shimmer(dB)': safe_call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            'Shimmer:APQ3': safe_call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            'Shimmer:APQ5': safe_call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            'MDVP:APQ': safe_call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            'Shimmer:DDA': safe_call([sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            'NHR': safe_call(sound, "Get noise to harmonics ratio", 0, 0),
            'HNR': safe_call(sound, "Get harmonicity (cc)", 0.01, 50, 0.1, 1.0),
            'RPDE': 0.0, 'DFA': 0.0, 'spread1': 0.0,
            'spread2': 0.0, 'D2': 0.0, 'PPE': 0.0
        }
        
        feature_values = [features_dict.get(name, 0.0) for name in FEATURE_NAMES]
        feature_array = np.array([feature_values], dtype=float).reshape(1, -1)
        
        return feature_array, features_dict
        
    except Exception as e:
        return None, None

# -------------------------
# Train + save model
# -------------------------
def train_and_save_model():
    if not os.path.exists(DATA_FILE):
        st.error(f"Dataset Error: The required file '{DATA_FILE}' was not found in the current directory.")
        st.error("Please place the voice data CSV file in the same folder as this script and try again.")
        return None, None

    data = pd.read_csv(DATA_FILE)
    if 'name' in data.columns:
        data = data.drop(columns=['name'])
    if 'status' not in data.columns:
        st.error("Dataset must have 'status' column to train the model.")
        return None, None

    data_features = set(data.drop(columns=['status']).columns)
    required_features = set(FEATURE_NAMES)
    
    missing_features = required_features - data_features
    if missing_features:
        st.error(f"Dataset missing required features: {missing_features}")
        return None, None
        
    X = data[FEATURE_NAMES].copy()
    y = data['status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

    gb = GradientBoostingClassifier(random_state=42)
    param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 4]}
    grid = GridSearchCV(gb, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    with st.spinner("Training model and tuning hyperparameters..."):
        grid.fit(X_train_res, y_train_res)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    joblib.dump(best_model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    st.success(f"Model trained and saved successfully. Test Accuracy: {acc*100:.2f}%")
    st.write("Best parameters:", grid.best_params_)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Parkinson'], 
                yticklabels=['Healthy', 'Parkinson'], ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).T.style.format("{:.3f}"))

    return best_model, scaler

# -------------------------
# Predict
# -------------------------
def predict_from_audio(model, scaler, audio_path):
    features_array, features_dict = extract_acoustic_features(audio_path)
    
    if features_array is None or features_dict is None:
        return -1, 0.0, None

    scaled = scaler.transform(features_array)
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][pred]
    
    return int(pred), float(prob), features_dict

# -------------------------
# Streamlit UI with Styling
# -------------------------
st.set_page_config(page_title="Parkinson's Voice Analyzer", page_icon="🔬", layout="centered")

# Custom CSS for aesthetic improvements
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 50px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    h1 {
        color: #1f2937;
        font-weight: 700;
        padding-bottom: 10px;
    }
    h2, h3 {
        color: #374151;
        font-weight: 600;
    }
    .success-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #d1fae5;
        border-left: 4px solid #10b981;
        margin: 20px 0;
    }
    .warning-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #fee2e2;
        border-left: 4px solid #ef4444;
        margin: 20px 0;
    }
    .info-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 15px 0;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Parkinson's Voice Analyzer")
st.markdown("Analyze voice features to detect potential indicators of Parkinson's disease")

# Load model silently
model, scaler = None, None
if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
    try:
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
    except Exception as e:
        st.warning(f"Could not load model/scaler: {e}. Please retrain.")
        model, scaler = None, None

st.markdown("---")

# Model Training Section
with st.expander("Model Training", expanded=False):
    if st.button("Retrain Model", use_container_width=True):
        model, scaler = train_and_save_model()

st.markdown("---")

# Voice Input Section
st.subheader("Voice Input")

tab1, tab2 = st.tabs(["Record Voice", "Upload Audio"])

with tab1:
    st.markdown("Record a sustained vowel sound (e.g., 'Aaaah') for 5 seconds")
    if st.button("Start Recording", use_container_width=True, type="primary"):
        st.info(f"Recording for {RECORD_DURATION} seconds... Please say 'Aaaah'")
        try:
            recording = sd.rec(int(RECORD_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
            sd.wait()
            write(LIVE_WAV_FILE, SAMPLE_RATE, recording)
            st.success("Recording completed successfully")
            st.session_state['audio_file'] = LIVE_WAV_FILE
        except Exception as e:
            st.error(f"Recording failed: {e}. Please check your microphone setup.")

with tab2:
    uploaded_file = st.file_uploader("Upload a WAV audio file", type=['wav'])
    if uploaded_file is not None:
        with open("uploaded_voice_sample.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File uploaded successfully")
        st.session_state['audio_file'] = "uploaded_voice_sample.wav"

st.markdown("---")

# Analysis Section
if 'audio_file' in st.session_state and os.path.exists(st.session_state['audio_file']):
    st.subheader("Audio Preview")
    st.audio(st.session_state['audio_file'])
    
    if st.button("Analyze Voice", use_container_width=True, type="primary"):
        if model is None or scaler is None:
            st.warning("Model not loaded. Please train the model first.")
        else:
            with st.spinner("Analyzing voice features..."):
                pred, prob, features_dict = predict_from_audio(model, scaler, st.session_state['audio_file'])
            
            if pred == -1:
                st.error("Error, try again")
            else:
                st.markdown("### Analysis Result")
                if pred == 1:
                    st.markdown(f"""
                    <div class="warning-box">
                        <h3 style="color: #dc2626; margin-top: 0;">Parkinson's Detected</h3>
                        <p style="font-size: 1.2em; margin-bottom: 0;">Confidence: <strong>{prob*100:.2f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="success-box">
                        <h3 style="color: #059669; margin-top: 0;">Healthy Voice</h3>
                        <p style="font-size: 1.2em; margin-bottom: 0;">Confidence: <strong>{prob*100:.2f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("### Extracted Acoustic Features")
                features_df = pd.DataFrame.from_dict(features_dict, orient='index', columns=['Value'])
                features_df['Value'] = features_df['Value'].apply(lambda x: f"{x:.4f}")
                st.dataframe(features_df, use_container_width=True, height=400)
                
                if st.button("Save Prediction Log"):
                    try:
                        with open("prediction_log.txt", "a") as f:
                            f.write(f"Prediction: {'Parkinson' if pred==1 else 'Healthy'} | Confidence: {prob*100:.2f}%\n")
                            f.write(f"Features: {features_dict}\n")
                            f.write("-" * 50 + "\n")
                        st.success("Log saved to prediction_log.txt")
                    except Exception as e:
                        st.error(f"Could not save log file: {e}")

