import streamlit as st
import pandas as pd
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write, read
from scipy import signal
import joblib
import parselmouth
from parselmouth.praat import call
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, Normalizer, 
    MaxAbsScaler, QuantileTransformer, PowerTransformer
)
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.base import clone
from imblearn.over_sampling import SMOTE
import os
import tempfile
import matplotlib.pyplot as plt
from datetime import datetime

# Supervised Classification Models
from sklearn.ensemble import (
    GradientBoostingClassifier, 
    RandomForestClassifier, 
    AdaBoostClassifier,
    ExtraTreesClassifier,
    BaggingClassifier,
    HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# --- CONFIGURATION ---
DATA_FILE = 'pd_voice_data.csv'
RECORD_DURATION = 3
SAMPLE_RATE = 44100

# =========================================================================
# AUDIO PREPROCESSING
# =========================================================================

def preprocess_audio(audio_file_path):
    """Normalize audio to match training data conditions."""
    try:
        sr, audio = read(audio_file_path)
        
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        
        if sr != 44100:
            num_samples = int(len(audio) * 44100 / sr)
            audio = signal.resample(audio, num_samples)
            sr = 44100
        
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        
        pre_emphasis = 0.97
        audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
        
        normalized_path = audio_file_path.replace('.wav', '_normalized.wav')
        write(normalized_path, sr, (audio * 32767).astype(np.int16))
        
        return normalized_path
    except Exception as e:
        return audio_file_path

# =========================================================================
# FEATURE EXTRACTION
# =========================================================================

def extract_acoustic_features(audio_file_path):
    """Extracts the 22 features required by the trained model."""
    try:
        normalized_path = preprocess_audio(audio_file_path)
        
        sound = parselmouth.Sound(normalized_path)
        
        if sound.get_total_duration() == 0:
            st.error("Audio file is empty or invalid")
            return None, None
            
        pitch = call(sound, "To Pitch", 0.0, 75, 500)
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
        
        mean_f0 = call(pitch, "Get mean", 0, 0, "Hertz")
        max_f0 = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
        min_f0 = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
        
        if mean_f0 == 0 or np.isnan(mean_f0):
            st.error("Unable to extract valid pitch. Please try recording again with a clearer sound.")
            return None, None
        
        try:
            jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_abs = jitter_local * mean_f0 / 1000000
            jitter_rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_ppq = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_ddp = call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
        except:
            jitter_local = jitter_abs = jitter_rap = jitter_ppq = jitter_ddp = 0.0
        
        try:
            shimmer_local = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_db = call([sound, point_process], "Get shimmer (local, dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq3 = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq5 = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_apq11 = call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_dda = call([sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        except:
            shimmer_local = shimmer_db = shimmer_apq3 = shimmer_apq5 = shimmer_apq11 = shimmer_dda = 0.0
        
        try:
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
            nhr = 1.0 / hnr if hnr > 0 else 0.0
        except:
            hnr = nhr = 0.0
        
        features_dict = {
            'MDVP:Fo(Hz)': mean_f0,
            'MDVP:Fhi(Hz)': max_f0,
            'MDVP:Flo(Hz)': min_f0,
            'MDVP:Jitter(%)': jitter_local * 100,
            'MDVP:Jitter(Abs)': jitter_abs * 1000,
            'MDVP:RAP': jitter_rap,
            'MDVP:PPQ': jitter_ppq,
            'Jitter:DDP': jitter_ddp,
            'MDVP:Shimmer': shimmer_local,
            'MDVP:Shimmer(dB)': shimmer_db,
            'Shimmer:APQ3': shimmer_apq3,
            'Shimmer:APQ5': shimmer_apq5,
            'MDVP:APQ': shimmer_apq11,
            'Shimmer:DDA': shimmer_dda,
            'NHR': nhr,
            'HNR': hnr,
            'RPDE': 0.45,
            'DFA': 0.72,
            'spread1': -65,
            'spread2': 0.2,
            'D2': 2.3,
            'PPE': 0.15,
        }
        
        feature_array = np.array([value for value in features_dict.values()]).reshape(1, -1)
        
        return feature_array, features_dict

    except Exception as e:
        st.error(f"Error during feature extraction: {e}")
        return None, None

# =========================================================================
# SCALER OPTIONS
# =========================================================================

def get_scaler_options():
    """Returns available scaler options."""
    return {
        "Standard Scaler": StandardScaler(),
        "MinMax Scaler": MinMaxScaler(),
        "Robust Scaler": RobustScaler(),
        "Normalizer": Normalizer(),
        "MaxAbs Scaler": MaxAbsScaler(),
        "QuantileTransformer": QuantileTransformer(output_distribution='uniform')
    }

# =========================================================================
# MODEL CONFIGURATIONS
# =========================================================================

def get_model_configs():
    """Returns dictionary of classification models with anti-overfitting parameters."""
    configs = {
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                'n_estimators': [30, 50, 70],
                'learning_rate': [0.01, 0.02, 0.05],
                'max_depth': [2, 3],
                'min_samples_split': [20, 30],
                'min_samples_leaf': [10, 15],
                'subsample': [0.7, 0.8],
                'max_features': ['sqrt']
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, 20],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [2, 4],
                'max_features': ['sqrt', 'log2']
            }
        },
        "XGBoost": {
            "model": XGBClassifier(random_state=42, eval_metric='logloss'),
            "params": {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 4, 5],
                'min_child_weight': [3, 5],
                'subsample': [0.7, 0.8],
                'colsample_bytree': [0.7, 0.8]
            }
        },
        "LightGBM": {
            "model": LGBMClassifier(random_state=42, verbose=-1),
            "params": {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5],
                'num_leaves': [15, 31],
                'min_child_samples': [20, 30]
            }
        },
        "Support Vector Machine": {
            "model": SVC(probability=True, random_state=42),
            "params": {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        },
        "Linear SVC": {
            "model": LinearSVC(random_state=42, max_iter=2000),
            "params": {
                'C': [0.1, 1, 10],
                'penalty': ['l2'],
                'loss': ['hinge', 'squared_hinge']
            }
        },
        "Logistic Regression": {
            "model": LogisticRegression(random_state=42, max_iter=2000),
            "params": {
                'C': [0.01, 0.1, 1, 10],
                'solver': ['lbfgs', 'liblinear'],
                'penalty': ['l2']
            }
        },
        "K-Nearest Neighbors": {
            "model": KNeighborsClassifier(),
            "params": {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {
                'max_depth': [5, 10, 15],
                'min_samples_split': [10, 20, 30],
                'min_samples_leaf': [5, 10],
                'criterion': ['gini', 'entropy']
            }
        },
        "AdaBoost": {
            "model": AdaBoostClassifier(random_state=42),
            "params": {
                'n_estimators': [30, 50, 100],
                'learning_rate': [0.5, 1.0, 1.5]
            }
        },
        "Naive Bayes": {
            "model": GaussianNB(),
            "params": {
                'var_smoothing': [1e-9, 1e-8, 1e-7]
            }
        }
    }
    
    return configs

# =========================================================================
# TRAINING FUNCTION
# =========================================================================

def train_model(model_name, selected_scaler, scaler_name, use_pca=False, n_components=10, use_smote=False):
    """Trains model with proper pipeline: Scale → SMOTE → PCA → Train"""
    
    if not os.path.exists(DATA_FILE):
        st.error(f"Training data file '{DATA_FILE}' not found!")
        return None, None, None, None
    
    with st.spinner(f"Training {model_name}..."):
        try:
            data = pd.read_csv(DATA_FILE)
            data = data.drop('name', axis=1, errors='ignore')
            X = data.drop('status', axis=1)
            y = data['status']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            scaler = clone(selected_scaler)
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            if use_smote:
                k_neighbors = min(3, min((y_train==0).sum(), (y_train==1).sum()) - 1)
                sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_train_scaled, y_train = sm.fit_resample(X_train_scaled, y_train)
            
            pca = None
            if use_pca:
                pca = PCA(n_components=n_components, random_state=42)
                X_train_final = pca.fit_transform(X_train_scaled)
                X_test_final = pca.transform(X_test_scaled)
            else:
                X_train_final = X_train_scaled
                X_test_final = X_test_scaled
            
            model_configs = get_model_configs()
            config = model_configs[model_name]
            
            grid_search = GridSearchCV(
                estimator=config["model"],
                param_grid=config["params"],
                scoring='accuracy',
                cv=5,
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train_final, y_train)
            best_model = grid_search.best_estimator_
            
            y_train_pred = best_model.predict(X_train_final)
            y_test_pred = best_model.predict(X_test_final)
            
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            overfitting_gap = train_accuracy - test_accuracy
            
            conf_matrix = confusion_matrix(y_test, y_test_pred)
            class_report = classification_report(y_test, y_test_pred, output_dict=True)
            
            results = {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'overfitting_gap': overfitting_gap,
                'best_params': grid_search.best_params_,
                'confusion_matrix': conf_matrix,
                'classification_report': class_report,
                'scaler_name': scaler_name,
                'pca_enabled': use_pca,
                'n_components': n_components if use_pca else None,
                'explained_variance': np.sum(pca.explained_variance_ratio_) * 100 if use_pca else None,
                'pca_variance_ratio': pca.explained_variance_ratio_.tolist() if use_pca else None,
                'smote_used': use_smote,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            model_filename = f"{model_name.replace(' ', '_').lower()}_model.pkl"
            scaler_filename = f"{model_name.replace(' ', '_').lower()}_scaler.pkl"
            pca_filename = f"{model_name.replace(' ', '_').lower()}_pca.pkl"
            
            joblib.dump(best_model, model_filename)
            joblib.dump(scaler, scaler_filename)
            if use_pca:
                joblib.dump(pca, pca_filename)
            
            return best_model, scaler, pca, results
            
        except Exception as e:
            st.error(f"Training error: {e}")
            import traceback
            st.code(traceback.format_exc())
            return None, None, None, None

# =========================================================================
# STREAMLIT APP
# =========================================================================

def main():
    st.set_page_config(page_title="Parkinson's Disease Detection", layout="wide")
    
    st.title("Parkinson's Disease Detection System")
    st.markdown("### Voice-based ML Classification System with PCA")
    
    if 'current_model' not in st.session_state:
        st.session_state.current_model = None
    if 'current_scaler' not in st.session_state:
        st.session_state.current_scaler = None
    if 'current_pca' not in st.session_state:
        st.session_state.current_pca = None
    if 'active_model_name' not in st.session_state:
        st.session_state.active_model_name = None
    if 'model_history' not in st.session_state:
        st.session_state.model_history = []
    
    with st.sidebar:
        st.header("Model Selection")
        st.markdown("Train different ML models and compare their performance")
        
        if os.path.exists(DATA_FILE):
            st.success(f"Data file found: {DATA_FILE}")
        else:
            st.error(f"Data file not found: {DATA_FILE}")
    
    tab1, tab2, tab3 = st.tabs(["Model Training", "Voice Prediction", "Model Comparison"])
    
    # ============= TAB 1: MODEL TRAINING =============
    with tab1:
        st.header("Train Classification Models")
        
        if st.session_state.active_model_name:
            st.success(f"Currently Active Model: *{st.session_state.active_model_name}*")
        
        st.subheader("Select Scaler")
        scaler_options = get_scaler_options()
        
        col1, col2, col3 = st.columns(3)
        scaler_names = list(scaler_options.keys())
        
        for idx, scaler_name in enumerate(scaler_names):
            col = [col1, col2, col3][idx % 3]
            with col:
                if st.button(scaler_name, key=f"scaler_{scaler_name}", use_container_width=True):
                    st.session_state.selected_scaler_name = scaler_name
                    st.session_state.selected_scaler = scaler_options[scaler_name]
        
        if 'selected_scaler_name' not in st.session_state:
            st.session_state.selected_scaler_name = "Standard Scaler"
            st.session_state.selected_scaler = scaler_options["Standard Scaler"]
        
        st.info(f"Selected Scaler: *{st.session_state.selected_scaler_name}*")
        
        st.markdown("---")
        
        st.subheader("Class Balancing")
        use_smote = st.checkbox(
            "Apply SMOTE (Synthetic Minority Over-sampling)", 
            value=False
        )
        
        st.markdown("---")
        
        st.subheader("PCA Configuration")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            use_pca = st.checkbox("Apply PCA", value=False)
        
        with col2:
            if use_pca:
                n_components = st.slider(
                    "Number of PCA Components",
                    min_value=2,
                    max_value=22,
                    value=10,
                    step=1
                )
                st.caption(f"Reducing from 22 to {n_components} components")
            else:
                n_components = 22
                st.caption("Using all 22 features")
        
        st.markdown("---")
        
        model_configs = get_model_configs()
        
        st.subheader("Select Model to Train")
        cols = st.columns(3)
        model_names = list(model_configs.keys())
        
        for idx, model_name in enumerate(model_names):
            col = cols[idx % 3]
            
            with col:
                is_active = st.session_state.active_model_name == model_name
                button_label = f"{'✅' if is_active else '🔵'} {model_name}"
                
                if st.button(button_label, key=f"btn_{model_name}", use_container_width=True):
                    model, scaler, pca, results = train_model(
                        model_name, 
                        st.session_state.selected_scaler,
                        st.session_state.selected_scaler_name,
                        use_pca=use_pca,
                        n_components=n_components,
                        use_smote=use_smote
                    )
                    
                    if model is not None:
                        st.session_state.current_model = model
                        st.session_state.current_scaler = scaler
                        st.session_state.current_pca = pca
                        st.session_state.active_model_name = model_name
                        
                        model_entry = {
                            'model_name': model_name,
                            'model': model,
                            'scaler': scaler,
                            'pca': pca,
                            'results': results
                        }
                        st.session_state.model_history.append(model_entry)
                        
                        st.success(f"✅ {model_name} trained successfully!")
                        st.rerun()
        
        if st.session_state.model_history:
            st.markdown("---")
            st.subheader("View Model Performance Reports")
            
            for entry in st.session_state.model_history:
                model_name = entry['model_name']
                results = entry['results']
                
                config_signature = f"{model_name} | {results['scaler_name']}"
                if results['pca_enabled']:
                    config_signature += f" | PCA-{results['n_components']}"
                if results['smote_used']:
                    config_signature += " | SMOTE"
                
                with st.expander(f"{config_signature} - Performance Report", expanded=False):
                    
                    if results['pca_enabled']:
                        st.info(f"PCA: {results['n_components']} components | {results['explained_variance']:.2f}% variance")
                    
                    if results['smote_used']:
                        st.info(f"SMOTE: Used")
                    
                    st.markdown("### Model Accuracy")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Training Accuracy", f"{results['train_accuracy']*100:.2f}%")
                    
                    with col2:
                        st.metric("Testing Accuracy", f"{results['test_accuracy']*100:.2f}%")
                    
                    with col3:
                        st.metric("Overfitting Gap", f"{results['overfitting_gap']*100:.2f}%")
                    
                    st.markdown("---")
                    
                    download_cols = st.columns(3 if results['pca_enabled'] else 2)
                    model_filename = f"{model_name.replace(' ', '_').lower()}_model.pkl"
                    scaler_filename = f"{model_name.replace(' ', '_').lower()}_scaler.pkl"
                    pca_filename = f"{model_name.replace(' ', '_').lower()}_pca.pkl"
                    
                    with download_cols[0]:
                        if os.path.exists(model_filename):
                            with open(model_filename, 'rb') as f:
                                st.download_button(
                                    label="Download Model",
                                    data=f,
                                    file_name=model_filename,
                                    mime='application/octet-stream',
                                    key=f"download_model_{model_name}_{results['timestamp']}",
                                    use_container_width=True
                                )
                    
                    with download_cols[1]:
                        if os.path.exists(scaler_filename):
                            with open(scaler_filename, 'rb') as f:
                                st.download_button(
                                    label="Download Scaler",
                                    data=f,
                                    file_name=scaler_filename,
                                    mime='application/octet-stream',
                                    key=f"download_scaler_{model_name}_{results['timestamp']}",
                                    use_container_width=True
                                )
                    
                    if results['pca_enabled']:
                        with download_cols[2]:
                            if os.path.exists(pca_filename):
                                with open(pca_filename, 'rb') as f:
                                    st.download_button(
                                        label="Download PCA",
                                        data=f,
                                        file_name=pca_filename,
                                        mime='application/octet-stream',
                                        key=f"download_pca_{model_name}_{results['timestamp']}",
                                        use_container_width=True
                                    )
                    
                    st.markdown("---")
                    
                    with st.expander("View Additional Details"):
                        if results['pca_enabled']:
                            st.markdown("*PCA Explained Variance:*")
                            pca_var_df = pd.DataFrame({
                                'Component': [f"PC{i+1}" for i in range(len(results['pca_variance_ratio']))],
                                'Variance (%)': [v * 100 for v in results['pca_variance_ratio']],
                                'Cumulative (%)': np.cumsum([v * 100 for v in results['pca_variance_ratio']])
                            })
                            st.dataframe(pca_var_df, use_container_width=True, hide_index=True)
                            st.markdown("---")
                        
                        st.markdown("*Best Hyperparameters:*")
                        st.json(results['best_params'])
                        
                        st.markdown(f"*Scaler Used:* {results['scaler_name']}")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("*Confusion Matrix:*")
                            cm_df = pd.DataFrame(
                                results['confusion_matrix'],
                                columns=['Predicted Healthy', 'Predicted PD'],
                                index=['Actual Healthy', 'Actual PD']
                            )
                            st.dataframe(cm_df, use_container_width=True)
                        
                        with col2:
                            st.markdown("*Classification Report:*")
                            report_df = pd.DataFrame(results['classification_report']).transpose()
                            st.dataframe(report_df.round(4), use_container_width=True)
                    
                    if st.button(f"Use {config_signature} for Predictions", key=f"use_{model_name}_{results['timestamp']}"):
                        st.session_state.current_model = entry['model']
                        st.session_state.current_scaler = entry['scaler']
                        st.session_state.current_pca = entry['pca']
                        st.session_state.active_model_name = model_name
                        st.success(f"{config_signature} is now active!")
                        st.rerun()
    
    # ============= TAB 2: VOICE PREDICTION =============
    with tab2:
        st.header("Voice-based Prediction")
        
        if not st.session_state.active_model_name:
            st.warning("Please train a model first from the Model Training tab.")
            return
        
        active_model_name = st.session_state.active_model_name
        st.success(f"Active Model: *{active_model_name}*")
        
        input_method = st.radio(
            "Choose Input Method:",
            ["Record Voice", "Upload Audio File"],
            horizontal=True
        )
        
        audio_file_path = None
        
        if input_method == "Record Voice":
            st.markdown(f"Click 'Start Recording' and say a sustained vowel sound (e.g., 'Aaaah') for {RECORD_DURATION} seconds.")
            
            has_recording = 'recorded_audio' in st.session_state and st.session_state.recorded_audio is not None
            
            if has_recording:
                col1, col2 = st.columns(2)
            else:
                col1, col2 = st.columns([1, 3])
            
            with col1:
                if st.button("Start Recording", type="primary", use_container_width=True):
                    with st.spinner(f"Recording for {RECORD_DURATION} seconds..."):
                        try:
                            recording = sd.rec(
                                int(RECORD_DURATION * SAMPLE_RATE),
                                samplerate=SAMPLE_RATE,
                                channels=1,
                                dtype='float64'
                            )
                            sd.wait()
                            
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                                audio_file_path = tmp_file.name
                                write(audio_file_path, SAMPLE_RATE, recording)
                            
                            st.session_state.recorded_audio = audio_file_path
                            st.success("Recording completed!")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Recording error: {e}")
            
            if has_recording:
                with col2:
                    if st.button("Clear & Re-record", type="secondary", use_container_width=True):
                        if 'recorded_audio' in st.session_state:
                            try:
                                if os.path.exists(st.session_state.recorded_audio):
                                    os.remove(st.session_state.recorded_audio)
                            except:
                                pass
                            del st.session_state.recorded_audio
                        st.rerun()
            
            if has_recording:
                audio_file_path = st.session_state.recorded_audio
                
                st.markdown("---")
                st.subheader("Your Recorded Audio")
                
                if os.path.exists(audio_file_path):
                    with open(audio_file_path, 'rb') as audio_file:
                        audio_bytes = audio_file.read()
                        st.audio(audio_bytes, format='audio/wav')
                    
                    st.download_button(
                        label="Download Recording",
                        data=audio_bytes,
                        file_name=f"voice_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav",
                        mime="audio/wav",
                        use_container_width=True
                    )
                else:
                    st.error("Recording file not found. Please record again.")
                    if 'recorded_audio' in st.session_state:
                        del st.session_state.recorded_audio
        
        else:
            uploaded_file = st.file_uploader(
                "Upload your voice sample (.wav format)",
                type=['wav']
            )
            
            if uploaded_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    audio_file_path = tmp_file.name
                
                st.markdown("---")
                st.subheader("Uploaded Audio")
                st.audio(uploaded_file, format='audio/wav')
        
        st.markdown("---")
        if audio_file_path and st.button("Analyze Voice Sample", type="primary", use_container_width=True):
            with st.spinner("Analyzing voice features..."):
                features, features_dict = extract_acoustic_features(audio_file_path)
                
                if features is not None:
                    model = st.session_state.current_model
                    scaler = st.session_state.current_scaler
                    pca = st.session_state.current_pca
                    
                    features_scaled = scaler.transform(features)
                    
                    if pca is not None:
                        features_scaled = pca.transform(features_scaled)
                    
                    prediction = model.predict(features_scaled)[0]
                    probability = model.predict_proba(features_scaled)[0]
                    
                    st.markdown("---")
                    st.subheader("Prediction Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Healthy Probability", f"{probability[0]*100:.2f}%")
                    with col2:
                        st.metric("Parkinson's Probability", f"{probability[1]*100:.2f}%")
                    
                    st.markdown("---")
                    
                    if prediction == 1:
                        st.error("### Parkinson's Disease Detected")
                        confidence = probability[1] * 100
                        st.metric("Detection Confidence", f"{confidence:.2f}%")
                        st.markdown("This is a screening tool only, not a diagnostic device. Consult a qualified neurologist for proper diagnosis.")
                    else:
                        st.success("### Healthy Voice Characteristics")
                        confidence = probability[0] * 100
                        st.metric("Healthy Confidence", f"{confidence:.2f}%")
                        st.markdown("This is a screening tool only. Regular health checkups are recommended.")
                    
                    st.markdown("---")
                    
                    st.subheader("Extracted Voice Features")
                    
                    if pca is None:
                        features_df = pd.DataFrame({
                            'Feature': list(features_dict.keys()),
                            'Value': [f"{val:.6f}" for val in features_dict.values()]
                        })
                        
                        col1, col2 = st.columns(2)
                        mid_point = len(features_df) // 2
                        
                        with col1:
                            st.dataframe(
                                features_df.iloc[:mid_point],
                                use_container_width=True,
                                hide_index=True
                            )
                        
                        with col2:
                            st.dataframe(
                                features_df.iloc[mid_point:],
                                use_container_width=True,
                                hide_index=True
                            )
                    else:
                        features_df = pd.DataFrame({
                            'Feature': list(features_dict.keys()),
                            'Value': [f"{val:.6f}" for val in features_dict.values()]
                        })
                        
                        with st.expander("View Original Features"):
                            col1, col2 = st.columns(2)
                            mid_point = len(features_df) // 2
                            
                            with col1:
                                st.dataframe(
                                    features_df.iloc[:mid_point],
                                    use_container_width=True,
                                    hide_index=True
                                )
                            
                            with col2:
                                st.dataframe(
                                    features_df.iloc[mid_point:],
                                    use_container_width=True,
                                    hide_index=True
                                )
                        
                        st.markdown(f"PCA-Transformed Features ({pca.n_components_} components):")
                        
                        pca_df = pd.DataFrame({
                            'Component': [f"PC{i+1}" for i in range(pca.n_components_)],
                            'Value': [f"{val:.6f}" for val in features_scaled[0]]
                        })
                        
                        col1, col2 = st.columns(2)
                        mid_point = len(pca_df) // 2
                        
                        with col1:
                            st.dataframe(
                                pca_df.iloc[:mid_point],
                                use_container_width=True,
                                hide_index=True
                            )
                        
                        with col2:
                            st.dataframe(
                                pca_df.iloc[mid_point:],
                                use_container_width=True,
                                hide_index=True
                            )
                    
                    csv = features_df.to_csv(index=False)
                    st.download_button(
                        label="Download Extracted Features",
                        data=csv,
                        file_name="extracted_features.csv",
                        mime="text/csv"
                    )
    
    # ============= TAB 3: MODEL COMPARISON =============
    with tab3:
        st.header("Model Comparison")
        
        if not st.session_state.model_history:
            st.info("No models have been trained yet. Train some models in the Model Training tab to see comparisons.")
            return
        
        st.markdown("### All Trained Models")
        
        comparison_data = []
        for idx, entry in enumerate(st.session_state.model_history):
            results = entry['results']
            
            config_id = f"{entry['model_name']}"
            if results['pca_enabled']:
                config_id += f" | PCA-{results['n_components']}"
            if results['smote_used']:
                config_id += " | SMOTE"
            config_id += f" | {results['scaler_name']}"
            
            comparison_data.append({
                'Model': entry['model_name'],
                'Configuration': config_id,
                'Scaler': results['scaler_name'],
                'PCA': f"{results['n_components']} comp" if results['pca_enabled'] else "No",
                'SMOTE': "Yes" if results['smote_used'] else "No",
                'Train Acc (%)': f"{results['train_accuracy']*100:.2f}",
                'Test Acc (%)': f"{results['test_accuracy']*100:.2f}",
                'Overfitting Gap (%)': f"{results['overfitting_gap']*100:.2f}",
                'Status': 'SEVERE OVERFIT' if results['train_accuracy'] > 0.95 else ('MODERATE OVERFIT' if results['overfitting_gap'] > 0.10 else 'GOOD'),
                'Timestamp': results['timestamp']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Model": st.column_config.TextColumn("Model Name", width="medium"),
                "Configuration": st.column_config.TextColumn("Full Configuration", width="large"),
                "Scaler": st.column_config.TextColumn("Scaler", width="small"),
                "PCA": st.column_config.TextColumn("PCA", width="small"),
                "SMOTE": st.column_config.TextColumn("SMOTE", width="small"),
                "Train Acc (%)": st.column_config.NumberColumn("Train Acc (%)", width="small"),
                "Test Acc (%)": st.column_config.NumberColumn("Test Acc (%)", width="small"),
                "Overfitting Gap (%)": st.column_config.NumberColumn("Gap (%)", width="small"),
                "Status": st.column_config.TextColumn("Status", width="medium"),
                "Timestamp": st.column_config.TextColumn("Trained At", width="medium")
            }
        )
        
        st.markdown("---")
        
        st.subheader("Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("*Best Test Accuracy:*")
            best_test = max(comparison_data, key=lambda x: float(x['Test Acc (%)']))
            st.success(f"{best_test['Configuration']}: {best_test['Test Acc (%)']}%")
        
        with col2:
            st.markdown("*Lowest Overfitting Gap:*")
            best_gap = min(comparison_data, key=lambda x: float(x['Overfitting Gap (%)']))
            st.success(f"{best_gap['Configuration']}: {best_gap['Overfitting Gap (%)']}%")
        
        st.markdown("---")
        
        st.markdown("### Export Comparison")
        csv_export = comparison_df.to_csv(index=False)
        st.download_button(
            label="Download Comparison Table (CSV)",
            data=csv_export,
            file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

if __name__ == "__main__":
    main()