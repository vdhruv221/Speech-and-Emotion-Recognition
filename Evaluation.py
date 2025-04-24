import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import librosa
import pickle
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

# Argument parser for command-line options
parser = argparse.ArgumentParser(description="Evaluate emotion recognition model on test data")
parser.add_argument('--test_dir', type=str, default=r'C:\Users\dhruv\OneDrive\Desktop\EE708\Test',
                    help='Path to the test directory containing actor folders')
args = parser.parse_args()

# Paths to model and preprocessing tools
test_dir = args.test_dir
model_path = r"C:\Users\dhruv\OneDrive\Desktop\EE708\improved_emotion_recognition_model.h5"
normalizer_path = r"C:\Users\dhruv\OneDrive\Desktop\EE708\feature_normalizer.pkl"
label_encoder_path = r"C:\Users\dhruv\OneDrive\Desktop\EE708\label_encoder.pkl"

# Emotion mapping (RAVDESS dataset)
emotion_mapping = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

print("\nLoading model and preprocessing tools...")
# Load the trained model and preprocessing tools
model = load_model(model_path)
with open(normalizer_path, 'rb') as f:
    normalizer = pickle.load(f)
with open(label_encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)

# Print the model summary
print("\nModel Summary:")
model.summary()

def extract_features(file_path, max_pad_len=64):
    """Extract audio features from a file."""
    try:
        y, sr = librosa.load(file_path, sr=22050)
        y = librosa.effects.preemphasis(y, coef=0.97)
        y, _ = librosa.effects.trim(y, top_db=25)
        
        # Extract features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=1024, hop_length=512)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
        zcr = librosa.feature.zero_crossing_rate(y)
        harmonic = librosa.effects.harmonic(y)
        chroma = librosa.feature.chroma_stft(y=harmonic, sr=sr)

        # Pad or truncate features to the same length
        def pad_trunc(feature, max_len):
            pad_width = max_len - feature.shape[1]
            if pad_width > 0:
                return np.pad(feature, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                return feature[:, :max_len]
        
        # Apply padding/truncation
        mfcc = pad_trunc(mfcc, max_pad_len)
        mfcc_delta = pad_trunc(mfcc_delta, max_pad_len)
        mfcc_delta2 = pad_trunc(mfcc_delta2, max_pad_len)
        spectral_contrast = pad_trunc(spectral_contrast, max_pad_len)
        spectral_centroid = pad_trunc(spectral_centroid, max_pad_len)
        spectral_bandwidth = pad_trunc(spectral_bandwidth, max_pad_len)
        spectral_rolloff = pad_trunc(spectral_rolloff, max_pad_len)
        zcr = pad_trunc(zcr, max_pad_len)
        chroma = pad_trunc(chroma, max_pad_len)

        # Combine all features into a single array
        features = np.concatenate([
            mfcc,
            mfcc_delta,
            mfcc_delta2,
            spectral_contrast,
            spectral_centroid,
            spectral_bandwidth,
            spectral_rolloff,
            zcr,
            chroma
        ], axis=0)

        return features

    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

def preprocess_features(features_test):
    """Preprocess test features using the loaded normalizer."""
    n_samples, n_features, n_timesteps = features_test.shape
    reshaped_features_test = features_test.reshape(n_samples, -1)

    # Normalize using the previously fitted normalizer
    normalized_features_test = normalizer.transform(reshaped_features_test)

    # Reshape back to original dimensions with an added channel dimension
    normalized_features_test_reshaped = normalized_features_test.reshape(n_samples, n_features, n_timesteps).astype(np.float32)

    return normalized_features_test_reshaped

def process_actor_folder(actor_dir):
    """Process a folder containing audio files for a single actor."""
    if not os.path.isdir(actor_dir):
        return None, None
    
    wav_files = [f for f in os.path.join(actor_dir, f) for f in os.listdir(actor_dir) if f.endswith('.wav')]
    
    test_files = []
    test_labels = []
    
    for file in wav_files:
        file_parts = os.path.basename(file).split('-')
        
        if len(file_parts) >= 7:
            emotion_code = file_parts[2]
            if emotion_code in emotion_mapping:
                emotion_label = emotion_mapping[emotion_code]
                features = extract_features(file)

                if features is not None:
                    test_files.append(features)
                    test_labels.append(emotion_label)

    return np.array(test_files) if test_files else None, test_labels

def process_test_directory(test_dir):
    """Process all actor folders in the test directory."""
    all_test_files = []
    all_test_labels = []
    
    # List all subdirectories in the test directory (Actor folders)
    actor_folders = [os.path.join(test_dir, folder) for folder in os.listdir(test_dir) 
                     if os.path.isdir(os.path.join(test_dir, folder)) and 'Actor_' in folder]
    
    if not actor_folders:
        print(f"No actor folders found in {test_dir}")
        return None, None
    
    print(f"\nProcessing {len(actor_folders)} actor folders...")
    
    for actor_folder in actor_folders:
        print(f"Processing {os.path.basename(actor_folder)}...")
        
        # Find all WAV files in the actor folder
        wav_files = [os.path.join(actor_folder, f) for f in os.listdir(actor_folder) if f.endswith('.wav')]
        
        for file in wav_files:
            file_parts = os.path.basename(file).split('-')
            
            if len(file_parts) >= 7:
                emotion_code = file_parts[2]
                if emotion_code in emotion_mapping:
                    emotion_label = emotion_mapping[emotion_code]
                    features = extract_features(file)

                    if features is not None:
                        all_test_files.append(features)
                        all_test_labels.append(emotion_label)
    
    if not all_test_files:
        return None, None
        
    return np.array(all_test_files), all_test_labels

def evaluate_model(X_test_normalized_reshaped, y_test_onehot):
    """Evaluate the pre-trained model and display all relevant metrics."""
    print("\n=== Evaluating Pre-Trained Model ===")
    
    # Evaluate the model on the test set
    loss_and_metrics = model.evaluate(X_test_normalized_reshaped[..., np.newaxis], y_test_onehot, verbose=0)
    print("\nPre-Trained Model Metrics:")
    print(f"Loss: {loss_and_metrics[0]:.4f}")
    print(f"Accuracy: {loss_and_metrics[1]:.4f}")
    
    # Get predictions
    y_pred_probs = model.predict(X_test_normalized_reshaped[..., np.newaxis], verbose=0)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    y_true_labels = np.argmax(y_test_onehot, axis=1)
    
    # Calculate additional metrics
    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    precision = precision_score(y_true_labels, y_pred_labels, average="weighted", zero_division=0)
    recall = recall_score(y_true_labels, y_pred_labels, average="weighted", zero_division=0)
    f1 = f1_score(y_true_labels, y_pred_labels, average="weighted", zero_division=0)
    
    # Display metrics
    print("\nDetailed Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true_labels, y_pred_labels, target_names=label_encoder.classes_, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

def main():
    # Process entire test directory and extract test data and labels
    X_test_raw, y_test_raw_labels = process_test_directory(test_dir)

    if X_test_raw is None or len(X_test_raw) == 0:
        print("No valid test files found.")
        return

    print(f"\nProcessed {len(X_test_raw)} audio files for testing.")

    # Preprocess extracted features
    X_test_normalized_reshaped = preprocess_features(X_test_raw)

    # Encode labels into one-hot format
    y_test_encoded_labels = label_encoder.transform(y_test_raw_labels)
    y_test_onehot_labels = tf.keras.utils.to_categorical(y_test_encoded_labels)

    # Evaluate the pre-trained model on the test data
    evaluate_model(X_test_normalized_reshaped, y_test_onehot_labels)

if __name__ == "__main__":
    main()