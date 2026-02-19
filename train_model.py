import os
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import matplotlib.pyplot as plt

# Parameters
DATASET_PATH = "Dataset"
VISUALIZATION_PATH = "Spectrograms_Visualization"
SAMPLE_RATE = 22050
DURATION = 3 # Seconds
N_MELS = 128
MAX_PAD_LEN = 130 # Sample width for spectrogram (depends on duration & hop_length)

# Ensure Visualization Directory Exists
if not os.path.exists(VISUALIZATION_PATH):
    os.makedirs(VISUALIZATION_PATH)

def save_spectrogram_image(log_mel_spectrogram, file_path, label):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_mel_spectrogram, sr=SAMPLE_RATE, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel-Spectrogram - {label}')
    plt.tight_layout()
    
    # Create filename based on original file
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    save_name = f"{label}_{base_name}.png"
    save_path = os.path.join(VISUALIZATION_PATH, save_name)
    
    plt.savefig(save_path)
    plt.close()

def extract_features(file_path, label=None):
    try:
        audio, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Padding/Truncating to ensure consistent length
        target_length = SAMPLE_RATE * DURATION
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
            
        # Compute Mel-Spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=N_MELS)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Save Visualization if label is provided
        if label:
            save_spectrogram_image(log_mel_spectrogram, file_path, label)
        
        # Ensure shape (N_MELS, fixed_width)
        if log_mel_spectrogram.shape[1] < MAX_PAD_LEN:
             log_mel_spectrogram = np.pad(log_mel_spectrogram, ((0, 0), (0, MAX_PAD_LEN - log_mel_spectrogram.shape[1])))
        else:
             log_mel_spectrogram = log_mel_spectrogram[:, :MAX_PAD_LEN]
             
        return log_mel_spectrogram
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}: {e}")
        return None

def load_data(dataset_path):
    features = []
    labels = []
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                # Assuming folder name is the label
                label = os.path.basename(root)
                
                print(f"Processing {file_path}...")
                data = extract_features(file_path, label=label)
                
                if data is not None:
                    features.append(data)
                    labels.append(label)
                    
    return np.array(features), np.array(labels)

def create_model(input_shape, num_classes):
    model = Sequential()
    
    # CNN architecture
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    print("Loading dataset...")
    X, y = load_data(DATASET_PATH)
    
    if len(X) == 0:
        print("No audio files found! Please check the Dataset directory.")
        exit()

    # Encode Labels
    le = LabelEncoder()
    y = to_categorical(le.fit_transform(y))
    
    # Reshape for CNN (Batch, Rows, Cols, Channels)
    # Mel Spectrogram shape is (N_MELS, Time Steps)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training on {len(X_train)} samples, validating on {len(X_test)} samples.")
    
    # Create Model
    input_shape = (X.shape[1], X.shape[2], 1)
    num_classes = y.shape[1]
    model = create_model(input_shape, num_classes)
    
    model.summary()
    
    # Train
    model.fit(X_train, y_train, epochs=30, batch_size=4, validation_data=(X_test, y_test))
    
    # Save Model
    print("Saving model...")
    model.save('speaker_recognition_model.h5')
    
    # Save Label Encoder pairs
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
        
    print("Model training complete.")
