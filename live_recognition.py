import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import pickle
import queue
import sys

# Parameters match training
SAMPLE_RATE = 22050
DURATION = 3 # Seconds
N_MELS = 128
MAX_PAD_LEN = 130 

def preprocess_audio(audio, sample_rate=22050):
    # Ensure 1D
    if len(audio.shape) > 1:
        audio = audio.flatten()
        
    # Truncate/Pad to 3 seconds
    target_length = sample_rate * DURATION
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        # Take the middle part if longer, or just the beginning?
        # Training used librosa.load which truncates or just takes first DURATION.
        # Let's take the first DURATION samples to be consistent.
        audio = audio[:target_length]
        
    # Compute Mel-Spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=N_MELS)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Pad/Truncate frequency axis (width)
    if log_mel_spectrogram.shape[1] < MAX_PAD_LEN:
         log_mel_spectrogram = np.pad(log_mel_spectrogram, ((0, 0), (0, MAX_PAD_LEN - log_mel_spectrogram.shape[1])))
    else:
         log_mel_spectrogram = log_mel_spectrogram[:, :MAX_PAD_LEN]
         
    return log_mel_spectrogram

def record_audio():
    q = queue.Queue()
    
    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    print("Type 'start' to begin recording...")
    while True:
        user_input = input().strip().lower()
        if user_input == 'start':
            break

    print("Recording... Type 'stop' to end.")
    
    # Start stream
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback)
    stream.start()
    
    while True:
        user_input = input().strip().lower()
        if user_input == 'stop':
            break
            
    stream.stop()
    stream.close()
    
    print("Recording finished.")
    
    # Collect data from queue
    audio_data = []
    while not q.empty():
        audio_data.append(q.get())
        
    return np.concatenate(audio_data, axis=0)

def main():
    # Load Model
    try:
        model = tf.keras.models.load_model('speaker_recognition_model.h5')
        print("Model loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please run train_model.py first.")
        return

    # Load Label Encoder
    try:
        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        print("Label encoder loaded.")
    except Exception as e:
        print(f"Error loading label encoder: {e}")
        return

    while True:
        audio = record_audio()
        
        if len(audio) == 0:
            print("No audio recorded.")
            continue
            
        print("Processing audio...")
        features = preprocess_audio(audio)
        
        # Reshape for Model (1, N_MELS, MAX_PAD_LEN, 1)
        features = features.reshape(1, features.shape[0], features.shape[1], 1)
        
        prediction = model.predict(features)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class = le.inverse_transform([predicted_class_index])[0]
        confidence = prediction[0][predicted_class_index]
        
        print(f"\nResult: Sound recorded matched with: {predicted_class} (Confidence: {confidence:.2f})")
        
        print("\nDo you want to try again? (yes/no)")
        retry = input().strip().lower()
        if retry != 'yes':
            break

if __name__ == "__main__":
    main()
