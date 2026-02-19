# Beginner's Guide to `train_model.py`

This guide explains every part of the `train_model.py` script. The goal of this script is to teach the computer how to recognize different speakers by "looking" at the sound of their voice.

## The Big Idea: How it Works
1.  **Sound to Image**: Computers are very good at recognizing images (like a cat vs. a dog). We turn sound into an image called a **Spectrogram**.
2.  **Learning**: We show the computer many spectrograms of different people (e.g., "This image is Rishav," "This image is Utkarsh").
3.  **Model**: The computer builds a brain (Neural Network) to find patterns in these images.

---

## Code Explanation

### 1. Importing Libraries
Think of libraries as toolboxes. We don't want to build a hammer from scratch; we just buy one.

```python
import os
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
# ... (other imports)
import matplotlib.pyplot as plt
```

-   **`os`**: Helps us work with files and folders on your computer (like finding where the audio files are).
-   **`numpy`**: The ultimate math toolbox. It handles big lists of numbers (arrays) very fast.
-   **`librosa`**: A specialized toolbox for analyzing **audio and music**. It's what we use to load sound files.
-   **`tensorflow` & `keras`**: Google's toolbox for **Artificial Intelligence**. This is what builds and trains the "brain" (Model).
-   **`sklearn` (scikit-learn)**: A helper for data science. We use it to shuffle our data and split it into training/testing sets.
-   **`pickle`**: A way to save Python objects (like our label list) to a file so we can use them later.
-   **`matplotlib`**: A drawing library we use to create and save the spectrogram images.

### 2. Setting Parameters
These are the settings or rules for our project.

```python
DATASET_PATH = "Dataset"
VISUALIZATION_PATH = "Spectrograms_Visualization"
SAMPLE_RATE = 22050
DURATION = 3 
N_MELS = 128
MAX_PAD_LEN = 130 
```

-   **`DATASET_PATH`**: Where your audio files live.
-   **`SAMPLE_RATE = 22050`**: The quality of the audio. It means we take 22,050 measurements of sound every second.
-   **`DURATION = 3`**: We only look at exactly 3 seconds of audio. If a file is shorter, we add silence; if longer, we cut it.
-   **`N_MELS = 128`**: The "height" of our spectrogram image. It represents frequency resolution (pitch).
-   **`MAX_PAD_LEN = 130`**: The "width" of our spectrogram image. It represents time.

### 3. Extracting Features (`extract_features` function)
This is the **most important function**. It turns an audio file into numbers the AI can understand.

```python
def extract_features(file_path, label=None):
    try:
        # 1. Load the audio file
        audio, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
```
-   `librosa.load`: Opens the `.wav` file. It loads the sound wave into `audio` (a list of numbers).

```python
        # 2. Pad or Truncate
        target_length = SAMPLE_RATE * DURATION
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
```
-   **Why?** The AI needs every input to be the *exact same size*.
-   If the audio is too short, we add zeros (silence) to the end (`np.pad`).
-   If it's too long, we chop off the end.

```python
        # 3. Create Mel-Spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=N_MELS)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
```
-   **Spectrogram**: Converts sound wave (time vs amplitude) into a heat map (time vs frequency vs loudness).
-   **Mel Scale**: Humans hear pitch logarithmically (we distinguish low notes better than high ones). The "Mel" scale adjusts the math to match human hearing.
-   **`power_to_db`**: Converts energy to Decibels (dB). This makes the quiet sounds visible and loud sounds not too overwhelming.

```python
        # 4. Save Image (Visualization)
        if label:
            save_spectrogram_image(log_mel_spectrogram, file_path, label)
```
-   If we know who the speaker is (`label`), we draw the spectrogram and save it as a picture so you can see it.

```python
        # 5. Fix Shape
        if log_mel_spectrogram.shape[1] < MAX_PAD_LEN:
             # ... paddding code ...
        return log_mel_spectrogram
```
-   Ensures the "image" width is exactly `130` pixels wide.

### 4. Loading Data (`load_data`)
This function acts like a librarian. It goes through every folder, picks up every file, processes it, and organizes it.

```python
def load_data(dataset_path):
    features = []  # List of all spectrograms (X)
    labels = []    # List of all names (Y)
    
    for root, dirs, files in os.walk(dataset_path):
        # ... loops through every file ...
        data = extract_features(file_path, label=label)
        features.append(data)
        labels.append(label)
```
-   At the end, we have two lists:
    -   `features`: A pile of spectrogram images.
    -   `labels`: A pile of names corresponding to those images.

### 5. Creating the Brain (`create_model`)
This defines the structure of our **Convolutional Neural Network (CNN)**.

```python
def create_model(input_shape, num_classes):
    model = Sequential()
    
    # Layer 1
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
```
-   **`Conv2D` (Convolution)**: This is the "eye". It scans the image looking for simple shapes like lines or curves (which in audio might be a rising pitch or a constant tone).
-   **`MaxPooling2D`**: This simplifies the image. It shrinks the image by keeping only the most important (loudest) features. It makes processing faster and reduces noise.
-   **`relu`**: An activation function. It filters out negative values (like turning off pixels that don't matter).

```python
    # Flattening
    model.add(Flatten())
```
-   Turns the 2D image data into a single long line of numbers, so the final decision-making layers can read it.

```python
    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))
```
-   **`Dense`**: A standard neural network layer where every neuron is connected to every other neuron.
-   **`softmax`**: This converts the final numbers into percentages (probabilities). e.g., "70% Rishav, 20% Utkarsh, 10% Sawant".

### 6. The Main Execution (`if __name__ == "__main__":`)
This is where the script actually starts running.

1.  **`load_data`**: Get all the audio and turn into numbers.
2.  **`LabelEncoder`**: Computers don't understand text like "Rishav". We convert names to numbers:
    -   Rishav -> 0
    -   Utkarsh -> 1
    -   Sawant -> 2
3.  **`to_categorical`**: Converts numbers to vectors:
    -   0 -> `[1, 0, 0]`
    -   1 -> `[0, 1, 0]`
4.  **`train_test_split`**: We hide 20% of the data. We train on 80% and then use the hidden 20% to "quiz" the model and see how smart it is.
5.  **`model.fit`**: The actual **Training**. The model looks at the data, makes a guess, checks the answer, and corrects itself. It does this 30 times (`epochs=30`) over the data.
6.  **`model.save`**: We save the trained "brain" to a file (`.h5`) so we can use it later without training again.
