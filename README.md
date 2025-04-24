# Speech Emotion Recognition Model

This repository contains a deep learning model for speech emotion recognition using audio features extracted from the RAVDESS dataset.

## Model Performance

The model achieves the following performance metrics on the test dataset:

```
Accuracy: 0.4500
Precision: 0.4695
Recall: 0.4500
F1 Score: 0.4364
```

## Project Structure

- `Model.py`: Script for training the emotion recognition model
- `Evaluation.py`: Script for evaluating the trained model on test data
- `improved_emotion_recognition_model.h5`: Trained model file
- `feature_normalizer.pkl`: Saved feature normalizer for preprocessing
- `label_encoder.pkl`: Label encoder for emotion classes

## Features

- Multi-feature extraction from audio files (MFCCs, spectral features, chroma)
- Data augmentation techniques (time/frequency masking, noise addition)
- CNN architecture with residual connections for improved learning
- Cyclic learning rate and advanced training strategies
- Comprehensive evaluation metrics

## Requirements

- Python 3.7+
- TensorFlow 2.x
- librosa
- scikit-learn
- numpy
- matplotlib
- seaborn
- soundfile

You can install the required packages using:

```bash
pip install tensorflow librosa scikit-learn numpy matplotlib seaborn soundfile
```

## Usage

### Training the Model

To train the model from scratch, run:

```bash
python Model.py
```

This will:
1. Process audio files from the training dataset
2. Extract and normalize audio features
3. Train the CNN model
4. Save the trained model and preprocessing tools

### Evaluating the Model

To evaluate the trained model on test data, run:

```bash
python Evaluation.py --test_dir "PATH_TO_TEST_DIRECTORY"
```

Where `PATH_TO_TEST_DIRECTORY` is the path to your test dataset directory containing actor folders with WAV files.

Example:
```bash
python Evaluation.py --test_dir "C:\Users\dhruv\OneDrive\Desktop\EE708\Test"
```

The evaluation script will:
1. Load the pre-trained model and preprocessing tools
2. Process audio files from the test directory
3. Extract and normalize features
4. Perform prediction and evaluation
5. Display detailed metrics and confusion matrix

## Dataset Structure

The model expects audio files following the RAVDESS naming convention:
- File naming format: `modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav`
- Example: `03-01-04-01-02-01-12.wav`

Emotion codes:
- 01: neutral
- 02: calm
- 03: happy
- 04: sad
- 05: angry
- 06: fearful
- 07: disgust
- 08: surprised

## Model Architecture

The model uses a CNN architecture with:
- Multiple convolutional layers with batch normalization
- Residual connections to prevent gradient vanishing
- Dropout for regularization
- Dense layers with L2 regularization

## Citation

If you use this model in your research, please cite the RAVDESS dataset:

```
Livingstone SR, Russo FA (2018) The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): 
A dynamic, multimodal set of facial and vocal expressions in North American English.
PLoS ONE 13(5): e0196391. https://doi.org/10.1371/journal.pone.0196391
```
