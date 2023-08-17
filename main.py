# Import necessary libraries
from flask import Flask, request, jsonify, render_template
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import soundfile as sf
import librosa
import librosa.feature
import librosa.display
from sklearn.preprocessing import StandardScaler

# Feature Extraction
def extract_feature(file_name, mfcc, chroma, mel):
    with sf.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result

# Load the data
def load_data(test_size=0.2):
    x, y = [], []
    emotions = {
        'neutral': 'neutral',
        'happy': 'happy',
        'sad': 'sad',
        'angry': 'angry',
        'fear': 'fear',
        'disgust': 'disgust',
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
    observed_emotions = ['happy', 'sad', 'angry', 'fear', 'disgust']

    # Load data from Ravdess dataset
    for file in glob.glob("D:/M_Documents/Varsity/8th Semester/Project/Dataset/ser-ravdess-dataset/Actor_*/*.wav"):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)

    # Load data from TESS dataset
    for file in glob.glob("D:/M_Documents/Varsity/8th Semester/Project/Dataset/ser-tess-dataset/*AF_*/*.wav"):
        file_name = os.path.basename(file)
        emotion = file_name.split("_")[2][:-4]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)

    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

# Load data for the emotions from both datasets
x_train, x_test, y_train, y_test = load_data(test_size=0.2)

print("Training set shape:", x_train.shape)
print("Testing set shape:", x_test.shape)
print("Features extracted:", x_train.shape[1])

# Perform feature normalization
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Initialize the SVM classifier
model = SVC(kernel='rbf', C=10, gamma='scale')

# Train the model
model.fit(x_train, y_train)

# Predict for the test set
y_pred = model.predict(x_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

# Print the model accuracy
print("Model Accuracy: {:.2f}%".format(accuracy * 100))

# Calculate the accuracy of the model for each emotion
emotions = ['happy', 'sad', 'angry', 'fear', 'disgust']
accuracy_dict = {}
for emotion in emotions:
    indices = np.where(np.array(y_test) == emotion)[0]
    y_true_emotion = np.array(y_test)[indices]
    y_pred_emotion = y_pred[indices]
    accuracy = accuracy_score(y_true=y_true_emotion, y_pred=y_pred_emotion)
    accuracy_dict[emotion] = accuracy

# Print the accuracy for each emotion
for emotion, accuracy in accuracy_dict.items():
    print("{} Accuracy: {:.2f}%".format(emotion.capitalize(), accuracy * 100))

# Generate Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Generate the Classification Report
cr = classification_report(y_test, y_pred)
print("Classification Report:")
print(cr)


# Function to handle the uploaded audio file
def handle_audio_upload(change):
    audio_file = change.new
    audio_path = next(iter(audio_file))
    audio_content = audio_file[audio_path]["M_Documents"]
    audio_temp_path = "D:/M_Documents/Varsity/8th Semester/Project/Dataset/temp_audio/tmp*.wav"

    with open(audio_temp_path, "wb") as f:
        f.write(audio_content)

    feature = extract_feature(audio_temp_path, mfcc=True, chroma=True, mel=True)
    feature = scaler.transform([feature])  # Normalize the feature
    predicted_emotion = model.predict(feature)[0]
    print("Predicted Emotion:", predicted_emotion)

    # Calculate the F1 score for the predicted emotion
    y_true = np.array([predicted_emotion])
    y_pred = np.array([predicted_emotion])
    f1 = f1_score(y_true, y_pred, average='weighted')

    print("F1 Score:", f1)


# Create Flask app
app = Flask(__name__)

accuracy_dict = {'happy': 0.85, 'sad': 0.89, 'angry': 0.96, 'fear': 0.96, 'disgust': 0.97}
f1_dict = {'happy': 0.86, 'sad': 0.92, 'angry': 0.95, 'fear': 0.98, 'disgust': 0.92}

# Load data and train the model when the app starts
x_train, x_test, y_train, y_test = load_data(test_size=0.2)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
model = SVC(kernel='rbf', C=10, gamma='scale')
model.fit(x_train, y_train)


# Function to predict emotion from uploaded audio
@app.route('/predict', methods=['POST'])
def predict_emotion():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    audio_file = request.files['file']
    audio_temp_path = "tmp_uploaded.wav"

    audio_file.save(audio_temp_path)

    feature = extract_feature(audio_temp_path, mfcc=True, chroma=True, mel=True)
    feature = scaler.transform([feature])  # Normalize the feature
    predicted_emotion = model.predict(feature)[0]

    os.remove(audio_temp_path)

    # Calculate accuracy and F1 score for the predicted emotion
    accuracy = accuracy_dict.get(predicted_emotion, 0)
    f1 = f1_dict.get(predicted_emotion, 0)

    return render_template('index.html', predicted_emotion=predicted_emotion, accuracy=accuracy, f1=f1)


# Route for the home page with the upload button
@app.route('/')
def home():
    # Clear the cached template and initially display empty string for predicted emotion
    app.jinja_env.cache = {}
    # Initially, display empty string for predicted emotion
    return render_template('index.html', predicted_emotion='', accuracy='', f1='')


if __name__ == '__main__':
    app.run(debug=True)