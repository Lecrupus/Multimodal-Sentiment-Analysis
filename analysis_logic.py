# File: analysis_logic.py

# --- Imports from all your cells ---
import cv2
import collections
import librosa
import torch
from deepface import DeepFace
from transformers import pipeline

# --- Global models (Load them once) ---
# We load the models here so they don't reload on every request, which is slow.
print("Loading models... This may take a moment.")
text_sentiment_pipeline = pipeline("sentiment-analysis")
audio_emotion_classifier = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")
print("Models loaded successfully.")


# --- 1. Text Analysis Logic (from cell 65ed0661) ---
def analyze_text(text_to_analyze):
    try:
        results = text_sentiment_pipeline(text_to_analyze)
        # Format the result for the website
        result = results[0]
        return f"Sentiment: {result['label']} (Score: {result['score']:.4f})"
    except Exception as e:
        return f"An error occurred during text analysis: {e}"


# --- 2. Audio Analysis Logic (from cell 64b2ff82) ---
def analyze_audio(file_path):
    try:
        # Load and resample the audio (as in your code)
        speech_array, sampling_rate = librosa.load(file_path, sr=16000)
        
        # Analyze the audio
        results = audio_emotion_classifier(speech_array)
        
        # Format the results
        formatted_results = "--- Emotion Analysis Results (from Tone) ---<br>"
        for result in results:
            formatted_results += f"Emotion: {result['label']}, Score: {result['score']:.4f}<br>"
        return formatted_results
        
    except Exception as e:
        return f"An error occurred during audio analysis: {e}"


# --- 3. Image Analysis Logic (from cells 5ae15c95 / a9e4e260) ---
def analyze_image(file_path):
    try:
        # We use img_path directly, as in your code
        analysis = DeepFace.analyze(img_path = file_path, actions = ['emotion'])
        
        # Format the results
        dominant_emotion = analysis[0]['dominant_emotion']
        emotions = analysis[0]['emotion']
        
        formatted_results = f"<b>Dominant Emotion: {dominant_emotion}</b><br><br>"
        formatted_results += "--- Full Emotion Analysis ---<br>"
        for emotion, score in emotions.items():
            formatted_results += f"{emotion}: {score:.2f}%<br>"
        return formatted_results

    except Exception as e:
        return f"An error occurred: {e} (This often happens if a face isn't detected.)"


# --- 4. Video Analysis Logic (from cell f46b9aae) ---
def analyze_video(file_path):
    try:
        cap = cv2.VideoCapture(file_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frame_skip_interval = int(frame_rate) # 1 frame per second
        
        frame_count = 0
        detected_emotions = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break # End of video

            if frame_count % frame_skip_interval == 0:
                try:
                    analysis = DeepFace.analyze(frame,
                                                actions=['emotion'],
                                                enforce_detection=False)
                    if analysis and isinstance(analysis, list):
                        detected_emotions.append(analysis[0]['dominant_emotion'])
                except Exception as e:
                    pass # Skip frames that fail

            frame_count += 1
            
        cap.release()

        if not detected_emotions:
            return "Could not detect any faces or emotions in the video."
        
        # Tally up
        emotion_counts = collections.Counter(detected_emotions)
        formatted_results = "--- Video Facial Emotion Analysis Complete ---<br>"
        formatted_results += "Dominant emotions found:<br>"
        for emotion, count in emotion_counts.most_common():
            formatted_results += f"- {emotion}: {count} times<br>"
        return formatted_results

    except Exception as e:
        return f"An error occurred during video processing: {e}"