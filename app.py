from flask import Flask, request, jsonify
from flask_cors import CORS  # Enable CORS for cross-origin requests
import os
import gdown
import concurrent.futures
from moviepy.editor import VideoFileClip
import whisper
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from collections import Counter
import re

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Lazy-load the heavy Whisper model
whisper_model = None
def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        whisper_model = whisper.load_model("medium")
    return whisper_model

# Load emotion model at startup (if possible, consider lazy-loading if this model is very large)
FER_MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_model.h5")
emotion_model = load_model(FER_MODEL_PATH)
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Function to download video from Google Drive
def download_video(drive_link, save_path="video.mp4"):
    gdown.download(drive_link, save_path, quiet=False)
    return save_path

# Function to extract audio from video
def extract_audio(video_path):
    audio_path = "extracted_audio.wav"
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, codec='pcm_s16le', fps=16000)
    return audio_path

# Function to transcribe audio using Whisper
def transcribe_audio(audio_path, language):
    model = get_whisper_model()
    result = model.transcribe(audio_path, language=language)
    return result.get("text", "")

# Function to filter only English words
def filter_english_words(text):
    return " ".join(re.findall(r'\b[a-zA-Z]+\b', text))

# Function to check answer correctness based on expected keywords
def check_answer_correctness(transcribed_text, expected_keywords):
    transcribed_text = filter_english_words(transcribed_text)
    matched_keywords = [word for word in expected_keywords if word.lower() in transcribed_text.lower()]
    correctness = (len(matched_keywords) / len(expected_keywords)) * 100 if expected_keywords else 0
    return correctness

# Function to analyze video for engagement and emotion detection
def analyze_video(video_path):
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
    cap = cv2.VideoCapture(video_path)
    engagement_frames, total_frames = 0, 0
    emotions_detected = []
    frame_skip = 5
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1
        if frame_index % frame_skip != 0:
            continue
        
        total_frames += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            engagement_frames += 1
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_face = cv2.resize(gray_frame, (48, 48)) / 255.0
            emotion_prediction = emotion_model.predict(np.expand_dims(resized_face, axis=0))[0]
            detected_emotion = EMOTION_LABELS[np.argmax(emotion_prediction)]
            emotions_detected.append(detected_emotion)

    cap.release()
    engagement_ratio = engagement_frames / total_frames if total_frames > 0 else 0
    nervousness_score = "Low" if engagement_ratio > 0.7 else ("Moderate" if engagement_ratio > 0.4 else "High")
    most_frequent_emotion = Counter(emotions_detected).most_common(1)[0][0] if emotions_detected else "Neutral"
    return {
        "Overall Confidence Level": "High" if engagement_ratio > 0.7 else ("Moderate" if engagement_ratio > 0.4 else "Low"),
        "Nervousness": nervousness_score,
        "Engagement Level": "High" if engagement_ratio > 0.7 else ("Moderate" if engagement_ratio > 0.4 else "Low"),
        "Most Frequent Emotion": most_frequent_emotion
    }

# API endpoint to process video and analyze results
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    video_link = data.get("video_link")
    language = data.get("language", "english")
    expected_keywords = data.get("keywords", [])
    
    if not video_link:
        return jsonify({"error": "Missing video link"}), 400

    video_path = download_video(video_link)
    
    # Run audio and video processing concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        audio_future = executor.submit(extract_audio, video_path)
        video_future = executor.submit(analyze_video, video_path)
        
        audio_path = audio_future.result()
        transcribed_text = transcribe_audio(audio_path, language)
        correctness = check_answer_correctness(transcribed_text, expected_keywords)
        video_analysis = video_future.result()
    
    result = {
        "Audio Analysis": {
            "Transcription": transcribed_text,
            "Correctness": correctness
        },
        "Video Analysis": video_analysis
    }
    return jsonify(result)

# For Vercel, expose the WSGI application
def handler(event, context):
    return app(event, context)

if __name__ == "__main__":
    # For local testing only; Vercel will use the exposed WSGI app
    app.run(host="0.0.0.0", port=5000)
