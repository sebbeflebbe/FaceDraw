import cv2
import time
from deepface import DeepFace

def detect_faces(frame, face_cascade):
    scaleFactor = 1.05
    minNeighbors = 20
    faces = face_cascade.detectMultiScale(frame, scaleFactor, minNeighbors)
    return faces

def detect_expression(frame, face_cascade):
    faces = detect_faces(frame, face_cascade)
    emotions = []
    if len(faces) == 0:
        return "No face detected"
    
    for (x, y, w, h) in faces:
        face_frame = frame[y:y+h, x:x+w]
        try:
            analysis_results = DeepFace.analyze(face_frame, actions=['emotion'], enforce_detection=False)
            
            if isinstance(analysis_results, list) and len(analysis_results) > 0:
                analysis = analysis_results[0]
                emotion_scores = analysis['emotion']
                dominant_emotion = max(emotion_scores, key=emotion_scores.get)
                emotions.append(dominant_emotion)
            else:
                emotions.append("Emotion data not found")

        except Exception as e:
            print(f"Error in emotion detection: {e}")
            emotions.append("Error in detection")

    return ", ".join(emotions) if emotions else "No face/emotion detected"

def main():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    last_analysis_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if time.time() - last_analysis_time >= 5:
            last_analysis_time = time.time()
            expression = detect_expression(frame, face_cascade)
            print("Dominant Emotion:", expression)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
