import cv2
from deepface import DeepFace

def detect_faces(frame, face_cascade):
    # If you want to use color frames, skip the conversion to grayscale
    # You can still use grayscale for better performance if color information is not necessary
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Adjust the scaleFactor and minNeighbors parameters as needed
    scaleFactor = 1.7  # Example: try 1.05 for finer scale steps
    minNeighbors = 2   # Example: try 2 or 3 for fewer restrictions
    
    faces = face_cascade.detectMultiScale(frame, scaleFactor, minNeighbors)
    return faces


def detect_expression(frame, face_cascade):
    faces = detect_faces(frame, face_cascade)
    if len(faces) == 0:
        return "No face detected"
    
    for (x, y, w, h) in faces:
        face_frame = frame[y:y+h, x:x+w]

        if face_frame.size == 0:
            print("Empty face frame.")
            continue

        try:
            analyses = DeepFace.analyze(face_frame, actions=['emotion'], enforce_detection=False)

            # Now iterate over the list of analyses
            for analysis in analyses:
                # Debug: Print each analysis result
                print("Analysis Result:", analysis)

                if 'emotion' in analysis:
                    emotion_scores = analysis['emotion']
                    dominant_emotion = max(emotion_scores, key=emotion_scores.get)
                    print("Dominant emotion:", dominant_emotion)
                else:
                    print("Emotion data not found in analysis")

        except Exception as e:
            print(f"Error in emotion detection: {e}")

    return "Emotion analysis complete"




def main():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        expression = detect_expression(frame, face_cascade)
        print(expression)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
