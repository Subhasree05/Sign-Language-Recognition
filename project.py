import os
from flask import Flask, render_template, Response
import cv2
import pickle
import numpy as np
import mediapipe as mp

# Change to project directory
with open('model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

with open('data.pickle', 'rb') as f:
    data = pickle.load(f)


# Flask application setup
app = Flask(__name__)

# Load the pre-trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Set up MediaPipe for hand landmark detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, max_num_hands=1)

# Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Dictionary to map model predictions to letters
labels_dict = {0: 'A', 1: 'B', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'J', 7: 'K', 8: 'L', 9: 'M',
               10: 'N', 11: 'O', 12: 'P', 13: 'Q', 14: 'R', 15: 'S', 16: 'T', 17: 'U', 18: 'W',
               19: 'C', 20: 'H', 21: 'V', 22: 'X', 23: 'Y', 24: 'Z'}

# Function to generate frames from the webcam
def generate_frames():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Error: Could not read frame.")
                break

            # Convert frame to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            # Process hand landmarks if detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    x_, y_, data_aux = [], [], []

                    for lm in hand_landmarks.landmark:
                        x_.append(lm.x)
                        y_.append(lm.y)

                    # Normalize landmarks
                    box_width = max(x_) - min(x_)
                    box_height = max(y_) - min(y_)

                    for lm in hand_landmarks.landmark:
                        data_aux.append((lm.x - min(x_)) / box_width)
                        data_aux.append((lm.y - min(y_)) / box_height)

                    if len(data_aux) == 42:  # Ensure correct size for the model
                        try:
                            prediction = model.predict([np.asarray(data_aux)])
                            predicted_character = labels_dict[int(prediction[0])]

                            # Display prediction on the frame
                            cv2.putText(frame, predicted_character, (10, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
                        except Exception as e:
                            print(f"Prediction Error: {e}")

            # Encode the frame and yield it
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        # Release the webcam on exit
        cap.release()

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=False)
