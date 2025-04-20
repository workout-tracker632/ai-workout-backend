import cv2
import PoseModule2 as pm
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
import time

# Initialize Mediapipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Define relevant landmark indices for analysis
relevant_landmarks_indices = [
    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    mp_pose.PoseLandmark.LEFT_ELBOW.value,
    mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    mp_pose.PoseLandmark.LEFT_WRIST.value,
    mp_pose.PoseLandmark.RIGHT_WRIST.value,
    mp_pose.PoseLandmark.LEFT_HIP.value,
    mp_pose.PoseLandmark.RIGHT_HIP.value,
    mp_pose.PoseLandmark.LEFT_KNEE.value,
    mp_pose.PoseLandmark.RIGHT_KNEE.value,
    mp_pose.PoseLandmark.LEFT_ANKLE.value,
    mp_pose.PoseLandmark.RIGHT_ANKLE.value
]

# -------------------------
# Helper function definitions
# -------------------------
def calculate_angle(a, b, c):
    if np.any(np.array([a, b, c]) == 0):
        return -1.0  # placeholder for missing landmarks
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0/np.pi)
    return angle if angle <= 180.0 else 360 - angle

def calculate_distance(a, b):
    if np.any(np.array([a, b]) == 0):
        return -1.0
    return np.linalg.norm(np.array(a) - np.array(b))

def calculate_y_distance(a, b):
    if np.any(np.array([a, b]) == 0):
        return -1.0
    return np.abs(a[1] - b[1])

def draw_styled_text(frame, text, position,
                     font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.55,
                     font_color=(255,255,255), font_thickness=2,
                     bg_color=(0,0,0), padding=5):
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_x, text_y = position
    box_coords = ((text_x - padding, text_y + padding),
                  (text_x + text_size[0] + padding, text_y - text_size[1] - padding))
    cv2.rectangle(frame, box_coords[0], box_coords[1], bg_color, cv2.FILLED)
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

# -------------------------
# Repetition counting logic
# -------------------------
def count_repetition_push_up(detector, img, landmark_list, stage, counter, exercise_instance):
    right_arm_angle = detector.find_angle(img, 12, 14, 16)
    left_arm_angle = detector.find_angle(img, 11, 13, 15)
    exercise_instance.visualize_angle(img, right_arm_angle, landmark_list[12][1:])
    exercise_instance.visualize_angle(img, left_arm_angle, landmark_list[11][1:])
    if left_arm_angle < 220:
        stage = "down"
    if left_arm_angle > 240 and stage == "down":
        stage = "up"
        counter += 1
    return stage, counter

def count_repetition_squat(detector, img, landmark_list, stage, counter, exercise_instance):
    right_leg_angle = detector.find_angle(img, 24, 26, 28)
    left_leg_angle = detector.find_angle(img, 23, 25, 27)
    exercise_instance.visualize_angle(img, right_leg_angle, landmark_list[26][1:])
    if right_leg_angle > 160 and left_leg_angle < 220:
        stage = "down"
    if right_leg_angle < 140 and left_leg_angle > 210 and stage == "down":
        stage = "up"
        counter += 1
    return stage, counter

def count_repetition_bicep_curl(detector, img, landmark_list, stage_right, stage_left, counter, exercise_instance):
    right_arm_angle = detector.find_angle(img, 12, 14, 16)
    left_arm_angle = detector.find_angle(img, 11, 13, 15)
    exercise_instance.visualize_angle(img, right_arm_angle, landmark_list[14][1:])
    exercise_instance.visualize_angle(img, left_arm_angle, landmark_list[13][1:])
    if 160 < right_arm_angle < 200:
        stage_right = "down"
    if 140 < left_arm_angle < 200:
        stage_left = "down"
    if (stage_right == "down" and (right_arm_angle > 310 or right_arm_angle < 60)
       and (left_arm_angle > 310 or left_arm_angle < 60) and stage_left == "down"):
        stage_right = "up"
        stage_left = "up"
        counter += 1
    return stage_right, stage_left, counter

def count_repetition_shoulder_press(detector, img, landmark_list, stage, counter, exercise_instance):
    right_arm_angle = detector.find_angle(img, 12, 14, 16)
    left_arm_angle = detector.find_angle(img, 11, 13, 15)
    exercise_instance.visualize_angle(img, right_arm_angle, landmark_list[14][1:])
    if right_arm_angle > 280 and left_arm_angle < 80:
        stage = "down"
    if right_arm_angle < 240 and left_arm_angle > 120 and stage == "down":
        stage = "up"
        counter += 1
    return stage, counter

# -------------------------
# The main Exercise class
# -------------------------
class Exercise:
    def __init__(self):
        # Attempt to load classification model/scaler/encoder
        try:
            self.lstm_model = load_model('models/final_forthesis_bidirectionallstm_and_encoders_exercise_classifier_model.h5')
        except Exception as e:
            print(f"Error loading LSTM model: {e}")
            self.lstm_model = None
        
        try:
            self.scaler = joblib.load('models/thesis_bidirectionallstm_scaler.pkl')
        except Exception as e:
            print(f"Error loading scaler: {e}")
            self.scaler = None
        
        try:
            self.label_encoder = joblib.load('models/thesis_bidirectionallstm_label_encoder.pkl')
            self.exercise_classes = self.label_encoder.classes_
        except Exception as e:
            print(f"Error loading label encoder: {e}")
            self.label_encoder = None
            self.exercise_classes = []

    def extract_features(self, landmarks):
        features = []
        if len(landmarks) == len(relevant_landmarks_indices) * 3:
            # Angles
            features.append(calculate_angle(landmarks[0:3],  landmarks[6:9],  landmarks[12:15]))
            features.append(calculate_angle(landmarks[3:6],  landmarks[9:12],  landmarks[15:18]))
            features.append(calculate_angle(landmarks[18:21], landmarks[24:27], landmarks[30:33]))
            features.append(calculate_angle(landmarks[21:24], landmarks[27:30], landmarks[33:36]))
            features.append(calculate_angle(landmarks[0:3],  landmarks[18:21], landmarks[24:27]))
            features.append(calculate_angle(landmarks[3:6],  landmarks[21:24], landmarks[27:30]))
            features.append(calculate_angle(landmarks[18:21], landmarks[0:3],  landmarks[6:9]))
            features.append(calculate_angle(landmarks[21:24], landmarks[3:6],  landmarks[9:12]))
            # Distances
            distances = [
                calculate_distance(landmarks[0:3],  landmarks[3:6]),
                calculate_distance(landmarks[18:21], landmarks[21:24]),
                calculate_distance(landmarks[18:21], landmarks[24:27]),
                calculate_distance(landmarks[21:24], landmarks[27:30]),
                calculate_distance(landmarks[0:3],  landmarks[18:21]),
                calculate_distance(landmarks[3:6],  landmarks[21:24]),
                calculate_distance(landmarks[6:9],  landmarks[24:27]),
                calculate_distance(landmarks[9:12], landmarks[27:30]),
                calculate_distance(landmarks[12:15], landmarks[0:3]),
                calculate_distance(landmarks[15:18], landmarks[3:6]),
                calculate_distance(landmarks[12:15], landmarks[18:21]),
                calculate_distance(landmarks[15:18], landmarks[21:24])
            ]
            y_distances = [
                calculate_y_distance(landmarks[6:9],  landmarks[0:3]),
                calculate_y_distance(landmarks[9:12], landmarks[3:6])
            ]
            # Normalize
            normalization_factor = next((d for d in [
                calculate_distance(landmarks[0:3],  landmarks[18:21]),
                calculate_distance(landmarks[3:6],  landmarks[21:24]),
                calculate_distance(landmarks[18:21], landmarks[24:27]),
                calculate_distance(landmarks[21:24], landmarks[27:30])
            ] if d > 0), 0.5)

            normalized_distances = [
                d / normalization_factor if d != -1.0 else d for d in distances
            ]
            normalized_y_distances = [
                d / normalization_factor if d != -1.0 else d for d in y_distances
            ]
            features.extend(normalized_distances)
            features.extend(normalized_y_distances)
        else:
            print(f"Insufficient landmarks: expected {len(relevant_landmarks_indices)}, got {len(landmarks)//3}")
            features = [-1.0] * 22
        return features

    def visualize_angle(self, img, angle, landmark):
        # Draw the integer angle near the landmark position
        pos = tuple(np.multiply(landmark, [img.shape[1], img.shape[0]]).astype(int))
        cv2.putText(img, str(int(angle)), pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)

    def repetitions_counter(self, img, counter):
        # Overlay the current rep count in big text near top center
        text = str(counter)
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        text_x = (img.shape[1]-text_width) // 2
        text_y = 80
        cv2.putText(img, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

    def are_hands_joined(self, landmark_list, is_video=False):
        # Example logic to end counting if user joins hands
        left_wrist = landmark_list[15][1:]
        right_wrist = landmark_list[16][1:]
        distance = np.linalg.norm(np.array(left_wrist) - np.array(right_wrist))
        if distance < 30 and not is_video:
            print("JOINED HANDS => stopping count.")
            return True
        return False

    def process_video(self, cap, out, is_video, count_repetition_function,
                      multi_stage=False, counter=0, stage=None, stage_right=None, stage_left=None):
        detector = pm.posture_detector()
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break
            frame = detector.find_person(frame)
            landmark_list = detector.find_landmarks(frame, draw=False)
            if landmark_list:
                # If multi_stage is True => bicep curl is a two-arm logic
                if multi_stage:
                    stage_right, stage_left, counter = count_repetition_function(
                        detector, frame, landmark_list, stage_right, stage_left, counter, self
                    )
                else:
                    stage, counter = count_repetition_function(
                        detector, frame, landmark_list, stage, counter, self
                    )
                self.repetitions_counter(frame, counter)
                if self.are_hands_joined(landmark_list, is_video=is_video):
                    # If user joins hands, stop early
                    break
            # Write the processed frame if out is provided
            if out is not None:
                out.write(frame)
        return counter

    # Exercise-specific methods
    def push_up(self, cap, out, is_video=False, counter=0, stage=None):
        return self.process_video(cap, out, is_video, count_repetition_push_up,
                                  counter=counter, stage=stage)

    def squat(self, cap, out, is_video=False, counter=0, stage=None):
        return self.process_video(cap, out, is_video, count_repetition_squat,
                                  counter=counter, stage=stage)

    def bicept_curl(self, cap, out, is_video=False, counter=0, stage_right=None, stage_left=None):
        return self.process_video(cap, out, is_video, count_repetition_bicep_curl,
                                  multi_stage=True, counter=counter,
                                  stage_right=stage_right, stage_left=stage_left)

    def shoulder_press(self, cap, out, is_video=False, counter=0, stage=None):
        return self.process_video(cap, out, is_video, count_repetition_shoulder_press,
                                  counter=counter, stage=stage)

    # Live classification method
    def classify_and_count_live(self, flat_landmarks, current_stages, current_counters):
        """
        Use LSTM model for classification, then update repetition counters.
        """
        if len(flat_landmarks) != len(relevant_landmarks_indices)*3:
            print("Insufficient landmarks for classification")
            return "Unknown", current_counters, current_stages

        # Extract + scale features
        features = self.extract_features(flat_landmarks)
        scaled_features = self.scaler.transform([features]) if self.scaler else [features]

        # Predict label
        if self.lstm_model:
            pred = self.lstm_model.predict(scaled_features)
            predicted_index = pred.argmax(axis=1)[0]
            predicted_exercise = self.exercise_classes[predicted_index]
        else:
            predicted_exercise = "Unknown"

        # Use a dummy frame to run repetition logic
        dummy_frame = np.zeros((480,640,3), dtype=np.uint8)
        detector = pm.posture_detector()

        if predicted_exercise == "Push Up":
            stage, new_count = count_repetition_push_up(
                detector,
                dummy_frame,
                flat_landmarks,
                current_stages.get("Push Up"),
                current_counters.get("Push Up"),
                self
            )
            current_stages["Push Up"] = stage
            current_counters["Push Up"] = new_count

        elif predicted_exercise == "Squat":
            stage, new_count = count_repetition_squat(
                detector,
                dummy_frame,
                flat_landmarks,
                current_stages.get("Squat"),
                current_counters.get("Squat"),
                self
            )
            current_stages["Squat"] = stage
            current_counters["Squat"] = new_count

        elif predicted_exercise == "Bicept Curl":
            bc_stages = current_stages.get("Bicept Curl", {"stage_right": None, "stage_left": None})
            sr = bc_stages.get("stage_right")
            sl = bc_stages.get("stage_left")
            sr, sl, new_count = count_repetition_bicep_curl(
                detector,
                dummy_frame,
                flat_landmarks,
                sr,
                sl,
                current_counters.get("Bicept Curl"),
                self
            )
            current_stages["Bicept Curl"] = {"stage_right": sr, "stage_left": sl}
            current_counters["Bicept Curl"] = new_count

        elif predicted_exercise == "Shoulder Press":
            stage, new_count = count_repetition_shoulder_press(
                detector,
                dummy_frame,
                flat_landmarks,
                current_stages.get("Shoulder Press"),
                current_counters.get("Shoulder Press"),
                self
            )
            current_stages["Shoulder Press"] = stage
            current_counters["Shoulder Press"] = new_count

        else:
            predicted_exercise = "Unknown"

        return predicted_exercise, current_counters, current_stages
