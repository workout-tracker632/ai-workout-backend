# gym_live.py
import cv2
import numpy as np
import mediapipe as mp
import time
import base64

INITIAL_COUNTERS = {'curl': 0, 'squat': 0, 'pushup': 0}

ACTIONS = ['curl', 'squat', 'pushup', 'none']

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

class GymLiveProcessor:
    def __init__(self):
        self.counters = INITIAL_COUNTERS.copy()
        self.action = 'none'
        self.reps = 0
        self.msg = ''
        self.time_diff = None
        self.stage = None
        self.last_time = { act: None for act in INITIAL_COUNTERS }

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

        self.cap = cv2.VideoCapture(0)

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(img_rgb)
            image_out = frame.copy()

            if results.pose_landmarks:
                self.mp_draw.draw_landmarks(image_out, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                lm = results.pose_landmarks.landmark

                # Curl angles
                l_sh, l_el, l_wr = [lm[i] for i in [self.mp_pose.PoseLandmark.LEFT_SHOULDER.value, self.mp_pose.PoseLandmark.LEFT_ELBOW.value, self.mp_pose.PoseLandmark.LEFT_WRIST.value]]
                r_sh, r_el, r_wr = [lm[i] for i in [self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value, self.mp_pose.PoseLandmark.RIGHT_ELBOW.value, self.mp_pose.PoseLandmark.RIGHT_WRIST.value]]
                l_curl = calculate_angle([l_sh.x, l_sh.y], [l_el.x, l_el.y], [l_wr.x, l_wr.y])
                r_curl = calculate_angle([r_sh.x, r_sh.y], [r_el.x, r_el.y], [r_wr.x, r_wr.y])

                # Squat angles
                r_hip, r_knee, r_ank = [lm[i] for i in [self.mp_pose.PoseLandmark.RIGHT_HIP.value, self.mp_pose.PoseLandmark.RIGHT_KNEE.value, self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]]
                l_hip, l_knee, l_ank = [lm[i] for i in [self.mp_pose.PoseLandmark.LEFT_HIP.value, self.mp_pose.PoseLandmark.LEFT_KNEE.value, self.mp_pose.PoseLandmark.LEFT_ANKLE.value]]
                r_squat = calculate_angle([r_hip.x, r_hip.y], [r_knee.x, r_knee.y], [r_ank.x, r_ank.y])
                l_squat = calculate_angle([l_hip.x, l_hip.y], [l_knee.x, l_knee.y], [l_ank.x, l_ank.y])

                prev = self.action
                if l_curl < 40 or r_curl < 40:
                    self.action = 'curl'
                elif r_squat < 110 or l_squat < 110:
                    self.action = 'squat'
                else:
                    self.action = 'none'

                if self.action != prev:
                    self.stage = None
                    self.reps = 0
                    self.last_time[self.action] = None

                now = time.time()
                if self.action == 'curl':
                    if l_curl > 160 and r_curl > 160:
                        self.stage = 'down'
                        self.last_time['curl'] = self.last_time['curl'] or now
                    if (l_curl < 40 or r_curl < 40) and self.stage == 'down':
                        self.stage = 'up'
                        self.time_diff = round(now - self.last_time['curl'], 2)
                        self.msg = f"Speed: {self.time_diff}s"
                        self.counters['curl'] += 1
                        self.reps = self.counters['curl']
                        self.last_time['curl'] = now

                if self.action == 'squat':
                    if l_squat > 160 and r_squat > 160:
                        self.stage = 'down'
                        self.last_time['squat'] = self.last_time['squat'] or now
                    if (l_squat < 110 or r_squat < 110) and self.stage == 'down':
                        self.stage = 'up'
                        self.time_diff = round(now - self.last_time['squat'], 2)
                        self.msg = f"Speed: {self.time_diff}s"
                        self.counters['squat'] += 1
                        self.reps = self.counters['squat']
                        self.last_time['squat'] = now
            else:
                self.msg = '⚠️ Please move into frame.'

            ret2, buf = cv2.imencode('.jpg', image_out)
            img_b64 = base64.b64encode(buf.tobytes()).decode('utf-8') if ret2 else ''

            yield {
                'counters': self.counters,
                'action': self.action,
                'reps': self.reps,
                'msg': self.msg,
                'time_diff': self.time_diff,
                'image': img_b64
            }
            time.sleep(0.1)

    def release(self):
        self.cap.release()
        self.pose.close()