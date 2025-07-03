import os
import cv2
import json
import mediapipe as mp
from collections import deque
import numpy as np

# Load and validate reference angles
angleFile = os.path.join(os.path.dirname(__file__), "angles.json")
if not os.path.exists(angleFile) or os.path.getsize(angleFile) == 0:
    raise FileNotFoundError(f"{angleFile} is missing or empty")

with open(angleFile) as f:
    allAngles = json.load(f)

poseList = list(allAngles.keys())
print("Select a yoga pose:\n")
for i, name in enumerate(poseList, 1):
    print(f"{i}. {name}")

try:
    selected = int(input("\nEnter pose number: "))
    if not (1 <= selected <= len(poseList)):
        raise ValueError
except ValueError:
    print("Invalid selection. Exiting.")
    exit()

poseName = poseList[selected - 1]
angleRef = allAngles[poseName]

# Ask user for visualization options
enable_grid = input("Show grid lines? (y/n): ").strip().lower() == 'y'
show_pose_landmarks = input("Show pose landmarks? (y/n): ").strip().lower() == 'y'

# Joint definitions
mpPose = mp.solutions.pose
JOINTS = {
    "left_knee": ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
    "right_knee": ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
    "left_elbow": ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
    "right_elbow": ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
    "left_shoulder": ("LEFT_ELBOW", "LEFT_SHOULDER", "LEFT_HIP"),
    "right_shoulder": ("RIGHT_ELBOW", "RIGHT_SHOULDER", "RIGHT_HIP"),
    "left_hip": ("RIGHT_HIP", "LEFT_HIP", "LEFT_KNEE"),
    "right_hip": ("LEFT_HIP", "RIGHT_HIP", "RIGHT_KNEE"),
    "spine": ("LEFT_HIP", "RIGHT_HIP", "NOSE"),
    "neck": ("LEFT_SHOULDER", "NOSE", "RIGHT_SHOULDER"),
}

def getAngle(a, b, c):
    a, b, c = map(np.array, (a, b, c))
    cos = np.clip(np.dot(a - b, c - b) / (np.linalg.norm(a - b) * np.linalg.norm(c - b)), -1, 1)
    return np.degrees(np.arccos(cos))

def getCoords(lms, p1, p2, p3):
    fetch = lambda p: [lms[mpPose.PoseLandmark[p].value].x, lms[mpPose.PoseLandmark[p].value].y]
    return fetch(p1), fetch(p2), fetch(p3)

buffers = {j: deque(maxlen=5) for j in JOINTS}

cap = cv2.VideoCapture(0)
cv2.namedWindow("Yoga Pose Validator", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Yoga Pose Validator", 960, 540)

# Countdown before capture
for i in range(3, 0, -1):
    ok, frame = cap.read()
    if not ok: continue
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, f"Get Ready: {i}", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 150, 255), 4)
    cv2.imshow("Yoga Pose Validator", frame)
    cv2.waitKey(1000)

# Pose validation loop
with mpPose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    holdTime = int(fps * 5)
    timer = 0
    wrongCount = 0
    wrongLimit = 10

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        issues = []

        result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if result.pose_landmarks:
            lms = result.pose_landmarks.landmark

            # Draw landmarks
            if show_pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    result.pose_landmarks,
                    mpPose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0,128,255), thickness=2)
                )

            for joint, ref in angleRef.items():
                if joint not in JOINTS:
                    continue
                try:
                    a, b, c = getCoords(lms, *JOINTS[joint])
                    buffers[joint].append(getAngle(a, b, c))
                    avg = sum(buffers[joint]) / len(buffers[joint])

                    if "angle_min" in ref and "angle_max" in ref:
                        if not (ref["angle_min"] <= avg <= ref["angle_max"]):
                            issues.append(ref["tip"])
                    elif "angle" in ref:
                        tol = ref.get("tolerance", 10) * 1.5
                        if abs(avg - ref["angle"]) > tol:
                            issues.append(ref["tip"])
                except:
                    pass

        if issues:
            wrongCount += 1
            if wrongCount >= wrongLimit:
                timer = 0

            boxW, pad = 320, 20
            x1 = 20
            y1 = frame.shape[0] - (30 + len(issues) * 25) - pad
            y2 = frame.shape[0] - pad
            x2 = x1 + boxW
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 1)
            cv2.putText(frame, "Adjust Your Posture", (x1 + 10, y1 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            for i, tip in enumerate(issues):
                cv2.putText(frame, f"- {tip}", (x1 + 10, y1 + 50 + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        else:
            wrongCount = 0
            timer += 1
            if timer >= holdTime:
                frame[:] = (255, 255, 255)
                cv2.putText(frame, "GOOD JOB!", (100, 180), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 128, 0), 5)
                cv2.putText(frame, f"You held the '{poseName}' pose correctly!", (60, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                cv2.putText(frame, "Press any key to close.", (100, 310),
                            cv2.FONT_HERSHEY_PLAIN, 1.8, (0, 0, 0), 2)
                cv2.imshow("Yoga Pose Validator", frame)
                cv2.waitKey(0)
                break

        # Draw grid if enabled
        if enable_grid:
            h, w = frame.shape[:2]
            cv2.line(frame, (w//2, 0), (w//2, h), (200, 200, 200), 1)
            cv2.line(frame, (0, h//2), (w, h//2), (200, 200, 200), 1)

        timeLeft = max(0, holdTime - timer) / fps
        cv2.putText(frame, f"Hold still: {int(timeLeft)}s", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 120, 255), 2)
        cv2.putText(frame, f"Pose: {poseName}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        cv2.imshow("Yoga Pose Validator", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
