import os
import cv2
import json
import numpy as np
import mediapipe as mp
from collections import defaultdict

# Folder and file names
dataPath = os.path.join(os.path.dirname(__file__), "dataset")
outputFile = os.path.join(os.path.dirname(__file__), "angles.json")
margin = 5.0  # angle range buffer

# MediaPipe pose model
poseModel = mp.solutions.pose
jointMap = {
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

def calculateAngle(a, b, c):
    a, b, c = map(np.array, (a, b, c))
    cosVal = np.clip(np.dot(a - b, c - b) / (np.linalg.norm(a - b) * np.linalg.norm(c - b)), -1, 1)
    return np.degrees(np.arccos(cosVal))

def getPoint(lmList, name):
    point = lmList[poseModel.PoseLandmark[name].value]
    return [point.x, point.y, point.z]

poseData = {}

with poseModel.Pose(static_image_mode=True) as pose:
    for poseName in sorted(os.listdir(dataPath)):
        folder = os.path.join(dataPath, poseName)
        if not os.path.isdir(folder):
            continue

        print(f"Scanning {poseName}")
        jointAngles = defaultdict(list)

        for image in os.listdir(folder):
            if not image.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            imgPath = os.path.join(folder, image)
            img = cv2.imread(imgPath)
            if img is None:
                continue

            result = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not result.pose_landmarks:
                continue

            lmList = result.pose_landmarks.landmark
            for joint, (p1, p2, p3) in jointMap.items():
                try:
                    a = getPoint(lmList, p1)
                    b = getPoint(lmList, p2)
                    c = getPoint(lmList, p3)
                    jointAngles[joint].append(calculateAngle(a, b, c))
                except:
                    continue

        poseSummary = {}
        for joint, angles in jointAngles.items():
            if not angles:
                continue
            arr = np.array(angles)
            minAngle, maxAngle = arr.min(), arr.max()
            adjMin, adjMax = minAngle + margin, maxAngle - margin
            if adjMin > adjMax:
                adjMin, adjMax = minAngle, maxAngle
            poseSummary[joint] = {
                "angle_min": round(adjMin, 1),
                "angle_max": round(adjMax, 1),
                "tip": f"Adjust your {joint.replace('_', ' ')}."
            }

        poseData[poseName] = poseSummary
        print(f"[OK] Saved {len(poseSummary)} joints for {poseName}")

with open(outputFile, "w") as f:
    json.dump(poseData, f, indent=4)

print(f"\n[SUCCESS] Saved angles to {outputFile}")
