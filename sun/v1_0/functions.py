import cv2
import numpy as np

NUM_FRAMES  = 32    # 32 frames representativos por video (video de 4-6 seg a 30 fps)
IMG_WIDTH = 256
IMG_HEIGHT = 144

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return np.zeros((NUM_FRAMES, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)

    indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
    indices = set(indices)
    frames = []
    frame_idx = 0

    while True:
        success, frame = cap.read()
        if not success:
            break
        if frame_idx in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        frame_idx += 1
        if len(frames) == NUM_FRAMES:
            break
    cap.release()
    while len(frames) < NUM_FRAMES:
        frames.append(frames[-1])
    return np.array(frames, dtype=np.float32)