import sys
import os
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# ============================================================
# 1️⃣ MONKEY-PATCH mediapipe BEFORE importing the library
# ============================================================
class FakeSolutions:
    pass

class FakeMP:
    solutions = FakeSolutions()

sys.modules["mediapipe"] = FakeMP()


# ============================================================
# 2️⃣ Import your recognizer AFTER patching mediapipe
# ============================================================
from facial_recognition.recognizer import SimpleFaceRecognizer


# ============================================================
# 3️⃣ Manual face-crop function (CelebA images are aligned)
# ============================================================
def fixed_crop(img):
    """
    Crops the face region using CelebA's standard aligned format:
    The face is centered and already aligned. We crop 15% borders.
    """
    h, w = img.shape[:2]
    x1 = int(0.15 * w)
    x2 = int(0.85 * w)
    y1 = int(0.15 * h)
    y2 = int(0.85 * h)
    return img[y1:y2, x1:x2]


# ============================================================
# 4️⃣ Dataset paths
# ============================================================
DATASET = r"Path to the extracted folder img_align_celeba containing the image files"
IDENTITY_FILE = r"Path to the text file identity_CelebA.txt that comes along with this dataset"

# ============================================================
# 5️⃣ Load id → list_of_images
# ============================================================
pid_images = defaultdict(list)

with open(IDENTITY_FILE) as f:
    for line in f:
        img, pid = line.strip().split()
        pid_images[pid].append(img)

print("[INFO] Total identities:", len(pid_images))

# ============================================================
# 6️⃣ Create one shared recognizer instance
# ============================================================
recognizer = SimpleFaceRecognizer()

# ============================================================
# 7️⃣ Compute reference embeddings
# ============================================================
reference = {}

def compute_ref(pid, img_list):
    img_path = os.path.join(DATASET, img_list[0])
    img = cv2.imread(img_path)
    if img is None:
        return None

    crop = fixed_crop(img)
    emb = recognizer.embed(crop)
    return pid, emb


print("[INFO] Computing reference embeddings...")
with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
    for result in tqdm(ex.map(lambda p: compute_ref(p[0], p[1]), pid_images.items()),
                       total=len(pid_images)):
        if result:
            pid, emb = result
            reference[pid] = emb


# ============================================================
# 8️⃣ Recognition test for remaining images
# ============================================================
correct = 0
total = 0

def test_one(pid, img_name, ref_emb):
    global correct, total

    img_path = os.path.join(DATASET, img_name)
    img = cv2.imread(img_path)
    if img is None:
        return (0, 0)

    crop = fixed_crop(img)
    emb = recognizer.embed(crop)

    score = np.dot(emb, ref_emb)

    # 0.3 = your default threshold
    if score > 0.3:
        return (1, 1)
    else:
        return (0, 1)


print("[INFO] Testing recognition in parallel...")
with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
    tasks = []
    for pid, imgs in pid_images.items():
        if pid not in reference:
            continue
        ref_emb = reference[pid]

        for img_name in imgs[1:]:  # skip the first (reference)
            tasks.append((pid, img_name, ref_emb))

    for correct_add, total_add in tqdm(
            ex.map(lambda t: test_one(t[0], t[1], t[2]), tasks),
            total=len(tasks)):
        correct += correct_add
        total += total_add


# ============================================================
# 9️⃣ Final accuracy
# ============================================================
accuracy = 100 * correct / total
print("\n======================================")
print(" CelebA Face Recognition Accuracy")
print("======================================")
print(f"Correct: {correct}")
print(f"Total:   {total}")
print(f"Accuracy: {accuracy:.2f}%")
print("======================================")
