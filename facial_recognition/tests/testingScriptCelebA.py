import os
import cv2
from facial_recognition import SimpleFaceRecognizer
from tqdm import tqdm

DATASET = r"Path to the extracted folder img_align_celeba containing the image files"
IDENTITY_FILE = r"Path to the text file identity_CelebA.txt that comes along with this dataset"

recognizer = SimpleFaceRecognizer()

# Load identities
id_map = {}
with open(IDENTITY_FILE) as f:
    for line in f:
        img, pid = line.strip().split()
        id_map[img] = pid

# Pick 1 image per person for registration
registered = {}

correct = 0
total = 0

for img_name, pid in id_map.items():
    img_path = os.path.join(DATASET, img_name)

    if pid not in registered:
        recognizer.add_person(pid, img_path)
        registered[pid] = img_path
        continue

    result = recognizer.recognize_image(img_path, save_output=False)
    if result and result[0]["name"] == pid:
        correct += 1
    total += 1

print(f"\nCelebA Accuracy = {correct/total * 100:.2f}%")
