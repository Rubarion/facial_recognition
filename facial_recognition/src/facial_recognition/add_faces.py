import os
from .recognizer import SimpleFaceRecognizer


def add_faces_from_folder(folder_path=None):
    """
    Adds all faces from a specified folder (or the current script's folder by default)
    to the known faces database.
    
    Args:
        folder_path (str, optional): Path to the folder containing images.
                                     If None, uses the folder of the calling script.
    """
    # --- Determine folder ---
    if folder_path is None:
        # Use the folder where the calling script resides, not the package
        try:
            import __main__
            folder_path = os.path.dirname(os.path.abspath(__main__.__file__))
        except AttributeError:
            # Fallback for interactive sessions (e.g., notebooks or Python shell)
            folder_path = os.getcwd()

    print(f"[INFO] Scanning folder: {folder_path}")
    if not os.path.exists(folder_path):
        print(f"[ERROR] Folder not found: {folder_path}")
        return

    # --- Load recognizer ---
    recognizer = SimpleFaceRecognizer()

    # --- Find images ---
    image_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    if not image_files:
        print("[WARN] No image files found in this folder.")
        return

    # --- Add each image ---
    for filename in image_files:
        name = os.path.splitext(filename)[0]
        file_path = os.path.join(folder_path, filename)

        print(f"[ADD] Adding {name} from {file_path}")
        try:
            recognizer.add_person(name, file_path)
        except Exception as e:
            print(f"[ERROR] Failed to add {name}: {e}")

    print("[INFO] Face database updated successfully.")
