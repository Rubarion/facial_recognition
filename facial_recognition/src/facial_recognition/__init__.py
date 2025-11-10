from .recognizer import SimpleFaceRecognizer
from .remove_faces import remove_face_database
from .add_faces import add_faces_from_folder


__all__ = [
    "SimpleFaceRecognizer",
    "remove_face_database",
    "add_person",
    "recognize_image",
    "add_faces_from_folder"
]


def add_person(name, image_path):
    """
    Convenience wrapper to add a person to the face database
    without manually creating a recognizer instance.
    """
    recognizer = SimpleFaceRecognizer()
    recognizer.add_person(name, image_path)
    print(f"[INFO] Added '{name}' successfully to the known faces database")


def recognize_image(image_path, threshold=None, save_output=True, return_embeddings=False):
    """
    Convenience wrapper for quick recognition using the saved database.
    Uses the default threshold from recognizer.py unless overridden.
    """
    recognizer = SimpleFaceRecognizer()

    if threshold is not None:
        print(f"[INFO] Using custom recognition threshold = {threshold}")
        results = recognizer.recognize_image(
            input_image=image_path,
            threshold=threshold,
            save_output=save_output,
            return_embeddings=return_embeddings
        )
    else:
        print("[INFO] Using default recognition threshold from recognizer.py")
        results = recognizer.recognize_image(
            input_image=image_path,
            save_output=save_output,
            return_embeddings=return_embeddings
        )

    # Print summary for quick feedback
    if not results:
        print("[WARN] No faces detected.")
    else:
        for r in results:
            print(f"[RESULT] {r['name']} (score={r['score']:.2f})")

            if return_embeddings:
                print(f"    → Embedding shape: {r['embedding'].shape}")

    return results
