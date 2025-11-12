import argparse
import os
from .recognizer import SimpleFaceRecognizer
from .add_faces import add_faces_from_folder
from .remove_faces import remove_face_database
from .setup_model import main as setup_model_main


def main():
    parser = argparse.ArgumentParser(
        description="facial_recognition CLI Tool — Add, recognize, or remove faces"
    )
    subparsers = parser.add_subparsers(dest="command")
    
    # --- Subcommand: setup ---
    setup_parser = subparsers.add_parser(
        "setup",
        help="Download or verify the ArcFace model"
    )

    # --- Subcommand: add_faces ---
    add_parser = subparsers.add_parser(
        "add_faces",
        help="Add all faces from the current folder to the database"
    )
    
    add_parser.add_argument(
    "--folder",
    default=os.getcwd(),
    help="Path to folder containing face images (default: current directory)"
    )
    # --- Subcommand: recognize ---
    rec_parser = subparsers.add_parser(
        "recognize",
        help="Recognize faces from an image"
    )
    rec_parser.add_argument(
        "image_path",
        help="Path to the image file for recognition"
    )
    rec_parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Cosine similarity threshold for recognition (default: 0.3)"
    )

    # --- Subcommand: remove_faces ---
    subparsers.add_parser(
        "remove_faces",
        help="Remove the local face database file"
    )

    # Parse arguments
    args = parser.parse_args()

    # --- No command provided ---
    if not args.command:
        parser.print_help()
        return

    # --- Handle commands ---
    if args.command == "setup":
        setup_model_main()  # ✅ Calls your setup_model.py:main()

    elif args.command == "add_faces":
        folder = args.folder if hasattr(args, "folder") else os.getcwd()
        add_faces_from_folder(folder)

    elif args.command == "recognize":
        recognizer = SimpleFaceRecognizer()
        recognizer.recognize_image(args.image_path, threshold=args.threshold)

    elif args.command == "remove_faces":
        remove_face_database()


if __name__ == "__main__":
    main()
