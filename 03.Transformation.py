import argparse
import cv2
import sys

from lib.transformation import Transformation


def main(file_path):
    try:
        if (file_path):
            image = cv2.imread(sys.argv[1])
            Transformer = Transformation(image)
            Transformer.transform_one_image(sys.argv[1])
        else:
            print(f"Usage: {sys.argv[0]} [path_to_image]")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A program to display image transformation."
    )

    parser.add_argument(
        "file_path",
        type=str,
        nargs="?",
        help="Image file path."
    )

    args = parser.parse_args()
    main(args.file_path)
