import argparse
import cv2
import sys
import os

from lib.transformation import Transformation


def transform(args) -> None:
    transforms = []
    for key, value in args._get_kwargs():
        if value is True:
            transforms.append(key)

    root, dirs, files = next(os.walk(args.src_path))
    dirname = os.path.dirname(args.src_path)

    for file in files:
        image_path = f"{dirname}/{file}"
        image = cv2.imread(image_path)

        image_dst_path = f"{args.dst_path}/{file}"
        Transformer = Transformation(image)
        Transformer.transform_one_image(image_dst_path, transforms)


def main(args) -> None:
    try:
        if (args.src_path):
            if os.path.isdir(args.src_path):
                if os.path.isdir(args.dst_path):
                    transform(args)
                else:
                    print(f"Usage: {sys.argv[0]} \
                          -src [SRC_PATH] -dst [DST_PATH]")
            else:
                image = cv2.imread(args.src_path)
                Transformer = Transformation(image)
                Transformer.plot_transformation_image()
        else:
            print(f"Usage: {sys.argv[0]} -src [SRC_PATH]")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A program to display image transformation."
    )

    parser.add_argument(
        "-src",
        "--src_path",
        type=str,
        required=True,
        nargs="?",
        help="Image file path."
    )
    parser.add_argument(
        "-dst",
        "--dst_path",
        type=str,
        nargs="?",
        help="Destination directory path."
    )
    parser.add_argument(
        "-gaussian",
        "--gaussian_blur",
        action="store_true",
        help="Gaussian Transform"
    )
    parser.add_argument(
        "-mask",
        action="store_true",
        help="Mask Transform"
    )
    parser.add_argument(
        "-roi",
        "--roi_objects",
        action="store_true",
        help="Roi Transform"
    )
    parser.add_argument(
        "-analyze",
        "--analyze_object",
        action="store_true",
        help="Analyze Transform"
    )
    parser.add_argument(
        "-pseudo",
        "--pseudolandmarks",
        action="store_true",
        help="Psudolandmark Transform"
    )
    parser.add_argument(
        "-hist",
        "--color_histogram",
        action="store_true",
        help="Color histogram Transform"
    )
    args = parser.parse_args()
    main(args)
