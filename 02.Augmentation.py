import argparse
import os
from torchvision.io import read_image
from lib.images import analyze, flip, rotate, perspective
from lib.images import brightness, contrast, saturation, IMAGE_FOLDER
from lib.print import warning
import random


def print_summary(before, after):
    """
    Print the number of images in each subdirectory
    """
    print("========================")
    print("Summary of augmentation:")
    print("========================")
    for key, value in before.items():
        print(f"{key}: {value} -> {after[key]}")


def random_transform(img, file_path):
    """
    Random augmentation
    """

    transforms = [flip, rotate, perspective, brightness, contrast, saturation]
    num = random.randint(0, 5)
    transforms[num](img, file_path)


def augment(category: str):
    """
    Augment images in the directory
    """
    print(f'\nAugmenting "{category}" images...')
    counts_before = analyze(category)
    counts_after = counts_before.copy()
    max_num = max(counts_after.values())

    root, dirs, files = next(os.walk(IMAGE_FOLDER))
    for dirname in dirs:
        if not dirname.lower().startswith(category.lower()):
            continue
        dir_path = os.path.join(root, dirname)

        while (counts_after[dirname] < max_num):
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                img = read_image(file_path)
                random_transform(img, file_path)
                counts_after[dirname] += 1
                if (counts_after[dirname] == max_num):
                    break
            # Reset the number of images with real number of files.
            counts_after = analyze(category)

    counts_after = analyze(category)
    print_summary(counts_before, counts_after)


def main(file_path):
    try:
        if (file_path):
            img = read_image(file_path)
            flip(img, file_path)
            rotate(img, file_path)
            perspective(img, file_path)
            brightness(img, file_path)
            contrast(img, file_path)
            saturation(img, file_path)
        else:
            print(warning("Auto image augmentation..."))
            augment("Apple")
            augment("Grape")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A program to augment images samples\
            by applying 6 types of transformation."
    )

    # Adding argument for the directory
    parser.add_argument(
        'file_path',
        type=str,
        nargs='?',
        help='Image file path to transform to 6 different types.',
    )

    args = parser.parse_args()
    main(args.file_path)
