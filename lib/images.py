from collections import Counter
import os
import torch
from torchvision.utils import save_image
from torchvision.transforms import v2

IMAGE_FOLDER = 'images'
IMAGE_URL = 'https://cdn.intra.42.fr/document/document/17547/leaves.zip'

def analyze(directory):
    """
    Analyze images in the directory and
    return a dictionary of counts per plant type.
    """
    plant_counts = Counter()

    # Traverse through the directory and
    # count the number of images in each subdirectory
    base = os.path.abspath(os.getcwd())

    for root, dirs, files in os.walk(IMAGE_FOLDER):
        directory = os.path.normpath(os.path.join(base, root, directory))
        for dirname in dirs:
            dirname = os.path.normpath(os.path.join(base, root, dirname))
            if not dirname.lower().startswith(directory.lower()):
                continue
            dir_path = os.path.join(base, root, dirname)
            num_images = len([
                file for file in os.listdir(dir_path) if file.lower().endswith(
                    ('.png', '.jpg', '.jpeg')
                )
            ])
            plant_counts[dirname] = num_images
    return plant_counts


def _transform_and_save(img, file_path, transforms, transform_type):
    img = transforms(img)
    filename, file_extension = os.path.splitext(file_path)
    try:
        save_image(img, f"{filename}_{transform_type}{file_extension}")
    except Exception as e:
        print(e)


def flip(img, file_path):
    """
    Horizontal flip
    """

    # Apply horizontal flip with probability 1 (always flip)
    transforms = v2.Compose([v2.RandomHorizontalFlip(p=1), v2.ToDtype(torch.float32, scale=True),])
    _transform_and_save(img, file_path, transforms, 'Flip')


def rotate(img, file_path):
    """
    Rotate
    """

    # Apply rotation with range (45, 180)
    transforms = v2.Compose([v2.RandomRotation((10, 180)), v2.ToDtype(torch.float32, scale=True),])
    _transform_and_save(img, file_path, transforms, 'Rotate')


def perspective(img, file_path):
    """
    Random perspective
    """

    # Apply random perspective
    transforms = v2.Compose([v2.RandomPerspective(distortion_scale=0.6, p=1.0), v2.ToDtype(torch.float32, scale=True),])
    _transform_and_save(img, file_path, transforms, 'Perspective')


def brightness(img, file_path):
    """
    Brightness
    """

    transforms = v2.Compose([v2.ColorJitter(brightness=(0.5, 1.5)), v2.ToDtype(torch.float32, scale=True),])
    _transform_and_save(img, file_path, transforms, 'Brightness')


def contrast(img, file_path):
    """
    Contrast
    """

    transforms = v2.Compose([v2.ColorJitter(contrast=(0.5, 3)), v2.ToDtype(torch.float32, scale=True),])
    _transform_and_save(img, file_path, transforms, 'Contrast')


def saturation(img, file_path):
    """
    Saturation
    """

    transforms = v2.Compose([v2.ColorJitter(saturation=(0.5, 3)), v2.ToDtype(torch.float32, scale=True),])
    _transform_and_save(img, file_path, transforms, 'Saturation')
