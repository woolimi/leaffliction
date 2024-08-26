import torch
import sys
import argparse
import cv2
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from model import LeafClassifier, class_to_idx
from lib.transformation import Transformation
from PIL import Image
import os 


def get_class_name(file_path: str):
    parent_folder_path = os.path.dirname(file_path)
    parent_folder_name = os.path.basename(parent_folder_path)
    return parent_folder_name


def plot_result(img1: Image, img2: Image, original: str, predicted: str):
    fig, axarr = plt.subplots(1, 2, figsize=(10, 5))
    
    axarr[0].imshow(img1)
    axarr[0].axis('off') 

    axarr[1].imshow(img2)
    axarr[1].axis('off')

    fig.patch.set_facecolor('black')
    fig.text(0.5, 0.12, "=== DL Classification ===", ha='center', fontsize=16, fontweight='bold', color='white')
    is_correct = original.lower() == predicted.lower()
    fig.text(0.5, 0.05, f"Class predicted {'OK' if is_correct else 'KO'}: {predicted}", ha='center', fontsize=12, color='green' if is_correct else 'red')
    plt.subplots_adjust(bottom=0.2)
    plt.show()
    fig.savefig('predicted.png', bbox_inches='tight')


def predict(file_path):
    original_image = Image.open(file_path)
    test_image = transform(original_image)
    test_image = test_image.unsqueeze(0)

    with torch.no_grad():
        output = model(test_image)
        _, predicted = torch.max(output, 1)

    predicted_label = idx2class[predicted.item()]
    original_label = get_class_name(file_path)
    masked_image = Transformation(cv2.imread(file_path)).transform_mask()
    plot_result(original_image, Image.fromarray(masked_image), original_label, predicted_label)


def main(file_path):
    try:
        if (file_path):
            predict(sys.argv[1])
        else:
            print(f"Usage: {sys.argv[0]} [path_to_image]")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A program to predict a type of disease specified in the leaf."
    )

    parser.add_argument(
        "file_path",
        type=str,
        nargs="?",
        help="Image file path."
    )

    args = parser.parse_args()
    model = LeafClassifier()
    model.load_state_dict(torch.load("./model_20240823_170439_4", weights_only=True))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,], std=[0.5,])
    ])
    
    idx2class = {v: k for k, v in class_to_idx.items()}
    main(args.file_path)