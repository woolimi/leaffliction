import torch
import sys
import argparse

from PIL import Image
from torchvision import transforms
from model import LeafClassifier, class_to_idx

def predict(file_path):
    test_image = Image.open(file_path)
    test_image = transform(test_image)
    test_image = test_image.unsqueeze(0)

    with torch.no_grad():
        output = model(test_image)
        _, predicted = torch.max(output, 1)

    print(f"Class predicted: {idx2class[predicted.item()]}")

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