import torch
import sys
import argparse
from tqdm import tqdm

from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from model import dataset, LeafClassifier, class_to_idx


def predict(val_loader):
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (vinputs, vlabels) in enumerate(
                tqdm(val_loader, desc="Validation Progress")
        ):
            voutputs = model(vinputs)
            _, predicted = torch.max(voutputs, 1)
            total += vlabels.size(0)
            correct += (predicted == vlabels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the validation set: {accuracy:.2f}%')


def main(file_path):
    try:
        if (file_path):
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            gen = torch.Generator().manual_seed(42)
            _, val_data = random_split(
                dataset, [train_size, val_size], generator=gen
            )
            val_loader = DataLoader(val_data, batch_size=256, shuffle=False)
            predict(val_loader)
        else:
            print(f"Usage: {sys.argv[0]} [path_to_image]")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A program to classify a type of leaf from validation set."
    )

    parser.add_argument(
        "folder_path",
        type=str,
        nargs="?",
        help="Image folder path.",
        default="images"
    )

    args = parser.parse_args()
    model = LeafClassifier()
    model.load_state_dict(
        torch.load("./model_20240823_170439_4", weights_only=True)
    )
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,], std=[0.5,])
    ])

    idx2class = {v: k for k, v in class_to_idx.items()}
    main(args.folder_path)
