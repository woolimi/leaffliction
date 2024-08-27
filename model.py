import torch
import torch.nn as nn

from torchvision import datasets, transforms
from lib.images import IMAGE_FOLDER

# Load Dataset
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # image = (image - mean) / std, range [-1, 1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = datasets.ImageFolder(root=IMAGE_FOLDER, transform=transform)

NUM_CLASSES = len(dataset.classes)
class_to_idx = dataset.class_to_idx


# Define model
class LeafClassifier(nn.Module):
    """
    Leaf Classifier that classifies the type of disease specified in the leaf

    Inputs: leaf Images, shape of (256, 256, 3)
    Labels: types of disease specified in the leaf
    """
    def __init__(self):
        super(LeafClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, NUM_CLASSES)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.relu(self.conv4(x))
        x = self.pool(x)

        # Flatten tensor
        x = x.view(-1, 256 * 14 * 14)

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
