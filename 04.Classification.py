import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torchvision import datasets, transforms
from lib.images import IMAGE_FOLDER

"""
TODO:
    - visualize output
"""
# Set hyperparameters
lr = 0.01
batch_size = 256
epochs = 10


# Load Dataset
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # image = (image - mean) / std, range [-1, 1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = datasets.ImageFolder(root=IMAGE_FOLDER, transform=transform)

NUM_CLASSES = len(dataset.classes)
# Get the mapping of class names to their corresponding labels
class_to_idx = dataset.class_to_idx

print(f"Number of classes: {NUM_CLASSES}")
print(f"Class to index mapping: {class_to_idx}")
print("\n", dataset)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size, shuffle=False)
print(f"Dataset has been loaded - batch size:{batch_size}", end=" ")

print(f"train_data: {len(train_loader)}, val_data: {len(val_loader)}")

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

# Loss and Optimizer
model = LeafClassifier() 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

#
def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.0
    last_loss = 0.0

    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # print(inputs.shape, outputs.shape, labels.shape)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print(f"\tBatch [{i + 1}] - loss {running_loss}")
        if i % batch_size == batch_size - 1:
            last_loss = running_loss / batch_size
            print(f"batch {i + 1} loss: {last_loss}")
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0

    last_loss = running_loss / len(train_loader)
    tb_x = epoch_index * len(train_loader) + i + 1
    tb_writer.add_scalar("Loss/train", last_loss, tb_x)
    return last_loss

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(f"./runs/leaf_trainer_{timestamp}")
epoch_number = 0
best_vloss = 1_000_000.

# 
print(f"\nStart training model - epochs:{epochs}")
for epoch in range(epochs):
    print(f"Epoch [{epoch + 1}/{epochs}]")

    # train
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    # validation
    model.eval()
    running_vloss = 0.0
    with torch.no_grad():
        for i, (vinputs, vlabels) in enumerate(val_loader):
            voutputs = model(vinputs)
            vloss = criterion(voutputs, vlabels)
            running_vloss += vloss
    avg_vloss = running_vloss / (i + 1)
    print(f"LOSS train {avg_loss:.4f} val {avg_vloss:.4f}")

    # Tensorboard
    writer.add_scalars("Training vs. Validation Loss", 
                    {"Training": avg_loss, "Validation": avg_vloss},
                    epoch_number + 1)
    writer.flush()

    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        torch.save(model.state_dict(), f"./model_{timestamp}_{epoch_number}")
    epoch_number += 1