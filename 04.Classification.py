import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split

"""
TODO:
    - load dataset
    - eval model
    - save model
    - visualize
"""

class LeafClassifier(nn.Module):
    """
    Leaf Classifier that classifies the type of disease specified in the leaf

    Input: leaf Image, shape of (256, 256, 3)
    Label: a type of disease specified in the leaf (black rot, Esca, spot, and healthy)
    """
    def __init__(self):
        super(LeafClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256 * 15 * 15, 512)
        self.fc2 = nn.Linear(512, 4)

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
        x = x.view(-1, 256 * 15 * 15)

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Set hyperparameters
lr = 0.001
batch_size = 32
epochs = 30


# Dataset
dataset = # TODO: load dataset 
train_size = int(0.8 * len(dataset))
val_size = int(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size, shuffle=False)


# Loss and Optimizer
model = LeafClassifier() 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


# Train model
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch + 1}/{epochs}] loss - {running_loss/len(train_loader):.4f}")