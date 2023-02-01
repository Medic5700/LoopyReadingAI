"""
Simple program to test the basics of the development stack

Reference
    https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
"""

from typing import Any, Callable, Generic, Literal, Optional, Type, TypeVar
from PIL import Image, ImageDraw, ImageFont
import random
import torch
from torch import nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ==============================================================================

def trainingTextGenerator(fontName : str = "calibri") -> tuple[str, Image.Image]:
    while True:
        text : str = str(random.randint(0,9))

        font : ImageFont.FreeTypeFont = ImageFont.truetype(fontName, 32)
        fontSize : list[int, int, int, int] = font.getbbox(text)
        x : int = (32 - (fontSize[2] - fontSize[0])) // 2
        y : int = -2 # hand tuned, #TODO

        x += random.randint(-1, 1)
        y += random.randint(-1, 1)

        image : Image.Image = Image.new(mode = '1', size = (32,32), color = (1))
        ImageDraw.Draw(image).text((x, y), text, fill = (0), font = font, align = "center")

        # print(text)
        # image.show()

        yield text, image

class CustomDataset(Dataset):
    def __init__(self, size : int = 100):
        self.dataset : list[tuple[torch.Tensor, int]] = []
        numberGenerator = trainingTextGenerator()
        convertImage2Tensor : Callable[[Image.Image], torch.Tensor] = transforms.ToTensor()
        # convertTensor2Image : Callable[[torch.Tensor], Image.Image] = transforms.ToPILImage()
        for _ in range(size):
            temp = next(numberGenerator)
            classification : str = temp[0]
            image : Image.Image = temp[1]
            self.dataset.append((convertImage2Tensor(image), int(classification)))
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index : int) -> tuple[Image.Image, int]:
        return self.dataset[index][0], self.dataset[index][1]

training_data = CustomDataset(1000)
test_data = CustomDataset(100)
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

"""
# Display an image and label from the dataset?
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
"""

# ==============================================================================

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32*32, 512), # input is 32x32 image
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

# ==============================================================================

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

"""
epochs = 1
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done Training!?")
"""

for i in range(64):
    print(f"Training Round {i+1}\n-------------------------------")
    training_data = CustomDataset(10000)
    test_data = CustomDataset(1000)
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done Training")

# ==============================================================================

# ==============================================================================

model.eval()
for i in range(16):
    x, y = test_data[i][0], test_data[i][1]
    with torch.no_grad():
        pred = model(x)
        predicted, actual = int(pred[0].argmax(0)), str(y)
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
