import torch.nn.functional as F
from torch import nn, optim
from pytorch_lightning import LightningModule


class SimpleCNN(LightningModule):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        # Criterium for computing the loss

        self.criterium = nn.CrossEntropyLoss()

    def forward(self, x):
        # Apply convolutional and pooling layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # Flatten the input before the fully connected layers
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, val_data):
        data, target = val_data
        preds = self(data)
        loss = self.criterium(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.003)


class PartialModel(nn.Module):
    def __init__(self, original_model, stop_layer_name):
        super(PartialModel, self).__init__()
        self.features = nn.Sequential()
        self.stop_layer_name = stop_layer_name

        for name, module in original_model.features.named_children():
            if name == stop_layer_name:
                break
            self.features.add_module(name, module)

    def forward(self, x):
        x = self.features(x)
        return x
