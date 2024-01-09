from pytorch_lightning import Trainer, loggers
from models.model_lightning import SimpleCNN
import torch


model = SimpleCNN()
trainer = Trainer(max_epochs=5, logger=loggers.WandbLogger(project="dtu_mlops"))
data = torch.load("data/processed/processed_tensor.pt")
train_loader = data["train_loader"]
val_loader = data["val_loader"]
trainer.fit(model, train_loader, val_loader)
