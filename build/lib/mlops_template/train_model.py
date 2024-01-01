import torch

from models.model import SimpleCNN
from torch import optim
import matplotlib.pyplot as plt 
import os 

import click 
@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=0.03, help="learning rate to use for training")
@click.option("--training_name", help="name of the training for generating subfolders")
def train(lr, training_name) : 

    # Instantiate the model
    model = SimpleCNN()

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    data = torch.load("data/processed/processed_tensor.pt")
    train_loader = data["train_loader"]
    # Training loop
    num_epochs = 5  
    loss_list = []
    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0

        for inputs, labels in train_loader :
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update the weights

            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)
        loss_list.append(average_loss)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}')
    plt.plot(loss_list)
    if os.isdir("mlops_template/reports/figures/{}".format(training_name)) == False : 
        os.system("mkdir mlops_template/reports/figures/{}".format(training_name))
    plt.savefig("mlops_template/reports/figures/{}/training.png".format(training_name))
    if os.isdir("mlops_template/models/{}".format(training_name)) == False : 
        os.system("mkdir mlops_template/models/{}".format(training_name))
    torch.save(model.state_dict(), 'mlops_template/models/{}/checkpoint.pth'.format(training_name))


cli.add_command(train)

if __name__ == "__main__":
    cli()
