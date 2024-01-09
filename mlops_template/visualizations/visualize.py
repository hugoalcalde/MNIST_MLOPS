import torch


from models.model import SimpleCNN, PartialModel
import matplotlib.pyplot as plt


import click


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--model_name", help="Name of the trained model we want to visualize")
def visualize(model_name):
    # Open the model

    model_checkpoint = torch.load("models/{}/checkpoint.pth".format(model_name))
    model = SimpleCNN()

    # Creating a model that stops in the previous layer :

    model.load_state_dict(model_checkpoint)
    model_partial = PartialModel(model, "fc1")

    # Getting some training images to visualize the network :

    data = torch.load("data/processed/processed_tensor.pt")
    train_loader = data["train_loader"]

    model.eval()
    model_partial.eval()

    for inputs, labels in train_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        plt.imshow(inputs[0])
        plt.xlabel("Predicted : {}. Real : {} ".format(predicted[0], labels[0]))
        break
    plt.savefig("reports/figures/{}/visualizations.png".format(model_name))


cli.add_command(visualize)

if __name__ == "__main__":
    cli()
