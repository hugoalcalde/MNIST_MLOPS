from tests import _PATH_DATA
import torch


def test_sets_dimmensions():
    processed_tensor = torch.load(_PATH_DATA + "processed_tensor.pt")

    train_dataloader = len(processed_tensor["train_loader"].dataset)
    val_dataloader = len(processed_tensor["val_loader"].dataset)
    test_dataloader = len(processed_tensor["test_loader"].dataset)

    print(train_dataloader)

    N = 35000

    assert train_dataloader > val_dataloader, "Validation set is bigger than train set"
    assert train_dataloader + test_dataloader + val_dataloader == N, "There was some data leak in the processing"
