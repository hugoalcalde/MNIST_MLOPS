import pytest 




@pytest.mark.parametrize("lr", [1, 0.001, -3])
def test_learning_rate(lr) : 
    if lr <= 0 : 
        raise ValueError("Incorrectly specified learning rate, the learning rate should be greater than 0.")