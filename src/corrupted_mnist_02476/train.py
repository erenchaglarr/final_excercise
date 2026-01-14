from corrupted_mnist_02476.model import Model
from corrupted_mnist_02476.data import MyDataset

def train():
    dataset = MyDataset("data/raw")
    model = Model()
    # add rest of your training code here

if __name__ == "__main__":
    train()
