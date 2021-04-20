from prepare import init
from train import training
from prediction import predict

if __name__ == "__main__":
    init()
    x_train, y_train = training()
    predict(x_train, y_train)
