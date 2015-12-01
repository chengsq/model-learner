from sgd_train import sgd_optimization_mnist
from predict import *

if __name__ == '__main__':
	# train
    sgd_optimization_mnist()

    # predict
    predict()