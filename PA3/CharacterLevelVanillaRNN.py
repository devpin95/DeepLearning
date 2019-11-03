from FreshRNN import RNN
from DataPreprocessor import DataPreprocessor
import numpy as np

datasetsPath = 'data'

vocab = 256
sequence_length = 100
trainx = []
trainy = []
testx = []
testy = []

if __name__ == "__main__":
    dataset = []
    DataPreprocessor.get_dataset(dataset, datasetsPath)
    trainx, trainy = DataPreprocessor.data_targets(dataset, sequence_length)
    trainx = np.array(trainx)
    trainy = np.array(trainy)

    model = RNN(vocab)

    losses = model.train_with_sgd(model, trainx[:1000], trainy[:1000], nepoch=5, eval_loss_after=1)

    test_seqx = trainx[6]

    o = model.predict(test_seqx)

    print("Output:")
    for char in o:
        print(chr(char), end='')
