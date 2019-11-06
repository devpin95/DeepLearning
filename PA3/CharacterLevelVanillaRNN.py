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
    DataPreprocessor.get_dataset(dataset, datasetsPath, clean=True)
    del dataset[0]

    trainx, trainy = DataPreprocessor.data_targets(dataset, sequence_length)
    trainx = np.array(trainx)
    trainy = np.array(trainy)

    print(trainx[0])
    print(trainy[0])

    model = RNN(vocab)

    losses = model.train_with_sgd(model, trainx[:500], trainy[:500], nepoch=5, eval_loss_after=1)
    model.gradient_check(['D', 'o', 'g', 's'], ['o', 'g', 's', ' '])

    test_seqx = trainx[-1]

    while True:
        print("input: ", end='')
        input1 = input()

        generated_string = input1

        for i in range(20):
            print(str(i) + " ", end='')
            seq = DataPreprocessor.one_hot_vector(generated_string, vocab)
            o = model.predict(seq)
            print(o)
            generated_string = generated_string + chr(o[-1])

        print("Final String: ", end='')
        print(generated_string, end='\n\n')

    o = model.predict(test_seqx)

    print("Output:")
    print(o)
    for char in o:
        print(chr(char), end='')
