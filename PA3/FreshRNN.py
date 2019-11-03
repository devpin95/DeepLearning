from DataPreprocessor import DataPreprocessor
import numpy as np
import math
from datetime import datetime
import sys


class RNN:
    def __init__(self, in_vocab_dim, in_hidden_dim=100, in_bptt_truncate=4):
        self.input_dim = in_vocab_dim
        self.hidden_dim = in_hidden_dim
        self.bptt_truncate = in_bptt_truncate

        # weights for input to hidden layer
        wxh_rand_inteval = self.random_interval(self.input_dim)
        wxh_dim = (self.hidden_dim, self.input_dim)
        self.U = np.random.uniform(wxh_rand_inteval[0], wxh_rand_inteval[1], wxh_dim)

        # weights for hidden layer to hidden layer
        whh_rand_interval = self.random_interval(self.hidden_dim)
        whh_dim = (self.hidden_dim, self.hidden_dim)
        self.W = np.random.uniform(whh_rand_interval[0], whh_rand_interval[1], whh_dim)

        # weights for hidden layers to output
        why_rand_interval = self.random_interval(self.hidden_dim)
        why_dim = (self.input_dim, self.hidden_dim)
        self.V = np.random.uniform(why_rand_interval[0], why_rand_interval[1], why_dim)

    def forward_propogation(self, x):
        # Number of steps to take
        T = len(x)

        # states of each hidden node
        # each layer will need to know the state of the previous layer
        s = np.zeros((T+1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)

        # output of each hidden node
        o = np.zeros((T, self.input_dim))

        # Step through the input and get the hidden layer state and it's output
        for t in range(T):
            a_t = self.U[:, t] + self.W.dot(s[t-1])
            s[t] = np.tanh(a_t)
            vht = self.V.dot(s[t])
            o[t] = self.softmax(vht)

        return [o, s]

    def predict(self, x):
        o, s = self.forward_propogation(x)
        return np.argmax(o, axis=1)

    def cross_entropy_sum(self, y):
        return -1 * sum(np.log(y))

    def calculate_total_loss(self, x, y):
        loss = 0

        for i in range(len(y)):
            if i % (len(y) / 4) == 0:
                print("(" + str(i) + "/" + str(len(y)) + ") ", end='')
                sys.stdout.flush()
            # Do the one hot here as we get the sequences so that we dont run out of memory
            xoh = DataPreprocessor.one_hot_vector(x[i], self.input_dim)
            yoh = DataPreprocessor.one_hot_vector(y[i], self.input_dim)
            o, _ = self.forward_propogation(xoh)

            yoh = np.array(yoh)

            correct_characters_predicted = o[np.arange(len(yoh)), np.argmax(yoh, axis=1)]

            loss += self.cross_entropy_sum(correct_characters_predicted)

        return loss

    def calculate_loss(self, x, y):
        print("Calculating Loss: ", end='')
        sys.stdout.flush()
        N = sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x, y) / N

    def bptt(self, x, y):
        T = len(y)
        xoh = DataPreprocessor.one_hot_vector(x, self.input_dim)
        yoh = DataPreprocessor.one_hot_vector(y, self.input_dim)

        o, s = self.forward_propogation(x)
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(yoh)), np.argmax(yoh, axis=1)] -= 1

        # Go backwards through time (::-1 is the reverse of time)
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)

            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t]**2))

            for bptt_step in np.arange(max(0, t - self.bptt_truncate), t+1)[::-1]:
                dLdW += np.outer(delta_t, s[bptt_step-1])
                dLdU[:, np.argmax(xoh[bptt_step])] += delta_t
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step - 1]**2)
        return [dLdU, dLdV, dLdW]

    def sgd_step(self, x, y, learning_rate):
        dLdU, dLdV, dLdW = self.bptt(x, y)
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW

    @staticmethod
    def train_with_sgd(model, x_train, y_train, learning_rate=0.005, nepoch=100, eval_loss_after=5):
        losses = []
        num_examples_seen = 0

        for epoch in range(nepoch):
            print("\n\nEpoch " + str(epoch+1) + "/" + str(nepoch))
            if epoch % eval_loss_after == 0:
                loss = model.calculate_loss(x_train, y_train)
                losses.append((num_examples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("\n%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))

                if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                    learning_rate = learning_rate * 0.5
                    print("Setting learning rate to %f" % learning_rate)
                sys.stdout.flush()

            print("BPTT: ", end='')
            sys.stdout.flush()
            for i in range(len(y_train)):
                if i % (len(y_train) / 4) == 0:
                    print("(" + str(i) + "/" + str(len(y_train)) + ") ", end='')
                    sys.stdout.flush()
                model.sgd_step(x_train[i], y_train[i], learning_rate)
                num_examples_seen += 1

    @staticmethod
    def random_interval(n):
        return -math.sqrt(1 / n), math.sqrt(1 / n)


    @staticmethod
    def softmax(x):
        xt = np.exp(x - np.max(x))
        return xt / np.sum(xt)
