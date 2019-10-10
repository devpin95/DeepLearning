from sklearn import preprocessing as prep
from sklearn.metrics import confusion_matrix, classification_report
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
from keras import backend as K
from keras import callbacks
import time
from zipfile import ZipFile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from prettytable import PrettyTable
import seaborn as sn
import pandas as pd


# Metric calculations found at
# https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
def recall_m( y_true, y_pred ):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / ( possible_positives + K.epsilon() )


def precision_m( y_true, y_pred ):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / ( predicted_positives + K.epsilon() )


def f1_m( y_true, y_pred ):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ( (precision*recall) / (precision + recall + K.epsilon()) )


def oneHotEncoder(x):
    tempx = [[0] * 10 for i in range(len(x))]

    for i in range(0, len(x)):
        tempx[i][x[i]] = 1

    return np.asarray(tempx)


def printMetrics(loss, accuracy, recall, precision, f1):
    t = PrettyTable(['Loss', "Accuracy", 'Recall', 'Precision', 'F1'])
    t.add_row([round(loss, 4), round(accuracy, 4), round(recall, 4), round(precision, 4), round(f1, 4)])
    print(t)


def lossPlot(loss):
    x = [i for i in range(0, len(loss))]

    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.plot(x, loss)
    plt.title('Epoch-Loss', fontdict={'fontsize': 15}, pad=20)
    plt.xlabel('Epoch', fontdict={'fontsize': 11}, labelpad=20)
    plt.ylabel('Loss', fontdict={'fontsize': 11}, labelpad=20)
    plt.savefig('EpochLoss.png', bbox_inches='tight', pad_inches=0.5)
    plt.show()


def timePlot(times):
    x = [i for i in range(0, len(times))]

    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.plot(x, times)
    plt.title('Loss-Wall', fontdict={'fontsize': 15}, pad=20)
    plt.xlabel('Epoch', fontdict={'fontsize': 11}, labelpad=20)
    plt.ylabel('Time', fontdict={'fontsize': 11}, labelpad=20)
    plt.savefig('LossWall.png', bbox_inches='tight', pad_inches=0.5)
    plt.show()


def confusionMatrix(cm):
    print(cm)
    df_cm = pd.DataFrame(cm, index=[i for i in range(0, 10)],
                         columns=[i for i in range(0, 10)])
    plt.figure(figsize=(10, 7))
    ax = sn.heatmap(df_cm, annot=True, cmap="Blues", fmt='g')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.title('Confusion Matrix', fontdict={'fontsize': 15}, pad=20)
    plt.xlabel('Predicted', fontdict={'fontsize': 11}, labelpad=20)
    plt.ylabel('True', fontdict={'fontsize': 11}, labelpad=20)
    plt.savefig('ConfusionMatrix.png', bbox_inches=None, pad_inches=0.5)
    plt.show()


def printClassMetrics(metrics):
    t = PrettyTable(['Class', 'Precision', 'Recall', 'F1'])
    for key in metrics:
        if key not in ['accuracy', 'macro avg', 'weighted avg']:
            t.add_row([key, round(metrics[key]['precision'], 4), round(metrics[key]['recall'], 4), round(metrics[key]['f1-score'], 4)])
    print(t)


class LoadDataModule(object):
    def __init__(self):
        self.DIR = './'
        pass

    # Returns images and labels corresponding for training and testing. Default mode is train.
    # For retrieving test data pass mode as 'test' in function call.
    def load(self, mode='train'):
        label_filename = mode + '_labels'
        image_filename = mode + '_images'
        label_zip = self.DIR + label_filename + '.zip'
        image_zip = self.DIR + image_filename + '.zip'
        with ZipFile(label_zip, 'r') as lblzip:
            labels = np.frombuffer(lblzip.read(label_filename), dtype=np.uint8, offset=8)
        with ZipFile(image_zip, 'r') as imgzip:
            images = np.frombuffer(imgzip.read(image_filename), dtype=np.uint8, offset=16).reshape(len(labels), 784)
        return images, labels


class TimeHistory(callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.time() - self.epoch_time_start)


def main():
    # csv_logger = CSVLogger('training.log')
    time_callback = TimeHistory()

    # Load the images
    loader = LoadDataModule()
    train_images, train_labels = loader.load()
    test_images, test_labels = loader.load(mode='test')

    # Scale the data
    scalar = prep.MinMaxScaler()
    scalar = scalar.fit(train_images)
    train_images_normalized = scalar.transform(train_images)
    test_images_normalized = scalar.transform(test_images)

    train_images_normalized = train_images_normalized.reshape(60000, 28, 28, 1)
    # test_images_normalized.reshape(28, 28, 60000)

    # Run the classes through onehot encoder
    train_labels = oneHotEncoder(train_labels)

    # Training parameters
    minibatch_size = 200
    epochs = 1

    # Set up the model
    model = Sequential()
    model.add(Conv2D(40, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=train_images_normalized.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    print("\nModel summary...")
    model.summary()

    print("\nCompiling model...")
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy', recall_m, precision_m, f1_m])

    print("\nFitting model...")
    history = model.fit(
        train_images_normalized, train_labels,
        epochs=epochs, batch_size=minibatch_size,
        callbacks=[time_callback]
    )

    times = time_callback.times
    epochloss = history.history['loss']
    lossPlot(epochloss)
    timePlot(times)

    print("\nRunning against training data...")

    loss, accuracy, recall, precision, f1 = model.evaluate(train_images_normalized, train_labels, batch_size=minibatch_size)
    print('\nTraining Metrics')
    printMetrics(loss, accuracy, recall, precision, f1)

    print('\nTest Metrics (Per Class)')
    predictions = model.predict_classes(test_images_normalized, minibatch_size)
    printClassMetrics(classification_report(test_labels, predictions, output_dict=True))

    print("\nConfusion Matrix")
    confusionMatrix(confusion_matrix(test_labels, predictions))

    print('\nEpoch-Loss graph saved in EpochLoss.png')
    print('Confusion matrix saved in ConfusionMatrix.png')


main()
