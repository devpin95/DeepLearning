from sklearn import preprocessing as prep
from numpy import loadtxt
from keras.layers import Dense
from keras.models import Sequential
from keras import backend as K

# Metric calculations found at
# https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
def recall_m( y_true, y_pred ):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0 ,1)))
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


batch_size = 128
epochs = 100
usecols = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12]

save_model = False

print("\nReading data from file...")

# Data cleaned in PrepareData.py
x = prep.scale(loadtxt('dataset_clean.csv', delimiter=',', skiprows=1, usecols=usecols, dtype=float))
y = loadtxt('dataset_clean.csv', delimiter=',', skiprows=1, usecols=[13], dtype=None)

input_dim = len(x[0])
nn = str(input_dim) + "x5x1 (TxRxS)"

x_train = x[:7199]
y_train = y[:7199]
x_test = x[7200:]
y_test = y[7200:]

print("\nSetting up model...")
model = Sequential()

model.add(Dense(batch_size, input_dim=input_dim, activation="relu"))
model.add(Dense(3, activation="tanh"))
model.add(Dense(1, activation="sigmoid"))

print("\nModel summary...")
model.summary()

print("\nCompiling model...")
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy', recall_m, precision_m, f1_m])

print("\nFitting model...")
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

print("\nRunning against test data...")
loss, accuracy, recall, precision, f1 = model.evaluate(x_test, y_test, batch_size=batch_size)

print("\nTest Data -------------------------------------------------------------------------------------- ")
print("Loss: " + str(round(loss*100, 2)) + "%")
print("Accuracy: " + str(round(accuracy*100, 2)) + "%")
print("Recall: " + str(round(recall*100, 2)) + "%")
print("Precision: " + str(round(precision*100, 2)) + "%")
print("F1: " + str(round(f1*100, 2)) + "%")

loss, accuracy, recall, precision, f1 = model.evaluate(x_train, y_train, batch_size=batch_size)

print("\nTraining Data -------------------------------------------------------------------------------------- ")
print("Loss: " + str(round(loss*100, 2)) + "%")
print("Accuracy: " + str(round(accuracy*100, 2)) + "%")
print("Recall: " + str(round(recall*100, 2)) + "%")
print("Precision: " + str(round(precision*100, 2)) + "%")
print("F1: " + str(round(f1*100, 2)) + "%")

if save_model:
    print("\nSaving model...")

    model_json = model.to_json()
    with open("model.json", "w") as json:
        json.write(model_json)
    model.save_weights("model.h5")

    print("\nModel saved...")
else:
    print("\nMODEL NOT SAVED")

print("\nEND")