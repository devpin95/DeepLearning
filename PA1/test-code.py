from sklearn import preprocessing as prep
from numpy import loadtxt
from keras.models import model_from_json

json_file = open("model.json")
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

usecols = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12]
xnew = prep.scale(loadtxt('judge_clean.csv', delimiter=',', skiprows=1, usecols=usecols, dtype=float))
ynew = loaded_model.predict_classes(xnew)


exited = 0
for row in ynew:
    if row[0] == 1:
        exited = exited + 1

print("Total: " + str(len(ynew)))
print("Exited: " + str(exited))
print("Stayed: " + str(len(ynew) - exited))
print("%: " + str((exited/len(ynew))*100))

with open("judge-pred.csv", "w+") as pred:
    cust = loadtxt('judge.csv', delimiter=',', skiprows=1, usecols=[0], dtype=int)
    pred.write("CustomerId,Exited\n")
    for i in range(0, len(cust)):
        pred.write(str(cust[i]) + "," + str(ynew[i][0]) + "\n")