# first neural network with keras tutorial
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

dataset = loadtxt("pima-indians-diabetes.data.csv", delimiter=",")

X = dataset[:,0:8]
y= dataset[:,8]

#define model
model = Sequential()
model.add(Dense(12, input_shape=(8,), activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
