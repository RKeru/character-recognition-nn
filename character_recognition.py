import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical, np_utils
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Read the data...
data = pd.read_csv("./A_Z Handwritten Data.csv").astype('float32')

# The data is represented as images 28x28[px] in the form of csv file
# In the csv file there are 785 columns:
#   the first one is the predicted label, the others are each pixel of the image
# Each row represent a data sample

# Split data:
# # - X: Our Data
# # - Y: The predict label
X = data.drop('0', axis=1)
y = data['0']

# Reshaping the data in the csv file so that it can be displayed as an image
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
train_x = np.reshape(train_x.values, (train_x.shape[0], 28, 28))
test_x = np.reshape(test_x.values, (test_x.shape[0], 28, 28))

print(f"Train data shape: {train_x.shape}")
print(f"Test data shape: {test_x.shape}")

# Set the dictionary of alphabets in the dataset
word_dict = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J',
    10: 'K',
    11: 'L',
    12: 'M',
    13: 'N',
    14: 'O',
    15: 'P',
    16: 'Q',
    17: 'R',
    18: 'S',
    19: 'T',
    20: 'U',
    21: 'V',
    22: 'W',
    23: 'X',
    24: 'Y',
    25: 'Z'
}

# {GUI Needed}
# # Plotting the number of alphabets in the dataset
# train_yint = np.intp(y)
# count = np.zeros(26, dtype='int')
# for i in train_yint:
#     count[i] += 1

# alphabets = []
# for i in word_dict.values():
#     alphabets.append(i)

# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# ax.barh(alphabets, count)

# plt.xlabel("Number of elements")
# plt.ylabel("Alphabets")
# plt.grid()
# plt.show()

# {GUI Needed}
# # Shuffling the data
# shuff = shuffle(train_x[:100])

# fig, ax = plt.subplots(3, 3, figsize=(10, 10))
# axes = ax.flatten()

# for i in range(9):
#     axes[i].imshow(np.reshape(shuff[i], (28, 28)), cmap="Greys")
# plt.show()

# Reshaping the training & test data so that it can be put in the model
train_X = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)
print(f"New shape of the train data: {train_X.shape}")
test_X = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2], 1)
print(f"New shape of the test data: {test_X.shape}")

# Reshaping the labels to categorical values
train_yOHE = to_categorical(train_y, num_classes=26, dtype='int')
print(f"New shape of train labels: {train_yOHE.shape}")
test_yOHE = to_categorical(test_y, num_classes=26, dtype='int')
print(f"New shape of test labels: {test_yOHE.shape}")

# CNN Model
model = Sequential()

# First layer
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

# Second layer
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

# Thrid layer
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

# Flatten the input (Add extra dimension to match the output) 
model.add(Flatten())

# Fourth Layer
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))

# Output Layer
model.add(Dense(26, activation='softmax'))

model.compile(optimizer= Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

print('Model compiled')

history = model.fit(train_X, train_yOHE, epochs=1, callbacks=[reduce_lr, early_stop], validation_data=(test_X, test_yOHE))

model.summary()
model.save(r'model_hand.h5')

# Displaying the accuracies & losses for train & validation set...

print("The validation accuracy is :", history.history['val_accuracy'])
print("The training accuracy is :", history.history['accuracy'])
print("The validation loss is :", history.history['val_loss'])
print("The training loss is :", history.history['loss'])
print("-------------------------------------------------------------------------")

# {GUI Needed}
# Making Model Predictions
pred = model.predict(test_X[:9])
print(test_X.shape)

# Display some of the test images & their prediction labels
fig, axes = plt.subplots(3, 3, figsize=(8, 9))
axes = axes.flatten()

for i, ax in enumerate(axes):
    img = np.reshape(test_X[i], (28, 28))
    ax.imshow(img, cmap="Greys")
    pred = word_dict[np.argmax(test_yOHE[i])]
    ax.set_title(f"Prediction: {pred}")
    ax.grid()

