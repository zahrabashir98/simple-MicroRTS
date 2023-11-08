import itertools
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, BatchNormalization, Dropout
import numpy as np
from matplotlib import pyplot as plt
import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization

def possibleAttackPos(minUnitPosX, minUnitPosY): #find closeset
    chance = np.random.randint(2) 
    if chance ==0:
        y1 = minUnitPosY
        if minUnitPosX <=1:
            x1 = minUnitPosX + 2
            
        elif minUnitPosX >=2:
            x1 = minUnitPosX - 2

    elif chance ==1:
        x1 = minUnitPosX
        if minUnitPosY <=1:
            y1 = minUnitPosY + 2
            
        elif minUnitPosY >=2:
            y1 = minUnitPosY - 2

    return x1, y1
def calculateDistance(x1, x2, y1, y2):
    return math.sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2))

def canAttack(x, y, myX, myY):
    if calculateDistance(x, myX, y, myY) <=2:
        return True
    else:
        return False

def generateLabel(otherPlayers, myPlayer):
    label = np.zeros(21)
    myX = int(myPlayer/4)
    myY = myPlayer%4
    distances = []
    for each in otherPlayers:
        x = int(each/4)
        y = each%4
        distances.append(math.sqrt(pow((x - myX), 2) + pow((y - myY), 2)))
    
    # print(distances)
    targetUnitIndex = distances.index(min(distances))
    minUnitPos = otherPlayers[targetUnitIndex]
    minUnitPosX = int(minUnitPos/4)
    minUnitPosY = minUnitPos%4

    if canAttack(minUnitPosX, minUnitPosY, myX, myY):
        # print("YES")
        label[minUnitPos] = 1

    else:
        # print("NO")
        x1, y1 = possibleAttackPos(minUnitPosX, minUnitPosY) #fix this
        # print(x1, y1)
        # print(myX, myY)
        diffX = x1 - myX
        diffY = y1 - myY
        chance = np.random.randint(2) 
        if chance == 0:
            if diffX >0: #right
                label[19] = 1
            elif diffX<0:
                label[18] = 1
            elif diffY >0:
                label[17] = 1
            elif diffY<0:
                label[16] = 1
        elif chance == 1:
            if diffY >0:
                label[17] = 1
            elif diffY<0:
                label[16] = 1   
            elif diffX >0: #right
                label[19] = 1
            elif diffX<0:
                label[18] = 1   
            
    return label
def choose(num):
    return list(itertools.combinations(list(range(16)), num))

dataset= []
labels = []
for num in range(2,6):
    combinations = choose(num)
    
    for each in combinations:
        for i in range(len(each)):
            data = np.zeros((4,4,2))
            x = int(each[i]/4)
            y = each[i]%4
            data[x][y][0] = 1
            tmp = list(each)
            tmp.remove(each[i])
            for rest in tmp:
                x = int(rest/4)
                y = rest%4
                data[x][y][1] = 1
            dataset.append(data)
            label = generateLabel(tmp, each[i])
            labels.append(label)


print(len(dataset))
print(len(labels))
c = list(zip(dataset, labels))
np.random.shuffle(c)
dataset, labels = zip(*c)

num_filters = 16
filter_size = 2
pool_size = 1
x = np.array(dataset[0:9*int(len(dataset)/10)])
y = np.array(labels[0:9*int(len(labels)/10)])

valData = np.array(dataset[8*int(len(dataset)/10): 9*int(len(dataset)/10)])
valLabels = np.array(labels[8*int(len(labels)/10): 9*int(len(labels)/10)])

testData = np.array(dataset[9*int(len(dataset)/10): 10*int(len(dataset)/10)])
testLabels = np.array(labels[9*int(len(labels)/10): 10*int(len(labels)/10)])

model = Sequential([
  Conv2D(num_filters, filter_size, input_shape=(4, 4, 2)),
  BatchNormalization(),
  Dropout(0.25),
  MaxPooling2D(pool_size=pool_size),
  Conv2D(16, 2),
  Dropout(0.1),
  Flatten(),
  Dense(21, activation='softmax'),
])

model.compile(
  'adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

history = model.fit(
x,
y,
epochs=100,
validation_data = (valData, valLabels),
shuffle = True
)


test_loss, test_acc = model.evaluate(testData,  testLabels, verbose=2)
print(test_loss)
print(test_acc)

import matplotlib.pyplot as plt

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold as SKF
from sklearn import metrics

testData = np.array(valData)
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
testData = testData.reshape(testData.shape[0], testData.shape[1]*testData.shape[2]*testData.shape[3])
wholeData = np.concatenate((x, testData))
print(wholeData.shape)
wholeLabels = []
for i in range(len(x)):
    wholeLabels.append(1)
for i in range(len(testData)):
    wholeLabels.append(0)

wholeLabels = np.array(wholeLabels)
print(wholeLabels.shape)

m = RandomForestClassifier(n_jobs=-1, max_depth=5, min_samples_leaf = 5)
predictions = np.zeros(wholeLabels.shape) #creating an empty prediction array

x = wholeData
y = wholeLabels

skf = SKF(n_splits=20, shuffle=True, random_state=100)
for fold, (train_idx, test_idx) in enumerate(skf.split(x, y)):
    X_train, X_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    m.fit(X_train, y_train)
    probs = m.predict_proba(X_test)[:, 1] #calculating the probability
    predictions[test_idx] = probs


print("ROC-AUC for train and test distributions:", metrics.auc(y, predictions)) 
# if it is not very high(>0.8), it is ok, and means that we don't have covariate shift in our dataset
# based on: https://towardsdatascience.com/how-dis-similar-are-my-train-and-test-data-56af3923de9b

