import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import warnings

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
data = pd.read_csv(r'...\data.csv')

X = data.iloc[:,0:36].values
Y = data.iloc[:,36].values.reshape(-1,1)

encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(Y)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

xtrain, xtest, ytrain, ytest = train_test_split(X , y , test_size=0.30 , random_state=0)

model = Sequential()
model.add(Dense(36, input_shape=(36,), activation='sigmoid', name='fc1'))
model.add(Dense(72, activation='sigmoid', name='fc2'))
model.add(Dropout(0.1))
model.add(Dense(36, activation='sigmoid', name='fc3'))
model.add(Dense(5, activation='softmax', name = 'output'))
optimizer = Adam(lr=0.001)

model.compile(optimizer, loss='mean_squared_error', metrics=['accuracy'])
print('Neural Network Model Summary: ')
print(model.summary())

history = model.fit(xtrain, ytrain, verbose=2, batch_size=8, epochs=40, validation_data=(xtest,ytest))

test_result = model.evaluate(xtest, ytest)
train_result = model.evaluate(xtrain, ytrain)
print('Final test set loss: {:4f}'.format(test_result[0]))
print('Final test set accuracy: {:4f}'.format(test_result[1]))

print('Final train set loss: {:4f}'.format(train_result[0]))
print('Final train set accuracy: {:4f}'.format(train_result[1]))


# Visualize Accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Visualize Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
