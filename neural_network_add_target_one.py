import pandas
import numpy
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

numpy.random.seed(7)

datas = pandas.read_csv('train.csv')
datas_one = datas[datas['target'] == 1]
datas_one = pandas.concat([datas_one] * 25)
datas = pandas.concat([datas, datas_one])

target = datas[['target']]
features = datas.drop(['id', 'target'], axis=1)

target_one = target[target['target'] == 1]
target_zero = target[target['target'] == 0]

print(target_one.shape)
print(target_zero.shape)

features_bin = features.filter(regex='bin$')
features_cat = features.filter(regex='cat$')
features_ord = features.drop(features_bin, axis=1).drop(features_cat, axis=1)
features_ord = features_ord.replace(-1, 0)

features_bin = features_bin.values
features_cat = features_cat.values
features_ord = features_ord.values
target = target.values

scaler = MinMaxScaler(feature_range=(0, 1))
features_ord_normalize = scaler.fit_transform(features_ord)

features_input = numpy.concatenate((features_bin, features_ord_normalize), axis=1)

hidden_neuro = 512

X_train, X_test, Y_train, Y_test = train_test_split(features_input, target, test_size=0.33, random_state=42)

model = Sequential()
model.add(Dense(hidden_neuro, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(hidden_neuro, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(hidden_neuro, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(hidden_neuro, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='tanh'))
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.summary()
model.fit(X_train, Y_train, batch_size=128, epochs=100, validation_data=(X_test, Y_test))
