import pandas
import numpy
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

datas = pandas.read_csv('train.csv')

target = datas[['id', 'target']]
features = datas.drop(['target'], axis=1)

target_one = target[target['target'] == 1]
target_zero = target[target['target'] == 0]
target_zero = target_zero.sample(target_one.shape[0])

features = features[features['id'].isin(target_one['id']) | features['id'].isin(target_zero['id'])]
target = pandas.concat([target_one, target_zero])

features = features.sort_values(['id'])
target = target.sort_values(['id'])

features = features.drop(['id'], axis=1)
target = target.drop(['id'], axis=1)

features_bin = features.filter(regex='bin$')
features_cat = features.filter(regex='cat$')
features_ord = features.drop(features_bin, axis=1).drop(features_cat, axis=1)

features_bin = features_bin.values
features_cat = features_cat.values
features_ord = features_ord.values
target = target.values

scaler = MinMaxScaler(feature_range=(0, 1))
features_ord_normalize = scaler.fit_transform(features_ord)

features_input = numpy.concatenate((features_bin, features_ord_normalize), axis=1)

hidden_neuro = 256

model = Sequential()
model.add(Dense(hidden_neuro, input_shape=(features_input.shape[1],), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(hidden_neuro, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(hidden_neuro, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(hidden_neuro, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(hidden_neuro, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(hidden_neuro, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(hidden_neuro, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(hidden_neuro, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.fit(features_input, target, batch_size=64, epochs=100, validation_split=0.2)
