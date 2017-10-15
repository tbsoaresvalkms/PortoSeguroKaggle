import pandas
import numpy
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

datas = pandas.read_csv('train.csv')

target = datas[['target']]
features = datas.drop(['id', 'target'], axis=1)

features_ind = features.filter(regex='ind')
features_reg = features.filter(regex='reg')
features_car = features.filter(regex='car')
features_calc = features.filter(regex='calc')

features_ind_bin = features_ind.filter(regex='bin$')
features_ind_cat = features_ind.filter(regex='cat$')
features_ind_ord = features_ind.drop(features_ind_bin, axis=1).drop(features_ind_cat, axis=1)

features_reg_bin = features_reg.filter(regex='bin$')
features_reg_cat = features_reg.filter(regex='cat$')
features_reg_ord = features_reg.drop(features_reg_bin, axis=1).drop(features_reg_cat, axis=1)

features_car_bin = features_car.filter(regex='bin$')
features_car_cat = features_car.filter(regex='cat$')
features_car_ord = features_car.drop(features_car_bin, axis=1).drop(features_car_cat, axis=1)

features_calc_bin = features_calc.filter(regex='bin$')
features_calc_cat = features_calc.filter(regex='cat$')
features_calc_ord = features_calc.drop(features_calc_bin, axis=1).drop(features_calc_cat, axis=1)

features_bin = features_ind_bin.values
features_cat = features_ind_cat.values
features_ord = features_ind_ord.values.astype('float32')
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
model.fit(features_input, target, batch_size=128, epochs=1, validation_split=0.2)

predict = model.predict(features_input)
