import pandas
import numpy
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras import utils

numpy.random.seed(7)

datas = pandas.read_csv('train.csv')
datas_one = datas[datas['target'] == 1]
datas_one = pandas.concat([datas_one] * 25)
datas = pandas.concat([datas, datas_one])

target = datas[['target']]
features = datas.drop(['id', 'target'], axis=1)
features = features.replace(-1, 0)

target_one = target[target['target'] == 1]
target_zero = target[target['target'] == 0]
print(target_one.shape)
print(target_zero.shape)

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

features_ind_bin = features_ind_bin.values
features_ind_cat = features_ind_cat.values
features_ind_ord = features_ind_ord.values

features_reg_bin = features_reg_bin.values
features_reg_cat = features_reg_cat.values
features_reg_ord = features_reg_ord.values

features_car_bin = features_car_bin.values
features_car_cat = features_car_cat.values
features_car_ord = features_car_ord.values

features_calc_bin = features_calc_bin.values
features_calc_cat = features_calc_cat.values
features_calc_ord = features_calc_ord.values

features_categorical_ind = [utils.to_categorical(features_ind_cat[:, x]) for x in range(features_ind_cat.shape[1])]
features_categorical_reg = [utils.to_categorical(features_reg_cat[:, x]) for x in range(features_reg_cat.shape[1])]
features_categorical_car = [utils.to_categorical(features_car_cat[:, x]) for x in range(features_car_cat.shape[1])]
features_categorical_calc = [utils.to_categorical(features_calc_cat[:, x]) for x in range(features_calc_cat.shape[1])]

features_bin = features_car
features_cat = features_ind_cat
features_ord = features_car_ord.astype('float32')
target = target.values

scaler = MinMaxScaler(feature_range=(0, 1))
features_ord_normalize = scaler.fit_transform(features_ord)

features_input = numpy.concatenate(
    (features_bin,
     features_ord_normalize,
     features_categorical_car[0],
     features_categorical_car[1],
     features_categorical_car[2],
     features_categorical_car[3],
     features_categorical_car[4],
     features_categorical_car[5],
     features_categorical_car[6],
     features_categorical_car[7],
     features_categorical_car[8],
     features_categorical_car[9],
     features_categorical_car[10]),
    axis=1)

print(features_input.shape)

hidden_neuro = 128

X_train, X_test, Y_train, Y_test = train_test_split(features_input, target, test_size=0.33, random_state=42)

model = Sequential()
model.add(Dense(hidden_neuro, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(hidden_neuro, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(hidden_neuro, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(hidden_neuro, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='tanh'))
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.summary()
model.fit(X_train, Y_train, batch_size=128, epochs=100, validation_data=(X_test, Y_test))
