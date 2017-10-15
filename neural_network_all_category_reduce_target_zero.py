import pandas
import numpy
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras import utils

numpy.random.seed(7)


def get_target_reduced():
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
    features = features.replace(-1, 0)

    target_one = target[target['target'] == 1]
    target_zero = target[target['target'] == 0]
    print(target_one.shape)
    print(target_zero.shape)

    return features, target


def get_target_real():
    datas = pandas.read_csv('train.csv')
    target = datas[['target']]
    features = datas.drop(['id', 'target'], axis=1)
    features = features.replace(-1, 0)

    return features, target


features, target = get_target_real()

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

features_ind_ord = features_ind_ord.astype('float32')
features_reg_ord = features_reg_ord.astype('float32')
features_car_ord = features_car_ord.astype('float32')
features_calc_ord = features_calc_ord.astype('float32')

target = target.values

scaler_ind = MinMaxScaler(feature_range=(0, 1))
features_ind_ord_normalize = scaler_ind.fit_transform(features_ind_ord)

scaler_reg = MinMaxScaler(feature_range=(0, 1))
features_reg_ord_normalize = scaler_reg.fit_transform(features_reg_ord)

scaler_car = MinMaxScaler(feature_range=(0, 1))
features_car_ord_normalize = scaler_car.fit_transform(features_car_ord)

scaler_calc = MinMaxScaler(feature_range=(0, 1))
features_calc_ord_normalize = scaler_calc.fit_transform(features_calc_ord)

features_input = numpy.concatenate(
    (features_ind_bin,
     features_reg_bin,
     features_car_bin,
     features_calc_bin,
     features_ind_ord_normalize,
     features_reg_ord_normalize,
     features_car_ord_normalize,
     features_calc_ord_normalize,
     features_categorical_ind[0],
     features_categorical_ind[1],
     features_categorical_ind[2],
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

hidden_neuro_1 = 64
hidden_neuro_2 = 32

X_train, X_test, Y_train, Y_test = train_test_split(features_input, target, test_size=0.33, random_state=42)

model = Sequential()
model.add(Dense(hidden_neuro_1, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(hidden_neuro_2, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.summary()
model.fit(X_train, Y_train, batch_size=128, epochs=500, validation_data=(X_test, Y_test))
model.evaluate(X_test, Y_test)
