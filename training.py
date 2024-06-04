
import keras
import pandas as pd
from keras import Input
from keras.src.losses import mean_squared_error
from keras.src.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

import input_data

params = {
    'activation_function': ('relu', 'tanh', 'tanh'),
    'optimizer': 'Adam',
    'num_of_layers': 3,
    'num_of_neurons': (4,4,32),
    'batch_size': 64,
    'epochs': 100,
    'patience': 10
}
# one_layer = {
#     'activation_function': ('relu',),
#     'optimizer': 'Adam',
#     'num_of_layers': 1,
#     'num_of_neurons': (32,),
#     'batch_size': 64,
#     'epochs': 100,
#     'patience': 10
# }
# two_layers = {
#     'activation_function': ('relu', 'relu'),
#     'optimizer': 'Adam',
#     'num_of_layers': 2,
#     'num_of_neurons': (4, 8),
#     'batch_size': 64,
#     'epochs': 100,
#     'patience': 10
# }
# three_layers = {
#     'activation_function': ('relu', 'tanh', 'tanh'),
#     'optimizer': 'Adam',
#     'num_of_layers': 3,
#     'num_of_neurons': (4, 4, 32),
#     'learning_rate': 0.001,
#     'batch_size': 64,
#     'epochs': 100
# }
while(1):
    training_data = input_data.load__training_data()
    test_data = input_data.load_test_data()

    X_train = training_data[['measured_x', 'measured_y']]
    Y_train = training_data[['expected_x', 'expected_y']]

    X_test = test_data[['measured_x', 'measured_y']]
    Y_test = test_data[['expected_x', 'expected_y']]

    X_scaler = StandardScaler()
    Y_scaler = StandardScaler()

    X_train = X_scaler.fit_transform(X_train)
    Y_train = Y_scaler.fit_transform(Y_train)

    X_test = X_scaler.transform(X_test)
    Y_test = Y_scaler.transform(Y_test)

    X_train = np.nan_to_num(X_train)
    Y_train = np.nan_to_num(Y_train)
    X_test = np.nan_to_num(X_test)
    Y_test = np.nan_to_num(Y_test)

    # blad sredniokwadratowy dla danych testowych
    X_T = X_test.T # transpozycja
    Y_T = Y_test.T
    x_mse_wo_correction = np.mean((X_T[0] - Y_T[0]) ** 2) # liczenie mse dla x
    y_mse_wo_correction = np.mean((X_T[1] - Y_T[1]) ** 2) # liczenie mse dla y
    mse_wo_correction = (x_mse_wo_correction + y_mse_wo_correction) / 2
    f = open('mse_wo_correction.txt', 'w')
    f.write(str(mse_wo_correction))

    #definiowanie modelu
    model = Sequential()
    model.add(Input(shape=(2,)))
    for i in range(params['num_of_layers']):
        model.add(Dense(params['num_of_neurons'][i], activation=params['activation_function'][i]))
    model.add(Dense(2, activation="linear"))
    model.compile(optimizer=params['optimizer'], loss="mse")

    early_stopping = EarlyStopping(monitor='loss', patience=params['patience'], restore_best_weights=True)
    history = model.fit(X_train, Y_train, epochs=params['epochs'], batch_size=params['batch_size'], callbacks=[early_stopping],validation_data=(X_test, Y_test),verbose=1)

    loss = model.evaluate(X_test, Y_test)
    print(loss)
    history_df = pd.DataFrame(history.history)
    history_df.columns = ['training_mse', 'test_mse']
    history_df.to_csv('history3.csv', index=False)
    predictions = model.predict(X_test)

    X_train = pd.DataFrame(X_scaler.inverse_transform(X_train))
    Y_train = pd.DataFrame(Y_scaler.inverse_transform(Y_train))

    X_test = pd.DataFrame(X_scaler.inverse_transform(X_test))
    Y_test = pd.DataFrame(Y_scaler.inverse_transform(Y_test))

    predictions = Y_scaler.inverse_transform(predictions)

    pred_df = pd.DataFrame(predictions)

    pred_df.columns=['predicted_x', 'predicted_y']
    pred_df.to_csv('predictions3.csv', index=False)
    model.save('threelayers.keras')
    break


