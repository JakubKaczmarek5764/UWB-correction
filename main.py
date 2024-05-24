import glob
import random

import pandas as pd
import tensorflow as tf
from keras.src.layers import Dropout, BatchNormalization, LeakyReLU
from keras.src.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError
import numpy as np
from tensorflow.keras.initializers import HeNormal
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.ops.init_ops import he_normal

import new


def load__training_data():
    f8_training_files = glob.glob('f8/stat/*.csv')
    f8_dfs = [pd.read_csv(file, header=None) for file in f8_training_files]
    f10_training_files = glob.glob('f10/stat/*.csv')
    f10_dfs = [pd.read_csv(file, header=None) for file in f10_training_files]
    combined_df = pd.concat(f8_dfs + f10_dfs, ignore_index=True)
    combined_df.columns = ['measured_x', 'measured_y', 'expected_x', 'expected_y']
    return combined_df

def load_test_data():
    f8_test_files = glob.glob('f8/dyn/*.csv')
    f8_dfs = [pd.read_csv(file, header=None) for file in f8_test_files]
    f10_test_files = glob.glob('f10/dyn/*.csv')


    f10_dfs = [pd.read_csv(file, header=None) for file in f10_test_files]


    combined_df = pd.concat(f8_dfs + f10_dfs, ignore_index=True)
    combined_df.columns = ['measured_x', 'measured_y', 'expected_x', 'expected_y']
    return combined_df



# combination_file = open('threelayers/combinations.txt', 'a')
# combinations = new.return_combinations(3)
# random.shuffle(combinations)

one_layer = {
    'activation_function': ('relu',),
    'optimizer': 'Adam',
    'num_of_layers': 1,
    'num_of_neurons': (32,),
    'learning_rate': 0.001,
    'batch_size': 64,
    'epochs': 100
}
two_layers = {
    'activation_function': ('relu', 'relu'),
    'optimizer': 'Adam', 'num_of_layers': 2,
    'num_of_neurons': (4, 8),
    'learning_rate': 0.001,
    'batch_size': 64,
    'epochs': 100
}
three_layers = {
    'activation_function': ('relu', 'tanh', 'tanh'),
    'optimizer': 'Adam',
    'num_of_layers': 3,
    'num_of_neurons': (4, 4, 32),
    'learning_rate': 0.001,
    'batch_size': 64,
    'epochs': 100
}

combinations = [one_layer, two_layers, three_layers]

for index, combination in enumerate(combinations):
    while 1:
        # loading data
        # combination_file.write(f'{index}' + str(combination) + '\n')
        training_data = load__training_data()
        test_data = load_test_data()

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


        # print(X_train)
        #
        # print(Y_train)
        #
        # print(X_test)
        #
        # print(Y_test)



        # funkcje aktywacji, optimizery, liczba warstw, liczba neuronow, learning rate, batch size, epochs
        # start = timer()
        # activation_functions = ['relu', 'sigmoid', 'tanh']
        # optimizers = [Adam, SGD]
        # num_of_layers = [1,2,3]
        # num_of_neurons = [2, 4, 8, 32, 64]
        # learning_rates = [0.1, 0.01, 0.001, 0.0001]
        # batch_sizes = [32, 64, 128]
        # epochs = [10, 20, 50]
        # Define the model
        model = Sequential()

        model.add(Dense(2, input_dim=2))
        for i in range(combination['num_of_layers']):
            model.add(Dense(units=combination['num_of_neurons'][i], activation=combination['activation_function'][i]))
        model.add(Dense(2, activation="linear"))

        if combination['optimizer'] == 'Adam':
            optimizer = Adam(learning_rate=combination['learning_rate'])
        elif combination['optimizer'] == "SGD":
            optimizer = SGD(learning_rate=combination['learning_rate'])

        model.compile(optimizer=optimizer, loss='mean_squared_error')

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        model.fit(X_train, Y_train, epochs=combination['epochs'], batch_size=combination['batch_size'], validation_split=.1, callbacks=[early_stopping])

        loss = model.evaluate(X_test, Y_test)
        print(f'Test Loss: {loss}')
        # model.save('first_good.h5')
        # model = load_model('first_good.h5')

        # Make predictions
        predictions = model.predict(X_test)

        # Inverse transform the predictions to get them back to the original scale
        X_train = pd.DataFrame(X_scaler.inverse_transform(X_train))
        Y_train = pd.DataFrame(Y_scaler.inverse_transform(Y_train))

        X_test = pd.DataFrame(X_scaler.inverse_transform(X_test))
        Y_test = pd.DataFrame(Y_scaler.inverse_transform(Y_test))

        predictions = Y_scaler.inverse_transform(predictions)
        y_test = Y_scaler.inverse_transform(Y_test)

        pred_df = pd.DataFrame(predictions, columns=['predicted_x', 'predicted_y'])
        # pred_df.to_csv('output.csv')

        # print(predictions)
        # print(X_test)
        # print(pred_df)

        if loss < 0.03:
            plt.scatter(x=X_test[0], y=X_test[1], s=10)
            plt.scatter(x=pred_df[['predicted_x']], y=pred_df[['predicted_y']], s=10)
            plt.scatter(x=Y_test[0], y=Y_test[1], s=10)

            plt.yticks(np.linspace(0, 3000, num=4))
            plt.title(f"{index}: {loss}")

            plt.savefig(f"threelayers/wykres{index}.png")
            plt.show()
            model.save(f'model{index+1}.h5')
            break
        # stop = timer()
        # print(stop - start)