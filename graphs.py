import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import input_data

training_data = input_data.load__training_data()
test_data = input_data.load_test_data()

X_train = training_data[['measured_x', 'measured_y']]
Y_train = training_data[['expected_x', 'expected_y']]

X_test = test_data[['measured_x', 'measured_y']]
Y_test = test_data[['expected_x', 'expected_y']]

mse_wo_correction = float(open('mse_wo_correction.txt').readline())

# tu beda 3 serie danych dla 1, 2, 3 sieci ukrytych
mse_df1 = pd.read_csv('history1.csv')
mse_train_values1 = mse_df1['training_mse']
mse_test_values1 = mse_df1['test_mse']


mse_df2 = pd.read_csv('history2.csv')
mse_train_values2 = mse_df2['training_mse']
mse_test_values2 = mse_df2['test_mse']

mse_df3 = pd.read_csv('history3.csv')
mse_train_values3 = mse_df3['training_mse']
mse_test_values3 = mse_df3['test_mse']

# tu beda 3 serie danych dla 1, 2, 3 sieci ukrytych
predictions1 = pd.read_csv('predictions1.csv')
predictions2 = pd.read_csv('predictions2.csv')
predictions3 = pd.read_csv('predictions3.csv')

plt.plot(range(1, len(mse_train_values1) + 1), mse_train_values1, marker='', label="Sieć o 1 warstwie ukrytej")

plt.plot(range(1, len(mse_train_values2) + 1), mse_train_values2, marker='', label="Sieć o 2 warstwach ukrytych")

plt.plot(range(1, len(mse_train_values3) + 1), mse_train_values3, marker='', label="Sieć o 3 warstwach ukrytych")
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('MSE Over Epochs Train')
plt.ylim(0.015, 0.045)
plt.legend(loc="best")
plt.grid(True)
plt.show()


plt.plot(range(1, len(mse_test_values1) + 1), mse_test_values1, marker='', label="Sieć o 1 warstwie ukrytej")
plt.plot(range(1, len(mse_test_values2) + 1), mse_test_values2, marker='', label="Sieć o 2 warstwach ukrytych")
plt.plot(range(1, len(mse_test_values3) + 1), mse_test_values3, marker='', label="Sieć o 3 warstwach ukrytych")
plt.plot([1, len(mse_test_values2) + 1], [mse_wo_correction, mse_wo_correction])
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('MSE Over Epochs Test')
plt.ylim(0.015, 0.045)
plt.legend(loc="best")
plt.grid(True)
plt.show()

def euclidean_distance(p1, p2):
    return math.sqrt((p1.iloc[0] - p2.iloc[0]) ** 2 + (p1.iloc[1] - p2.iloc[1]) ** 2)

errors1 = [euclidean_distance(Y_test.iloc[i], predictions1.iloc[i]) for i in range(len(predictions1))]
errors1.sort()
cumulative_probabilities1 = np.arange(1, len(errors1) + 1) / len(errors1)

errors2 = [euclidean_distance(Y_test.iloc[i], predictions2.iloc[i]) for i in range(len(predictions2))]
errors2.sort()
cumulative_probabilities2 = np.arange(1, len(errors2) + 1) / len(errors2)

errors3 = [euclidean_distance(Y_test.iloc[i], predictions3.iloc[i]) for i in range(len(predictions3))]
errors3.sort()
cumulative_probabilities3 = np.arange(1, len(errors3) + 1) / len(errors3)

# Plot the CDF
plt.plot(errors1, cumulative_probabilities1, marker='', linestyle='-', label="Sieć o 1 warstwie ukrytej")
plt.plot(errors2, cumulative_probabilities2, marker='', linestyle='-', label="Sieć o 2 warstwach ukrytych")
plt.plot(errors3, cumulative_probabilities3, marker='', linestyle='-', label="Sieć o 3 warstwach ukrytych")

#jeszcze jeden tutaj

plt.xlabel('Error')
plt.ylabel('Cumulative Frequency')
plt.title('CDF of Prediction Errors')
plt.legend(loc="best")
plt.grid(True)
plt.show()


plt.scatter(x=X_test[['measured_x']], y=X_test[['measured_y']], s=10, label="Pomiary dynamiczne")
plt.scatter(x=predictions1[['predicted_x']], y=predictions1[['predicted_y']], s=10, label="Skorygowane")
plt.scatter(x=Y_test[['expected_x']], y=Y_test[['expected_y']], s=10, label="Rzeczywiste")
plt.legend(loc="best")
plt.yticks(np.linspace(0, 3000, num=4))
plt.show()