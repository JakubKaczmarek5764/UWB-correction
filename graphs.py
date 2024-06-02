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
mse_df = pd.read_csv('history.csv')
mse_train_values = mse_df['training_mse']
mse_test_values = mse_df['test_mse']

# tu beda 3 serie danych dla 1, 2, 3 sieci ukrytych
predictions = pd.read_csv('predictions.csv')

plt.plot(range(1, len(mse_train_values) + 1), mse_train_values, marker='')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('MSE Over Epochs Train')
plt.grid(True)
plt.show()


plt.plot(range(1, len(mse_test_values) + 1), mse_test_values, marker='')
plt.plot([1, len(mse_test_values) + 1], [mse_wo_correction, mse_wo_correction])
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('MSE Over Epochs Test')
plt.grid(True)
plt.show()

def euclidean_distance(p1, p2):
    return math.sqrt((p1.iloc[0] - p2.iloc[0]) ** 2 + (p1.iloc[1] - p2.iloc[1]) ** 2)

errors = [euclidean_distance(Y_test.iloc[i], predictions.iloc[i]) for i in range(len(predictions))]
errors.sort()
cumulative_probabilities = np.arange(1, len(errors) + 1) / len(errors)


# Plot the CDF
plt.plot(errors, cumulative_probabilities, marker='', linestyle='-')
plt.xlabel('Error')
plt.ylabel('Cumulative Frequency')
plt.title('CDF of Prediction Errors')
plt.grid(True)
plt.show()


plt.scatter(x=X_test[['measured_x']], y=X_test[['measured_y']], s=10)
plt.scatter(x=predictions[['predicted_x']], y=predictions[['predicted_y']], s=10)
plt.scatter(x=Y_test[['expected_x']], y=Y_test[['expected_y']], s=10)

plt.yticks(np.linspace(0, 3000, num=4))
plt.show()