import itertools
from itertools import product

def generate_combinations(activation_functions, optimizers, num_of_layers, num_of_neurons, learning_rates, batch_sizes, epochs):
    combinations = []
    for af, opt, num_layers, num_neurons, lr, batch_size, epoch in product(activation_functions, optimizers, num_of_layers, num_of_neurons, learning_rates, batch_sizes, epochs):
        combinations.append({
            'activation_function': af,
            'optimizer': opt,
            'num_of_layers': num_layers,
            'num_of_neurons': num_neurons,
            'learning_rate': lr,
            'batch_size': batch_size,
            'epochs': epoch
        })
    return combinations

# Example usage:



# combinations = generate_combinations(activation_functions, optimizers, num_of_layers, num_of_neurons, learning_rates, batch_sizes, epochs)


def return_combinations(num_of_layers):
    activation_functions = itertools.combinations_with_replacement(['relu', 'sigmoid', 'tanh'], num_of_layers)
    optimizers = ['Adam', 'SGD']

    num_of_neurons = itertools.combinations_with_replacement([4, 8, 32], num_of_layers)
    learning_rates = [0.01, 0.001, 0.0001]
    batch_sizes = [64]
    epochs = [200]
    return generate_combinations(activation_functions, optimizers, [num_of_layers], num_of_neurons, learning_rates, batch_sizes,
                          epochs)
def return_all_combinations():
    return return_combinations(1) + return_combinations(2) + return_combinations(3)
combinations = return_all_combinations()
# print(combinations)
f = open('new.txt', 'w')
for c in combinations:
    f.write(str(c) + "\n")