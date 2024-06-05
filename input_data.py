import glob

import pandas as pd


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
