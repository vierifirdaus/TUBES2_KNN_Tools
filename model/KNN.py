import dill
import pandas as pd
def load_function(filename):
    with open(filename, 'rb') as file:
        return dill.load(file)

iterate, iterate_k, classify_point, distance, find_correlation, normalize,pointbiserialr = load_function('model/KNN.pkl')

train=pd.read_csv('data/data_train.csv')
inputs=pd.read_csv('data/test.csv')
result = iterate(inputs[:10], train, 3)
print(result)