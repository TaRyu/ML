import pandas as pd

FTRAIN = ''

def load_data():
    data = pd.read_csv(FTRAIN)
    return data
