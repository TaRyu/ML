import numpy as np
import pandas as pd

FTRAIN = '/home/chlau/data/facial_keypoints/training.csv'


def load_data():
    data = pd.read_csv(FTRAIN)
    data.Image = data.Image.apply(lambda x: np.fromstring(x, sep=' '))/255
    return data
