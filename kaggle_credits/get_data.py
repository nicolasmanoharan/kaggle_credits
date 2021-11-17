import pandas as pd


def get_data(nrows=10_000):
    df = pd.read_csv("raw_data/test.csv")



if __name__ == '__main__' :
    df = get_data()
