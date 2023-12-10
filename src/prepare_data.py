
import pandas as pd
from omegaconf import OmegaConf
def prepare_data(config):
    print("preparing data")
    df= pd.read_csv(config.data.csv_file_path)
    print(df.head())






if __name__=="__main__":
    config= OmegaConf.load("params.yaml")
    prepare_data(config)

