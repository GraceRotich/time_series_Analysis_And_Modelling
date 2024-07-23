# importing relevant libraries

# Analysis libraries
import subprocess

# Install pandas using pip
subprocess.check_call(["pip", "install", "pandas"])

# Now you can import pandas
import pandas as pd


class data_loading:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        df = pd.read_csv(self.file_path)
        
        #shape of the dataframe
        print("\nShape of the dataframe:")
        print(df.shape)
        
        #column datatypes
        print("\nColumn datatypes:")
        print(df.dtypes)
        
        #describe the dataframe
        print("\nDescription of the dataframe:")
        print(df.describe())
        
        return df
