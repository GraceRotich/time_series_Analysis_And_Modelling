# importing relevant libraries
import pandas as pd


class Data_Loader:
    def __init__(self):
        pass

    def load_data(self, file_path):
        # Loading the Dataset
        df = pd.read_csv(file_path)
        
        return df

class Data_Informer(Data_Loader):
    def __init__(self):
        super().__init__()

    def print_info(self, df):
        # Shape of the dataframe
        print("\nShape of the dataset:")
        print(df.shape)
        
        # Column data Information
        print("\nInformation about the Dataset:")
        print(df.info())
        
        #Data Types
        data_types = df.dtypes
        print("\nColumns and their data types:")
        for column, dtype in data_types.items():
            print(f"{column}: {dtype}")
    
class Data_cleaning:
    def __init__(self):
        pass
    
class Analysis:
    def __init__(self):
        pass
    
class Modeling:
    def __init__(self):
        pass
    
class Evaluation:
    def __init__(self):
        pass
        
   
