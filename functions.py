# importing relevant libraries
import pandas as pd
import numpy as np


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
    
    

class DataCleaner:
    def __init__(self, data):
        self.data = data
    
    def remove_null_values(self):
        self.data = self.data.dropna()
        return self.data
    
    def remove_outliers(self, column_name):
        Q1 = self.data[column_name].quantile(0.25)
        Q3 = self.data[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        self.data = self.data[~((self.data[column_name] < lower_bound) | (self.data[column_name] > upper_bound))]
        return self.data
    
    def fill_missing_values(self, strategy='mean'):
        if strategy == 'mean':
            self.data = self.data.fillna(self.data.mean())
        elif strategy == 'median':
            self.data = self.data.fillna(self.data.median())
        elif strategy == 'mode':
            self.data = self.data.fillna(self.data.mode().iloc[0])
        return self.data



    
class DataPreparer:
    def __init__(self, data):
        self.data = data
    
    def melt_data(self):
        melted = pd.melt(self.data, id_vars=['RegionID', 'RegionName', 'City', 'State', 'Metro', 'CountyName', 'SizeRank'], var_name='time', value_name='value')
        melted['time'] = pd.to_datetime(melted['time'], infer_datetime_format=True)
        melted = melted.dropna(subset=['value'])
        return melted
    
    def check_null_values(self):
        null_counts = self.data.isnull().sum()
        return null_counts
    
    def missing_values_percentage(self):
        total = self.data.isnull().sum().sort_values(ascending=False)
        percent = (self.data.isnull().sum() / self.data.isnull().count() * 100).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        return missing_data
    
    def check_outliers(self, column_name):
        Q1 = self.data[column_name].quantile(0.25)
        Q3 = self.data[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = self.data[(self.data[column_name] < lower_bound) | (self.data[column_name] > upper_bound)]
        return outliers
    
    def modify_column_names(self):
        self.data.columns = [col.lower().replace(' ', '_') for col in self.data.columns]
        return self.data
       
class Analysis:
    def __init__(self):
        pass
    
class Modeling:
    def __init__(self):
        pass
    
class Evaluation:
    def __init__(self):
        pass
        
   
