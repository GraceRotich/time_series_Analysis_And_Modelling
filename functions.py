# importing relevant libraries
import pandas as pd
import numpy as np
# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

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

class TimeSeriesAnalyzer:
    def __init__(self, dataframe, value_column):
        self.df = dataframe
        self.value_column = value_column
    
    def plot_overall(self, xlim=None, ylim=None, figsize=(12, 4), title=None):
        plt.figure(figsize=figsize)
        sns.lineplot(x=self.df.index, y=self.df[self.value_column])
        
        if xlim:
            plt.xlim(pd.Timestamp(xlim[0]), pd.Timestamp(xlim[1]))
        if ylim:
            plt.ylim(ylim)
        
        plt.xlabel('Date')
        plt.ylabel(self.value_column)
        plt.title(title)
        plt.show()
        
    def plot_series(self, start=None, end=None, xlim=None, ylim=None, title= None):
        plt.figure(figsize=(14, 7))
        plt.plot(self.df[start:end].index, self.df[start:end][self.value_column], label=self.value_column)
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title(title)
        plt.legend()
        plt.show()

    def plot_section(self, xlim, ylim=None, figsize=(12, 4), title=None):
        self.plot_overall(xlim=xlim, ylim=ylim, figsize=figsize, title=title)
    
    def plot_resampled(self, rule= None, figsize=(12, 4), title= None):
        resampled_df = self.df.resample(rule=rule).mean()
        plt.figure(figsize=figsize)
        sns.lineplot(x=resampled_df.index, y=resampled_df[self.value_column])
        
        plt.xlabel('Date')
        plt.ylabel(self.value_column)
        plt.title(title)
        plt.show()

    def plot_smoothing(self, window_sma=3, span_ewma=0.3, figsize=(12, 4), title='Smoothing Plot'):
        sma = self.df[self.value_column].rolling(window=window_sma).mean()
        ewma = self.df[self.value_column].ewm(span=span_ewma).mean()
        
        plt.figure(figsize=figsize)
        sns.lineplot(x=self.df.index, y=self.df[self.value_column], label='Original')
        sns.lineplot(x=self.df.index, y=sma, label=f'SMA (window={window_sma})')
        sns.lineplot(x=self.df.index, y=ewma, label=f'EWMA (span={span_ewma})')
        
        plt.xlabel('Date')
        plt.ylabel(self.value_column)
        plt.title(title)
        plt.legend()
        plt.show()

    def plot_seasonal_decomposition(self, df, column ,model='additive', period=12, figsize=(12, 8), title='Seasonal Decomposition'):
        decomposition = seasonal_decompose(df[column], model=model, period=period)
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        ax1.plot(df.index, df[column], label='Original')
        ax1.legend(loc='upper left')
        ax1.set_title(title)
        
        ax2.plot(df.index, decomposition.trend, label='Trend')
        ax2.legend(loc='upper left')
        
        ax3.plot(df.index, decomposition.seasonal, label='Seasonal')
        ax3.legend(loc='upper left')
        
        ax4.plot(df.index, decomposition.resid, label='Residual')
        ax4.legend(loc='upper left')
        
        plt.xlabel('Date')
        plt.show()

class Modeling:
    def __init__(self):
        pass
    
class Evaluation:
    def __init__(self):
        pass
        
   
