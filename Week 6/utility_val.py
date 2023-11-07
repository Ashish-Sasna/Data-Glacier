
import os
import re
import yaml
import time
import logging
import pandas as pd

def read_yaml_file(file):
    with open(file, 'r') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            logging.error(e)
            
def validate(df, config_data):
    
    if 'columns' not in config_data:
        print('Error: Columns dictionary missing in the config file. Can\'t validate')
    else:
        config_cols = sorted(config_data['columns'])
        df_cols = sorted(df.columns)
        
        # Remove leading or trailing white spaces
        df.columns = list(map(lambda x:x.strip(), list(df.columns)))
        
        # Convert to lowercase
        df.columns = list(map(lambda x:x.lower(), list(df.columns)))
        
        # Remove any special characters
        df.columns = list(map(lambda x:re.sub('[^a-z_]+', '', x) if x != 'Unnamed: 0' else x, list(df.columns)))
        
        if len(config_cols) != len(df_cols):
            print('Error: Invalid number of columns in either config_file/dataset.')
            return 0
        elif list(config_cols) != list(df_cols):
            print('Error: Column names not matching as per config file.')
            return 0
        else:
            print('All tests passed.')
            return 1
