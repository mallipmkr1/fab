#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime
from clickhouse_driver import Client

import warnings
warnings.filterwarnings("ignore")


# In[4]:


mapper = {'object': 'String', 
          'float64':'Float64',
          'int64':'Int64',
          'datetime64[ns]':'Date'}


# In[5]:


def send_data(data,table_name,workspace):
    '''
    This function writes data to given table_name in workspace.
    data : Pandas Dataframe for numpy nDarray 
    table_name : string table name 
    workspace : string workspaces name like 'w42'
    '''
    client = Client('localhost', settings={'use_numpy': True})
    print(data.columns)
    if 'date_ts' in data.columns:
            data['date_ts'] = pd.to_datetime(data['date_ts'])
    if 'item_id' in data.columns:
            data['item_id'] = data['item_id'].astype(str)
    
    #get col types and replace with clickhouse nomenclature
    cols_types = data.dtypes
    cols_types = cols_types.replace(mapper)
    col_names = data.columns
    
    data = pd.DataFrame(data, dtype = object)
    data[data.isna()] = None
    #create table creation query
    t = ''
    for i in range(data.shape[1]):
        if cols_types[i] == 'String':
            data[col_names[i]] = data[col_names[i]].astype(str)
        t=t + (f'{col_names[i]}    {cols_types[i]},' )
    t = t[:-1]
    
    
    #clean and write to tables
    client.execute(f"""DROP TABLE IF EXISTS {workspace}.{table_name}""")
    
    
    client.execute(f"""
        CREATE TABLE IF NOT EXISTS {workspace}.{table_name}
        (
            {t}
        )
            Engine = MergeTree
                Order by {col_names[0]}
           """)
    
    client.insert_dataframe(f'''INSERT INTO {workspace}.{table_name} VALUES''', data)

