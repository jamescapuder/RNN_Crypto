import pandas as pd
import numpy as np



def getData(path):    
    hdfstore = pd.HDFStore(path)
    full_set = hdfstore.select('table')
    full_s = full_set.reset_index(drop=True)
    
    dates = full_s.pop('Date')
    full_s.index = dates
    return full_s
