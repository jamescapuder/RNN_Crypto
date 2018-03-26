import json
from datetime import datetime
import time
import pandas as pd
import requests

class cryptoWrapper:
    URL = "https://api.cryptowat.ch/markets/prices"
    SAVE_PATH="data/allMarkets"


    def __init__(self):
        self.SAVE_PATH += str(datetime.now()) + ".h5"
        self.df = self.request()

    
    def request(self):
        request = requests.get(self.URL)
        jsn = json.loads(request.content)
        results = jsn["result"]
        results["Date"] = datetime.now()
        results["DPrice"] = results["bitfinex:btcusd"]
        df = pd.DataFrame.from_dict([results])
        return df

    def update(self):
        req = self.request()
        self.df = pd.concat([self.df, req])
        return req

    def save(self):
        df = self.df.drop_duplicates()
        df.to_hdf(self.SAVE_PATH, key='table')
        
