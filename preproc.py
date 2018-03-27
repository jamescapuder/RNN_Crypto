import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import prepscript
import os
import keras
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from sklearn import metrics

class Setup:
    def __init__(self,valcol,num_units, modpath="",second_layer=False, optim="rmsprop"):
        self.unNormed = self.get_frames()
        self.frames = {}
        self.aggregate=[]
        self.valcol = valcol
        self.chunks={}
        self.maybe_chunks = []
        for k,v in self.unNormed.items():
            temp_frame = self.get_cols(v)
            #self.aggregate.extend([float(a) for a in temp_frame[[self.valcol]].values])
            #normed = self.normalize_frame(temp_frame)
            shifted = self.add_labels(temp_frame, -1)
            self.aggregate.append(shifted)
            self.frames[k] = shifted
            self.unNormed[k]=temp_frame
        #print(self.aggregate)
        concatted_frame=pd.concat(self.aggregate)
        self.master_frame = concatted_frame.dropna(how='any')
        self.mean, self.std = self.normalize_master()
        if modpath=="":
            self.mod = self.build_mod(num_units,second_layer,optim)
        else:
            self.mod = keras.models.load_model(modpath)
            
    def break_into_chunks(self):
        for k,v in self.unNormed.items():
            keyList = np.array_split(v, 25)
            #self.chunks[k] = keyList
            toext = [x[[self.valcol]].values for x in keyList]
            self.maybe_chunks.extend(toext)

    
        
    def do_run(self,frame_name):
        focus_data = self.frames[frame_name]
        train, test = self.train_test_split(focus_data)
        trainReshaped = self.reshape_train(train)
        hist = self.fit_model(train,trainReshaped)
        pred_results = self.get_test_pred(test)
        #print(pred_results)
        #self.mod.save("models/test_mod.h5")
        return hist, self.un_norm_mse(frame_name, test, pred_results)
        
    def un_norm_mse(self, frame_name,test, test_pred):
        mn = self.unNormed[frame_name].mean().values[0]
        sdev = self.unNormed[frame_name].std().values[0]
        testY = test[["label"]].values.tolist()
        testYflat = [b[0] for b in testY]
        testPredList = test_pred.tolist()
        testPredFlat = [b[0] for b in testPredList]
        normedPred = [x*sdev+mn for x in testPredFlat]
        normedY = [x*sdev+mn for x in testYflat]
        retdict ={"Predicted Values": normedPred, "Actual Values": normedY, "RMSE": math.sqrt(metrics.mean_squared_error(normedY[:-2], normedPred[:-2])), "Normalized Actual": testYflat, "Normalized Predictions": testPredFlat} 
        return retdict
        
        
    def get_frames(self):
        root = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(root, "data")
        sequences_dict = {}
        for p,s,f in os.walk(data_path):
            for name in f:
                if name.endswith(".h5"):
                    data = prepscript.getData("data/"+name)
                    sequences_dict[name]=data
        return sequences_dict

    def get_cols(self, frame):
        return frame[[self.valcol]]

    def normalize_frame(self,frame):
        newframe = (frame-frame.mean())/frame.std()
        return newframe

    def normalize_master(self):
        mean = self.master_frame[[self.valcol]].mean().values[0]
        std = self.master_frame[[self.valcol]].std().values[0]
        self.master_frame[[self.valcol]] = (self.master_frame[[self.valcol]]-mean)/std
        self.master_frame[['label']] = (self.master_frame[['label']]-mean)/std
        return mean, std
        

    def add_labels(self,frame,shift):
        frame[['label']] = frame[[self.valcol]].shift(shift)
        return frame

    def train_test_split(self,frame):
        train = frame.iloc[0:int(0.7*len(frame.index))]
        test = frame.iloc[int(0.7*len(frame.index)):len(frame.index)]
        return train, test

    def reshape_train(self,train):
        trainX = train[[self.valcol]].values
        trainXreshaped = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        return trainXreshaped

    def build_mod(self,num_units,second_layer=False,optim="rmsprop"):
        mod = Sequential()
        if second_layer:
            mod.add(LSTM(num_units, return_sequences=True,input_shape=(None, 1)))
            mod.add(LSTM(num_units))
        else:
            mod.add(LSTM(num_units, input_shape=(None, 1)))
        mod.add(Dense(1))
        mod.compile(loss="mse", optimizer=optim)
        return mod
    
    def fit_model(self,train, trainXreshaped):
        return self.mod.fit(trainXreshaped, train[["label"]], epochs=100, batch_size=1, verbose=2)

    def get_test_pred(self, test):
        testXreshaped = self.reshape_train(test)
        return self.mod.predict(testXreshaped)

# def main():
#     util = Setup('bitfinex:btcusd')
#     longest_frame_name = ""
#     for k,v in util.frames.items():
#         if longest_frame_name in util.frames:
#             if len(util.frames[k].index)>len(util.frames[longest_frame_name].index):
#                 longest_frame_name = k
#         else:
#             longest_frame_name = k
    
#     predictions, y_vals, rmse,predflat, yflat= util.do_run(longest_frame_name)
    
#     plt.plot(y_vals)
#     plt.plot(predictions)
#     plt.show()
def main():
    util = Setup('bitfinex:btcusd',6)
    print(util.master_frame)
main()
