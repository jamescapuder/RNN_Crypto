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

#Biggest Difference is the use of adam vs rmsprop

class Setup:
    def __init__(self,valcol,num_units, modpath="", second_layer=False, optim="rmsprop", epochs=1500, batch_size = 20):
        self.unNormed = self.get_frames()
        self.frames = {}
        self.valcol = valcol
        self.aggregate=[]
        self.chunks={}
        self.maybe_chunks = []
        for k,v in self.unNormed.items():
            #first = self.get_cols(v)
            temp_frame = self.get_cols(v) #self.make_diffs(first)
            shifted = self.add_labels(temp_frame, -1)
            #print(len(temp_frame.index))
            self.aggregate.append(shifted)
            self.frames[k] = shifted
            self.unNormed[k]=temp_frame
        concatted_frame=pd.concat(self.aggregate)
        self.master_frame = concatted_frame.dropna(how='any')
        self.mean, self.std = self.normalize_master()
        self.epochs = epochs
        self.batch_size = batch_size
        #self.max_x = self.normalize_master()
        if modpath=="":
            self.mod = self.build_mod(num_units,second_layer,optim)
        else:
            self.mod = keras.models.load_model(modpath)        

    def make_diffs(self, frame):
        frame = frame - frame.values.tolist()[0][0]
        return frame
            
    def master_run(self):
        train, test = self.train_test_split(self.master_frame)
        trainReshaped = self.reshape_train(train)
        hist = self.fit_model(train, trainReshaped)
        pred_results = self.get_test_pred(test)
        testY = test[["label"]].values.tolist()
        testYflat = [b[0] for b in testY]
        testPredList = pred_results.tolist()
        testPredFlat = [b[0] for b in testPredList]
        testX = test[[self.valcol]].values.tolist()
        testXflat = [b[0] for b in testX]
        normedX = [x*self.std + self.mean for x in testXflat]
        normedPred = [x*self.std + self.mean for x in testPredFlat]
        normedY = [x*self.std+self.mean for x in testYflat]
        # normedX = [x*self.max_x for x in testXflat]
        # normedPred = [x*self.max_x for x in testPredFlat]
        # normedY = [x*self.max_x for x in testYflat]
        retdict ={"Predicted Values": normedPred, "Actual Values": normedY, "RMSE": math.sqrt(metrics.mean_squared_error(normedY[:-2], normedPred[:-2])), "Normalized Actual": testYflat, "Normalized Predictions": testPredFlat, "X Values": normedX}
        return hist, retdict
        
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
        # max_x = self.master_frame[[self.valcol]].max().values[0]
        # self.master_frame[[self.valcol]] = (self.master_frame[[self.valcol]])/max_x
        # self.master_frame[['label']] = (self.master_frame[['label']])/max_x
        # return max_x

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
        return self.mod.fit(trainXreshaped, train[["label"]], epochs=self.epochs, batch_size=self.batch_size, verbose=1)

    def get_test_pred(self, test):
        testXreshaped = self.reshape_train(test)
        return self.mod.predict(testXreshaped)

def main():
    util = Setup('bitfinex:btcusd', 2)
main()
