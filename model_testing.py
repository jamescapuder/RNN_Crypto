import preproc
import random
from pathlib import Path
import os
import time

def get_longest_name(util):
    longest_frame_name = ""
    for k,v in util.frames.items():
        if longest_frame_name in util.frames:
            if len(util.frames[k].index)>len(util.frames[longest_frame_name].index):
                longest_frame_name = k
        else:
            longest_frame_name = k
    return longest_frame_name

def save_info(fname, hist, num_units, output_vector,targ_name, batch_size, epochs):
    delim_string="##################################\n"
    with open(targ_name, 'w') as f:
        f.write(delim_string)
        f.write("File Name: "+fname+"\n")
        f.write("Number of units: "+num_units+"\n")
        f.write("Epochs: " + str(epochs)+"\n")
        f.write("Batch Size: " +str(batch_size)+"\n")
        f.write("Z-Score Normalized\n")
        f.write(delim_string)
        f.write("History:\n")
        f.write(delim_string)
        for k,v in hist.history.items():
            f.write(k+":\n")
            for val in v:
                f.write(str(float(val))+"\n")
            f.write(delim_string)
        f.write("Model Performance\n")
        f.write("RMSE: {0:.5f}\n".format(float(output_vector["RMSE"])))
        f.write(delim_string)
        f.write("{0:20}, {1:20}, {2:20}, {3:20}\n".format("X Values", "Actual Values", "Predicted Values", "Actual-Pred"))
        for i,v in zip(output_vector["Actual Values"], output_vector["Predicted Values"]):
            f.write("{0:25}, {1:25}, {2:25}\n".format(float(i),float(v),float(i-v) ) )
        f.write(delim_string)
        f.write("{0:25}, {1:25}, {2:25}\n".format("Normalized Actual", "Normalized Predictions", "Actual-Pred"))
        for i,v in zip(output_vector["Normalized Actual"], output_vector["Normalized Predictions"]):
            f.write("{0:25}, {1:25}, {2:25}\n".format(float(i),float(v), float(i-v)))


def n_unit(n, strat):
    util = preproc.Setup('bitfinex:btcusd',n, optim=strat)
    #longest_name = get_longest_name(util)
    hist, outvec = util.master_run()
    mod_name = "models/test_"+str(n)+"_unit.h5"
    util.mod.save(mod_name)
    targ_name = str(n)+"unit_data_"+strat
    fpath = Path(targ_name+".txt")
    if fpath.is_file():
        targ_name += str(hash(random.randint(0,1000)))
    save_info("master_frame_run", hist, str(n),outvec,targ_name+".txt", util.batch_size, util.epochs)

def multi_layer(n):
    util = preproc.Setup('bitfinex:btcusd',n,second_layer=True)
    hist, outvec = util.master_run()
    mod_name = "models/test_"+str(n)+"_unit_2layer.h5"
    util.mod.save(mod_name)
    targ_name = str(n)+"unit_data_2layer.txt"
    save_info("master_frame_run", hist, str(n),outvec,targ_name)

def main():
    #multi_layer(6)
    n_unit(4, "adam")
main()
