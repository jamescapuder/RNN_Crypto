import matplotlib.pyplot as plt
from itertools import takewhile
import fire

def get_results_from_file(fname):
    delim_string = "##################################"
    loss = []
    performance_tab = {"Actual": [], "Pred": [], "Diff":[]}
    with open(fname, "r") as f:
        for line in f:
            if line.startswith("loss"):
                loss_dat = takewhile(lambda x: not x.startswith(delim_string), f)
                loss.extend([float(z.strip()) for z in loss_dat])
            if line.startswith("Actual Values"):
                perf_tab = takewhile(lambda x: not x.startswith(delim_string), f)
                for l in perf_tab:
                    a,p,d = l.strip().split(",")
                    performance_tab["Actual"].append(float(a))
                    performance_tab["Pred"].append(float(p))
                    performance_tab["Diff"].append(float(d))
    return loss, performance_tab

def performance_plot(targ_name, dest_name):
    loss, performance_tab = get_results_from_file(targ_name)
    perf_fig = plt.figure(figsize=(8,8),dpi=800)
    plt.plot(performance_tab["Actual"])
    plt.plot(performance_tab["Pred"])
    plt.ylabel("Price")
    plt.title(targ_name + " Performance")
    perf_fig.savefig(dest_name, bbox_inches='tight')

def diff_plot(targ_name, dest_name):
    loss, performance_tab = get_results_from_file(targ_name)
    perf_fig = plt.figure(figsize=(15,8),dpi=1200)
    plt.plot(performance_tab["Diff"])
    #plt.plot(performance_tab["Pred"])
    plt.ylabel("Diff")
    plt.title(targ_name + "Diff")
    perf_fig.savefig(dest_name, bbox_inches='tight')
    
def loss_plot(targ_name, dest_name):
    loss, performance_tab = get_results_from_file(targ_name)
    #loss2, perf2 = get_results_from_file()
    loss_fig = plt.figure(dpi=1200)
    plt.plot(loss)
    #plt.axis([0,100,0,1])
    #plt.plot(loss)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.title(targ_name + " Loss")
    loss_fig.savefig(dest_name, bbox_inches='tight')


if __name__=='__main__':
    fire.Fire()
