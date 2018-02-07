import numpy as np
import scipy.stats as st


def flatten(l): return flatten(l[0]) + (flatten(l[1:]) if len(l) > 1 else []) if type(l) is list else [l]

#load the data

sgd = np.loadtxt("mean_swimmer_sgd.txt")
sn = np.loadtxt("mean_swimmer_sn.txt")
svrg = np.loadtxt("mean_swimmer_svrg.txt")


mean_sgd = np.mean(sgd)
mean_sn = np.mean(sn)
mean_svrg = np.mean(svrg)

var_sgd = np.var(sgd)
var_sn = np.var(sn)
var_svrg = np.var(svrg)

int_sgd = st.t.interval(0.95, len(sgd)-1, loc=np.mean(sgd), scale=st.sem(sgd))
int_sn = st.t.interval(0.95, len(sn)-1, loc=np.mean(sn), scale=st.sem(sn))
int_svrg = st.t.interval(0.95, len(svrg)-1, loc=np.mean(svrg), scale=st.sem(svrg))

np.savetxt("ci.txt",[int_sgd, int_sn, int_svrg])