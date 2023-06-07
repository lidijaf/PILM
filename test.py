
import numpy as np
from iterative_methods import *
from evaluations import *
from splitting import *
from read_data import *
import time
import os
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

K = size # number of subproblems that we split into, has to be positive

#only the master process reads and prepares the data
if rank==0:

    # choose the problem that we want to solve
    cwd = os.getcwd()
    folderpath = 'timetest_010_1000/'

    # select the file containing the coordinate observations
    #st_dev and countP determine which coordinate observation file (initial guess) we want to use 
    st_dev = 1.0
    countP = 0
    PO_file = "project_"+str(st_dev)+'_'+str(countP)+".tco"

    observations, N_pt, adj_list, X0 = setup_problem(folderpath, PO_file, weights = [1,3,3])

    # if splitting, call clustering, which calls Metis to partition the graph and sets up the splitting
    cl_variables, cl_labels, cl_var_dic = clustering(adj_list, K, splitting = 'metis')
    
    # %%timeit -r1 -n1
    # call the solver
    # RHS_mat defines the way we approximate the direction d
    #        = 'none' block diagonal matrix with no RHS correction
    #        = 'B' block diagonal matrix with RHS correction given by the non-diagonal blocks
    #        = 'inner_it' d computed with fixed-point inner iterations
else:
    X0 = None
    N_pt = None
    observations = None
    cl_variables = None
    cl_var_dic = None
    cl_labels = None

X0 = comm.bcast(X0, root=0)
N_pt = comm.bcast(N_pt, root=0)
observations = comm.bcast(observations, root=0)
cl_variables = comm.bcast(cl_variables, root=0)
cl_var_dic = comm.bcast(cl_var_dic, root=0)
cl_labels = comm.bcast(cl_labels, root=0)

comm.Barrier()
start = MPI.Wtime()
X, it, R = LM(X0, N_pt, observations, cl_variables,cl_var_dic, cl_labels, K, maxit=1000, RHS_mat='inner_it', PO_as_obs=1)
comm.Barrier()
end = MPI.Wtime()
local_time = end -start
maxTime = 0
maxTime = comm.reduce(local_time, op=MPI.MAX, root=0)
if rank==0:
    print("Elapsed time:")
    print(maxTime)
    np.savetxt("Solution.txt", X)
