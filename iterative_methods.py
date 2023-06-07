import numpy as np
from evaluations import *
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve
from splitting import *
import pypardiso
from mpi4py import MPI

def LM(X, N_pt, observations, cl_variables, cl_var_dic, cl_labels, K, maxit=50, RHS_mat = 'B',  PO_as_obs = 0):
    """
    :param X: initial guess
           N_pt: number of points
           observations: list of observations (described in read_data.setup_problem() )
           cl_variables, cl_var_dic, cl_labels: result of splitting.clustering() - only used if K>1
           K: number of subproblems that we split into
           maxit: maximum number of Levenberg-Marquardt iterations
           RHS_mat: matrix used for the RHS correction - only used if K>1
           PO_as_obs: if 0 the coordinate observations are only used as initial guess,
                      if 1 they are also used as actual observations

    :return X: final vector of variables
             it: number of Levenberg-Marquardt iterations performed
             R: residual vector at the last iteration
    """
    
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    cl_eq = None
    obs_kind = None
    if rank==0:    
        print('STARTING SOLVER')
    count_all = 0
    mu = 1.e5

    mu_max = 1.e10
    mu_min = 1.e-10
    
    rate = 0.5
    ready = False
    Mgrippo = 1

    obs_variables, obs_value, obs_kind, obs_variance = observations
    X = np.array(X)

    obs_variables_aux = []
    obs_value_aux = []
    obs_variance_aux = []
    obs_kind_aux = []

    use_EQ = 1
    if use_EQ==1:
        list_obs_kind = ['TD', 'AN', 'BE', 'PL', 'EQ', 'PO']
    else:
        list_obs_kind = ['TD', 'AN', 'BE', 'PL', 'PO']

    for k in list_obs_kind:
        obs_variables_aux.extend([eq for i, eq in enumerate(obs_variables) if obs_kind[i] == k])
        obs_value_aux.extend([eq for i, eq in enumerate(obs_value) if obs_kind[i] == k])
        obs_variance_aux.extend([eq for i, eq in enumerate(obs_variance) if obs_kind[i] == k])
        obs_kind_aux.extend([eq for i, eq in enumerate(obs_kind) if obs_kind[i] == k])


    observations = [obs_variables_aux, obs_value_aux, obs_kind_aux, obs_variance_aux]
    obs_variables, obs_value, obs_kind, obs_variance = observations


    # cl_eq[k] contains the list of observations involving only variables in subproblem k
    # eq_compl contains all the observations that involve variables in different subproblems
    # everything in the following IF statement is to create these two lists
    
        
    if K>1:
        cl_eq = [[] for k in range(K)]
        eq_compl = []
        for m in range(np.shape(obs_variables)[0]):
            eq = obs_variables[m]
            if obs_kind[m] == 'TD':
                l0 = cl_labels[int(eq[0] / 2)]
                l1 = cl_labels[int(eq[2] / 2)]
                if l0 == l1:
                    cl_eq[l0].append(m)
                else:
                    eq_compl.append([m, l0, l1])
            if obs_kind[m] == 'BE':
                l0 = cl_labels[int(eq[0] / 2)]
                l1 = cl_labels[int(eq[2] / 2)]
                if l0 == l1:
                    cl_eq[l0].append(m)
                else:
                   eq_compl.append([m, l0, l1])
            if obs_kind[m] == 'EQ':
                l0 = cl_labels[int(eq[0] / 2)]
                l1 = cl_labels[int(eq[1] / 2)]
                if l0 == l1:
                   cl_eq[l0].append(m)
                else:
                   eq_compl.append([m, l0, l1])
            if obs_kind[m] == 'AN' or obs_kind[m] == 'PL':
                l0 = cl_labels[int(eq[0] / 2)]
                l1 = cl_labels[int(eq[2] / 2)]
                l2 = cl_labels[int(eq[4] / 2)]
                if l0 == l1 and l1 == l2:
                    cl_eq[l0].append(m)
                    if obs_kind == 'AN':
                        print('accepted',eq)
                else:
                    eq_compl.append([m, l0, l1, l2])

            if obs_kind[m] == 'PO':
                l0 = cl_labels[int(eq[0] / 2)]
                cl_eq[l0].append(m)

        if rank==0:  
            print('   variables in each subproblem   ', [len(cl_variables[k]) for k in range(K)])
            print('   observations in each subproblem', [len(cl_eq[k]) for k in range(K)])
            print('   approximated observations', len(eq_compl), len(eq_compl) / len(obs_kind))
   

    # ind_XX contains the indices of observations that are of XX kind
    # we use this for vectorized evaluations of R and J
    ind_TD = [j for j, k in enumerate(obs_kind) if k == 'TD']
    ind_PL = [j for j, k in enumerate(obs_kind) if k == 'PL']
    ind_AN = [j for j, k in enumerate(obs_kind) if k == 'AN']
    ind_BE = [j for j, k in enumerate(obs_kind) if k == 'BE']
    ind_EQ = [j for j, k in enumerate(obs_kind) if k == 'EQ']
    ind_PO = [j for j, k in enumerate(obs_kind) if k == 'PO']
    ind_REG = [j for j, k in enumerate(obs_kind) if k == 'REG']

    if PO_as_obs == 0:
        ind_PO = []
    ind_kind = [ind_TD, ind_AN, ind_BE, ind_PL, ind_EQ, ind_PO, ind_REG]

    # similar to ind_XX above but relative to each subproblem
    # cl_ind_kind[k] = [cl_ind_TD, cl_ind_AN, cl_ind_BE, cl_ind_PL,cl_ind_EQ,cl_ind_PO]
    # where cl_ind_XX contains the indices of observations in subproblem k that are of XX kind
    
    cl_eq_local = cl_eq[rank]#comm.scatter(cl_eq, root=0)
    #obs_kind = comm.bcast(obs_kind, root=0)
    #PO_as_obs = comm.bcast(PO_as_obs,root=0)

    cl_ind_kind = []
    cl_ind_TD = [j for j in cl_eq_local if obs_kind[j] == 'TD']
    cl_ind_PL = [j for j in cl_eq_local if obs_kind[j] == 'PL']
    cl_ind_AN = [j for j in cl_eq_local if obs_kind[j] == 'AN']
    cl_ind_BE = [j for j in cl_eq_local if obs_kind[j] == 'BE']

    cl_ind_EQ = [j for j in cl_eq_local if obs_kind[j] == 'EQ']
    cl_ind_PO = [j for j in cl_eq_local if obs_kind[j] == 'PO' or obs_kind[j] == 'AP']
    cl_ind_AP = [j for j in cl_eq_local if obs_kind[j] == 'AP']
    if PO_as_obs == 0:
        cl_ind_PO = cl_ind_AP
    cl_ind_kind = [cl_ind_TD, cl_ind_AN, cl_ind_BE, cl_ind_PL,cl_ind_EQ,cl_ind_PO]
    
    
    cl_indices = None
    cl_ptr = None
    compl_indices = None
    compl_ptr = None
    
    # easier to have variances and values as vectors rather than lists
    obs_variance = np.array(obs_variance)
    obs_value = np.array(obs_value)
    observations = obs_variables, obs_value, obs_kind, obs_variance

    if rank==0:
        # create the necessary vectors for the definition of J as a sparse matrix
        cl_indices, cl_ptr, compl_indices, compl_ptr = split_indices(observations, K, cl_var_dic, cl_eq, eq_compl)

   
    cl_indices_local = comm.scatter(cl_indices, root=0)
    cl_ptr_local = comm.scatter(cl_ptr, root=0)
    compl_indices_local = comm.scatter(compl_indices, root=0)
    compl_ptr_local = comm.scatter(compl_ptr, root=0)

    X = np.array(X)
  
    it = 0
    if maxit==0:
        if rank==0:
            R = residual_vector_vectorized(X, observations, ind_kind, PO_as_obs)
        else:
            R = None
        return X, it, R

    innerIterNum = 5   
    for it in range(maxit):
        # in the first iteration we need to compute R, for all others, we already computed R during the linesearch
        if it == 0:
            R = residual_vector_vectorized(X, observations, ind_kind, PO_as_obs)
            norm_R = np.linalg.norm(R)
            F_0 = norm_R**2
        else:
            R = R_new
            norm_R = norm_new
        if it==0:
            grippo = np.ones(Mgrippo)*norm_R
            grippo = list(grippo)
        else:
            grippo.pop(0)
            grippo.append(norm_R)
        
        if rank==0:
            print('   computing Jacobian')
        # cl_J[k] contains the jacobian for the subproblem k
        # compl_J contains the jacobian of all the observations in eq_compl (obs between different subproblems)
      
        cl_J, compl_J = Splitted_Sparse_Jacobian_evaluation_vectorized(X, observations, cl_ind_kind, K, cl_indices_local, cl_ptr_local, eq_compl, compl_indices_local, compl_ptr_local)  
       
        res = norm_R**2 #np.linalg.norm(R)**2 # value of the objetive function at the current X
        if rank==0:
            print('OUTER ITERATION',it, ': mu =', mu, 'R^2 = %10.3e' % (res))
        inner_it = -1
     
        while True:
            inner_it+=1
            if rank==0:
                print('-----inner iteration:', inner_it)
                print('   solving the linear system')
          
            # d is the direction
            # cl_G[k] is the gradient for subproblem k
           
            d, cl_G = split_LM_step(N_pt, K, R, cl_J, compl_J, mu, cl_variables, cl_eq_local, eq_compl, innerIterNum, RHS_mat)          
                
            shouldBreak = False          
            if rank==0:    
                print('   linear system solved')
                print('   starting line_search')
           
            t = 1
            armijo = 1.e-4

            grad = np.zeros(2 * N_pt)
            for k in range(K):
                grad[cl_variables[k]] = cl_G[k]
                
            RHS = armijo*grad@grad            
            Rt = residual_vector_vectorized(X + t*d, observations, ind_kind, PO_as_obs=PO_as_obs)
            wasBacktrack = False
            a = 1
            b= 2
            eps_term = a*F_0/((it+1)**b)
          
            while t>1.e-3 and np.linalg.norm(Rt)**2>np.max(grippo)**2+t**2*RHS+eps_term:
                wasBacktrack = True
                t = t/2
                Rt = residual_vector_vectorized(X + t * d, observations, ind_kind, PO_as_obs=PO_as_obs)
            d = t*d

            X_new = X + d #new candidate point
            R_new = Rt
            norm_new = np.linalg.norm(R_new)        

            ready= False
            shouldBreak = False
            if norm_new**2<=np.max(grippo)**2+t*RHS:
                # if the linesearch terminated because the Armijo condition was satisfied
                # candidate point accepted
                it = it + 1

                if t<=0.5:
                    #if the stepsize is small we increase mu
                    mu = mu/rate
                    if rank==0:
                        print("mu updated to", mu)
                else:
                    #otherwise we decrease mu
                    mu = max(mu*rate, mu_min)
                    if rank==0:
                        print("mu updated to", mu)
                X = X_new

                acc_1 = 100*len([r for r in R_new if abs(r) < 1]) / len(R_new)
                acc_2 = 100*len([r for r in R_new if abs(r) < 2]) / len(R_new)
                acc_3 = 100*len([r for r in R_new if abs(r) < 3]) / len(R_new)
                if rank==0:
                    print('   percentages', acc_1, acc_2, acc_3)
                if acc_1>68.0 and acc_2>95.0 and acc_3>99.0:
                     if rank==0:
                        print('ACCURACY REACHED')
                     ready = True    
                break
            else:
                # if the linesearch terminated because we reached the maximum number of bactracking
                # (that is, linesearch terminated but Armijo not satisfied)
                # candidate point rejected, increase mu
                mu = mu/rate
                if rank==0:
                    print('   rejected step', mu, inner_it)
                  
            if mu>mu_max:
                # if mu_max is reached, LM terminates
                ready = True
                if rank==0:    
                    print('MAXIMUM MU REACHED')
                    
                break
            if inner_it>10:  # this is probably useless
                # if the maximum number of inner iterations is reached, increase mu
                mu = mu*2
                break
   
      
        if ready:
            if rank==0:
                print('FINAL:', it, 'mu =', mu, 'R^2 = %10.3e' % (norm_new**2))
            break

    return X, it, R


