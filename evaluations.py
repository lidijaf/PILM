import numpy as np
import scipy.sparse as sp
from mpi4py import MPI

def residual_vector_vectorized(X, observations, ind_kind, PO_as_obs=0):
    obs_variables, obs_value, obs_kind, obs_variance = observations
    ind_TD, ind_AN, ind_BE, ind_PL, ind_EQ, ind_PO, ind_REG = ind_kind

    R = []
    # TD observations
    if ind_TD:
        v = np.array([X[obs_variables[i]] for i in ind_TD])
        dx12 = v[:, 0] - v[:, 2]
        dy12 = v[:, 1] - v[:, 3]
        res = np.sqrt(dx12 ** 2 + dy12 ** 2) - obs_value[ind_TD]
        R = np.concatenate((R, res / obs_variance[ind_TD]))
    # BE observations
    if ind_BE:
        v = np.array([X[obs_variables[i]] for i in ind_BE])
        dx = v[:, 2] - v[:, 0]
        dy = v[:, 3] - v[:, 1]
        res = np.arctan2(dy, dx)*180.0/np.pi - obs_value[ind_BE]
        R = np.concatenate((R, res / obs_variance[ind_BE]))
    # AN observations
    if ind_AN:
        v = np.array([X[obs_variables[i]] for i in ind_AN])
        xik = v[:, 2] - v[:, 0]
        yik = v[:, 3] - v[:, 1]
        xjk = v[:, 4] - v[:, 0]
        yjk = v[:, 5] - v[:, 1]

        res = np.arctan2(yjk, xjk)-np.arctan2(yik, xik)
        val = obs_value[ind_AN]

        res = np.remainder(res-val, 2*np.pi)
        np.seterr(all='raise')
        try:
            res[res > np.pi] = 2*np.pi - res[res > np.pi]
        except:
            pass
        R = np.concatenate((R, res / obs_variance[ind_AN]))

    # PL observations
    if ind_PL:
        v = np.array([X[obs_variables[i]] for i in ind_PL])

        dx12 = v[:, 0] - v[:, 2]
        dx13 = v[:, 0] - v[:, 4]
        # dy32 = v[:, 5] - v[:, 3]
        dy12 = v[:, 1] - v[:, 3]
        dy13 = v[:, 1] - v[:, 5]

        res = (dx13 * dy12 - dx12 * dy13) / np.sqrt(dx13 ** 2 + dy13 ** 2)
        R = np.concatenate((R, res / obs_variance[ind_PL]))

    # EQ observations
    if ind_EQ:
        v = np.array([X[obs_variables[i]] for i in ind_EQ])
        dxy12 = v[:, 0] - v[:, 1]
        res = dxy12
        R = np.concatenate((R, res / obs_variance[ind_EQ]))
    if PO_as_obs == 1:
        v = np.array([X[obs_variables[i][0]] for i in ind_PO])
        res = v-obs_value[ind_PO]
        R = np.concatenate((R, res / obs_variance[ind_PO]))
    if ind_REG:
        v = np.array([X[obs_variables[i][0]] for i in ind_REG])
        res = v
        R = np.concatenate((R, res / obs_variance[ind_REG]))

    return R


def find_indices(X, obs_variables,ind_kind, K):
    """
    This funciton builds the vectors indices, indptr necessary for the definition of J.

    In Sparse_Jacobian_evaluation_vectorized called by iterative_methods.LM
       the Jacobian J is computed as a scipy csr matrix, whose sparsity structure remains the same at all times

    We will have J = sp.csr_matrix((data,indices,indptr)) with indices and indptr such that the nonzero elements of J
       in row i are given by data[indptr[i]:indptr[i+1]] and in positions indices[indptr[i]:indptr[i+1]]
    """
    indices = []
    indptr = []
    if K>=1:
        indices = [j for k in range(6) for i in ind_kind[k] for j in obs_variables[int(i)]]
        indptr = [0]
        for k in range(6):
            for j in ind_kind[k]:
                eq = obs_variables[int(j)]
                indptr.append(indptr[-1]+len(eq))
    return indices, indptr


def Sparse_Jacobian_evaluation_vectorized(X, observations, ind_kind, indices, indptr):
    """
    evaluates the Jacobian matrix J
    """

    obs_variables,obs_value,obs_kind,obs_variance = observations
    data = []
    ind_TD, ind_AN, ind_BE, ind_PL, ind_EQ, ind_PO, ind_REG = ind_kind
    # TD
    if ind_TD:
        v = np.array([X[obs_variables[i]] for i in ind_TD])
        J_loc = TD_jac_eval_vectorized(v, obs_variance[ind_TD])
        data = np.concatenate((data,J_loc))

    # BE
    if ind_BE:
        v = np.array([X[obs_variables[i]] for i in ind_BE])
        J_loc = BE_jac_eval_vectorized(v, obs_variance[ind_BE])
        data = np.concatenate((data, J_loc))

    # AN
    if ind_AN:
        v = np.array([X[obs_variables[i]] for i in ind_AN])
        J_loc = AN_jac_eval_vectorized(v, obs_variance[ind_AN], obs_value[ind_AN])
        data = np.concatenate((data, J_loc))

    # PL
    if ind_PL:
        v = np.array([X[obs_variables[i]] for i in ind_PL])
        J_loc = PL_jac_eval_vectorized(v, obs_variance[ind_PL])
        data = np.concatenate((data, J_loc))

    # EQ
    if ind_EQ:
        res = [1,-1]*len(ind_EQ)
        res = res / obs_variance[ind_EQ[0]]

        data = np.concatenate((data, np.ravel(res)))

    # PO
    res = [1] * len(ind_PO)
    data = np.concatenate((data, res / obs_variance[ind_PO]))

    res = [1] * len(ind_REG)
    data = np.concatenate((data, res / obs_variance[ind_REG]))
    J = sp.csr_matrix((data,indices,indptr))

    return J


def Splitted_Sparse_Jacobian_evaluation_vectorized(X, observations,cl_ind_kind, K, cl_indices, cl_ptr, eq_compl, compl_indices, compl_ptr):
    """
    evaluates the Jacobian matrix taking into account the slitting
    cl_J[k] contains the Jacobian of the observations in subproblem k wrt the variables in subproblem k
    compl_J[k] contains the Jacobian of the observations in eq_compl wrt the variables in subproblem k
    """
    comm = MPI.COMM_WORLD


    obs_variables,obs_value,obs_kind,obs_variance = observations
    cl_data = [] 
    cl_J = [] 
    compl_J = [] 
    compl_data = [] 

    rank = comm.Get_rank()

    # TD
    if cl_ind_kind[0]:
        v = np.array([X[obs_variables[i]] for i in cl_ind_kind[0]])
        J_loc = TD_jac_eval_vectorized(v, obs_variance[cl_ind_kind[0]])
        cl_data = np.concatenate((cl_data, J_loc))
    # BE
    if cl_ind_kind[2]:
        v = np.array([X[obs_variables[i]] for i in cl_ind_kind[2]])
        J_loc = BE_jac_eval_vectorized(v, obs_variance[cl_ind_kind[2]])
        cl_data = np.concatenate((cl_data, J_loc))
    # AN
    if cl_ind_kind[1]:
        v = np.array([X[obs_variables[i]] for i in cl_ind_kind[1]])
        J_loc = AN_jac_eval_vectorized(v, obs_variance[cl_ind_kind[1]], obs_value[cl_ind_kind[1]])
        cl_data = np.concatenate((cl_data, J_loc))
    # PL
    if cl_ind_kind[3]:
        v = np.array([X[obs_variables[i]] for i in cl_ind_kind[3]])
        J_loc = PL_jac_eval_vectorized(v, obs_variance[cl_ind_kind[3]])
        cl_data = np.concatenate((cl_data, J_loc))
    # EQ
    if cl_ind_kind[4]:
        res = [1, -1] * len(cl_ind_kind[4])
        res = res / obs_variance[cl_ind_kind[4][0]]
        cl_data = np.concatenate((cl_data, np.ravel(res)  ))
    # PO
    if cl_ind_kind[5]:
        res = [1] * len(cl_ind_kind[5])
        cl_data = np.concatenate((cl_data, res / obs_variance[cl_ind_kind[5]]))

    cl_J = sp.csr_matrix((cl_data,cl_indices,cl_ptr))

    compl_data_glob = [[] for k in range(K)]
    
    for obs in eq_compl:
            i = obs[0]
            eq = obs_variables[i]
            if obs_kind[i] == 'TD':
                J_loc = TD_jac_eval(X, eq)
            if obs_kind[i] == 'BE':
                J_loc = BE_jac_eval(X, eq)
            if obs_kind[i] == 'AN':
                J_loc = AN_jac_eval(X, eq, obs_value[i])
            if obs_kind[i] == 'PL':
                J_loc = PL_jac_eval(X, eq)
            if obs_kind[i] == 'EQ':
                J_loc = np.array([1, -1])
            if obs_kind[i] == 'PO':
                J_loc = np.array([1])
            if len(obs) == 3:
                j,k0,k1 = obs
                aux = 0
                if observations[-2][j] == 'TD':
                    if k0==rank:
                        compl_data_glob[k0].append(J_loc[aux] / obs_variance[j])
                        aux += 1
                        compl_data_glob[k0].append(J_loc[aux] / obs_variance[j])
                        aux += 1
                    if observations[-2][j] == 'TD' and k1==rank:
                        aux = 2
                        compl_data_glob[k1].append(J_loc[aux] / obs_variance[j])
                        aux += 1
                        compl_data_glob[k1].append(J_loc[aux] / obs_variance[j])
                        aux += 1
                elif observations[-2][j] == 'BE':
                    if k0==rank:
                        compl_data_glob[k0].append(J_loc[aux] / obs_variance[j])
                        aux += 1
                        compl_data_glob[k0].append(J_loc[aux] / obs_variance[j])
                        aux += 1
                    if k1==rank:
                        aux = 2
                        compl_data_glob[k1].append(J_loc[aux] / obs_variance[j])
                        aux += 1
                        compl_data_glob[k1].append(J_loc[aux] / obs_variance[j])
                        aux += 1
                elif observations[-2][j] == 'EQ':
                    if k0==rank:
                        aux = 0
                        compl_data_glob[k0].append(J_loc[aux] / obs_variance[j])
                        aux += 1
                    if k1==rank:
                        aux = 1
                        compl_data_glob[k1].append(J_loc[aux] / obs_variance[j])
                        aux += 1
            else:
                j,k0,k1,k2 = obs
                aux = 0
                if k0==rank:
                    compl_data_glob[k0].append(J_loc[aux]/obs_variance[j])
                    aux+=1
                    compl_data_glob[k0].append(J_loc[aux]/obs_variance[j])
                    aux+=1
                if k1==rank:
                    aux=2
                    compl_data_glob[k1].append(J_loc[aux] / obs_variance[j])
                    aux += 1
                    compl_data_glob[k1].append(J_loc[aux] / obs_variance[j])
                    aux += 1
                if k2==rank:
                    aux = 4
                    compl_data_glob[k2].append(J_loc[aux] / obs_variance[j])
                    aux += 1
                    compl_data_glob[k2].append(J_loc[aux] / obs_variance[j])
                    aux += 1
    compl_data_loc = compl_data_glob[rank]#comm.scatter(compl_data_glob, root=0)
    compl_J = sp.csr_matrix((compl_data_loc, compl_indices, compl_ptr), shape=(np.shape(compl_ptr)[0]-1, np.shape(cl_J)[1]))

    return cl_J, compl_J


def PL_jac_eval(X, eq):
    J = np.zeros(6)
    v = [X[eq[0]],X[eq[1]],X[eq[2]],X[eq[3]],X[eq[4]],X[eq[5]]]
    dx32 = v[4] - v[2]
    dx12 = v[0] - v[2]
    dx13 = v[0] - v[4]
    dy32 = v[5] - v[3]
    dy12 = v[1] - v[3]
    dy13 = v[1] - v[5]

    nrm_13 = dx13 ** 2 + dy13 ** 2
    nrm_13_15 = nrm_13 ** 1.5
    tmp = dy13 * (dx32 * dx13 + dy13 * dy32)
    J[0] = tmp / nrm_13_15
    J[2] = dy13 / np.sqrt(nrm_13)
    tmp = -dy13 * (dx12 * dx13 + dy12 * dy13)
    J[4] = tmp / nrm_13_15

    tmp = -dx13 * (dy32 * dy13 + dx13 * dx32)
    J[1] = tmp / nrm_13_15
    J[3] = -dx13 / np.sqrt(nrm_13)
    tmp = dx13 * (dy12 * dy13 + dx12 * dx13)
    J[5] = tmp / nrm_13_15
    J_aux = []
    for j, eqj in enumerate(eq):
        J_aux.append(J[j])

    return np.array(J_aux)


def PL_jac_eval_vectorized(v, stdev):
    nr_obs = v.shape[0]
    vars_involved = 6
    res = np.zeros(shape=(nr_obs, vars_involved))

    x1, x2, x3 = v[:, 0], v[:, 2], v[:, 4]
    y1, y2, y3 = v[:, 1], v[:, 3], v[:, 5]

    dx32 = x3 - x2
    dx12 = x1 - x2
    dx13 = x1 - x3
    dy32 = y3 - y2
    dy12 = y1 - y2
    dy13 = y1 - y3

    nrm_13 = (dx13 ** 2 + dy13 ** 2)
    nrm_13_15 = nrm_13 ** (1.5)

    tmp = dy13 * (dx32 * dx13 + dy13 * dy32)
    res[:, 0] = tmp / nrm_13_15
    res[:, 2] = dy13 / np.sqrt(nrm_13)
    tmp = -dy13 * (dx12 * dx13 + dy12 * dy13)
    res[:, 4] = tmp / nrm_13_15

    tmp = -dx13 * (dy32 * dy13 + dx13 * dx32)
    res[:, 1] = tmp / nrm_13_15
    res[:, 3] = -dx13 / np.sqrt(nrm_13)
    tmp = dx13 * (dy12 * dy13 + dx12 * dx13)
    res[:, 5] = tmp / nrm_13_15

    res = res / np.expand_dims(stdev, axis=1)

    return np.ravel(res)


def TD_jac_eval(X, eq):
    xi = X[eq[0]]
    yi = X[eq[1]]
    xj = X[eq[2]]
    yj = X[eq[3]]
    dx = xi - xj
    dy = yi - yj
    N = np.sqrt(dx ** 2 + dy ** 2)
    # R = N - eq[4]
    J = np.zeros(4)
    J[0] = dx / N
    J[1] = dy / N
    J[2:] = -J[:2]
    J_aux = []
    for j, eqj in enumerate(eq):
        J_aux.append(J[j])

    return np.array(J_aux)


def TD_jac_eval_vectorized(v, stdev):

    nr_obs = v.shape[0]
    vars_involved = 4

    res = np.zeros(shape=(nr_obs, vars_involved))

    dx12 = v[:, 0] - v[:, 2]
    dy12 = v[:, 1] - v[:, 3]

    denominator = np.sqrt(dx12 ** 2 + dy12 ** 2)
    
    res[:, 0] = dx12 / denominator
    res[:, 2] = - res[:, 0]
    res[:, 1] = dy12 / denominator
    res[:, 3] = - res[:, 1]
    res = res / np.expand_dims(stdev, axis=1)

    return np.ravel(res)


def AN_jac_eval(X,eq, val):
    v = X[eq]
    res = np.zeros(6)

    xik = v[2] - v[0]
    yik = v[3] - v[1]
    xjk = v[4] - v[0]
    yjk = v[5] - v[1]
    Ni = (xik ** 2 + yik ** 2)
    Nj = (xjk ** 2 + yjk ** 2)

    res[2] = yik / Ni
    res[4] = -yjk / Nj
    res[0] = yjk / Nj - yik / Ni
    res[3] = -xik / Ni
    res[5] = xjk / Nj
    res[1] = -xjk / Nj + xik / Ni

    res1 = np.arctan2(yjk, xjk) - np.arctan2(yik, xik)
    res1 = np.remainder(res1 - val, 2 * np.pi)
    np.seterr(all='raise')
    try:
        res[res1 > np.pi, :] = - res[res1 > np.pi, :]
    except:
        pass

    return res


def BE_jac_eval(X,eq):
    v = X[eq]
    res = np.zeros(4)

    dx = v[2] - v[0]
    dy = v[3] - v[1]

    Ni = (dx ** 2 + dy ** 2)

    res[2] = dy / Ni
    res[0] = -dy / Ni
    res[3] = -dx / Ni
    res[1] = dx / Ni

    return res


def BE_jac_eval_vectorized(v, stdev):
    nr_obs = v.shape[0]
    vars_involved = 4

    res = np.zeros(shape=(nr_obs, vars_involved))

    dx = v[:, 2] - v[:, 0]
    dy = v[:, 3] - v[:, 1]

    Ni = np.sqrt(dx ** 2 + dy ** 2)

    res[:, 2] = dy / Ni
    res[:, 0] = - dy / Ni
    res[:, 3] = -dx / Ni
    res[:, 1] = dx / Ni

    return np.ravel(res)/ np.expand_dims(stdev, axis=1)



def AN_jac_eval_vectorized(v, stdev, val):
    nr_obs = v.shape[0]
    vars_involved = 6
    res = np.zeros(shape=(nr_obs, vars_involved))

    xik = v[:, 2] - v[:, 0]
    yik = v[:, 3] - v[:, 1]
    xjk = v[:, 4] - v[:, 0]
    yjk = v[:, 5] - v[:, 1]
    Ni = np.sqrt(xik**2+yik**2)
    Nj = np.sqrt(xjk**2+yjk**2)

    res[:, 2] = yik/Ni
    res[:, 4] = -yjk/Nj
    res[:, 0] = yjk/Nj-yik/Ni
    res[:, 3] = -xik/Ni
    res[:, 5] = xjk/Nj
    res[:, 1] = -xjk/Nj+xik/Ni

    res1 = np.arctan2(yjk, xjk) - np.arctan2(yik, xik)
    res1 = np.remainder(res1 - val, 2 * np.pi)
    np.seterr(all='raise')
    try:
        res[res1 > np.pi,:] = - res[res1 > np.pi,:]
    except:
        pass

    res = res / np.expand_dims(stdev, axis=1)

    return np.ravel(res)
