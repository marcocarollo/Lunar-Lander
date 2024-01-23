import numpy as np
import scipy
def discretize_state(state, observation_space):
    a, b, c, d, e, f, _, _ = observation_space
    discrete_state = (min(a//2, max(-a//2, int((state[0]) / 0.15))), \
                        min(b-2, max(-1, int((state[1]) / 0.2))), \
                        min(c//2, max(-c//2, int((state[2]) / 0.2))), \
                        min(d//2, max(-d//2, int((state[3]) / 0.2))), \
                        min(e//2, max(-e//2, int((state[4]) / 0.2))), \
                        min(f//2, max(-f//2, int((state[5]) / 0.2))), \
                        int(state[6]), \
                        int(state[7]))

    return discrete_state
def naive_discretize_state(state):
    '''
    Returns the indices of the bin of state, given the binning described by _binning_ and _Nbinning_.
    Binning denotes the ranges [min, max] in each dimension.
    Nbinning denotes the number of bins in each dimension.
    Values below (above) the min (max) range are put into the first (last) bin by default.
    ''' 
    Nbins = [11,5,5,5,5,5,2,2]
    bins = np.array([[-1.5, 1.5], [-0., 1.5], [-5., 5.] , [-5., 5.], [-3.1415927, 3.1415927], [-5., 5.], [-0., 1.], [-0., 1.]])

    #arrays of integers
    discrete_state = np.zeros(len(state), dtype=int)
    
    for i,s in enumerate(state):
        if s <= bins[i,0]:
            discrete_state[i] = 0
        elif s >= bins[i,1]:
            discrete_state[i] = Nbins[i]-1
        else :
            discrete_state[i] = np.floor((s-bins[i,0])/(bins[i,1]-bins[i,0])*Nbins[i])
    return tuple(discrete_state)

def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

#FIX EVERYTHING WITH SELF, PUT THAT ON ARGUMENTS, CFR VERSION DEL TIPO
def get_action_epsilon_greedy(Qvalues, s, eps, action_size):
        """
        Chooses action at random using an epsilon-greedy policy wrt the current Q(s,a).
        """
        ran = np.random.rand()

        if ran < eps:
            prob_actions = np.ones(action_size) / action_size

        else:
            best_value = np.max(Qvalues[(*s,)])
            #in case more than one action has the best value
            best_actions = Qvalues[(*s,)] == best_value

            prob_actions = best_actions / np.sum(best_actions)

        a = np.random.choice(action_size, p=prob_actions)
        return a


def nested_list(dim):
    """
        Creates a nested list of dimension dim. Can be seen as an len(dim)-dimensional matrix.
    Parameters
    ----------
    dim : list
        List of dimensions of the array.
    Returns
    -------
    len(dim)-dimensional matrix where each element is an empty list.
    """
    if len(dim) == 1:
        return [[] for i in range(dim[0])]
    else:
        return [nested_list(dim[1:]) for i in range(dim[0])]
    
def append_value_nested(nested_list, index, value):
    """
        append a value at a nested list given an index
    Parameters
    ----------
    nested_list : list
        Nested list.
    index : list
        List of indices.
    value : any
        Value to append.
    Returns
    -------
    Value of the nested list at the given index.
    """
    if len(index) == 1:
        nested_list[index[0]].append(value)
    else:
        append_value_nested(nested_list[index[0]], index[1:], value)

def get_value_nested(nested_list, index):
    """
        get a value at a nested list given an index
    Parameters
    ----------
    nested_list : list
        Nested list.
    index : list
        List of indices.
    Returns
    -------
    Value of the nested list at the given index. (if the value is a list, it returns the list)
    """
    if len(index) == 1:
        return nested_list[index[0]]
    else:
        return get_value_nested(nested_list[index[0]], index[1:])
    