import numpy as np

import utils

def init_experiment(func_name, d, seed=2, **init_args):
    args = {}
    args['func_name'] = func_name
    if func_name == "Reg":
#        np.random.seed(seed)
#        L = init_args['L']
#        mu = init_args['mu']
#        args['A'] = utils.generate_matrix(d, mu, L, seed)
#        args['b'] = np.random.random(d)
        args['A'] = np.asarray([[1.0]])
        args['b'] = np.asarray([[10.0]])
    else:
        pass
    
    return args
