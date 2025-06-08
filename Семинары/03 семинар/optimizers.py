import numpy as np
import tqdm
from importlib import reload

import utils
reload(utils)

class Optimizer:
    def __init__(self, args, x_0, max_iter=10**4, eps=0.000001, seed=18, err="grad", **params):
        np.random.seed(seed)
        self.learning_rate_k = params["gamma"]
        self.x_curr = np.copy(x_0)
        self.err = err
        self.args = args
        
        if self.args['func_name'] == "Reg":
            self.A = args['A']
            self.b = args['b']
        else:
            pass

        self.R0 = self.get_error(x_0)
        self.max_iter = max_iter
        self.eps = eps

    def step(self, x, k):
    
        pass

        return x_next
    
    def get_error(self, x):
        if self.err == "grad":
            if self.args['func_name'] == "Reg":
                error =  np.linalg.norm(utils.Reg_grad(x, A=self.A, b=self.b))
            else:
                pass
        else:
            pass
        return error
    
    def optimize(self):
        
        self.errors_list = [1.]
        
        for k in tqdm.trange(self.max_iter):
            self.x_curr = self.step(self.x_curr, k)
            
            error = self.get_error(self.x_curr) / self.R0
            self.errors_list.append(error)
            
            if error <= self.eps:
                print(f"Точность {self.eps} достигнута на шаге {k}!")
                break

class GDOptimizer(Optimizer):
    def __init__(self, args, x_0, max_iter=10**4, eps=0.000001, seed=18, err="grad", **params):
        super().__init__(args, x_0, max_iter, eps, seed, err, **params)
        self.name = "GD"
    
    def step(self, x, k):
    
        # Your code

        return x_next
