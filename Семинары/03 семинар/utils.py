import numpy as np
from matplotlib import pylab as plt

def generate_matrix(d, mu, L, s):
    
    # Your code
    
    return matrix

def Reg_grad(w, A, b):
    return A @ w - b

def make_err_plot(optimizers_list, labels, title, markers=None, colors=None):
    
    markers = ['o', 'v', 's', 'p', 'x', 'P', 'D', '^', '<', '>']
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'black', 'olive', 'pink', 'brown']

    x_label = "Количество итераций"
    
    if optimizers_list[0].err == "grad":
        y_label = r"$||\nabla f(x^k)||$"
    else:
        pass
        
    plt.title(title, fontsize=25)
    plt.xlabel(x_label, fontsize=25)
    plt.ylabel(y_label, fontsize=25)
    
    flag = False
    first = optimizers_list[0].name
    for optimizer in optimizers_list:
        if first != optimizer.name:
            flag = True
            break

    for optimizer, label, color, marker in zip(optimizers_list, labels, colors, markers):
        error = optimizer.errors_list
        if flag:
            plt.semilogy(np.array(range(len(error))), error, color=color, label=optimizer.name, markevery=0.05, marker=marker)
        else:
            plt.semilogy(np.array(range(len(error))), error, color=color, label='gamma = ' + label, markevery=0.05, marker=marker)

    plt.legend(fontsize=17)
    plt.grid(True)
        
    plt.show()
