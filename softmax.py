import torch
import torch.nn as nn
import numpy as np

def cross_entropy(actual, predicted):
    loss = np.sum(actual * np.log(predicted))
    return loss


Y = np.array([1,0,0])

Y_pred_good = np.array([0.7,.2,.1])
Y_pred_bad =  np.array([0.1,.3,.6])

l1 = cross_entropy(Y,Y_pred_good)
l2 = cross_entropy(Y,Y_pred_bad)

print(f'Loss1 numpy: {l1:.4f}')
print(f'Loss2 numpy: {l2:.4f}')
