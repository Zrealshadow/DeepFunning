'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-09-25 14:33:38
 * @desc 
'''

import torch


'''--------------------- Weighted Binary cross Entropy ----------------------'''

'''
In Torch BCELoss, weight is set to every element of input instead of to every class
'''
def weighted_binary_cross_entropy(output, target, weights=None):
    if weights is not None:
        assert len(weights) == 2
        
        loss = weights[1] * (target * torch.log(output)) + \
            weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))


''' ---------------------- Binary focal loss function -------------------------- '''

'''
In some degree, it can reduce the influence of imbalanced dataset
'''

def focal_loss(y_true,y_pred,device):
    alpha,gamma = torch.tensor(0.25).to(device) , torch.tensor(2.0).to(device)
    y_pred=torch.clamp(y_pred,1e-7,1-1e-7)
    return - alpha * y_true * torch.log(y_pred) * (1 - y_pred) ** gamma\
        - (1 - alpha) * (1 - y_true) * torch.log(1 -  y_pred) * y_pred

