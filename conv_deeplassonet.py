"""
implementation of LassoNet where the hierarchical penalty is applied to the convolutional filters.  

some snippets from: https://medium.com/dataseries/visualizing-the-feature-maps-and-filters-by-convolutional-neural-networks-e1462340518e
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Callable

from sklearn.metrics import accuracy_score, roc_auc_score

def hier_prox(v: torch.Tensor, u: torch.Tensor, lambda_: float, lambda_bar: float, M: float):
    """
    Copied (with minor adaptions) from:
        Louis Abraham, Ismael Lemhadri: https://github.com/lasso-net/lassonet/blob/master/lassonet/prox.py
    
    
    v has shape (k,) or (k, d)
    u has shape (K,) or (K, d)
    
    standard case described in the paper: v has size (1,d), u has size (K,d)
    
    """
    onedim = len(v.shape) == 1
    if onedim:
        v = v.unsqueeze(-1)
        u = u.unsqueeze(-1)

    u_abs_sorted = torch.sort(u.abs(), dim=0, descending=True).values
    k, d = u.shape
    s = torch.arange(k + 1.0).view(-1, 1).to(v)
    zeros = torch.zeros(1, d).to(u)

    a_s = lambda_ - M * torch.cat(
        [zeros, torch.cumsum(u_abs_sorted - lambda_bar, dim=0)]
    )

    norm_v = torch.norm(v, p=2, dim=0)
    x = F.relu(1 - a_s / norm_v) / (1 + s * M ** 2)
    w = M * x * norm_v
    intervals = soft_threshold(lambda_bar, u_abs_sorted)
    lower = torch.cat([intervals, zeros])

    idx = torch.sum(lower > w, dim=0).unsqueeze(0)

    x_star = torch.gather(x, 0, idx).view(1, d)
    w_star = torch.gather(w, 0, idx).view(1, d)
    
    beta_star = x_star * v
    theta_star = sign_binary(u) * torch.min(soft_threshold(lambda_bar, u.abs()), w_star)

    if onedim:
        beta_star.squeeze_(-1)
        theta_star.squeeze_(-1)

    return beta_star, theta_star

def metrics(y, y_hat, final: str="softmax"):
    if (torch.any(torch.isnan(y)) or not torch.all(torch.isfinite(y))):
        print('nan or inf', y)
        
    if (torch.any(torch.isnan(y_hat)) or not torch.all(torch.isfinite(y_hat))):
        print('nan or inf hat', y_hat)

    if final == 'sigmoid':
        y_pred = y_hat > 0.5
        auroc = roc_auc_score(y, y_hat, multi_class='ovr')
        acc = accuracy_score(y, y_pred)
    else:
        y_pred = np.argmax(y_hat, axis=1)
        y_true = np.argmax(y, axis=1)
        if y_true.max() > 1:
            auroc = roc_auc_score(y, y_hat, multi_class='ovr')
        else:
            auroc = roc_auc_score(y, y_hat)
        
        acc = accuracy_score(y_true, y_pred)

    return auroc, acc

# from: https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/6
def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    if type(h_w) is not tuple:
        h_w = (h_w, h_w)
    
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    
    if type(stride) is not tuple:
        stride = (stride, stride)
    
    if type(pad) is not tuple:
        pad = (pad, pad)
    
    h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1)// stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1)// stride[1] + 1
    
    return h, w#, w

class ConvLassoNet(nn.Module):
    def __init__(self, 
                 lambda_=1., 
                 M=1., 
                 D_in=(28,28), 
                 D_out=10, 
                 out_channels1=16, 
                 out_channels2=32, 
                 out_channels3=64,
                 out_channels4=64,
                 stride=1,
                 padding=2,
                 dilation=1,
                 final: str="sigmoid",
                 kernel_size: int=5,
                 sparse_layer: str="conv1"):
        """
        LassoNet applied after a first layer of convolutions. See https://jmlr.org/papers/volume22/20-848/20-848.pdf for details.

        Parameters
        ----------
        lambda_ : float, optional
            Penalty parameter for the skip layer. The default is 1.
            By setting to ``None``, the LassoNet penalty is deactivated.
        M : float, optional
            Penalty parameter for the hierarchical constraint. The default is 1.
        D_in : int, optional
            input dimension of the model. The default is 784.
        D_out : int, optional
            output dimension of the model. The default is 10.

        Returns
        -------
        None.

        """
        
        super(ConvLassoNet, self).__init__()
        
        # LassoNet penalty
        self.lambda_ = lambda_
        self.M = M
        if sum(D_in) < 28 * 3: #only for MedMNIST
            conv = nn.Conv2d
        else:
            conv = nn.Conv3d
        
        assert (self.lambda_ is None) or (self.lambda_ >= 0), "lambda_ must be None or positive"
        assert self.M > 0, "M needs to be positive (possibly np.inf)"
        
        # hyperparameters (works as long as input size can be divided by 4)
        self.kernel_size1 = kernel_size
        self.kernel_size2 = kernel_size
        self.out_channels1 = out_channels1
        self.out_channels2 = out_channels2
        self.out_channels3 = out_channels3
        self.out_channels4 = out_channels4
        
        self.D_in = D_in
        self.D_out = D_out
        self.sparse_layer = sparse_layer
        self.conv1_output_dim(stride, padding, dilation)
        self.conv2_output_dim(dilation)
        self.conv3_output_dim(dilation)
        self.conv4_output_dim(dilation)

        # first conv layer and skip layer
        # input pixels nxn, filter size fxf, padding p: output size (n + 2p — f + 1)
        self.conv1 = conv(1, self.out_channels1, kernel_size=self.kernel_size1, stride=stride, padding=padding, dilation=dilation)
        print('conv1', self.conv1.weight.shape)
        # remaining nonlinear part (after conv1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = conv(self.out_channels1, self.out_channels2, kernel_size=self.kernel_size2, stride=1, padding=2)
        self.conv3 = conv(self.out_channels2, self.out_channels3, kernel_size=self.kernel_size2, stride=1, padding=2)
        self.conv4 = conv(self.out_channels3, self.out_channels4, kernel_size=self.kernel_size2, stride=1, padding=2)
        self.drop_out = nn.Dropout()
        # downsampling twice by factor 2 --> 7x7 output, 32 channels
        # self.fc1 = nn.Linear(int((self.D_in[0]*self.D_in[1])/(4*4)) *self.out_channels2, 200)
        self.fc1 = nn.Linear((self.h_out4 // 2)* (self.w_out4 // 2) * self.out_channels4, 200)
        self.fc2 = nn.Linear(200, self.D_out)
        self.features = nn.Sequential(self.conv1, 
                                      self.relu, 
                                      self.maxpool, 
                                      self.conv2, 
                                      self.relu, 
                                      self.maxpool,
                                      self.conv3, 
                                      self.relu,
                                      self.maxpool,
                                      self.conv4,
                                      self.relu,
                                      self.maxpool)

        if final == 'sigmoid':
            self.final = nn.Sigmoid()
        else:
            self.final = nn.Softmax(dim=1)
        
        self.apply(self.init_params)
        return

    def init_params(self, m: torch.nn.Module):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=.1)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0.)

    def conv1_output_dim(self, stride, padding, dilation):
        """ computes output dimension of conv1; needed for dimensionality of skip layer
        """
        self.h_out, self.w_out = conv_output_shape(self.D_in, self.kernel_size1, stride, padding, dilation)
        
        return
    
    def conv2_output_dim(self, dilation):
        """ computes output dimension of conv1; needed for dimensionality of skip layer
        """
        self.h_out2, self.w_out2 = conv_output_shape(self.h_out//2, self.kernel_size2, 1, 2, dilation)
        
        return
    
    def conv3_output_dim(self, dilation):
        """ computes output dimension of conv1; needed for dimensionality of skip layer
        """
        self.h_out3, self.w_out4 = conv_output_shape(self.h_out2//2, self.kernel_size2, 1, 2, dilation)
        
        return
    
    def conv4_output_dim(self, dilation):
        """ computes output dimension of conv1; needed for dimensionality of skip layer
        """
        self.h_out4, self.w_out4 = conv_output_shape(self.h_out3//2, self.kernel_size2, 1, 2, dilation)
        
        return

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.relu(out) # wasn't defined before 220516
        z2 = self.fc2(out)
        return self.final(z2)
        

    def prox(self, lr):
        """
        New group-wise L1 penalty function: Soft threshold
        """
        for j in range(getattr(self, self.sparse_layer).out_channels):
            w_j = getattr(self, self.sparse_layer).weight.data[j,:,:,:]
            threshold = torch.clamp(torch.linalg.norm(w_j) - self.lambda_ * lr - self.M, min=0.)
            getattr(self, self.sparse_layer).weight.data[j,:,:,:] = w_j / torch.linalg.norm(w_j) * threshold

        return

    def show_norm(self):
        """
        Show 2-norm of filter weights, channel-wise
        """
        for j in range(self.out_channels1):
            w_j = self.conv1.weight[j,:,:,:]
            print(f"{j}'th channel 2-norm:", torch.linalg.norm(w_j))
            print(f"{j}'th channel 2-norm:", torch.clamp(torch.linalg.norm(w_j) - self.lambda_, min=0.))

        return

    def train_epoch(self, loss: torch.nn.Module, dl: DataLoader, opt: torch.optim.Optimizer=None, device='cpu'):
        """
        Trains one epoch.

        Parameters
        ----------
        loss : ``torch.nn.Module`` loss function
            Loss function for the model.
        dl : ``torch.utils.data.DataLoader``
            DataLoader with the training data.
        opt : from ``torch.optim``, optional
            Pytorch optimizer. The default is SGD with Nesterov momentum and learning rate 0.001.
        
        Returns
        -------
        info : dict
            Training loss and accuracy.

        """
        if opt is None:
            opt = torch.optim.SGD(self.parameters(), lr = 1e-3, momentum = 0.9, nesterov = True)
        
        info = {'train_loss':[],'train_acc':[], 'train_auroc': []}
        
        ################### START OF EPOCH ###################
        self.train()
        for inputs, targets in dl:
            inputs = inputs.to(device)
            targets = targets#.to(device)
            # forward pass
            y_pred = self.forward(inputs).cpu()
            print(y_pred.max(dim=1), y_pred.min(dim=1), y_pred.shape, 'targets:', targets.shape)
            # compute loss
            loss_val = loss(y_pred, targets)        
            # zero gradients
            opt.zero_grad()
            # backward pass
            loss_val.backward()
            # iteration
            opt.step()
            # step size
            alpha = opt.state_dict()['param_groups'][0]['lr']

            # threshold step
            self.prox(alpha)

            # self.show_norm()
            
            ## COMPUTE ACCURACY AND STORE 
            auroc, acc = metrics(targets, y_pred.data)
            
            # accuracy = torch.mean(accuracy)
            info['train_loss'].append(loss_val.item())
            info['train_acc'].append(acc)
            info['train_auroc'].append(auroc)
            
        return info
        



