import pandas as pd
import numpy as np
import os
from copy import deepcopy

import torch

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torch.optim.lr_scheduler import StepLR

from conv_lassonet import ConvLassoNet, metrics

from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms

batch_size = 2 ** 14

loss = torch.nn.BCELoss()

l1 = 1e1
M = 1e0
v_ = 0.5

n_epochs = 25
alpha0 = 1e-2
momentum = 0.95


if torch.cuda.is_available():     # Make sure GPU is available
    dev = torch.device("cuda:0")
    kwar = {'num_workers': 6, 'pin_memory': True}
    cpu = torch.device("cpu")
else:
    print("Warning: CUDA not found, CPU only.")
    dev = torch.device("cpu")
    kwar = {}
    cpu = torch.device("cpu")

print(dev)

np.random.seed(551)

# convert data to torch.FloatTensor (including scaling to [0,1])
transform = transforms.ToTensor()

class FashionMNISTDataset(FashionMNIST):
    def __init__(self, train: bool=True):
        super().__init__("/cluster/projects/radiomics/Temp/sejin/LassoNet", train, download=True)
    
    def __getitem__(self, index: int):
        image, label = super().__getitem__(index)

        image = transform(image)
        labels = torch.zeros(10)
        labels[label] = 1.

        return image, labels


def train_model(model, opt, loss, lr_schedule=None, n_epochs = 5, pretrained=None, device='cpu'):
    if pretrained is not None:
        model.load_state_dict(pretrained.state_dict())
    
    loss_hist = {'train_loss':[], 'valid_loss':[], 'train_acc':[], 'valid_acc':[], 'train_auroc':[], 'valid_auroc':[]}
    
    for j in range(n_epochs): 
        print(f"================== Epoch {j+1}/{n_epochs} ================== ")
        print(opt)  
        
        ### TRAINING
        epoch_info = model.train_epoch(loss, train_loader, opt=opt, device=device)
        loss_hist['train_loss'].append(np.mean(epoch_info['train_loss']))
        loss_hist['train_acc'].append(np.mean(epoch_info['train_acc']))
        loss_hist['train_auroc'].append(np.mean(epoch_info['train_auroc']))
        
        if lr_schedule is not None:
            lr_schedule.step()
        
        ### VALIDATION
        valid_loss, valid_acc = 0, 0
        info = {'val_loss':[],'val_acc':[], 'val_auroc': []}
        model.eval()
        for inputs, targets in test_loader:
            output = model.forward(inputs.to(device)).cpu()
            auroc, acc = metrics(targets, output.detach())
            info['val_loss'].append(loss(output, targets).item())
            info['val_acc'].append(acc)
            info['val_auroc'].append(auroc)
        
        loss_hist['valid_loss'].append(np.mean(info['val_loss']))
        loss_hist['valid_acc'].append(np.mean(info['val_acc']))
        loss_hist['valid_auroc'].append(np.mean(info['val_auroc']))
                 
        print(f"\t  train loss: {loss_hist['train_loss']}.")
        print(f"\t  validation loss: {loss_hist['valid_loss']}.")    

    return loss_hist

def plot_skip(model, label, cmap = plt.cm.cividis, vmin = None, vmax = None):
    skip_weight = model.skip.weight.data.numpy().copy()
    T = skip_weight[label,:]
        
    for j in np.arange(model.out_channels1):
        ax = axs.ravel()[j]
        ax.imshow(T[model.h_out*model.w_out*j:model.h_out*model.w_out*(j+1)].reshape(model.h_out,model.w_out),\
                  cmap = cmap, vmin = vmin, vmax = vmax)
        ax.axis('off')
        ax.set_title(f'filter {j}', fontsize = 8)

    return

def plot_filter_norm(mod, ax, color, layer: str='conv1'):
    X = getattr(mod, layer).weight.data.numpy()
    infnorm_ = np.linalg.norm(X, axis = (2,3)).T
    ax.plot(infnorm_, lw = 0.5, alpha = 0.2, c = color, marker = 'o')    

    return 

def plot_filter1(model, cmap = plt.cm.cividis, vmin = None, vmax = None):
    conv_filter = model.conv1.weight.data
    
    for j in np.arange(conv_filter.size(0)):
        ax = axs.ravel()[j]
        ax.imshow(torch.mean(conv_filter[j,:,:,:], axis=0), cmap = cmap, vmin = vmin, vmax = vmax) 
        ax.axis('off')
        ax.set_title(f'filter {j}', fontsize = 8)

    return 

def plot_filter2(model, cmap = plt.cm.cividis, vmin = None, vmax = None):
    conv_filter = model.conv2.weight.data

    for j in np.arange(conv_filter.size(0)):
        ax = axs.ravel()[j]
        ax.imshow(torch.mean(conv_filter[j,:,:,:], axis=0), cmap = cmap, vmin = vmin, vmax = vmax) 
        ax.axis('off')
        ax.set_title(f'filter {j}', fontsize = 8)

    return 


from tqdm.auto import tqdm
import scipy.ndimage as nd
from torch.autograd import Variable
from torch.optim import Adam
import matplotlib.pyplot as plt

# Octaver function
def octaver_fn(model, base_img, step_fn, octave_n=6, octave_scale=1.4, iter_n=10, **step_args):
    octaves = [base_img]

    for i in range(octave_n - 1):
        print(octaves[-1].shape)
        octaves.append(nd.zoom(octaves[-1], (1, 1.0 / octave_scale, 1.0 / octave_scale), order=1))

    detail = np.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]

        if octave > 0:
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0 * h / h1, 1.0 * w / w1), order=1)

        src = octave_base + detail

        for i in range(iter_n):
            src = step_fn(model, src, **step_args)

        detail = src.numpy() - octave_base

    return src

# Filter visualization
def filter_step(model, img, layer_index, filter_index, step_size=5, use_L2=False):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.zero_grad()

    img_var = Variable(torch.Tensor(img).to(device), requires_grad=True)
    optimizer = Adam([img_var], lr=step_size, weight_decay=1e-4)

    x = img_var
    for index, layer in enumerate(model.features):
        x = layer(x)
        if index == layer_index:
            break
    
    output = x[filter_index]
    loss =  -torch.mean(output)
    loss.backward()

    if use_L2 == True:
        # L2 normalization on gradients
        mean_square = torch.Tensor([torch.mean(img_var.grad.data ** 2) + 1e-5]).to(device)
        img_var.grad.data /= torch.sqrt(mean_square)
        img_var.data.add_(img_var.grad.data * step_size)

    else:
        optimizer.step()

    result = img_var.data.cpu().numpy()

    # result[0, :, :] = np.clip(result[0, :, :], -0.5, 0.5)

    return torch.Tensor(result)

def visualize_filter(model, base_img, layer_index, filter_index,
                     octave_n, octave_scale, iter_n,
                     step_size, display, use_L2):
    return octaver_fn(
        model, base_img, step_fn=filter_step,
        octave_n=octave_n, octave_scale=octave_scale,
        iter_n=iter_n, layer_index=layer_index, 
        filter_index=filter_index, step_size=step_size,
        use_L2=use_L2
    )

def init_image(size=(1, 28, 28)):
    img_np = np.random.normal(0, 1, size) * 0.5
    return img_np

# Show
def show_layer(model, layer_num, img_size, save_path, filter_start=10, filter_end=20, step_size= 0.05,  use_L2=False):
    img_np = init_image(size=img_size)

    for i in tqdm(range(filter_start, filter_end), desc="Feature Visualization", mininterval=0.01):
        title = "Layer {} Filter {}".format(layer_num, i)

        filter = visualize_filter(model, img_np, layer_num,
                                  filter_index=i,
                                  octave_scale=1.2,
                                  octave_n=6,
                                  iter_n=15,
                                  step_size=step_size,
                                  display=True,
                                  use_L2=use_L2)

        plt.imsave(save_path + title + ".png", np.stack([filter, filter, filter]).T, cmap='gray')

# l1_range = [*torch.linspace(0., 1, steps=6), *torch.linspace(2, 10, steps=5)]
l1_range = [0.2, 0.4, 0.6, 0.8, 1., 2., 4., 6., 8., 10.]
print(l1_range, "\n\n\n")

from datetime import date
plot_dir = f'plots_fashion/{date.today().strftime("%y%m%d")}'
train_data = FashionMNISTDataset(True)#FashionMNIST("D:/Research/MedMNIST", True)#, download=True)
test_data  = FashionMNISTDataset(False)#FashionMNIST("D:/Research/MedMNIST", False)#, download=True)

arr, lab = train_data[0]

img_shape = arr.shape[1:]
if np.sum(img_shape) < (28 * 2 + 2): #quick and dirty of checking if 3d
    try:
        os.makedirs(plot_dir)
    except:
        pass
else:
    pass


# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=512, num_workers=0)


# obtain one batch of training images
dataiter = iter(test_loader)
images, labels = dataiter.next()
images, labels = images.to(dev), labels.to(dev)
print(images.shape, labels.shape)

### Unconstrained
model_ = ConvLassoNet(lambda_=0., M=M, D_in=(28, 28), D_out=10, kernel_size=3, padding=0, sparse_layer='conv2').to(dev)
try:
    os.makedirs(f'{plot_dir}/random')
except:
    pass

show_layer(model_, 3, (1, 128, 128), save_path=f'{plot_dir}/random/act_max.png', filter_start=0, filter_end=32)

print(model_)
# test forward method, reshape input to vector with view
model_.forward(images).size()
# params of G are already included in params of model!
for param in model_.parameters():
    print(param.size())



loss_plots, acc_plots, auroc_plots = {}, {}, {}

opt_ = torch.optim.SGD(model_.parameters(), lr=alpha0, momentum=momentum, nesterov=True)
sched_ = StepLR(opt_, step_size=1, gamma=0.9)
# train first 75 epochs
train_model(model_, opt_, loss, lr_schedule=sched_, n_epochs=n_epochs*3, device=dev)
# record last 25 epochs
hist_ = train_model(model_, opt_, loss, lr_schedule=sched_, n_epochs=n_epochs, device=dev)
model0 = deepcopy(model_)

show_layer(model_, 3, (1, 128, 128), save_path=f'{plot_dir}/init/act_max.png', filter_start=0, filter_end=32)


loss_plots[0.]   = hist_['valid_loss']
acc_plots[0.]    = hist_['valid_acc']
auroc_plots[0.]  = hist_['valid_auroc']

### LassoNet
for M in [0.25, 0.5, 1, 2, 4]:
    for l1 in l1_range:
        model = ConvLassoNet(lambda_=l1, M=M, D_in=(28, 28), D_out=10, kernel_size=3, padding=0, final='softmax', sparse_layer='conv2').to(dev)

        opt = torch.optim.SGD(model.parameters(), lr=alpha0, momentum=momentum, nesterov=True)
        sched = StepLR(opt, step_size=1, gamma=0.9)
        
        hist = train_model(model, opt, loss, lr_schedule=sched, n_epochs=n_epochs, pretrained=model_, device=dev)
        
        loss_plots[l1]  = hist_['valid_loss']
        acc_plots[l1]   = hist_['valid_acc']
        auroc_plots[l1] = hist_['valid_auroc']
        
        
        v_, v2_ = 0.5, 0.1
        """ Let the Plotting Begin !"""
        try:
            os.makedirs(f'{plot_dir}/{M:.1f}_{l1:.2f}')
            print(f"plotting {plot_dir}, {M:.1f}, {l1:.2f}")
        except:
            pass

        ### unconstrained weights
        fig, axs = plt.subplots(4,4)
        plot_filter1(model0.to('cpu'), cmap=plt.cm.RdBu_r, vmin=-v_, vmax=v_)
        fig.suptitle('Filter1 weights unconstrained')
        fig.savefig(f'{plot_dir}/{M:.1f}_{l1:.2f}/conv_filter1_unc.png', dpi = 400)

        ### unconstrained weights
        fig, axs = plt.subplots(4,8)
        plot_filter2(model0.to('cpu'), cmap=plt.cm.RdBu_r, vmin=-v2_, vmax=v2_)
        fig.suptitle('Filter2 weights unconstrained')
        fig.savefig(f'{plot_dir}/{M:.1f}_{l1:.2f}/conv_filter2_unc.png', dpi = 400)

        # Loss plot
        fig, axs = plt.subplots(1,3, figsize=(20, 8))

        ax = axs[0]
        ax.plot(hist['train_loss'], c = '#002635', marker = 'o', label = 'Train loss ConvLassoNet')
        ax.plot(hist['valid_loss'], c = '#002635', marker = 'x', ls = '--', label = 'Validation loss ConvLassoNet')

        ax.plot(hist_['train_loss'], c = '#AB1A25', marker = 'o', label = 'Train loss unconstrained')
        ax.plot(hist_['valid_loss'], c = '#AB1A25', marker = 'x', ls = '--', label = 'Validation loss unconstrained')

        ax.set_xlabel('Epoch')
        ax.set_ylim(0,)
        ax.legend()

        ax = axs[1]
        ax.plot(hist['train_acc'], c = '#002635', marker = 'o', label = 'Train accuracy ConvLassoNet', alpha=0.7, markersize=4.)
        ax.plot(hist['valid_acc'], c = '#002635', marker = 'x', ls = '--', label = 'Validation accuracy ConvLassoNet', markersize=4.)
        
        ax.plot(hist_['train_acc'], c = '#AB1A25', marker = 'o', label = 'Train accuracy unconstrained', alpha=0.7, markersize=2.)
        ax.plot(hist_['valid_acc'], c = '#AB1A25', marker = 'x', ls = '--', label = 'Validation accuracy unconstrained', markersize=2.)

        ax.set_xlabel('Epoch')
        ax.set_ylim(0.,1.)
        ax.legend()

        ax = axs[2]
        ax.plot(hist['train_auroc'], c = '#002635', marker = 'o', label = 'Train AUROC ConvLassoNet', alpha=0.7, markersize=4.)
        ax.plot(hist['valid_auroc'], c = '#002635', marker = 'x', ls = '--', label = 'Validation AUROC ConvLassoNet', markersize=4.)
        
        ax.plot(hist_['train_auroc'], c = '#AB1A25', marker = 'o', label = 'Train AUROC unconstrained', alpha=0.7, markersize=2.)
        ax.plot(hist_['valid_auroc'], c = '#AB1A25', marker = 'x', ls = '--', label = 'Validation AUROC unconstrained', markersize=2.)

        ax.set_xlabel('Epoch')
        ax.set_ylim(0.,1.)
        ax.legend()

        fig.suptitle('Loss with and without LassoNet constraint')

        fig.savefig(f'{plot_dir}/{M:.1f}_{l1:.2f}/conv_loss.png', dpi = 400)

        fig, ax = plt.subplots()
        plot_filter_norm(model.to('cpu'), ax, color = '#002635')
        plot_filter_norm(model0.to('cpu'), ax, color = '#AB1A25')

        ax.set_ylabel('Max-norm of each filter channel')
        ax.set_xlabel('Conv1 output channel')
        ax.set_ylim(0,)

        p1 = mpatches.Patch(color='#002635', label='ConvLassoNet')
        p2 = mpatches.Patch(color='#AB1A25', label='unconstrained')
        ax.legend(handles=[p1,p2])

        fig.savefig(f'{plot_dir}/{M:.1f}_{l1:.2f}/conv1_filter_norm.png', dpi = 400)

        fig, ax = plt.subplots()
        plot_filter_norm(model.to('cpu'), ax, color = '#002635', layer='conv2')
        plot_filter_norm(model0.to('cpu'), ax, color = '#AB1A25', layer='conv2')

        ax.set_ylabel('Max-norm of each filter channel')
        ax.set_xlabel('Conv2 output channel')
        ax.set_ylim(0,)

        p1 = mpatches.Patch(color='#002635', label='ConvLassoNet')
        p2 = mpatches.Patch(color='#AB1A25', label='unconstrained')
        ax.legend(handles=[p1,p2])

        fig.savefig(f'{plot_dir}/{M:.1f}_{l1:.2f}/conv2_filter_norm.png', dpi = 400)

        fig, axs = plt.subplots(4,4)
        plot_filter1(model.to('cpu'), cmap=plt.cm.RdBu_r, vmin=-v_, vmax=v_)
        fig.suptitle('Filter1 weights ConvLassoNet')
        fig.savefig(f'{plot_dir}/{M:.1f}_{l1:.2f}/conv_filter1.png', dpi = 400)

        fig, axs = plt.subplots(4,8)
        plot_filter2(model.to('cpu'), cmap=plt.cm.RdBu_r, vmin=-v2_, vmax=v2_)
        fig.suptitle('Filter2 weights ConvLassoNet')
        fig.savefig(f'{plot_dir}/{M:.1f}_{l1:.2f}/conv_filter2.png', dpi = 400)
        
        # continue training with previous model
        model_ = deepcopy(model)

        pd.DataFrame.from_dict(loss_plots).to_csv("losses.csv")
        pd.DataFrame.from_dict(acc_plots).to_csv("accuracies.csv")
        pd.DataFrame.from_dict(auroc_plots).to_csv("aurocs.csv")

        show_layer(model.to(dev), 3, (1, 128, 128), save_path=f'{plot_dir}/{M:.1f}_{l1:.2f}/act_max.png', filter_start=0, filter_end=32)
    
        

# # %%
# img, _ = train_data[1]
# img.shape
# plt.imshow(img[0].T, cmap=plt.cm.bone)

# # %%
# arr = np.load(medmnists[3])['train_images']
# plt.imshow(arr[0])

# # %%



