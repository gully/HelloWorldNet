
########################################
########### IMPORT PACKAGES ############
########################################

### standard packages
import os
import numpy as np
import glob as glob
from tqdm import tqdm 
from random import random
import argparse
import pdb

### torch packages
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn

### sklearn packages
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

### plotting packages
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


########################################
########### PARSE ARGUMENTS ############
########################################

### an example input to command line:
### python extranet.py 225 64 1e-5 '/data/kepler/new' '/data/ensembling/exonet_xs_ensembling'

### parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("n_epochs", help="number of epochs to use for training", type=int)
parser.add_argument("n_batches", help="number of matches to use for training", type=int)
parser.add_argument("r_learn", help="learning rate for Adam optimizer", type=float)
parser.add_argument("d_path", help="path to data")
parser.add_argument("m_out", help="path for output plot")
parser.add_argument("--fixed_seed", help="set if wanting to fix the seed", action="store_true")
parser.add_argument("--XS", help="use ExtranetXS model", action="store_true")
args = parser.parse_args()

### set manual seed
if args.fixed_seed:
    torch.cuda.manual_seed(42)


########################################
########### DEFINE CLASSES  ############
########################################

class KeplerDataLoader(Dataset):
    
    '''
    
    PURPOSE: DATA LOADER FOR KERPLER LIGHT CURVES
    INPUT: PATH TO DIRECTOR WITH LIGHT CURVES + INFO FILES
    OUTPUT: LOCAL + GLOBAL VIEWS, LABELS
    
    '''

    def __init__(self, filepath):

        ### list of global, local, and info files (assumes certain names of files)
        self.flist_global = np.sort(glob.glob(os.path.join(filepath, '*global.npy')))
        self.flist_local = np.sort(glob.glob(os.path.join(filepath, '*local.npy')))
        self.flist_info = np.sort(glob.glob(os.path.join(filepath, '*info.npy')))
        
        ### list of whitened centroid files
        self.flist_global_cen = np.sort(glob.glob(os.path.join(filepath, '*global_cen_w.npy')))
        self.flist_local_cen = np.sort(glob.glob(os.path.join(filepath, '*local_cen_w.npy')))
        
        ### ids = {TIC}_{TCE}
        self.ids = np.sort([(x.split('/')[-1]).split('_')[1] + '_' + (x.split('/')[-1]).split('_')[2] for x in self.flist_global])

    def __len__(self):

        return self.ids.shape[0]

    def __getitem__(self, idx):

        ### grab local and global views
        data_global = np.load(self.flist_global[idx])
        data_local = np.load(self.flist_local[idx])

        ### grab centroid views
        data_global_cen = np.load(self.flist_global_cen[idx])
        data_local_cen = np.load(self.flist_local_cen[idx])
        
        ### info file contains: [0]kic, [1]tce, [2]period, [3]epoch, [4]duration, [5]label)
        data_info = np.load(self.flist_info[idx])
        
        return (data_local, data_global, data_local_cen, data_global_cen, data_info[6:]), data_info[5]


class ExtranetModel(nn.Module):

    '''
    
    PURPOSE: DEFINE EXTRANET MODEL ARCHITECTURE
    INPUT: GLOBAL + LOCAL LIGHT CURVES AND CENTROID CURVES, STELLAR PARAMETERS
    OUTPUT: BINARY CLASSIFIER
    
    '''
    
    def __init__(self):

        ### initialize model
        super(ExtranetModel, self).__init__()

        ### define global convolutional lalyer
        self.fc_global = nn.Sequential(
            nn.Conv1d(2, 16, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 16, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
            nn.Conv1d(16, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
            nn.Conv1d(32, 64, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
            nn.Conv1d(64, 128, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, 5, stride=1, padding=2),
            nn.MaxPool1d(5, stride=2),
            nn.Conv1d(128, 256, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(256, 256, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
        )

        ### define local convolutional lalyer
        self.fc_local = nn.Sequential(
            nn.Conv1d(2, 16, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 16, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(7, stride=2),
            nn.Conv1d(16, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(7, stride=2),
        )

        ### define fully connected layer that combines both views
        self.final_layer = nn.Sequential(
            nn.Linear(16586, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            ### need output of 1 because using BCE for loss
            nn.Linear(512, 1),
            nn.Sigmoid())

    def forward(self, x_local, x_global, x_local_cen, x_global_cen, x_star):

        ### concatonate light curve and centroid data
        x_local_all = torch.cat([x_local, x_local_cen], dim=1)
        x_global_all = torch.cat([x_global, x_global_cen], dim=1)
        
        ### get outputs of global and local convolutional layers
        out_global = self.fc_global(x_global_all)
        out_local = self.fc_local(x_local_all)
        
        ### flattening outputs from convolutional layers into vector
        out_global = out_global.view(out_global.shape[0], -1)
        out_local = out_local.view(out_local.shape[0], -1)

        ### concatonate global and local views with stellar parameters
        out = torch.cat([out_global, out_local, x_star.squeeze(1)], dim=1)
        out = self.final_layer(out)

        return out


class ExtranetXSModel(nn.Module):

    '''
    
    PURPOSE: DEFINE EXTRANET-XS MODEL ARCHITECTURE
    INPUT: GLOBAL + LOCAL LIGHT CURVES AND CENTROID CURVES, STELLAR PARAMETERS
    OUTPUT: BINARY CLASSIFIER
    
    '''

    def __init__(self):

        ### initializing the nn.Moduel (super) class
        ### (must do this first always)
        super(ExtranetXSModel, self).__init__()

        ### define global convolutional lalyer
        self.fc_global = nn.Sequential(
            nn.Conv1d(2, 16, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(16, 16, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(16, 32, 5, stride=1, padding=2),
            nn.ReLU(),
        )

        ### define the local convolutional layer
        self.fc_local = nn.Sequential(
            nn.Conv1d(2, 16, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(16, 16, 5, stride=1, padding=2),
            nn.ReLU(),
        )
                
        ### define fully connected layer that combines both views
        self.final_layer = nn.Sequential(
            nn.Linear(58, 1),
            nn.Sigmoid())
        
    ### define how to move forward through model
    def forward(self, x_local, x_global, x_local_cen, x_global_cen, x_starpars):

        ### concatonate light curve and centroid data
        x_local_all = torch.cat([x_local, x_local_cen], dim=1)
        x_global_all = torch.cat([x_global, x_global_cen], dim=1)
        out_global = self.fc_global(x_global_all)
        out_local = self.fc_local(x_local_all)
        
        ### do global pooling
        out_global = F.max_pool1d(out_global, out_global.shape[-1])
        out_local = F.max_pool1d(out_local, out_local.shape[-1])

        ### flattening outputs from convolutional layers into vector
        out_global = out_global.view(out_global.shape[0], -1)
        out_local = out_local.view(out_local.shape[0], -1)

        ### concatonate global and local views with stellar parameters
        out = torch.cat([out_global, out_local, x_starpars.squeeze(1)], dim=1)
        out = self.final_layer(out)

        return out


########################################
########## DEFINE FUNCTIONS ############
########################################

def invert_tensor(tensor):

    '''

    PURPOSE: FLIP A 1D TENSOR ALONG ITS AXIS
    INPUT: 1D TENSOR
    OUTPUT: INVERTED 1D TENSOR

    '''

    idx = [i for i in range(tensor.size(0)-1, -1, -1)]
    idx = torch.cuda.LongTensor(idx)
    inverted_tensor = tensor.index_select(0, idx)

    return inverted_tensor


def train_model(n_epochs, kepler_data_loader, model, criterion, optimizer):

    '''

    PURPOSE: TRAIN MODEL

    INPUTS:  num_epoch = number of epochs for training
             kepler_data_loader = data loader for Kepler dataset
             model = model use for training
             criterion = criterion for calculating loss

    OUTPUT:  epoch_{train/val}_loss = training set loss for each epoch
             epoch_val_acc = validation set accuracy for each epoch
             epoch_val_ap = validation set avg. precision for each epoch
             final_val_pred = validation predictions from final model
             final_val_gt = validation ground truths

    '''

    ### empty arrays to fill per-epoch outputs
    epoch_train_loss = []
    epoch_val_loss = []
    epoch_val_acc = []
    epoch_val_ap = []

    ### loop over number of epochs of training
    for epoch in tqdm(range(n_epochs)):

        ####################
        ### for training set
        
        ### loop over batches
        train_loss = torch.zeros(1).cuda()
        for x_train_data, y_train in kepler_data_loader:
            
            ### get local view, global view, and label for training
            x_train_local, x_train_global, x_train_local_cen, x_train_global_cen, x_train_star = x_train_data
            x_train_local = Variable(x_train_local).type(torch.FloatTensor).cuda()
            x_train_global = Variable(x_train_global).type(torch.FloatTensor).cuda()    
            x_train_local_cen = Variable(x_train_local_cen).type(torch.FloatTensor).cuda()
            x_train_global_cen = Variable(x_train_global_cen).type(torch.FloatTensor).cuda()
            x_train_star = Variable(x_train_star).type(torch.FloatTensor).cuda()        
            y_train = Variable(y_train).type(torch.FloatTensor).cuda()
                      
            ### randomly invert half of light curves
            for batch_ind in range(x_train_local.shape[0]):     
                
                ### add random gaussian noise
                sig_noise = np.random.uniform(0, 1.0)
                local_noise = Variable(x_train_local[batch_ind].data.new(x_train_local[batch_ind].size()).normal_(0.0, sig_noise))
                x_train_local[batch_ind] = x_train_local[batch_ind] + local_noise
                global_noise = Variable(x_train_global[batch_ind].data.new(x_train_global[batch_ind].size()).normal_(0.0, sig_noise))
                x_train_global[batch_ind] = x_train_global[batch_ind] + global_noise

                if random() < 0.5:
                    x_train_local[batch_ind] = invert_tensor(x_train_local[batch_ind])
                    x_train_global[batch_ind] = invert_tensor(x_train_global[batch_ind])
                    x_train_local_cen[batch_ind] = invert_tensor(x_train_local_cen[batch_ind])
                    x_train_global_cen[batch_ind] = invert_tensor(x_train_global_cen[batch_ind])

            ### fix dimensions for next steps
            x_train_local = x_train_local.unsqueeze(1)
            x_train_global = x_train_global.unsqueeze(1)
            x_train_local_cen = x_train_local_cen.unsqueeze(1)
            x_train_global_cen = x_train_global_cen.unsqueeze(1)
            x_train_star = x_train_star.unsqueeze(1)
            y_train = y_train.unsqueeze(1)

            ### calculate loss using model
            output_train = model(x_train_local, x_train_global, x_train_local_cen, x_train_global_cen, x_train_star)
            loss = criterion(output_train, y_train)
            train_loss += loss.data

            ### train model (zero gradients and back propogate results)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ### record training loss for this epoch (divided by size of training dataset)
        epoch_train_loss.append(train_loss.cpu().numpy() / len(kepler_data_loader.dataset))
        
        ######################
        ### for validation set
        
        ### loop over batches
        val_pred, val_gt, val_loss, num_corr = [], [], 0, 0
        for x_val_data, y_val in kepler_val_loader:
                        
            ### get local view, global view, and label for validating
            x_val_local, x_val_global, x_val_local_cen, x_val_global_cen, x_val_star = x_val_data
            x_val_local = Variable(x_val_local).type(torch.FloatTensor).cuda()
            x_val_global = Variable(x_val_global).type(torch.FloatTensor).cuda()
            x_val_local_cen = Variable(x_val_local_cen).type(torch.FloatTensor).cuda()
            x_val_global_cen = Variable(x_val_global_cen).type(torch.FloatTensor).cuda()
            x_val_star = Variable(x_val_star).type(torch.FloatTensor).cuda()
 
            ### fix dimensions for next steps
            y_val = Variable(y_val).type(torch.FloatTensor).cuda()
            x_val_local = x_val_local.unsqueeze(1)
            x_val_global = x_val_global.unsqueeze(1)
            x_val_local_cen = x_val_local_cen.unsqueeze(1)
            x_val_global_cen = x_val_global_cen.unsqueeze(1)
            x_val_star = x_val_star.unsqueeze(1)
            y_val = y_val.unsqueeze(1)

            ### calculate loss & add to sum over all batches
            output_val = model(x_val_local, x_val_global, x_val_local_cen, x_val_global_cen, x_val_star)
            loss_val = criterion(output_val, y_val)
            val_loss += loss_val.data

            ### get number of correct predictions using threshold=0.5
            ### & sum over all batches
            output_pred = output_val >= 0.5
            num_corr += output_pred.eq(y_val.byte()).sum().item()
                        
            ### record predictions and ground truth by model
            ### (used for AP per epoch; reset at each epoch; final values output)
            val_pred.append(output_val.data.cpu().numpy())
            val_gt.append(y_val.data.cpu().numpy())
            
        ### record validation loss calculate for this epoch (divided by size of validation dataset)
        epoch_val_loss.append(val_loss.cpu().numpy() / len(kepler_val_loader.dataset))
        
        ### record validation accuracy (# correct predictions in val set) for this epoch
        epoch_val_acc.append(num_corr / len(kepler_val_loader.dataset))
        
        ### calculate average precision for this epoch
        epoch_val_ap.append(average_precision_score(np.concatenate(val_gt).ravel(), np.concatenate(val_pred).ravel(), average=None))            
        
    ### grab final predictions and ground truths for validation set
    final_val_pred = np.concatenate(val_pred).ravel()
    final_val_gt = np.concatenate(val_gt).ravel()

    return epoch_train_loss, epoch_val_loss, epoch_val_acc, epoch_val_ap, final_val_pred, final_val_gt


########################################
############ TRAIN MODEL  ##############
########################################

### setup screen output
print("\nTRAINING MODEL...\n")

### initialize model; cuda puts it on GPU
if args.XS:
    model = ExtranetXSModel().cuda()
else:
    model = ExtranetModel().cuda()

### specify optimizer for learning to use for training
optimizer = torch.optim.Adam(model.parameters(), lr=args.r_learn)

### specify loss function to use for training
criterion = nn.BCELoss()

### specify batch size to use for training
batch_size = args.n_batches 

### number of epochs to use for training
n_epochs = args.n_epochs 

### grab data using data loader
kepler_train_data = KeplerDataLoader(filepath=os.path.join(args.d_path, 'train'))
kepler_val_data = KeplerDataLoader(filepath=os.path.join(args.d_path, 'val'))
kepler_data_loader = DataLoader(kepler_train_data, batch_size=batch_size, shuffle=True, num_workers=4)
kepler_val_loader = DataLoader(kepler_val_data, batch_size=batch_size, shuffle=False, num_workers=4)

### train model
loss_train_epoch, loss_val_epoch, acc_val_epoch, ap_val_epoch, pred_val_final, gt_val_final  = train_model(n_epochs, kepler_data_loader, model, criterion, optimizer)


########################################
####### CALCULATE STATISTICS ###########
########################################

### setup screen output
print("\nCALCULATING METRICS...\n")

### calculate average precision & precision-recall curves
AP = average_precision_score(gt_val_final, pred_val_final, average=None)
print("   average precision = {0:0.4f}\n".format(AP))
 
### calculate precision-recall curve
P, R, _ = precision_recall_curve(gt_val_final, pred_val_final)

### calculate confusion matrix based on different thresholds 
thresh = [0.5, 0.6, 0.7, 0.8, 0.9]
prec_thresh, recall_thresh = np.zeros(len(thresh)), np.zeros(len(thresh))
for n, nval in enumerate(thresh):
    pred_byte = np.zeros(len(pred_val_final))
    for i, val in enumerate(pred_val_final):
        if val > nval:
            pred_byte[i] = 1.0
        else:
            pred_byte[i] = 0.0
    prec_thresh[n] = precision_score(gt_val_final, pred_byte)
    recall_thresh[n] = recall_score(gt_val_final, pred_byte)
    print("   thresh = {0:0.2f}, precision = {1:0.2f}, recall = {2:0.2f}".format(thresh[n], prec_thresh[n], recall_thresh[n]))
    tn, fp, fn, tp = confusion_matrix(gt_val_final, pred_byte).ravel()
    print("      TN = {0:0}, FP = {1:0}, FN = {2:0}, TP = {3:0}".format(tn, fp, fn, tp))


########################################
######### OUTPUT MODEL + STATS  ########
########################################

### transform from loss per sample to loss per batch (multiple by batch size to compare to Chris')
loss_train_batch = [x.item()* batch_size for x in loss_train_epoch]
loss_val_batch = [x.item()* batch_size for x in loss_val_epoch]

### setup
from astropy.table import Table
run = 0

### output predictions & ground truth
pt_fname = 'r' + str(run).zfill(2) + '-i' + str(n_epochs) + '-pt.csv'
while os.path.isfile(os.path.join(args.m_out, pt_fname)):
    run +=1
    pt_fname = 'r' + str(run).zfill(2) + '-i' + str(n_epochs) + '-pt.csv'
t = Table()
t['gt'] = gt_val_final
t['pred'] = pred_val_final
t.write(os.path.join(args.m_out, pt_fname), format='csv')

### output per-iteration values
epochs_fname = 'r' + str(run).zfill(2) + '-i' + str(n_epochs) + '-epoch.csv'
t = Table()
t['loss_train'] = loss_train_batch
t['loss_val'] = loss_val_batch
t['acc_val'] = acc_val_epoch
t['ap_val'] = ap_val_epoch
t.write(os.path.join(args.m_out, epochs_fname), format='csv')

### save model
model_fname = 'r' + str(run).zfill(2) + '-i' + str(n_epochs) + '-model.pth'
torch.save(model.state_dict(), os.path.join(args.m_out, model_fname))
print("\nOUTPUTTING MODEL @ " + os.path.join(args.m_out, model_fname) + "\n")


########################################
################ MAKE PLOTS ############
########################################

### setup figure
fig = plt.figure(figsize=(7, 7))
ax = gridspec.GridSpec(2,2)
ax.update(wspace = 0.4, hspace = 0.4)
ax1 = plt.subplot(ax[0,0])
ax2 = plt.subplot(ax[0,1])
ax3 = plt.subplot(ax[1,0])
ax4 = plt.subplot(ax[1,1])

### plot precision-recall curve
ax1.set_xlabel('Precision', fontsize=10, labelpad=10)
ax1.set_ylabel('Recall', fontsize=10)
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.0])
ax1.plot(R, P, linewidth=3, color='black')

### plot loss curve for training and validation sets
ax2.set_xlabel('Epoch', fontsize=10, labelpad=10)
ax2.set_ylabel('Loss', fontsize=10)
ax2.set_xlim([0.0, n_epochs])
ax2.set_ylim([0.0, np.max(loss_train_batch)*1.5])
ax2.plot(np.arange(len(loss_train_batch)), loss_train_batch, linewidth=3, color='cadetblue')
ax2.plot(np.arange(len(loss_val_batch)), loss_val_batch, linewidth=3, color='orangered')

### plot average precision per epoch
ax3.set_xlabel('Epoch', fontsize=10, labelpad=10)
ax3.set_ylabel('Average Precision', fontsize=10)
ax3.plot(np.arange(len(ap_val_epoch)), ap_val_epoch, linewidth=1.0, color='orangered')
ax3.scatter(np.arange(len(ap_val_epoch)), ap_val_epoch, marker='o', edgecolor='orangered', facecolor='orangered', s=10, linewidth=0.5, alpha=0.5)

### plot accuracy per epoch
ax4.set_xlabel('Epoch', fontsize=10, labelpad=10)
ax4.set_ylabel('Accuracy', fontsize=10)
ax4.plot(np.arange(len(acc_val_epoch)), acc_val_epoch, color='orangered', linewidth=1.0)
ax4.scatter(np.arange(len(acc_val_epoch)), acc_val_epoch, marker='o', edgecolor='orangered', facecolor='orangered', s=10, linewidth=0.5, alpha=0.5)

### save plot
plot_fname = 'r' + str(run).zfill(2) + '-i' + str(n_epochs) + '-plot.pdf'
plt.savefig(os.path.join(args.m_out, plot_fname), bbox_inches='tight', dpi=200, rastersized=True, alpha=True)

