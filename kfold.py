import time
from apex import amp
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold

from config import Config
from model.metric import *
from model.model import mynet
from model.loss import weighted_CrossEntropyLoss
from data_loaders import *
from utils.util import *
import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
training_files = ['/home/deep1/17145213/data/edf20/edf20.npz']
#'/home/deep1/17145213/data/edf197/ST_all.npz'

# # 保存当前模型的权重，并且更新最佳的模型权重
# def save_ckpt(state, is_best, model_save_dir):
#     current_w = os.path.join(model_save_dir, cfg.current_w)
#     best_w = os.path.join(model_save_dir, cfg.best_w)
#     torch.save(state, current_w)
#     if is_best: shutil.copyfile(current_w, best_w)


# 保存当前模型的权重，并且更新最佳的模型权重
def save_ckpt(state, is_best, model_save):
    current_w = os.path.join(cfg.model_pth, cfg.current_w)
    best_w = model_save
    torch.save(state, current_w)
    if is_best: shutil.copyfile(current_w, best_w)
    
cfg = Config()
if not os.path.exists(cfg.model_pth):
    os.makedirs(cfg.model_pth)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = LoadDataset_from_numpy(training_files)

all_ys = train_dataset.y_data.tolist()
num_classes = len(np.unique(all_ys))
counts = [all_ys.count(i) for i in range(num_classes)]
print(len(train_dataset), counts)
weights_for_each_class = calc_class_weight(counts)

kfold = KFold(n_splits=cfg.num_folds, shuffle=True)

for fold, (train_ids, val_ids) in enumerate(kfold.split(train_dataset)):
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

    # Adding subsamples to dataloader to create batches of 100.
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=train_subsampler, drop_last=False)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=val_subsampler, drop_last=False)

    model = mynet().to(device)
    model.apply(weights_init_normal)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay, amsgrad=True)

    # model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    train_loss = MetricTracker()

    test_loss = MetricTracker()
    test_acc = MetricTracker()
    test_fscore = MetricTracker()
    test_kappa = MetricTracker()
    max_fscore, max_acc, max_k = -1, -1, -1
    
    
    #Each epoch runs over the complete dataset once.
    for epoch in range(cfg.epochs):
        model.train()        
        train_loss.reset()

        #Batch of 100 samples
        for data, target in tqdm(train_loader):
            data_t, data_f, target = data[0].type(torch.FloatTensor).to(device), data[1].type(torch.FloatTensor).to(device), target.to(device)

            optimizer.zero_grad()
            t, f, fin = model(data_t, data_f)

            loss = weighted_CrossEntropyLoss(
                fin, target, weights_for_each_class, device)

            loss.backward()
            optimizer.step()
            train_loss.update(loss.item())
            
            
        model.eval()
        test_loss.reset()
        test_acc.reset()
        test_fscore.reset()
        test_kappa.reset()
        with torch.no_grad():
            for i, (data, target) in enumerate(val_loader):
                data_t, data_f, target = data[0].type(torch.FloatTensor).to(device), data[1].type(torch.FloatTensor).to(device), target.to(device)            
                t, f, fin = model(data_t, data_f)
                
                loss = weighted_CrossEntropyLoss(
                    fin, target, weights_for_each_class, device)
                
                test_loss.update(loss.item())
                test_acc.update(accuracy(fin, target))
                test_fscore.update(f1(fin, target))
                test_kappa.update(kappa(fin, target))
            
            wf = open(cfg.result+'/20ST.txt', 'a+')
            wf.write(
                f'Fold: {fold:03d} / Epoch: {epoch:03d} | Loss: {test_loss.avg:.4f} | Acc: {test_acc.avg:.4f} | F1: {test_fscore.avg:.4f} | Kappa: {test_kappa.avg:.4f} \n')
            wf.close()    
        
        
        # save_ckpt(model.state_dict(), test_acc.avg > max_acc, os.path.join(cfg.model_pth, 'a.pth')) 
        # save_ckpt(model.state_dict(), test_fscore.avg > max_fscore, os.path.join(cfg.model_pth, 'f.pth'))        
        # save_ckpt(model.state_dict(), test_kappa.avg > max_k, os.path.join(cfg.model_pth, 'k.pth'))
            
        # max_acc = max(max_acc, test_acc.avg)
        # max_fscore = max(max_fscore, test_fscore.avg)
        # max_k = max(max_k, test_kappa.avg)
        
