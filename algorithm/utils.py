import numpy as np
from torch.utils import data
import torch
import math
from sklearn import svm
import warnings
warnings.filterwarnings("ignore")


def fairness_measure(label,pred_label,sensitive_feature,type="di"):
    pred_label = np.array(pred_label)
    sensitive_feature = np.array(sensitive_feature)
    label= np.array(label)

    a_idx,b_idx = np.where(sensitive_feature==0)[0], np.where(sensitive_feature==1)[0]
    a_ratio, b_ratio = (pred_label[a_idx] == label[a_idx]).sum()/len(a_idx), \
                        (pred_label[b_idx] == label[b_idx]).sum()/len(b_idx)
    

    if type == "dp":
        f_score = np.abs(a_ratio-b_ratio)

    elif type == "eo":
        y_pos = np.where(label==1)[0]
        pos_a,pos_b = np.intersect1d(y_pos,a_idx), np.intersect1d(y_pos,b_idx)
        a_eo,b_eo = (pred_label[pos_a]==1).sum()/(len(pos_a)+1e-15), (pred_label[pos_b]==1).sum()/(len(pos_b)+1e-15)
        f_score = np.abs(a_eo-b_eo)

    elif type == "di":
        f_score = np.min([(a_ratio/(b_ratio+1e-15)),(b_ratio/(a_ratio+1e-15))])

    else:
        print("NotImplement")
        f_score = None
    
    return f_score



def add_label_bias(yclean,rho,theta_dict,seed):
    """
    theta_0_p: P(Y=+1|Z=-1,A=0) rho_A 
    theta_0_m: P(Y=-1|Z=+1,A=0)
    theta_1_p: P(Y=+1|Z=-1,A=1)
    theta_1_m: P(Y=-1|Z=+1,A=1) rho_B
    """
    n = len(yclean)
    np.random.seed(seed)

    t_0_p, t_0_m, t_1_p,t_1_m = theta_dict['theta_0_p'],theta_dict['theta_0_m'],theta_dict['theta_1_p'],theta_dict['theta_1_m']


    def locate_group(label,sensitive_attr,a,y):
        return np.intersect1d(np.where(sensitive_attr==a)[0],np.where(label==y)[0])

    g_01, g_00 = locate_group(yclean,rho,0,1),locate_group(yclean,rho,0,0)
    g_11, g_10 = locate_group(yclean,rho,1,1),locate_group(yclean,rho,1,0)

    group = [g_01,g_00,g_11,g_10]
    theta = [t_0_m,t_0_p,t_1_m,t_1_p]
    tilde_y = [0,1,0,1]

    t = yclean.copy()

    for i in range(len(group)):
        for j in range(len(group[i])):
            p = np.random.uniform(0,1)
            if p < theta[i]:
                t[group[i][j]] = tilde_y[i]
            else:
                t[group[i][j]] = yclean[group[i][j]]


    return t


def transform_label(data,dict_theta,seed=42):
    y = data['y'].squeeze().long().cpu().detach().numpy()
    s = data['s'].cpu().detach().numpy()

    y_new = add_label_bias(y,s,dict_theta,seed)
    y_tilder = torch.tensor(y_new)
    data['yt'] = y_tilder
    clean_idx = np.where(np.abs(y-y_new)==0)[0]

    return data, clean_idx


def set_noisy_label(data, crop_type='sys', crop_ratio=0.4,seed = 38):
        if crop_type == "asy":
            dict_theta = {'theta_0_p':0,'theta_0_m':0,'theta_1_p':0,'theta_1_m':crop_ratio}
            data, clean_idx = transform_label(data,dict_theta,seed=seed)

        elif crop_type == "sys":
            dict_theta = {'theta_0_p':crop_ratio,'theta_0_m':0,'theta_1_p':0,'theta_1_m':crop_ratio}
            data, clean_idx = transform_label(data,dict_theta,seed=seed)
        else:
            raise NotImplementedError
        
        return data, clean_idx







