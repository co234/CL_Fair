import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from algorithm.math_utils import count_label
from sklearn.metrics import f1_score, accuracy_score
from algorithm.utils import fairness_measure
from algorithm.mlp_model import MLP


class CL_Fair_model():
    def __init__(self, 
                 input_dim, 
                 epoch = 50,
                 hidden_dim=10, 
                 output_dim=2, 
                 lr_1=0.01, 
                 lr_2 = 0.05,
                 sigma = 0.01,
                 n_s = 0.8):
        # Initialize network 
        self.net_A = MLP(input_dim, hidden_dim, output_dim)
        self.net_B = MLP(input_dim, hidden_dim, output_dim)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer1 = optim.Adam(self.net_A.parameters(), lr=lr_1)
        self.optimizer2 = optim.Adam(self.net_B.parameters(), lr=lr_1)

        self.net_clean = MLP(input_dim, hidden_dim, output_dim)
        self.optimizer = optim.Adam(self.net_clean.parameters(), lr=lr_2)

        self.sigma = sigma 
        self.n_s = n_s
        self.epoch = epoch



    def train_coteaching(self, dataloader):
        self.net_A.train()
        self.net_B.train()
        self.net_clean.train()
        n_a_true = 0
        n_b_true = 0
        n_total_true = 0
        n_a = 0
        n_b = 0
        n_total = 0
        for i in range(self.epoch):
            for batch in dataloader:
                
                data, target, s_idx, true_label = batch[1], batch[2], batch[3], batch[4]

                data_A, data_B = data[np.where(s_idx==1)[0]], data[np.where(s_idx==0)[0]]
                target_A, target_B = target[np.where(s_idx==1)[0]], target[np.where(s_idx==0)[0]]

                
                # Training Net A
                out_A = self.net_A(data_A)
                loss_A = self.criterion(out_A, target_A)
                self.optimizer1.zero_grad()
                loss_A.backward()
                self.optimizer1.step()
                
                # Training Model 2
                out_B = self.net_B(data_B)
                loss_B = self.criterion(out_B, target_B)
                self.optimizer2.zero_grad()
                loss_B.backward()
                self.optimizer2.step()

                
                # Co-teaching strategy
                r_a, r_b = self.net_A(data), self.net_B(data)
                c_a, c_b = torch.softmax(r_a, dim=1), torch.softmax(r_b, dim=1)

            
                cm_a, cbar_a, Q_a, tn_idx_a, tp_idx_a = count_label(target,c_a.detach().numpy(), self.sigma, self.n_s,base=False)
                cm_b, cbar_b, Q_b,tn_idx_b, tp_idx_b = count_label(target,c_b.detach().numpy(), self.sigma, self.n_s,base=False)
                
                true_neg_inter = np.intersect1d(tn_idx_a,tn_idx_b)
                true_pos_inter = np.intersect1d(tp_idx_a,tp_idx_b)

                d_a = np.union1d(tn_idx_a,tp_idx_a)
                d_b = np.union1d(tn_idx_b,tp_idx_b)

                true_idx = np.where(target==true_label)[0]

                for i in d_a:
                    if i in true_idx:
                        n_a_true+=1
                n_a += len(d_a)

                for i in d_b:
                    if i in true_idx:
                        n_b_true+=1
                n_b += len(d_b)
            

                selected_idx =list(true_neg_inter)+list(true_pos_inter)

                for i in selected_idx:
                    if i in true_idx:
                        n_total_true+=1
                n_total+=len(selected_idx)

                data_s = data[selected_idx]

                out = self.net_clean(data_s)
                loss = self.criterion(out, target[selected_idx])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        print("DA: {:.2f}, DB: {:.2f}, Total: {:.2f}".format((n_a_true/n_a)*100,
                                                             (n_b_true/n_b)*100,
                                                             (n_total_true/n_total)*100))


    def test(self,dataloader):
        self.net_clean.eval()
        predictions = []
        ground_truths = []
        sens_s = []

        with torch.no_grad():
            for batch in dataloader:
                data, target, s_idx = batch[1], batch[2], batch[3]
                out = self.net_clean(data)
                _, predicted = torch.max(out.data,1)
                predictions.extend(predicted.cpu().numpy())
                ground_truths.extend(target.cpu().numpy())
                sens_s.extend(s_idx.cpu().numpy())

        f1 = f1_score(predictions,ground_truths,average="weighted")
        acc = accuracy_score(predictions,ground_truths)
        di = fairness_measure(ground_truths,predictions,sens_s,type="di")
        eo = fairness_measure(ground_truths,predictions,sens_s,type="eo")
        dp = fairness_measure(ground_truths,predictions,sens_s,type="dp")
        print("Performance - F1_err: {:.2f}, ACC_err: {:.2f}".format((1-f1)*100,(1-acc)*100))
        print("Fairness - DI: {:.2f}, EO: {:.2f}, DP: {:.2f}".format(di,eo,dp))


        r_dict = {"err": (1-acc)*100,
                  "di": di,
                  "deo": eo,
                  "dp":dp}

        return r_dict



