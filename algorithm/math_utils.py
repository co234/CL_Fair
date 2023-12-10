import numpy as np 


def count_label(y, score, sigma=0.01, n_s=0.5, base = False):
    neg_idx, pos_idx = np.where(y==0)[0], np.where(y==1)[0]
    neg_sample = score[neg_idx]
    pos_sample = score[pos_idx]

    if base == True:
        t_0, t_1 = np.mean(score[:,0]), np.mean(score[:,1])
    else: 
        t_0 = score_function(score[:,0],sigma,n_s)
        t_1 = score_function(score[:,1],sigma,n_s)

    
    true_neg_idx, true_pos_idx = np.where(score[:,0]>t_0)[0], np.where(score[:,1]>t_1)[0]
    
    # compute y_t = 0, y = 0
    c_00 = len(np.where(neg_sample[:,0]>t_0)[0])
    
    # compute y_t = 0, y = 1
    c_01 = len(np.where(neg_sample[:,1]>t_1)[0])
    
    # compute y_t = 1, y = 0
    c_10 = len(np.where(pos_sample[:,0]>t_0)[0])
    
    # compute y_t = 1, y = 1
    c_11 = len(np.where(pos_sample[:,1]>t_1)[0])
    
    count_matrix = np.zeros((2,2))
    count_matrix[0][0] = c_00
    count_matrix[0][1] = c_01
    count_matrix[1][0] = c_10
    count_matrix[1][1] = c_11
    
    true_neg_idx = np.intersect1d(neg_idx,true_neg_idx)
    true_pos_idx = np.intersect1d(pos_idx,true_pos_idx)

    x_y_0 = len(neg_sample)
    x_y_1 = len(pos_sample)

    c_00_bar = x_y_0 * c_00 / (c_00+c_01+1e-10)
    c_01_bar = x_y_0 * c_01 / (c_01+c_00+1e-10)
    c_10_bar = x_y_1 * c_10 / (c_10+c_11+1e-10)
    c_11_bar = x_y_1 * c_11 / (c_10+c_11+1e-10)

    cbar = np.zeros((2,2))
    cbar[0][0] = c_00_bar
    cbar[0][1] = c_01_bar
    cbar[1][0] = c_10_bar
    cbar[1][1] = c_11_bar

    N_cbar = c_00_bar+c_01_bar+c_10_bar+c_11_bar
    Q = np.zeros((2,2))
    Q[0][0] = c_00_bar/N_cbar
    Q[0][1] = c_01_bar/N_cbar
    Q[1][0] = c_10_bar/N_cbar
    Q[1][1] = c_11_bar/N_cbar



    
    return count_matrix, cbar, Q, true_neg_idx, true_pos_idx
    
    
    

def phi_function(x):
    return np.log(1+x+x**2/2)


def score_function(p,sigma,n_s):
    n = p.shape[0]
    mu_t = np.mean(phi_function(p))
    s = sigma**2

    upper_num = s*(n+s*np.log(2*n)/n**2)
    lower_num = int(n_s*n) - s

    return mu_t - upper_num/lower_num
    


    
    


if __name__ == "__main__":
    score = np.array([[0.1,0.2,0.3,0.4,0.23],[0.5,0.5,0.5,0.4,0.5]])
    y = np.array([1,0,1,1,1])
    c_a,x,y = count_label(y,score)
    print(c_a)
    # sigma = 0.01
    # n_s = 4

    # s = score_function(score,sigma,n_s)
    # print(s)
    # print(np.mean(score))




    
