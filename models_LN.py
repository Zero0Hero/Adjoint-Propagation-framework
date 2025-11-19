import torch
import numpy as np

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

import copy


def dataset(config):
    ## Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),  # 
    ])

    if config.train_task == "FMNIST" :
        config.train_eta_global = 10e-4
        train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform)
    elif config.train_task == "MNIST":
        config.train_eta_global = 10e-4
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)
    elif config.train_task == "CIFAR10":
        config.train_eta_global = 2e-4
        config.n_input = 32*32*3
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)
    elif config.train_task == "CIFAR100":
        config.train_eta_global = 2e-4
        config.n_input = 32*32*3
        config.n_out = 100
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=config.train_batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.train_batch_size, shuffle=False)
    return train_loader,test_loader,config



class FRNN_LN(torch.nn.Module):
    def __init__(self, config):
        super(FRNN_LN, self).__init__()
        # 
        self.device = config.device if hasattr(config, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

        self.train_tmethod = config.train_tmethod 
        self.train_batch_size = config.train_batch_size
        self.train_lambda1 = config.train_lambda1
        self.train_task = config.train_task
        self.train_fbupdaterule = config.train_fbupdaterule
        self.f = config.f
        self.fd = config.fd

        self.flag_feedbackLearning = config.flag_feedbackLearning
        self.flag_RNNLearning = config.flag_RNNLearning
        self.flag_RNNBiasLearning = config.flag_RNNBiasLearning

        self.factor_gamma1 = config.factor_gamma1
        self.RNN_alpha1 = config.RNN_alpha1  
        self.RNN_t2sta = config.RNN_t2sta
        self.RNN_t2sta2 = config.RNN_t2sta2
        self.flag_ycz = config.flag_ycz
        self.RNN_SR = config.RNN_SR
        self.RNN_CR = config.RNN_CR

        self.n_NperU = config.n_NperU
        self.n_NU = config.n_NU
        self.n_hidd = config.n_hidd
        if self.n_NU<0: self.n_hidd = -config.n_NperU * config.n_NU
        else: self.n_hidd = config.n_NperU * config.n_NU

        self.nL_hidd = config.nL_hidd
        
        self.sc_forward = config.sc_forward
        self.sc_forward_f = config.sc_forward_f
        self.sc_bias = config.sc_bias
        self.sc_back = config.sc_back
        self.sc_back_f = config.sc_back_f 

        # self.factor_beta1 = config.factor_beta1 
        
        self.PC_gamma = config.PC_gamma
        self.n_RE = config.n_RE if hasattr(config, 'n_RE') else self.n_NperU
        self.n_SE = config.n_SE if hasattr(config, 'n_SE') else self.n_NperU
        self.n_SI = config.n_SI if hasattr(config, 'n_SI') else self.n_NperU
        self.n_RI = config.n_RI if hasattr(config, 'n_RI') else self.n_NperU
        
        self.ffsc = config.ffsc if hasattr(config, 'ffsc') else 1
        self.fbsc = config.fbsc if hasattr(config, 'fbsc') else 1

        self.n_input = config.n_input
        self.n_out = config.n_out

        
        self.FA_linearErr = config.FA_linearErr if hasattr(config, 'FA_linearErr') else True

        # self.LRA_beta = config.LRA_beta       


        self.WinX = ( torch.rand(self.n_input, self.n_RI, device=self.device)*2-1) * self.sc_forward * 1.0/np.sqrt(self.n_input)  # in
        self.Win, self.Wb = [], []
        for i in range(self.nL_hidd-1):
            self.Win.append( (torch.rand(self.n_SI, self.n_RI, device=self.device)*2-1)* self.sc_forward * 1.0/np.sqrt(self.n_SI) )   
            self.Wb.append( (torch.rand(self.n_SE, self.n_RE, device=self.device)*2-1) * self.sc_back * 1.0/np.sqrt(self.n_SE) )   
        
        self.Wr, self.Wr_s, self.bias_rnn = [], [], []
        for i in range(self.nL_hidd):
            self.Wr.append(self.initialize_Wr().to(self.device))
            self.Wr_s.append(self.Wr[i] != 0)
            self.bias_rnn.append(torch.zeros(self.n_hidd, device=self.device))

        self.W_f = (torch.rand(self.n_SI, self.n_out, device=self.device)*2-1) * self.sc_forward_f * 1.0/np.sqrt(self.n_SI) 
        self.bias_f = (torch.rand(self.n_out, device=self.device)*2-1) * self.sc_bias * 1.0/np.sqrt(self.n_NperU) 
        self.Wb_f = (torch.rand(self.n_out, self.n_RE, device=self.device)*2-1)* self.sc_back_f * 1.0/np.sqrt(self.n_out) 

        # Adam
        self.opt_m_WinX = torch.zeros_like(self.WinX, device=self.device)
        self.opt_v_WinX = torch.zeros_like(self.WinX, device=self.device)

        self.opt_m_Wb_f = torch.zeros_like(self.Wb_f, device=self.device)
        self.opt_v_Wb_f = torch.zeros_like(self.Wb_f, device=self.device)

        self.opt_m_Win = [torch.zeros_like(w, device=self.device) for w in self.Win]
        self.opt_v_Win = [torch.zeros_like(w, device=self.device) for w in self.Win]

        self.opt_m_Wb = [torch.zeros_like(w, device=self.device) for w in self.Wb]
        self.opt_v_Wb = [torch.zeros_like(w, device=self.device) for w in self.Wb]

        self.opt_m_Wr = [torch.zeros_like(w, device=self.device) for w in self.Wr]
        self.opt_v_Wr = [torch.zeros_like(w, device=self.device) for w in self.Wr]

        self.opt_m_b = [torch.zeros_like(b, device=self.device) for b in self.bias_rnn]
        self.opt_v_b = [torch.zeros_like(b, device=self.device) for b in self.bias_rnn]

        self.opt_m_Wout = torch.zeros_like(self.W_f, device=self.device)
        self.opt_v_Wout = torch.zeros_like(self.W_f, device=self.device)
        self.opt_m_bias_f = torch.zeros_like(self.bias_f, device=self.device)
        self.opt_v_bias_f = torch.zeros_like(self.bias_f, device=self.device)

        self.opt_beta1 = 0.9
        self.opt_beta2 = 0.999
        self.opt_epsilon = 1e-8
        self.opt_eta = config.train_eta_global
        self.opt_t = 0

        # 
        self.z = [torch.zeros(self.train_batch_size, b.size(0), device=self.device) for b in self.bias_rnn]
        self.y = [torch.zeros(self.train_batch_size, b.size(0), device=self.device) for b in self.bias_rnn]
        self.e = [torch.zeros(self.train_batch_size, b.size(0), device=self.device) for b in self.bias_rnn]

        self.dWr = [torch.zeros_like(w, device=self.device) for w in self.Wr]
        self.dbias = [torch.zeros_like(b, device=self.device) for b in self.bias_rnn]

        self.dWin = [torch.zeros_like(w, device=self.device) for w in self.Win]
        self.dWb = [torch.zeros_like(w, device=self.device) for w in self.Wb]

        self.alpha = torch.ones([self.nL_hidd+1])
        if self.train_tmethod == 'AP':
            self.forward = self.forward_AP 
            self.backward = self.backward_AP
            self.factor_beta1 = config.factor_beta1 if hasattr(config, 'factor_beta1') else 1
            self.flag_feedbackLearning = config.flag_feedbackLearning if hasattr(config, 'flag_feedbackLearning') else False
            self.P_FNN  = config.P_FNN if hasattr(config, 'P_FNN') else False


    #
    def initialize_Wr(self):
        Wr = rand_sparse_matrix(self.n_hidd, self.n_hidd, self.RNN_CR).to_dense()
        Wr = Wr / torch.abs( torch.linalg.eigvals(Wr)) .max() * self.RNN_SR
        return Wr
    
    ##
    #
    def forward_AP(self, x):
        x = x.to(self.device)  
        self.z = [torch.zeros(x.size(0), b.size(0), device=self.device) for b in self.bias_rnn]
        self.cbias = [torch.zeros(x.size(0), b.size(0), device=self.device) for b in self.bias_rnn]

        for i in range(0, self.nL_hidd):
            if i==0:
                self.cbias[i][:, Aindx2Nrange(self.n_hidd, self.n_NU, 1)[:self.n_RI]] = x.mm(self.WinX) *self.ffsc
            else:
                self.cbias[i][:, Aindx2Nrange(self.n_hidd, self.n_NU, 1)[:self.n_RI]] = self.z[i-1][:, Aindx2Nrange(self.n_hidd, self.n_NU, 2)[:self.n_SI]].mm(self.Win[i-1]) *self.ffsc

            if self.P_FNN:
                self.z[i] = self.f( self.bias_rnn[i] + self.cbias[i] )
            else:
                self.z[i] = self.It2staFout(self.Wr[i], self.z[i], self.bias_rnn[i] + self.cbias[i], self.RNN_t2sta)

        
        self.zfh = self.z[-1][:, Aindx2Nrange(self.n_hidd, self.n_NU, 2)[:self.n_SI]].mm(self.W_f) + self.bias_f
        self.zf = softmax(self.zfh)
        return self.zf

    
    def backward_AP(self, x, y, output):
        x = x.to(self.device)
        y = y.to(self.device)
        output = output.to(self.device)
        
        #
        if self.flag_ycz == True : self.y = [z for z in self.z]
        else: self.y = [torch.zeros_like(z, device=self.device) for z in self.z]
        for i in range(self.nL_hidd-1, -1, -1):
            if i == self.nL_hidd-1:
                self.e_f = output - y  # dz3
                self.dWout = self.z[self.nL_hidd-1][:, Aindx2Nrange(self.n_hidd, self.n_NU, 2)[:self.n_SI]].t().mm(self.e_f) / self.train_batch_size
                self.dbias_f = self.e_f.sum(0) / self.train_batch_size
                self.cbias[self.nL_hidd-1][:, Aindx2Nrange(self.n_hidd, self.n_NU, 3)[:self.n_RE]] += -self.factor_beta1 * self.e_f.mm(self.Wb_f)
            else:
                self.dWin[i] = self.z[i][:, Aindx2Nrange(self.n_hidd, self.n_NU, 2)[:self.n_SI]].t().mm(self.e[i+1][:, Aindx2Nrange(self.n_hidd, self.n_NU, 1)[:self.n_RI]]) / self.train_batch_size
                self.cbias[i][:, Aindx2Nrange(self.n_hidd, self.n_NU, 3)[:self.n_RE]] += -self.factor_beta1 * self.e[i+1][:, Aindx2Nrange(self.n_hidd, self.n_NU, 4)[:self.n_SE]].mm(self.Wb[i])*self.fbsc
            
            if self.P_FNN:
                self.y[i] = self.f( self.bias_rnn[i] + self.cbias[i] )
            else:
                self.y[i] = self.It2staFout(self.Wr[i], self.y[i], self.bias_rnn[i] + self.cbias[i], self.RNN_t2sta2)

            self.e[i] = self.z[i] - self.y[i]

            self.dWr[i] = self.z[i].t().mm(self.e[i]) / self.train_batch_size
            self.dbias[i] = self.e[i].sum(0) / self.train_batch_size

        self.dWinX = x.t().mm(self.e[0][:, Aindx2Nrange(self.n_hidd, self.n_NU, 1)[:self.n_RI]]) / self.train_batch_size

        
    def ret_error(self):
        dim = self.nL_hidd*self.n_NperU +self.n_out        
        error = np.zeros(dim)
        for i in range(self.nL_hidd+1):
            if i == self.nL_hidd:
                error[self.nL_hidd*self.n_NperU :self.nL_hidd*self.n_NperU +self.n_out] = torch.mean(torch.abs(self.e_f),dim=0).cpu()
            else:
                error[i*self.n_NperU :(i+1)*self.n_NperU] = torch.mean(torch.abs(self.e[i][:, Aindx2Nrange(self.n_hidd, self.n_NU, 2)]),dim=0).cpu()

        return error
    
    ##    
    def update_weights_adam(self):
        self.opt_t += 1

        dWinX, self.opt_m_WinX, self.opt_v_WinX   = adam_update(self.opt_m_WinX, self.opt_v_WinX, self.dWinX, self.opt_beta1, self.opt_beta2, self.opt_t)
        self.WinX -= self.opt_eta * self.alpha[0] * dWinX #+ self.WinX * self.train_lambda1 

        for i in range(self.nL_hidd-2, -1, -1):
            dWin, self.opt_m_Win[i], self.opt_v_Win[i]      = adam_update(self.opt_m_Win[i], self.opt_v_Win[i],   self.dWin[i], self.opt_beta1, self.opt_beta2, self.opt_t)
            self.Win[i] -= self.opt_eta * self.alpha[i+1] * dWin #+ self.Win[i] * self.train_lambda1 


        dWout, self.opt_m_Wout, self.opt_v_Wout    = adam_update(self.opt_m_Wout, self.opt_v_Wout, self.dWout, self.opt_beta1, self.opt_beta2, self.opt_t)
        self.W_f -= self.opt_eta * self.alpha[-1] * dWout #+ self.W_f * self.train_lambda1 


        if self.flag_RNNLearning: 
            for i in range(0, self.nL_hidd):
                dWr, self.opt_m_Wr[i], self.opt_v_Wr[i]       = adam_update(self.opt_m_Wr[i], self.opt_v_Wr[i],     self.dWr[i], self.opt_beta1, self.opt_beta2, self.opt_t)
                self.Wr[i] -= self.opt_eta * dWr * self.Wr_s[i] * self.RNN_alpha1 + self.Wr[i] * self.train_lambda1 

            
        if self.flag_RNNBiasLearning:
            for i in range(0, self.nL_hidd):
                dbias, self.opt_m_b[i], self.opt_v_b[i]   = adam_update(self.opt_m_b[i], self.opt_v_b[i],       self.dbias[i], self.opt_beta1, self.opt_beta2, self.opt_t)
                self.bias_rnn[i] -= self.opt_eta * self.alpha[i] * dbias + self.bias_rnn[i] * self.train_lambda1

            dbias_f, self.opt_m_bias_f, self.opt_v_bias_f    = adam_update(self.opt_m_bias_f, self.opt_v_bias_f, self.dbias_f, self.opt_beta1, self.opt_beta2, self.opt_t)
            self.bias_f -= self.opt_eta * self.alpha[-1] * dbias_f #+ self.bias_f * self.train_lambda1 

        if self.flag_feedbackLearning: 
            for i in range(self.nL_hidd-2, -1, -1):
                dWb, self.opt_m_Wb[i], self.opt_v_Wb[i]    = adam_update(self.opt_m_Wb[i], self.opt_v_Wb[i], self.dWb[i], self.opt_beta1, self.opt_beta2, self.opt_t)
                self.Wb[i] += self.opt_eta * self.factor_gamma1  * dWb #+ self.Wb[i] * self.train_lambda1 
                
            dWb_f, self.opt_m_Wb_f, self.opt_v_Wb_f    = adam_update(self.opt_m_Wb_f, self.opt_v_Wb_f, self.dWb_f, self.opt_beta1, self.opt_beta2, self.opt_t)
            self.Wb_f += self.opt_eta * self.factor_gamma1 * dWb_f #+ self.Wb_f * self.train_lambda1 

        return self.Wr
    
    ##    
    def update_weights(self):
        self.opt_t += 1

        self.WinX -= self.opt_eta * self.alpha[0] * self.dWinX #+ self.WinX * self.train_lambda1 

        for i in range(self.nL_hidd-2, -1, -1):
            self.Win[i] -= self.opt_eta  * self.alpha[i+1] * self.dWin[i] #+ self.Win[i] * self.train_lambda1 


        self.W_f -= self.opt_eta  * self.alpha[-1] * self.dWout #+ self.W_f * self.train_lambda1 

        if self.flag_RNNLearning: 
            for i in range(0, self.nL_hidd):
                self.Wr[i] -= self.opt_eta * self.dWr[i] * self.Wr_s[i] * self.RNN_alpha1 + self.Wr[i] * self.train_lambda1 

            
        if self.flag_RNNBiasLearning:
            for i in range(0, self.nL_hidd):
                self.bias_rnn[i] -= self.opt_eta *  self.dbias[i] + self.bias_rnn[i] * self.train_lambda1

            self.bias_f -= self.opt_eta * self.dbias_f #+ self.bias_f * self.train_lambda1 

        if self.flag_feedbackLearning: 
            for i in range(self.nL_hidd-2, -1, -1):
                self.Wb[i] += self.opt_eta * self.factor_gamma1  * self.dWb[i] #+ self.Wb[i] * self.train_lambda1 
                
            self.Wb_f += self.opt_eta * self.factor_gamma1 * self.dWb_f #+ self.Wb_f * self.train_lambda1 

        return self.Wr
    

    def It2staFout(self, Wr, h, bias, itsta):
        # h = torch.zeros_like(bias,device=self.device)
        for indx in range(itsta):
            h = self.f(h.mm(Wr)+bias)
        
        return h
    
    def It2staFin(self, Wr, h, bias, itsta):
        # h = torch.zeros_like(bias,device=self.device)
        for indx in range(itsta):
            h = self.f(h).mm(Wr)+bias
        
        return h
    
    
    def It2stanF(self, Wr, h, bias, itsta):
        # h = torch.zeros_like(bias,device=self.device)
        for indx in range(itsta):
            h = (h.mm(Wr)+bias)
        
        return h
    


def rand_sparse_matrix(rows, cols, connection_rate):

    assert 0 < connection_rate <= 1, "The connection rate must be between (0,1]"

    # Calculate the number of non-zero elements
    num_elements = rows * cols
    num_nonzero = int(num_elements * connection_rate)

    # Randomly generate the position of non-zero elements
    row_indices = torch.randint(0, rows, (num_nonzero,))
    col_indices = torch.randint(0, cols, (num_nonzero,))

    # Stack row and column indices into a two-dimensional tensor
    indices = torch.stack((row_indices, col_indices))

    # Randomly generate values for non-zero elements
    values = torch.rand(num_nonzero)*2-1

    # Create Sparse Matrix
    sparse_matrix = torch.sparse_coo_tensor(indices, values, (rows, cols))

    return sparse_matrix


def Aindx2Nrange(numnodes, div, Aindx):
    node_indx = []
    nodes_per_div = numnodes // abs(div)
    if isinstance(Aindx, list) == False: Aindx = [Aindx]

    if div == 1: node_indx = range(numnodes)
    elif div == 2:
        for indx in Aindx:
            # Calculate the start and end indices for the specified division
            if (indx == 1) or (indx == 2): indx = 1 
            elif (indx == 3) or (indx == 4): indx = 2 
            start_idx = (indx-1) * nodes_per_div
            end_idx = indx * nodes_per_div - 1
            
            # Create the output range
            node_indx.extend(range(start_idx, end_idx + 1))
    elif div == -2:
        for indx in Aindx:
            # Calculate the start and end indices for the specified division
            if (indx == 1) or (indx == 4): indx = 1 
            elif (indx == 3) or (indx == 2): indx = 2 
            start_idx = (indx-1) * nodes_per_div
            end_idx = indx * nodes_per_div - 1
            
            # Create the output range
            node_indx.extend(range(start_idx, end_idx + 1))
    else :
        for indx in Aindx:
            # Calculate the start and end indices for the specified division
            start_idx = (indx-1) * nodes_per_div
            end_idx = indx * nodes_per_div - 1
            
            # Create the output range
            node_indx.extend(range(start_idx, end_idx + 1))


    return node_indx


def adam_update(m, v, dw, beta1, beta2, t, epsilon=1e-8):
    # Update first-order moment estimation
    m = beta1 * m + (1 - beta1) * dw
    v = beta2 * v + (1 - beta2) * (dw ** 2)
    
    # First and Second Order Moment Estimation for Deviation Correction Calculation
    m_corr = m / (1 - beta1 ** t)
    v_corr = v / (1 - beta2 ** t)
    
    # Calculate update value
    update = m_corr / (torch.sqrt(v_corr) + epsilon)
    
    return update, m, v

def linear(x):
    return x

def linear_d(x):
    return torch.ones_like(x)

def tanh(x):
    return torch.tanh(x)

def tanh_d(x):
    return 1 - torch.tanh(x) ** 2  

def sign(x):
    return torch.sign(x)

def sign_d(x):
    return torch.zeros_like(x)

def hard_sigmoid(x):
    return torch.clamp(x,0,1)

def hard_sigmoid_d(x):
    return torch.where((x >= 0) & (x <= 1), torch.ones_like(x), torch.zeros_like(x))

def sigmoid(x):
    return torch.sigmoid(x)

def sigmoid_d(x):
    return (1-torch.sigmoid(x)) * torch.sigmoid(x)

def relu(x):
    return torch.max(x, torch.zeros_like(x))  

def relu_d(x):
    return torch.where(x >= 0, torch.ones_like(x), torch.zeros_like(x))

def relu6(x):
    return torch.max(torch.min(x, torch.ones_like(x) * 6), torch.zeros_like(x))  

def relu6_d(x):
    return torch.where((x > 0) & (x <= 6), torch.ones_like(x), torch.zeros_like(x))

def softmax(x):
    x = x - torch.max(x, dim=1, keepdim=True)[0]  # Subtract the maximum value to prevent overflow
    return torch.exp(x) / torch.sum(torch.exp(x), dim=1, keepdim=True)



# One-hot 
def one_hot(labels, n_out):
    one_hot_labels = torch.zeros(labels.size(0), n_out)
    one_hot_labels[torch.arange(labels.size(0)), labels] = 1
    return one_hot_labels

def cross_entropy_loss(output, target):
    epsilon = 1e-8  # avoid log(0)
    output = torch.clamp(output, epsilon, 1. - epsilon)  # Limit the output to [epsilon, 1-epsilon] 
    return -torch.mean(torch.sum(target * torch.log(output), dim=1))
