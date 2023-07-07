import torch.utils.data as data
from termcolor import colored
import torch.nn.functional as F
import torch
from numpy import genfromtxt
import numpy as np
from utils_math import Th_comp_matmul, Th_inv, Th_pinv
import neptune
import re
import torch.nn as nn
import time

# Database ####################################################################################################################
class Data_Reader(data.Dataset):
    def __init__(self, filename, Us, Mr, Nrf, K, Noise_pwr):

        print(colored('You select core dataset', 'cyan'))
        print(colored(filename, 'yellow'), 'is loading ... ')
        np_data = np.load(filename)

        self.channelR = np_data['channel'].real.astype(float)
        self.channelI = np_data['channel'].imag.astype(float)

        self.RSSI_N = np_data['RSSI_N'].real.astype(float)

        self.UR = np_data['U'].real.astype(float)
        self.UI = np_data['U'].imag.astype(float)

        self.AR = np_data['A'].real.astype(float)
        self.AI = np_data['A'].imag.astype(float)

        self.WR = np_data['W'].real.astype(float)
        self.WI = np_data['W'].imag.astype(float)

        self.Noise_pwr = Noise_pwr
        
        self.n_samples = self.channelR.shape[0]

    def __len__(self):
        return self.n_samples

    def uniq_clas(self):
        codes = np.load('Codebook_ij.npz')['codebook']
        NO_Class = len(codes)
        print(colored("The number of Unique AP in I1: ", "green"), NO_Class)
        return NO_Class
    
    def rate_calculator_3d_np(self, FDP, channel):  #      FDP = (i, Nt, Nu, 1)         H = (i, Nu, 1, Nt)
        W = np.einsum('nij,njk->nik', np.conj(channel), FDP)
        diag_W = np.diagonal(np.abs(W) ** 2, axis1=1, axis2=2)
        SINR = diag_W / (np.sum(np.abs(W) ** 2, 2) - diag_W + self.Noise_pwr)
        userRates = np.log2(1 + SINR)
        sumRate = userRates.sum(1)
        return sumRate
    
    def optimum_HBF(self):
        A = self.AR + 1j*self.AI
        W = self.WR + 1j*self.WI
        channel = self.channelR + 1j*self.channelI
        FDP_AW = np.einsum('nij,njk->nik', A, W)
        FDP_AW = FDP_AW / np.linalg.norm(FDP_AW, axis=(1,2), keepdims=True)
        sr_HBF = Data_Reader.rate_calculator_3d_np(self, FDP_AW, channel)
        return sr_HBF.mean()
    
    def optimum_FDP(self):
        U = self.UR + 1j*self.UI
        channel = self.channelR + 1j*self.channelI
        U = U / np.linalg.norm(U, axis=(1,2), keepdims=True)
        sr_FDP = Data_Reader.rate_calculator_3d_np(self, U, channel)
        return sr_FDP.mean()

    def __getitem__(self, index):
        return torch.Tensor(self.channelR[index]), torch.Tensor(self.channelI[index]), torch.Tensor(self.RSSI_N[index])

# readme reader for HBF initial parameters ####################################################################################
def md_reader(DB_name):
    md = genfromtxt('DATASET.md', delimiter='\n', dtype='str')
    Us = int(re.findall(r'\d+', md[1])[0])
    Mr = int(re.findall(r'\d+', md[2])[0])
    Nrf = int(re.findall(r'\d+', md[3])[0])
    Ass_n = int(re.findall(r'\d+', md[4])[0])
    Noise_pwr = float(''.join(('1e-', str(int(int(re.findall(r'\d+', md[6])[0]) / 10)))))
    return Us, Mr, Nrf, Ass_n, Noise_pwr

class Initialization_Model_Params(object):
    def __init__(self,
                 DB_name,
                 Us,
                 Mr,
                 Nrf,
                 K,
                 K_limited,
                 Noise_pwr,
                 device,
                 device_ids
                 ):
        self.DB_name = DB_name
        self.Us = Us
        self.Mr = Mr
        self.Nrf = Nrf
        self.K = K
        self.K_limited = K_limited
        self.Noise_pwr = Noise_pwr
        self.device = device
        self.dev_id = device_ids

    def Data_Load(self):
        DataBase = Data_Reader(''.join(('DataBase_', self.DB_name, '.npz')),
                               self.Us, self.Mr, self.Nrf, self.K, self.Noise_pwr)
        uniq_dis_label = DataBase.uniq_clas()
        sr_HBF, sr_FDP = DataBase.optimum_HBF(), DataBase.optimum_FDP()
        return DataBase, uniq_dis_label, sr_HBF, sr_FDP

    def Code_Read(self):
        codes = np.load('Codebook_ij.npz')['codebook']
        # codes = genfromtxt('Codebook_ij.csv', delimiter=',', dtype='complex', skip_header=0)
        label = np.arange(len(codes))
        self.n_output_clas = len(codes)
        print(colored("The length of the codebook: ", "green"), len(codes))
        Codes_idx = np.concatenate((label[:, np.newaxis], codes), axis=1)
        codeword_C = {}
        index_C = []
        for i in range(len(codes)):
            index_C = Codes_idx[i, 0].real.astype(int)
            icode_C = Codes_idx[i, 1:]
            codeword_C[index_C] = icode_C

        # torch tensor of codes
        codesr = torch.from_numpy(codes.real).type(torch.float)
        codesi = torch.from_numpy(codes.imag).type(torch.float)
        return codeword_C, len(codes), codesr, codesi

class Loss_FDP_Rate_Based(torch.nn.Module):
    def __init__(self, Us, Mr, Nrf, Noise_pwr):
        super(Loss_FDP_Rate_Based, self).__init__()
        self.Us = Us
        self.Mr = Mr
        self.Nrf = Nrf
        self.noise_power = Noise_pwr

    def rate_calculator(self, u_re, u_im, channelr, channeli):
        Wr, Wi = Th_comp_matmul(channelr, -channeli, u_re, u_im)
        W = Wr**2 + Wi**2
        diag_W = torch.diagonal(W, dim1=1, dim2=2)

        SINR = diag_W / (torch.sum(W, 2) - diag_W + self.noise_power)
        userRates = torch.log2(1 + SINR)
        sumRate = userRates.sum(1)

        return sumRate

    def forward(self, outr, outi, channelr, channeli):
        outr = outr.view(-1, self.Us, self.Mr).permute(0, 2, 1)
        outi = outi.view(-1, self.Us, self.Mr).permute(0, 2, 1)

        # power normalization over all antennas
        temp_pre = torch.sqrt(torch.sum(outr.flatten(1) ** 2 + outi.flatten(1) ** 2, dim=1))
        outr = (outr.flatten(1) / temp_pre.unsqueeze(1)).view(outr.shape)
        outi = (outi.flatten(1) / temp_pre.unsqueeze(1)).view(outi.shape)

        sum_rate = Loss_FDP_Rate_Based.rate_calculator(self, outr, outi, channelr, channeli)
        return -sum_rate.mean()

    def evaluate_rate(self, outr, outi, channelr, channeli):
        outr = outr.view(-1, self.Us, self.Mr).permute(0, 2, 1)
        outi = outi.view(-1, self.Us, self.Mr).permute(0, 2, 1)

        # power normalization over all antennas
        temp_pre = torch.sqrt(torch.sum(outr.flatten(1) ** 2 + outi.flatten(1) ** 2, dim=1))
        outr = (outr.flatten(1) / temp_pre.unsqueeze(1)).view(outr.shape)
        outi = (outi.flatten(1) / temp_pre.unsqueeze(1)).view(outi.shape)

        sum_rate = Loss_FDP_Rate_Based.rate_calculator(self, outr, outi, channelr, channeli)
        return sum_rate.mean()

class Loss_HBF_Rate_Based_4D(torch.nn.Module):
    def __init__(self, Us, Mr, Nrf, Noise_pwr):
        super(Loss_HBF_Rate_Based_4D, self).__init__()
        self.Us = Us
        self.Mr = Mr
        self.Nrf = Nrf
        self.noise_power = Noise_pwr

    def rate_calculator_4d(self, u_re, u_im, channelr, channeli):
        Wr, Wi = Th_comp_matmul(channelr, -channeli, u_re, u_im)
        W = Wr**2 + Wi**2
        diag_W = torch.diagonal(W, dim1=2, dim2=3)
        SINR = diag_W / (torch.sum(W, 3) - diag_W + self.noise_power)
        userRates = torch.log2(1 + SINR)
        sumRate = userRates.sum(2)
        return sumRate

    def forward(self, Wr, Wi, channelr, channeli, Ar, Ai):
        HBF_prer, HBF_prei = Th_comp_matmul(Ar.view(-1, len(channelr), self.Nrf, self.Mr).permute(0, 1, 3, 2),
                                            Ai.view(-1, len(channelr), self.Nrf, self.Mr).permute(0, 1, 3, 2), Wr, Wi)

        # power normalization over all antennas
        temp_pre = torch.sqrt(torch.sum(HBF_prer.flatten(2) ** 2 + HBF_prei.flatten(2) ** 2, dim=2))
        HBF_prer = (HBF_prer.flatten(2) / temp_pre.unsqueeze(2)).view(HBF_prer.shape)
        HBF_prei = (HBF_prei.flatten(2) / temp_pre.unsqueeze(2)).view(HBF_prei.shape)

        sum_rate = Loss_HBF_Rate_Based_4D.rate_calculator_4d(self, HBF_prer, HBF_prei, channelr, channeli)
        return sum_rate.T

    def evaluate_rate(self, Wr, Wi, channelr, channeli, Ar, Ai):
        HBF_prer, HBF_prei = Th_comp_matmul(Ar.view(-1, self.Nrf, self.Mr).permute(0, 2, 1),
            Ai.view(-1, self.Nrf, self.Mr).permute(0, 2, 1), Wr.permute(0, 2, 1), Wi.permute(0, 2, 1))

        # power normalization over all antennas
        temp_pre = torch.sqrt(torch.sum(HBF_prer.flatten(1) ** 2 + HBF_prei.flatten(1) ** 2, dim=1))
        HBF_prer = (HBF_prer.flatten(1) / temp_pre.unsqueeze(1)).view(HBF_prer.shape)
        HBF_prei = (HBF_prei.flatten(1) / temp_pre.unsqueeze(1)).view(HBF_prei.shape)

        sum_rate = Loss_FDP_Rate_Based.rate_calculator(self, HBF_prer, HBF_prei, channelr, channeli)

        return sum_rate.mean()

def FLP_loss(x, y):
    log_prob = - 1.0 * F.softmax(x, 1)
    temp = log_prob * y
    cel = temp.sum(dim=1)
    cel = cel.mean()
    return cel
