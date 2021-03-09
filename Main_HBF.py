import numpy as np
import os
import csv
import torch.nn as nn
import math
from numpy import genfromtxt
import torch as th
from networks_activation import Networks_activations
from utils import md_reader, Initialization_Model_Params, Loss_FDP_Rate_Based, Loss_HBF_Rate_Based_4D, FLP_loss
from utils_math import Th_pinv, Th_comp_matmul, Th_inv
from termcolor import colored
from torch.optim.lr_scheduler import ReduceLROnPlateau

###############################################################################
# Directory file
###############################################################################
DB_name = 'dataSet64x8x4_130dB_0129201820'

###############################################################################
# Processor selection GPU if available (using GPU is highly recommended)
###############################################################################
device = th.device("cuda:2" if th.cuda.is_available() else "cpu")
device_ids = [2, 1, 3]
print("Is Cuda available? ", colored('True', 'green')
    if th.cuda.is_available() else colored('False', 'red'))
print("Which devide?", colored(device, 'cyan'))

###############################################################################
# Setup Parameters
###############################################################################

# Beamforming approach  AFP_Net, HBF_NET   ####################################
BF_approach = 'HBF_Net'

###############################################################################
# Beamfroming system model and DNN Parameters
###############################################################################
os.chdir(os.path.dirname(os.path.abspath(__file__)))
Us, Mr, Nrf, K, Noise_pwr = md_reader(DB_name)                # Number of users, antenna, K, RF chains and noise power
K_limited = K                                                 # Number of SS as RSSI
batch_size = 500                                              # Batch size
epoch_size = 1000                                             # Number of training epoches
lr = 0.001                                                    # Learning rate
wd = 1e-6                                                     # Weight decay
n_input = Us * K_limited                                      # Input dimensions
n_hidden = 1024                                               # Size of FCL layers
out_channel = 16                                              # Size of CL channels
kernel_s = 3                                                  # Size of Kernels in CL
padding = 1                                                   # Size of padding in CL
p_dropout = 0.05                                              # Probability of dropout

if BF_approach == 'HBF_Net':
    n_output_reg = Us * Nrf
elif BF_approach == 'AFP_Net':
    n_output_reg = Us * Mr
else:
    raise Exception('BF_approach value is wrong !!')

###############################################################################
# Main Menu of configuration
###############################################################################
Main_Menu = Initialization_Model_Params(DB_name,
                                        Us,
                                        Mr,
                                        Nrf,
                                        K,
                                        K_limited,
                                        Noise_pwr,
                                        device,
                                        device_ids)

###############################################################################
# Reading Database
###############################################################################
DataBase, uniq_dis_label = Main_Menu.Data_Load()

###############################################################################
# Codeword dictionary
###############################################################################
codeword_C, n_output_clas, codesr, codesi = Main_Menu.Code_Read()

###############################################################################
# Training-set and test-set generation
###############################################################################
train_size = int(0.85 * len(DataBase))
test_size = len(DataBase) - train_size
train_dataset, test_dataset = th.utils.data.random_split(DataBase, [train_size, test_size])

print(colored('The size of training set is ', 'yellow'), len(train_dataset))
print(colored('The size of Test set is ', 'yellow'), len(test_dataset))

###############################################################################
# Dataloaders
###############################################################################
my_dataloader = th.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
my_testloader = th.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

###############################################################################
# DNN architecture parameters
###############################################################################
Networks_Main_Menu = Networks_activations(Us,
                                        Mr,
                                        Nrf,
                                        K,
                                        K_limited,
                                        Noise_pwr,
                                        device,
                                        device_ids,
                                        n_input,
                                        n_hidden,
                                        n_output_reg,
                                        n_output_clas,
                                        p_dropout,
                                        out_channel,
                                        kernel_s,
                                        padding)

Model_m_task = Networks_Main_Menu.Network_m_Task()

###############################################################################
# DNN OPTIMIZER
###############################################################################
optimizer_m_task = th.optim.Adam(Model_m_task.parameters(), lr=lr, weight_decay=wd)

###############################################################################
# scheduler lr
###############################################################################
scheduler_MT = ReduceLROnPlateau(optimizer_m_task, mode='max', factor=0.1, patience=5, verbose=True)

###############################################################################
# Main training loop
###############################################################################
if BF_approach == 'AFP_Net':
    # initialing the loss function
    criterium_clas_4d = Loss_HBF_Rate_Based_4D(Us, Mr, Nrf, Noise_pwr).to(device)
    criterium_reg = Loss_FDP_Rate_Based(Us, Mr, Nrf, Noise_pwr).to(device)
    for i in range(1, epoch_size):   # Main traning loop
        for k, (channelR, channelI, alpha, RSSI, UR, UI, AR, AI, index, WR, WI, deltaR, deltaI, userp) in enumerate(my_dataloader):  # Loading data from data loader

            # Input data dimension check (CNN)
            Inputs_MT = Networks_Main_Menu.Inp_MT(RSSI)

            # Loading the CSI (real and imaginary)
            channelR = channelR.view(-1, Us, Mr).to(device)
            channelI = channelI.view(-1, Us, Mr).to(device)

            # Set gradient to 0.
            optimizer_m_task.zero_grad()

            # Feed forward multi-tasking DNN
            Model_m_task.train()
            out1_reg, out2_reg, out_clas = Model_m_task(Inputs_MT)

            # Computing loss for FDP in AFP-Net eq(27) in the paper
            loss_reg = criterium_reg(out1_reg, out2_reg, channelR, channelI)

            # computing the loss fucntion for HBF using eq(20)
            xx_pr, xx_pi = Th_pinv(th.unsqueeze(codesr.unsqueeze(1), 2).repeat(1, len(RSSI), 1, 1).view(-1, len(RSSI), Nrf, Mr).to(device),
                th.unsqueeze(codesi.unsqueeze(1), 2).repeat(1, len(RSSI), 1, 1).view(-1, len(RSSI), Nrf, Mr).to(device))
            w_outr, w_outi = Th_comp_matmul(out1_reg.view(-1, Us, Mr), out2_reg.view(-1, Us, Mr), xx_pr, xx_pi)

            HBF_all_4d = criterium_clas_4d(w_outr.permute(0, 1, 3, 2), w_outi.permute(0, 1, 3, 2), channelR, channelI,
                th.unsqueeze(codesr.unsqueeze(1), 2).repeat(1, len(RSSI), 1, 1).to(device), th.unsqueeze(codesi.unsqueeze(1), 2).repeat(1, len(RSSI), 1, 1).to(device))

            loss_clas = FLP_loss(out_clas, HBF_all_4d)

            # total loss fucntion eq(29)
            loss = loss_clas + loss_reg

            # Gradient calculation.
            loss_clas.backward(retain_graph=True)
            loss_reg.backward(retain_graph=True)
            loss.backward()

            # Model weight modification based on the optimizer.
            optimizer_m_task.step()

            # iterate through test dataset
            if k == 0 or i % epoch_size == 0:
                del loss
                # No gardient in test mode
                with th.no_grad():
                    R_predicted_HBF = []
                    R_optimum_HBF = []
                    R_optimum_FDP = []
                    R_predicted_FDP = []
                    Rate_Ratio_HBF = []
                    Rate_Ratio_FDP = []
                    for (tchannelR, tchannelI, talpha, tRSSI, tUR, tUI, tAR, tAI, tindex, tWR, tWI, tdeltaR, tdeltaI, tup) in my_testloader:

                        # Input data dimension check (CNN)
                        testInputs_Reg = Networks_Main_Menu.Inp_MT(tRSSI)

                        # Loading the near-optimal digital precoder, CSI (real and imaginary)
                        T_wR = tWR.reshape(-1, Us, Nrf).to(device)
                        T_wI = tWI.reshape(-1, Us, Nrf).to(device)
                        T_channelR = tchannelR.reshape(-1, Us, Mr).to(device)
                        T_channelI = tchannelI.reshape(-1, Us, Mr).to(device)

                        # Forward pass test mode DNN
                        Model_m_task.eval()
                        pred1_reg, pred2_reg, pred_class = Model_m_task(testInputs_Reg)

                        # find the maximum probability as predication of classification
                        _, predicted = th.max(pred_class, 1)

                        # mapping in the codebook to find the corresponding analog precoder
                        An_Predr = codesr[predicted, :].to(device)
                        An_Predi = codesi[predicted, :].to(device)

                        # finding digital precoder using eq(20)
                        x_pr, x_pi = Th_pinv(An_Predr.view(-1, Nrf, Mr), An_Predi.view(-1, Nrf, Mr))
                        w_prer, w_prei = Th_comp_matmul(pred1_reg.view(-1, Us, Mr), pred2_reg.view(-1, Us, Mr), x_pr, x_pi)

                        # rate calculation
                        # DNN HBF
                        R_predicted_HBF.append(criterium_clas_4d.evaluate_rate(w_prer, w_prei, T_channelR, T_channelI, An_Predr, An_Predi))
                        # near-optimal HBF
                        R_optimum_HBF.append(criterium_clas_4d.evaluate_rate(T_wR, T_wI, T_channelR, T_channelI, tAR.to(device), tAI.to(device)))
                        # DNN FDP
                        R_predicted_FDP.append(criterium_reg.evaluate_rate(pred1_reg, pred2_reg, T_channelR, T_channelI))
                        # near-optimal HBF
                        R_optimum_FDP.append(criterium_reg.evaluate_rate(tUR.to(device), tUI.to(device), T_channelR, T_channelI))

                # Average over all mini-batches
                RATE_Predicted_HBF = sum(R_predicted_HBF) / len(R_predicted_HBF)
                RATE_Predicted_FDP = sum(R_predicted_FDP) / len(R_predicted_FDP)
                RATE_Optimum_HBF = sum(R_optimum_HBF) / len(R_optimum_HBF)
                RATE_Optimum_FDP = sum(R_optimum_FDP) / len(R_optimum_FDP)
                RATE_Ratie_HBF = 100 * RATE_Predicted_HBF / RATE_Optimum_HBF
                RATE_Ratie_FDP = 100 * RATE_Predicted_FDP / RATE_Optimum_FDP

                scheduler_MT.step(RATE_Predicted_HBF)

                print('Iter:==>{:3d} Loss_FDP:{:.3f} Loss_Class:{:.3f} Rate_opt_HBF:{:.2f} Rate_opt_FDP:{:.2f} Rate_pre_HBF:{:.2f} Rate_pre_FDP:{:.2f} Ratio_HBF:{:.2f}% Ratio_FDP:{:.2f}%'.
                    format(i, loss_reg, loss_clas, RATE_Optimum_HBF, RATE_Optimum_FDP, RATE_Predicted_HBF, RATE_Predicted_FDP, RATE_Ratie_HBF, RATE_Ratie_FDP))

elif BF_approach == 'HBF_Net':
    # initialing the loss function
    criterium_clas_4d = Loss_HBF_Rate_Based_4D(Us, Mr, Nrf, Noise_pwr).to(device)
    for i in range(1, epoch_size):
        for k, (channelR, channelI, alpha, RSSI, UR, UI, AR, AI, index, WR, WI, deltaR, deltaI, userp) in enumerate(my_dataloader):

            # Input data dimension check (CNN)
            Inputs_Reg = Networks_Main_Menu.Inp_MT(RSSI)

            # Loading the CSI (real and imaginary)
            channelR = channelR.view(-1, Us, Mr).to(device)
            channelI = channelI.view(-1, Us, Mr).to(device)

            # Set gradient to 0.
            optimizer_m_task.zero_grad()

            # Feed forward multi-tasking DNN
            Model_m_task.train()
            out1_reg, out2_reg, out_clas = Model_m_task(Inputs_Reg)

            # computing the loss fucntion for HBF using eq(25)
            w_outr, w_outi = out1_reg.view(-1, Us, Nrf), out2_reg.view(-1, Us, Nrf)
            HBF_all_4d = criterium_clas_4d(w_outr.permute(0, 2, 1), w_outi.permute(0, 2, 1), channelR, channelI,
                th.unsqueeze(codesr.unsqueeze(1), 2).repeat(1, len(RSSI), 1, 1).to(device),
                th.unsqueeze(codesi.unsqueeze(1), 2).repeat(1, len(RSSI), 1, 1).to(device))
            loss_clas = FLP_loss(out_clas, HBF_all_4d)

            # Gradient calculation.
            loss_clas.backward()

            # Model weight modification based on the optimizer.
            optimizer_m_task.step()

            # iterate through test dataset
            if k == 0 or i % epoch_size == 0:
                R_predicted_HBF = []
                R_optimum_HBF = []
                Rate_Ratio_HBF = []
                with th.no_grad():
                    for (tchannelR, tchannelI, talpha, tRSSI, tUR, tUI, tAR, tAI, tindex, tWR, tWI, tdeltaR, tdeltaI, tup) in my_testloader:

                        # Input data dimension check (CNN)
                        testInputs_Reg = Networks_Main_Menu.Inp_MT(tRSSI)

                        # Loading the near-optimal digital precoder, CSI (real and imaginary)
                        T_wR = tWR.reshape(-1, Us, Nrf).to(device)
                        T_wI = tWI.reshape(-1, Us, Nrf).to(device)
                        T_channelR = tchannelR.reshape(-1, Us, Mr).to(device)
                        T_channelI = tchannelI.reshape(-1, Us, Mr).to(device)

                        # Forward pass reg
                        Model_m_task.eval()
                        pred1_reg, pred2_reg, pred_class = Model_m_task(testInputs_Reg)

                        # find the maximum probability as predication of classification
                        _, predicted = th.max(pred_class, 1)

                        # mapping in the codebook to find the corresponding analog precoder
                        An_Predr = codesr[predicted, :].to(device)
                        An_Predi = codesi[predicted, :].to(device)
                        w_prer, w_prei = pred1_reg.view(-1, Us, Nrf), pred2_reg.view(-1, Us, Nrf)

                        # rate calculation
                        # DNN HBF
                        R_predicted_HBF.append(criterium_clas_4d.evaluate_rate(w_prer, w_prei, T_channelR, T_channelI, An_Predr, An_Predi))
                        # near-optimal HBF
                        R_optimum_HBF.append(criterium_clas_4d.evaluate_rate(T_wR, T_wI, T_channelR, T_channelI, tAR.to(device), tAI.to(device)))

                # Average over all mini-batches
                RATE_Predicted_HBF = sum(R_predicted_HBF) / len(R_predicted_HBF)
                RATE_Optimum_HBF = sum(R_optimum_HBF) / len(R_optimum_HBF)
                RATE_Ratie_HBF = 100 * RATE_Predicted_HBF / RATE_Optimum_HBF

                scheduler_MT.step(RATE_Predicted_HBF)

                print('Iter:==>{:3d} Loss_Class:{:.3f} Rate_opt_HBF:{:.2f} Rate_pre_HBF:{:.2f} Ratio_HBF:{:.2f}%'.
                    format(i, loss_clas, RATE_Optimum_HBF, RATE_Predicted_HBF, RATE_Ratie_HBF))

else:
    raise Exception('BF_approach is wrong !!')
