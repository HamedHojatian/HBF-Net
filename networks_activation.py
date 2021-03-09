from networks_arch import Net_m_task_CNN
import torch.nn as nn
import torch

class Networks_activations(object):
    def __init__(self,
                 Us,
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
                 padding
                 ):
        self.Us = Us
        self.Mr = Mr
        self.Nrf = Nrf
        self.K = K
        self.K_limited = K_limited
        self.Noise_pwr = Noise_pwr
        self.device = device
        self.dev_id = device_ids
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output_reg = n_output_reg
        self.n_output_clas = n_output_clas
        self.p_dropout = p_dropout
        self.out_channel = out_channel
        self.kernel_s = kernel_s
        self.padding = padding

    def Network_m_Task(self):
        if self.device.type == 'cuda':
            return nn.DataParallel(Net_m_task_CNN(self.n_input, self.n_hidden, self.n_output_reg, self.n_output_clas, self.p_dropout, self.Us, self.K_limited, self.out_channel, self.kernel_s, self.padding), device_ids=self.dev_id).to(self.device)
        else:
            return Net_m_task_CNN(self.n_input, self.n_hidden, self.n_output_reg, self.n_output_clas,
                self.p_dropout, self.Us, self.K_limited, self.out_channel, self.kernel_s, self.padding)

    def Inp_MT(self, RSSI):
        Inputs_MT = RSSI.reshape(len(RSSI), 1, self.Us, self.K)[:, :, :, 0:self.K_limited].float().to(self.device)
        return Inputs_MT
