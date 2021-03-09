import torch.nn as nn
import torch

class Net_m_task_CNN(nn.Module):
    def __init__(self, n_in, n_hidden, n_out_Reg, n_out_clas, p_dropout, U, Ass, out_channel, kernel_s, padding):
        super(Net_m_task_CNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out_channel, kernel_size=kernel_s, stride=1, padding=padding)
        self.relu1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(num_features=out_channel)
        self.do1 = nn.Dropout2d(p_dropout)

        self.cnn2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel_s, stride=1, padding=padding)
        self.relu2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(num_features=out_channel)
        self.do2 = nn.Dropout2d(p_dropout)

        self.cnn3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel/2, kernel_size=kernel_s, stride=1, padding=padding)
        self.relu3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(num_features=out_channel/2)
        self.do3 = nn.Dropout2d(p_dropout)

        # Fully connected 1 (readout)
        x_new = (U + 2 * padding - kernel_s) + 1
        y_new = (Ass + 2 * padding - kernel_s) + 1

        nn_in_fc = 8 * (x_new + 2 * padding - kernel_s + 1) * (y_new + 2 * padding - kernel_s + 1)

        self.fc20 = nn.Linear(nn_in_fc, n_hidden)
        self.bn20 = nn.BatchNorm1d(n_hidden)
        self.relu20 = nn.LeakyReLU()
        self.do20 = nn.Dropout(p_dropout)

        self.fc30 = nn.Linear(n_hidden, n_hidden)
        self.bn30 = nn.BatchNorm1d(n_hidden)
        self.relu30 = nn.LeakyReLU()
        self.do30 = nn.Dropout(p_dropout)

        # Linear function (readout)
        self.fc7R = nn.Linear(n_hidden, n_out_Reg)

        # Linear function (readout)
        self.fc8R = nn.Linear(n_hidden, n_out_Reg)

        # Linear function (readout)
        self.fc9C = nn.Linear(n_hidden, n_out_clas)

    def forward(self, x):  # always

        out = self.cnn1(x)
        out = self.do1(out)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.cnn2(out)
        out = self.do2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.cnn3(out)
        out = self.do3(out)
        out = self.bn3(out)
        out = self.relu3(out)

        out = out.view(out.size(0), -1)

        out = self.fc20(out)
        out = self.do20(out)
        out = self.relu20(out)
        out = self.bn20(out)

        out = self.fc30(out)
        out = self.do30(out)
        out = self.relu30(out)
        out = self.bn30(out)

        # Linear function (readout)  ****** LINEAR ******
        outR1 = self.fc7R(out)

        # Linear function (readout)  ****** LINEAR ******
        outR2 = self.fc8R(out)

        # Linear function (readout)  ****** LINEAR ******
        outC = self.fc9C(out)

        return outR1, outR2, outC
