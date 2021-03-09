import torch
from numpy import genfromtxt
import numpy as np

def Th_comp_matmul(Ar, Ai, Br, Bi):  # Complex matmul pytorch function ########
    if Ar.ndim == 3 and Br.ndim == 3:
        a_th = torch.cat((torch.cat((Ar, -Ai), dim=2), torch.cat((Ai, Ar), dim=2)), dim=1)
        b_th = torch.cat((torch.cat((Br, -Bi), dim=2), torch.cat((Bi, Br), dim=2)), dim=1)
        c_th = torch.matmul(a_th, b_th)
        c_th_r = c_th[:, 0:int(c_th.shape[1] / 2), 0:int(c_th.shape[2] / 2)]
        c_th_i = c_th[:, int(c_th.shape[1] / 2):, 0:int(c_th.shape[2] / 2)]
    elif Ar.ndim == 2 and Br.ndim == 2:
        a_th = torch.cat((torch.cat((Ar, -Ai), dim=1), torch.cat((Ai, Ar), dim=1)), dim=0)
        b_th = torch.cat((torch.cat((Br, -Bi), dim=1), torch.cat((Bi, Br), dim=1)), dim=0)
        c_th = torch.matmul(a_th, b_th)
        c_th_r = c_th[0:int(c_th.shape[0] / 2), 0:int(c_th.shape[1] / 2)]
        c_th_i = c_th[int(c_th.shape[0] / 2):, 0:int(c_th.shape[1] / 2)]
    elif Ar.ndim == 4 and Br.ndim == 4:
        a_th = torch.cat((torch.cat((Ar, -Ai), dim=3), torch.cat((Ai, Ar), dim=3)), dim=2)
        b_th = torch.cat((torch.cat((Br, -Bi), dim=3), torch.cat((Bi, Br), dim=3)), dim=2)
        c_th = torch.matmul(a_th, b_th)
        c_th_r = c_th[:, :, 0:int(c_th.shape[2] / 2), 0:int(c_th.shape[3] / 2)]
        c_th_i = c_th[:, :, int(c_th.shape[2] / 2):, 0:int(c_th.shape[3] / 2)]
    elif Ar.ndim == 5 and Br.ndim == 5:
        a_th = torch.cat((torch.cat((Ar, -Ai), dim=4), torch.cat((Ai, Ar), dim=4)), dim=3)
        b_th = torch.cat((torch.cat((Br, -Bi), dim=4), torch.cat((Bi, Br), dim=4)), dim=3)
        c_th = torch.matmul(a_th, b_th)
        c_th_r = c_th[:, :, :, 0:int(c_th.shape[3] / 2), 0:int(c_th.shape[4] / 2)]
        c_th_i = c_th[:, :, :, int(c_th.shape[3] / 2):, 0:int(c_th.shape[4] / 2)]
    elif Ar.ndim * Br.ndim == 12:
        if Ar.ndim == 4:
            a_th = torch.cat((torch.cat((Ar, -Ai), dim=3), torch.cat((Ai, Ar), dim=3)), dim=2)
            b_th = torch.cat((torch.cat((Br, -Bi), dim=2), torch.cat((Bi, Br), dim=2)), dim=1)
            c_th = torch.matmul(a_th, b_th)
            c_th_r = c_th[:, :, 0:int(c_th.shape[2] / 2), 0:int(c_th.shape[3] / 2)]
            c_th_i = c_th[:, :, int(c_th.shape[2] / 2):, 0:int(c_th.shape[3] / 2)]
        elif Br.ndim == 4:
            a_th = torch.cat((torch.cat((Ar, -Ai), dim=2), torch.cat((Ai, Ar), dim=2)), dim=1)
            b_th = torch.cat((torch.cat((Br, -Bi), dim=3), torch.cat((Bi, Br), dim=3)), dim=2)
            c_th = torch.matmul(a_th, b_th)
            c_th_r = c_th[:, :, 0:int(c_th.shape[2] / 2), 0:int(c_th.shape[3] / 2)]
            c_th_i = c_th[:, :, int(c_th.shape[2] / 2):, 0:int(c_th.shape[3] / 2)]
    elif Ar.ndim * Br.ndim == 20:
        if Ar.ndim == 5:
            a_th = torch.cat((torch.cat((Ar, -Ai), dim=4), torch.cat((Ai, Ar), dim=4)), dim=3)
            b_th = torch.cat((torch.cat((Br, -Bi), dim=3), torch.cat((Bi, Br), dim=3)), dim=2)
            c_th = torch.matmul(a_th, b_th)
            c_th_r = c_th[:, :, :, 0:int(c_th.shape[3] / 2), 0:int(c_th.shape[4] / 2)]
            c_th_i = c_th[:, :, :, int(c_th.shape[3] / 2):, 0:int(c_th.shape[4] / 2)]
        elif Br.ndim == 5:
            a_th = torch.cat((torch.cat((Ar, -Ai), dim=3), torch.cat((Ai, Ar), dim=3)), dim=2)
            b_th = torch.cat((torch.cat((Br, -Bi), dim=4), torch.cat((Bi, Br), dim=4)), dim=3)
            c_th = torch.matmul(a_th, b_th)
            c_th_r = c_th[:, :, :, 0:int(c_th.shape[3] / 2), 0:int(c_th.shape[4] / 2)]
            c_th_i = c_th[:, :, :, int(c_th.shape[3] / 2):, 0:int(c_th.shape[4] / 2)]
    else:
        raise Exception('the dimension is not defined for Th_comp_matmul.')

    return c_th_r, c_th_i

def Th_inv(Ar, Ai):  # Complex inverse pytorch function ########
    Ar_inv = torch.inverse(Ar + torch.matmul(torch.matmul(Ai, torch.inverse(Ar)), Ai))
    Ai_inv = - torch.matmul(torch.matmul(torch.inverse(Ar), Ai), Ar_inv)
    return Ar_inv, Ai_inv

def Th_pinv(Ar, Ai):  # Complex inverse pytorch function ########
    if Ar.ndim == 2:
        if Ar.shape[0] < Ar.shape[1]:
            Tempr, Tempi = Th_comp_matmul(Ar, Ai, Ar.T, -Ai.T)
            Ar_inv, Ai_inv = Th_inv(Tempr, Tempi)
            return Th_comp_matmul(Ar.T, -Ai.T, Ar_inv, Ai_inv)
        elif Ar.shape[0] > Ar.shape[1]:
            Tempr, Tempi = Th_comp_matmul(Ar.T, -Ai.T, Ar, Ai)
            Ar_inv, Ai_inv = Th_inv(Tempr, Tempi)
            return Th_comp_matmul(Ar_inv, Ai_inv, Ar.T, -Ai.T)
        elif Ar.shape[0] == Ar.shape[1]:
            return Th_inv(Ar, Ai)
    elif Ar.ndim == 3:
        if Ar.shape[1] < Ar.shape[2]:
            Tempr, Tempi = Th_comp_matmul(Ar, Ai, Ar.permute(0, 2, 1), -Ai.permute(0, 2, 1))
            Ar_inv, Ai_inv = Th_inv(Tempr, Tempi)
            return Th_comp_matmul(Ar.permute(0, 2, 1), -Ai.permute(0, 2, 1), Ar_inv, Ai_inv)
        elif Ar.shape[1] > Ar.shape[2]:
            Tempr, Tempi = Th_comp_matmul(Ar.permute(0, 2, 1), -Ai.permute(0, 2, 1), Ar, Ai)
            Ar_inv, Ai_inv = Th_inv(Tempr, Tempi)
            return Th_comp_matmul(Ar_inv, Ai_inv, Ar.permute(0, 2, 1), -Ai.permute(0, 2, 1))
        elif Ar.shape[1] == Ar.shape[2]:
            return Th_inv(Ar, Ai)
    elif Ar.ndim == 4:
        if Ar.shape[2] < Ar.shape[3]:
            Tempr, Tempi = Th_comp_matmul(Ar, Ai, Ar.permute(0, 1, 3, 2), -Ai.permute(0, 1, 3, 2))
            Ar_inv, Ai_inv = Th_inv(Tempr, Tempi)
            return Th_comp_matmul(Ar.permute(0, 1, 3, 2), -Ai.permute(0, 1, 3, 2), Ar_inv, Ai_inv)
        elif Ar.shape[2] > Ar.shape[3]:
            Tempr, Tempi = Th_comp_matmul(Ar.permute(0, 1, 3, 2), -Ai.permute(0, 1, 3, 2), Ar, Ai)
            Ar_inv, Ai_inv = Th_inv(Tempr, Tempi)
            return Th_comp_matmul(Ar_inv, Ai_inv, Ar.permute(0, 1, 3, 2), -Ai.permute(0, 1, 3, 2))
        elif Ar.shape[2] == Ar.shape[3]:
            return Th_inv(Ar, Ai)
    else:
        raise Exception('5-D is not defined for Th_pinv.')
