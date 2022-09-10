import torch
import numpy as np
import torch.nn as nn

def tesseract(device, param_list, net, old_direction, susp, cmax):
    flip_local = torch.zeros(len(param_list)).to(device)
    penalty = 1.0 - 2 * cmax / len(param_list)
    reward = 1.0 - penalty

    for i in range(len(param_list)):
        direction = torch.sign(param_list[i])
        flip = torch.sign(direction * (direction - old_direction.reshape(-1)))
        flip_local[i] = torch.sum(flip * (param_list[i] ** 2))
        del direction, flip
    argsorted = torch.argsort(flip_local).to(device)
    if (cmax > 0):
        susp[argsorted[cmax:-cmax]] = susp[argsorted[cmax:-cmax]] + reward
        susp[argsorted[:cmax]] = susp[argsorted[:cmax]] - penalty
        susp[argsorted[-cmax:]] = susp[argsorted[-cmax:]] - penalty
    weights = torch.exp(susp) / torch.sum(torch.exp(susp))
    global_params = torch.matmul(torch.transpose(param_list, 0, 1), weights.reshape(-1, 1))
    global_direction = torch.sign(global_params)

    with torch.no_grad():
        idx = 0
        for j, (param) in enumerate(net.named_parameters()):
            if param[1].requires_grad:
                param[1].data += global_params[idx:(idx + param[1].nelement())].reshape(param[1].shape)
                idx += param[1].nelement()
    del param_list, global_params

    return net, global_direction, susp

def FEDSGD(device, param_list, net, wts):

    global_params = torch.matmul(torch.transpose(param_list, 0, 1), wts.reshape(-1,1))
    with torch.no_grad():
        idx = 0
        for j, (param) in enumerate(net.named_parameters()):
            if param[1].requires_grad:
                param[1].data += global_params[idx:(idx+param[1].nelement())].reshape(param[1].shape)
                idx += param[1].nelement()  
    del param_list, global_params
    return net

def krum(device, param_list, net, cmax):
    k = len(param_list)-cmax-2
    dist = torch.zeros((len(param_list), len(param_list))).to(device)
    for i in range(len(param_list)):
        for j in range(i):
            dist[i][j] = torch.norm(param_list[i]-param_list[j])
            dist[j][i] = dist[i][j]       
    sorted_dist = torch.sort(dist)
    sum_dist = torch.sum(sorted_dist[0][:,:k+1], axis=1)
    model_selected = torch.argmin(sum_dist).item()
    with torch.no_grad():
        idx = 0
        for j, (param) in enumerate(net.named_parameters()):
            if param[1].requires_grad:
                param[1].data += param_list[model_selected][idx:(idx+param[1].nelement())].reshape(param[1].shape)
                idx += param[1].nelement()  
    del param_list
    return net   

def fltrust(device, param_list, net,server_params):
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    server_norm = torch.norm(server_params)
    ts = torch.zeros((len(param_list)))
    for i in range(len(param_list)):
        ts[i] = max(cos(server_params, param_list[i]), 0)
        param_list[i] = (server_norm/torch.norm(param_list[i])) * param_list[i] * ts[i]
    if torch.sum(ts)  == 0: global_params = torch.zeros(len(param_list[0]))
    else: global_params = torch.sum(param_list, dim=0) / torch.sum(ts)
    print(ts)
    global_params = global_params.to(device)
    del param_list
    with torch.no_grad():
        idx = 0
        for j, (param) in enumerate(net.named_parameters()):
            if param[1].requires_grad:
                param[1].data += global_params[idx:(idx+param[1].nelement())].reshape(param[1].shape)
                idx += param[1].nelement()  
    del global_params
    return net

def trim(device, param_list, net, cmax):
    sorted_array = torch.sort(param_list, axis=0)
    trimmed = torch.mean(sorted_array[0][cmax:len(param_list)-cmax,:], axis=0)

    with torch.no_grad():
        idx = 0
        for j, (param) in enumerate(net.named_parameters()):
            if param[1].requires_grad:
                param[1].data += trimmed[idx:(idx+param[1].nelement())].reshape(param[1].shape)
                idx += param[1].nelement()  
                
    del param_list, sorted_array, trimmed
    return net  

def median(device, param_list, net, cmax):
    sorted_array = torch.sort(param_list, axis=0)
    if (len(param_list)%2 == 1):
        med = sorted_array[0][int(len(param_list)/2),:]
    else:
        med = (sorted_array[0][int(len(param_list)/2)-1,:] + sorted_array[0][int(len(param_list)/2),:])/2

    with torch.no_grad():
        idx = 0
        for j, (param) in enumerate(net.named_parameters()):
            if param[1].requires_grad:
                param[1].data += med[idx:(idx+param[1].nelement())].reshape(param[1].shape)
                idx += param[1].nelement()
    del param_list, sorted_array
    return net


def foolsgold(device, param_list, net):
    num_workers = len(param_list)
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).to(device)
    cs = torch.zeros((num_workers, num_workers)).to(device)
    for i in range(num_workers):
        for j in range(i):
            cs[i, j] = cos(param_list[i], param_list[j])
            cs[j, i] = cs[i, j]
    v = torch.zeros(num_workers).to(device)
    for i in range(num_workers):
        v[i] = torch.max(cs[i])

    alpha = torch.zeros(num_workers).to(device)
    for i in range(num_workers):
        for j in range(num_workers):
            if (v[j] > v[i]):
                cs[i, j] = cs[i, j] * v[i] / v[j]
        alpha[i] = 1 - torch.max(cs[i])

    alpha[alpha > 1] = 1
    alpha[alpha < 0] = 0
    alpha = alpha / (torch.max(alpha))
    alpha[alpha == 1] = 0.99
    alpha = torch.log(alpha / (1 - alpha)) + 0.5
    alpha[alpha > 1] = 1
    alpha[ alpha == np.inf] = 1
    alpha[ alpha == -np.inf] = 0
    alpha[alpha < 0] = 0
    alpha = alpha / torch.sum(alpha).item()
    print(alpha)
    global_params = torch.matmul(torch.transpose(param_list, 0, 1), alpha.reshape(-1, 1))
    with torch.no_grad():
        idx = 0
        for j, (param) in enumerate(net.named_parameters()):
            if param[1].requires_grad:
                param[1].data += global_params[idx:(idx + param[1].nelement())].reshape(param[1].shape)
                idx += param[1].nelement()
    del param_list, global_params
    return net

