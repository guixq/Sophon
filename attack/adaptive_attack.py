import torch
import torch.nn as nn

def sophon (device, server_params, param_list):
        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)

        num_workers = len(param_list)
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
            alpha[i] = 1 - torch.max(cs[i])
        alpha = (alpha-torch.min(alpha))/(torch.max(alpha)-torch.min(alpha))

        ts = torch.zeros(num_workers).to(device)
        for i in range(len(param_list)):
            ts[i] = cos(server_params, param_list[i])
        ts_temp = ts.clone().detach()
        ts = (ts - torch.min(ts)) / (torch.max(ts) - torch.min(ts))

        factor1 = ts + alpha
        factor2 = ts * alpha
        weights = 2 * factor2 / factor1
        weights[factor1 == 0] = 0
        weights = weights / torch.sum(weights)
        del server_params
        del param_list

        return v,ts_temp,weights




def full_trim(device, lr, param_list, cmax):

    max_dim = torch.max(-param_list, axis=0)[0]
    min_dim = torch.min(-param_list, axis=0)[0]
    direction = torch.sign(torch.sum(-param_list, axis=0)).to(device)
    directed_dim = (direction > 0) * min_dim + (direction < 0) * max_dim
    for i in range(cmax):
        random_12 = 1 + torch.rand(len(param_list[0])).to(device)
        param_list[i] = -(directed_dim * ((direction * directed_dim > 0) / random_12 + (direction * directed_dim < 0) * random_12))
    return param_list

def adaptive_attack (device, lr, param_list, nbyz, server_params,v,q):
    if (nbyz==0):
        return param_list
    cs,ts,be_weights = sophon(device, server_params, param_list)
    ret_param_list = param_list.clone().detach()
    be_param_list = param_list.clone().detach()
    for i in range(len(be_param_list)):
        be_param_list[i] = (1 / torch.norm(be_param_list[i])) * be_param_list[i]
    be_global_params = torch.matmul(torch.transpose(be_param_list, 0, 1), be_weights.reshape(-1, 1)).to(device)

    server_norm = torch.norm(server_params)
    direction = torch.sign(be_global_params)
    param_list = full_trim(device, lr, param_list, nbyz)
    for i in range(len(param_list)):
        param_list[i] = (1 / torch.norm(param_list[i])) * param_list[i]

    P = len(param_list[0])
    gamma = 0.005
    eta = 0.01
    for i in range(v):
        for j in range(nbyz):
            cs,ts,factor1 = H(device,server_params, be_global_params, param_list,direction,cs,ts,j)
            for t in range(q):
                origine = param_list[j].clone().detach()
                u = torch.normal(mean=0.0, std=0.5, size=(1, P)).to(device)
                param_list[j] = param_list[j]+gamma*u
                cs,ts,factor2 = H(device,server_params, be_global_params, param_list,direction,cs,ts,j)
                gradient = (factor2 - factor1)/gamma * u
                param_list[j] = origine+eta*gradient
                param_list[j] = (1 / torch.norm(param_list[j])) * param_list[j]
                factor1 = factor2

    for i in range(nbyz):
        ret_param_list[i] = (server_norm / torch.norm(param_list[i])) * param_list[i]
    del param_list,be_param_list
    return ret_param_list



def H(device,server_params,be_global_params,param_list,s,cs,ts,ind):
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    num_workers = len(param_list)
    v = torch.zeros(num_workers).to(device)
    for i in range(num_workers):
        v[i] = cos(param_list[ind], param_list[i])
        if i == ind: v[i] = 0
        if v[i] > cs[i]: cs[i] = v[i]

    ts[ind] = cos(server_params,param_list[ind])

    alpha = torch.zeros(num_workers).to(device)
    for i in range(num_workers):
        alpha[i] = 1 - cs[i]
    alpha = (alpha - torch.min(alpha)) / (torch.max(alpha) - torch.min(alpha))
    tts = (ts - torch.min(ts)) / (torch.max(ts) - torch.min(ts))

    factor1 = tts + alpha
    factor2 = tts * alpha
    weights = 2 * factor2 / factor1
    weights[factor1 == 0] = 0
    weights = weights / torch.sum(weights)

    byz_global_params = torch.matmul(torch.transpose(param_list, 0, 1), weights.reshape(-1, 1)).to(device)

    differ = be_global_params-byz_global_params

    h = torch.matmul(s.T,differ)

    del server_params,be_global_params,tts,param_list
    return cs,ts,h