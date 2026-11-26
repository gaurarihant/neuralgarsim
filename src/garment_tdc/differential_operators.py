# Adapted from SIREN (https://github.com/vsitzmann/siren).
# Copyright (c) 2020 Vincent Sitzmann. MIT License; see THIRD_PARTY_NOTICES.md.

import torch
from torch.autograd import grad
from typing import Tuple

def jacobian(y: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
    ''' jacobian of y wrt x '''
    meta_batch_size, num_observations = y.shape[:2]
    jac = torch.zeros(meta_batch_size, num_observations, y.shape[-1], x.shape[-1]).to(y.device)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        y_flat = y[...,i].view(-1, 1)
        jac[:, :, i, :] = grad(y_flat, x, torch.ones_like(y_flat), create_graph=True)[0]

    status = 0
    if torch.any(torch.isnan(jac)):
        status = -1
    return jac, status

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    gradient = grad(y, x, grad_outputs=grad_outputs, create_graph=True)[0]
    return gradient

def hessian(y, x):
    g = gradient(y, x).squeeze(0)
    h0 = gradient(g[:,0], x).squeeze(0)[:,None,:]
    h1 = gradient(g[:,1], x).squeeze(0)[:,None,:]
    h2 = gradient(g[:,2], x).squeeze(0)[:,None,:]
    h = torch.cat((h0,h1,h2), dim=1)

    status = 0
    if torch.any(torch.isnan(h)):
        status = -1

    return h.unsqueeze(0), status