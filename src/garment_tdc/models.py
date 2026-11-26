import torch
import torch.nn as nn
import numpy as np
from .boundary_conditions import BoundaryCondition

class SineLayer(nn.Module):      
    def __init__(self, in_features: int, out_features: int, bias=True, is_first=False, omega_0=30.):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
   
class SirenDeformationField(nn.Module):
    def __init__(self, boundary: BoundaryCondition, in_features=3, hidden_features=512, hidden_layers=5, out_features=3, outermost_linear=True, first_omega_0=30., hidden_omega_0=30.):
        super().__init__()
        self.boundary = boundary
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, np.sqrt(6 / hidden_features) / hidden_omega_0)
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))    
        self.net = nn.Sequential(*self.net)
    
    def forward(self, rest_points: torch.Tensor) -> torch.Tensor:                

        normalized_coords = self.boundary.normalization(rest_points)      
        deformations = self.net(normalized_coords)
        deformations = self.boundary.dirichlet_condition(deformations, rest_points)

        return deformations    

# Below code is taken from DiffUDF (https://github.com/LIA-DiTella/DiffUDF).
# Copyright (c) 2024 Laboratorio de Inteligencia Artificial - UTDT. MIT License; see THIRD_PARTY_NOTICES.md.

class ReLuLayer(nn.Module):
    """A Sine non-linearity layer.
    """
    def __init__(self, w0=30):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return nn.functional.relu(self.w0 * x)

    def __repr__(self):
        return f"ReLuLayer(w0={self.w0})"
    
class SineLayerGarment(nn.Module):
    """A Sine non-linearity layer.
    """
    def __init__(self, w0=30):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)

    def __repr__(self):
        return f"SineLayer(w0={self.w0})"

@torch.no_grad()
def sine_init(m, w0):
    if hasattr(m, 'weight'):
        num_input = m.weight.size(-1)
        m.weight.uniform_(-np.sqrt(6 / num_input) / w0,
                          np.sqrt(6 / num_input) / w0)


@torch.no_grad()
def first_layer_sine_init(m):
    if hasattr(m, 'weight'):
        num_input = m.weight.size(-1)
        m.weight.uniform_(-1 / num_input, 1 / num_input)

class SirenGarment(nn.Module):
    def __init__(self, in_features, out_features, hidden_layer_config=[],
                 w0=30, ww=None, delay_init=False, activation='sine'):
        super().__init__()
        self.w0 = w0
        if ww is None:
            self.ww = w0
        else:
            self.ww = ww

        net = []
        net.append(nn.Sequential(
            nn.Linear(in_features, hidden_layer_config[0]),
            SineLayerGarment(self.w0) if activation == 'sine' else ReLuLayer(self.ww)
        ))

        for i in range(1, len(hidden_layer_config)):
            net.append(nn.Sequential(
                nn.Linear(hidden_layer_config[i-1], hidden_layer_config[i]),
                SineLayerGarment(self.ww) if activation == 'sine' else ReLuLayer(self.ww)
            ))

        net.append(nn.Sequential(
            nn.Linear(hidden_layer_config[-1], out_features),
        ))

        self.net = nn.Sequential(*net)
        if not delay_init and activation == 'sine':
            self.net[0].apply(first_layer_sine_init)
            self.net[1:].apply(lambda module: sine_init(module, self.ww))


    def forward(self, x):
        """Forward pass of the model.

        Parameters
        ----------
        x: torch.Tensor
            The model input containing of size Nx3

        Returns
        -------
        dict
            Dictionary of tensors with the input coordinates under 'model_in'
            and the model output under 'model_out'.
        """
        if not x.requires_grad:
            x = x.clone().requires_grad_(True)
        coords = x
        y = self.net(coords)
    
        return {"model_in": coords, "model_out": y}
