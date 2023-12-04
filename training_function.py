
from typing import List
from tqdm import tqdm
import math

import torch.nn.functional as F
import torch 

from model.unet import ConsistencyModel
from model.utils import kerras_boundaries


def calculate_loss( x, z, t1, t2, model,ema_model):
        x2 = x + z * t2[:, :, None, None]
        x2 = model(x2, t2)

        with torch.no_grad():
            x1 = x + z * t1[:, :, None, None]
            x1 = ema_model(x1, t1)

        return F.mse_loss(x1, x2) 


def trainCM_Issolation(dataloader, n_epochs=100, device="cuda:0", img_channels=1) :
    
    model = ConsistencyModel( img_channels=img_channels,time_emb_dim=256,base_channels=64)
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Define \theta_{-}, which is EMA of the params
    ema_model = ConsistencyModel(img_channels=img_channels,time_emb_dim=256,base_channels=64)
    ema_model.to(device)
    ema_model.load_state_dict(model.state_dict())
    for epoch in range(1, n_epochs):
        N = math.ceil(math.sqrt((epoch * (150**2 - 4) / n_epochs) + 4) - 1) + 1
        boundaries = kerras_boundaries(7.0, 0.002, N, 80.0).to(device)
        
        pbar = tqdm(dataloader)
        loss_ema = None
        model.train()
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)

            z = torch.randn_like(x)
            t = torch.randint(0, N - 1, (x.shape[0], 1), device=device)
            t_0 = boundaries[t]
            t_1 = boundaries[t + 1]

            loss_model = calculate_loss(x, z, t_0, t_1,model=model, ema_model=ema_model)

            loss_model.backward()
            if loss_ema is None:
                loss_ema = loss_model.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss_model.item()

            optim.step()
            with torch.no_grad():
                mu = math.exp(2 * math.log(0.95) / N)
                # update \theta_{-}
                for p, ema_p in zip(model.parameters(), ema_model.parameters()):
                    ema_p.mul_(mu).add_(p, alpha=1 - mu)

            pbar.set_description(f"loss: {loss_ema:.10f}, mu: {mu:.10f}")
