

from tqdm import tqdm
import math
import shutil

import torch.nn.functional as F
import torch 
from torchvision.utils import save_image, make_grid
import os 
from model.utils import karras_schedule, pad_dims_like, timesteps_schedule


def calculate_loss( x, z, t1, t2, model,ema_model):
        if t1.ndim==1:
            t1=torch.unsqueeze(t1,dim=-1)

        if t2.ndim==1:
            t2=torch.unsqueeze(t2,dim=-1) 
        x2 = x + z * t2[:, :, None, None]
        x2 = model(x2, t2)

        with torch.no_grad():
            x1 = x + z * t1[:, :, None, None]
            x1 = ema_model(x1, t1)

        return F.mse_loss(x1, x2) 


def sample(x,ts,model): 
    with torch.no_grad():
        x = model(x*ts[0], ts[0])
        for t in ts[1:]:
            z = torch.randn_like(x)
            xtn = x + math.sqrt(t**2 - model.eps**2) * z
            x = model(xtn, t)

        return x


def trainCM_Issolation(model,ema_model, dataloader,dbname,device ,lr=1e-4,n_epochs=100,s1=150,s0=2,hideProgressBar=False,
                       
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0, 
        sigma_data: float = 0.5,
        initial_timesteps: int = 2,
        final_timesteps: int = 150,
        total_training_steps: int =50000
                       ) :
     
    #model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    print('started.')
    # Define \theta_{-}, which is EMA of the params 
    #ema_model.to(device)
    ema_model.load_state_dict(model.state_dict())
 
    dataiter = iter(dataloader)

    isExist = os.path.exists(dbname+'_training_samples')

    if not isExist:
        os.mkdir(dbname+'_training_samples')
    else:
        shutil.rmtree(dbname+'_training_samples', ignore_errors=True)
        os.mkdir(dbname+'_training_samples')


    progress_val= 0

    perc_step=total_training_steps//100
    for current_training_step in range(total_training_steps):
        progress_val= (current_training_step*100)/total_training_steps
        progress_val= round(progress_val,2)
        
        num_timesteps= timesteps_schedule(current_training_step=1,total_training_steps=total_training_steps,final_timesteps=final_timesteps,initial_timesteps=initial_timesteps)
        # N or number of timesteps
        boundaries = karras_schedule(num_timesteps, sigma_min, sigma_max, rho, device=device) # karras boundaries 
        loss_ema=None
        try:
            x,_= next(dataiter)
            #x = dataiter.next()
        except StopIteration:
            
            dataiter = iter(dataloader)
            x,_= next(dataiter)
        
        noise = torch.randn_like(x).to(device=device)
        if not model.training:
            model.train() 
        optim.zero_grad()

        x = x.to(device)

        timesteps = torch.randint(0, num_timesteps - 1, (x.shape[0],), device=device) # uniform distribution

        current_sigmas = boundaries[timesteps].to(device=device)
        next_sigmas = boundaries[timesteps + 1].to(device=device)

        loss_model = calculate_loss(x, noise, current_sigmas, next_sigmas,model=model, ema_model=ema_model)

        loss_model.backward()
        if loss_ema is None:
            loss_ema = loss_model.item()
        else:
            
            loss_ema = 0.9 * loss_ema + 0.1 * loss_model.item()

        optim.step()
        with torch.no_grad():
                mu = math.exp(2 * math.log(0.95) / num_timesteps)
                # update \theta_{-}
                for p, ema_p in zip(model.parameters(), ema_model.parameters()):
                    ema_p.mul_(mu).add_(p, alpha=1 - mu)

        if current_training_step%(perc_step)==0 and current_training_step!=0: 
            model.eval()
            with torch.no_grad():
                # Sample 5 Steps
                time_list= list(reversed([5.0, 10.0, 20.0, 40.0, 80.0]))
                time_list= torch.tensor(time_list).unsqueeze(dim=-1).unsqueeze(dim=-1).to(device=device)
                xh =sample(
                    torch.randn_like(x).to(device=device)  ,
                    time_list,model
                )
                xh = (xh * 0.5 + 0.5).clamp(0, 1)
                grid = make_grid(xh, nrow=4)

                save_image(grid, f"{dbname}_training_samples/ct_{dbname}_sample_5step_{int(progress_val)}.png")


                time_list= list(reversed([2.0, 80.0]))
                time_list= torch.tensor(time_list).unsqueeze(dim=-1).unsqueeze(dim=-1).to(device=device)

                # Sample 2 Steps
                xh = sample(
                    torch.randn_like(x).to(device=device)  ,
                    time_list,model
                )
                xh = (xh * 0.5 + 0.5).clamp(0, 1)
                grid = make_grid(xh, nrow=4)
                save_image(grid, f"{dbname}_training_samples/ct_{dbname}_sample_2step_{int(progress_val)}.png")

                # save model
                torch.save(model.state_dict(), f"ct_{dbname}.pth")
                
                print(f"loss: {loss_ema:.10f}, mu: {mu:.10f} Time step completed: {current_training_step},  {progress_val}%")
        
        print(f"current_training_step: {current_training_step},  {progress_val}%")



  