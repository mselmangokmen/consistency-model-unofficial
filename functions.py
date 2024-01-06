

from tqdm import tqdm
import math
import shutil

from typing import Any, Callable, Iterable, Optional, Tuple, Union
from torch import Tensor, nn
import torch.nn.functional as F
import torch 
from torchvision.utils import save_image, make_grid
import os 
from model.utils import ema_decay_rate_schedule, karras_schedule, model_forward_wrapper, pad_dims_like, timesteps_schedule, update_ema_model_

import torchvision
from torchvision.utils import save_image
 
def loss_metric( x1,x2): 
        return F.mse_loss(x1, x2) 
 
 



def trainCM_Issolation(student_model,teacher_model, dataloader,model_name,device ,lr=1e-4,hideProgressBar=False,
                       
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0, 
        sigma_data: float = 0.5,
        initial_timesteps: int = 2,
        final_timesteps: int = 150,
        total_training_steps: int =50000  ) : 
    #model.to(device)
    optim = torch.optim.AdamW(student_model.parameters(), lr=1e-4)
    print('started.')
    # Define \theta_{-}, which is EMA of the params 
    #ema_model.to(device)
    teacher_model.load_state_dict(student_model.state_dict())
 
    dataiter = iter(dataloader)

    trFolderExist = os.path.exists('training_results')
    if not trFolderExist:
        os.mkdir('training_results')

    resultFilePath = 'training_results/'+ model_name+".txt"
    resultFileExist= os.path.isfile(resultFilePath)
    if resultFileExist:
        os.remove(resultFilePath)

    with open(resultFilePath, 'w') as fp:
        pass
    

    isExist = os.path.exists(model_name+'_training_samples')
    if not isExist:
        os.mkdir(model_name+'_training_samples')
    else:
        shutil.rmtree(model_name+'_training_samples', ignore_errors=True)
        os.mkdir(model_name+'_training_samples')


    progress_val= 0

    perc_step=total_training_steps//100
    for current_training_step in range(total_training_steps):

        try:
            x,_= next(dataiter)
            #x = dataiter.next()
        except StopIteration:
            
            dataiter = iter(dataloader)
            x,_= next(dataiter)
            
        x=x.to(device=device)

        if not student_model.training:
            student_model.train() 
        optim.zero_grad()

        progress_val= (current_training_step*100)/total_training_steps
        progress_val= round(progress_val,2)
        
        num_timesteps= timesteps_schedule(current_training_step=current_training_step,total_training_steps=total_training_steps,final_timesteps=final_timesteps,initial_timesteps=initial_timesteps)
        #print(num_timesteps)
        # N or number of timesteps
        boundaries = karras_schedule(num_timesteps, sigma_min, sigma_max, rho, device=device).to(device=device) # karras boundaries 
        #boundaries=torch.unsqueeze(boundaries,dim=-1)
        #print(boundaries)
        timesteps = torch.randint(0, num_timesteps - 1, (x.shape[0],), device=device) # uniform distribution

        current_sigmas = boundaries[timesteps].to(device=device)
        next_sigmas = boundaries[timesteps + 1].to(device=device)
        #current_sigmas=torch.unsqueeze(current_sigmas,dim=-1)
        #next_sigmas=torch.unsqueeze(next_sigmas,dim=-1)

        noise = torch.randn_like(x).to(device=device)
        current_noisy_data = x + pad_dims_like(current_sigmas,noise)* noise
        next_noisy_data = x + pad_dims_like(next_sigmas,noise) * noise
 
        #current_noisy_data = x + current_sigmas* noise
        #next_noisy_data = x + next_sigmas* noise

        student_model_prediction= model_forward_wrapper(x=next_noisy_data,sigma=next_sigmas,model=student_model)
        with torch.no_grad():
            teacher_model_prediction= model_forward_wrapper(x=current_noisy_data,sigma=current_sigmas,model=teacher_model)

        loss = loss_metric(student_model_prediction,teacher_model_prediction)
        
        loss.backward()
        with torch.no_grad():
            current_ema_decay_rate= ema_decay_rate_schedule(num_timesteps)
            update_ema_model_(ema_model=teacher_model,online_model=student_model,ema_decay_rate=current_ema_decay_rate)
        optim.step() 
         

        if current_training_step%(perc_step)==0 and current_training_step!=0:  
            with torch.no_grad():
                
                sigmas = torch.Tensor([2.0, 80.0]).to(device=device) 
                #sigmas = torch.unsqueeze(sigmas,dim=-1)

                sample_results= consistency_sampling(student_model,torch.zeros_like(x),sigmas.flip(0))
                sample_results = (sample_results * 0.5 + 0.5).clamp(0, 1) 
                save_tensor_as_grid(sample_results,f"{model_name}_training_samples/ct_{model_name}_sample_2step_{int(progress_val)}.png",nrow=sample_results.shape[0]//8) 

 
                sigmas = torch.Tensor([5.0, 10.0, 20.0, 40.0, 80.0]).to(device=device) 
                #sigmas = torch.unsqueeze(sigmas,dim=-1)

                sample_results= consistency_sampling(student_model,torch.zeros_like(x),sigmas.flip(0)) 
                sample_results = (sample_results * 0.5 + 0.5).clamp(0, 1) 
                save_tensor_as_grid(sample_results,f"{model_name}_training_samples/ct_{model_name}_sample_5step_{int(progress_val)}.png",nrow=sample_results.shape[0]//8) 
                 
                percent_result= "loss:"+str(loss.item())+", current_ema_decay_rate: "+str(current_ema_decay_rate)+"Time step completed: "+str(current_training_step)+",  "+str(progress_val)+"%"

                file1 = open(resultFilePath, "a")  # append mode
                file1.write(  percent_result+" \n")
                file1.close()
        print("current_training_step:"+ str(current_training_step), str(progress_val) +"%")


 

def consistency_sampling(
    model: nn.Module,
    y: torch.Tensor,
    sigmas: torch.Tensor,
    sigma_min: float = 0.002
) -> torch.Tensor:
    """
    A simplified sampling function that generates samples from noise using a given model.

    Parameters
    ----------
    model : nn.Module
        Model to sample from.
    y : torch.Tensor
        Noise sample.
    sigmas : torch.Tensor
        A tensor of standard deviations for noise, each row is a standard deviation value.
    sigma_min : float, default=0.002
        Minimum standard deviation of the noise.
    sigma_data : float, default=0.5
        Standard deviation of the data.

    Returns
    -------
    torch.Tensor
        Sampled sample.
    """
    # Initialize x with zeros
    x = torch.zeros_like(y)

    # Sample using the first standard deviation value
    first_sigma = sigmas[0]

    x = y + first_sigma * torch.randn_like(y)
    sigma = torch.full((x.shape[0],), first_sigma, dtype=x.dtype, device=x.device)
    #sigma= torch.unsqueeze(sigma,dim=-1)
    x = model_forward_wrapper(  model, x, sigma  ) 
    # Progressively denoise the sample
    for sigma_value in sigmas[1:]:
        sigma = torch.full((x.shape[0],), sigma_value, dtype=x.dtype, device=x.device)
        #sigma=torch.unsqueeze(sigma,dim=-1)
  
        std= pad_dims_like((sigma**2 - sigma_min**2) ** 0.5, x )
        x = x + std * torch.randn_like(x)
        x = model_forward_wrapper(  model, x, sigma  )

    return x
 


def save_tensor_as_grid(tensor: torch.Tensor, filename: str, nrow: int = 8) -> None:
    
    grid = torchvision.utils.make_grid(tensor, nrow=nrow)

    # Grid'i dosyaya kaydet
    save_image(grid, filename)