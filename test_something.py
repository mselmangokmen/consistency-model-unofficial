
from model.utils import karras_schedule, timesteps_schedule

import torch
total_training_steps=50000

initial_timesteps = 2
final_timesteps = 150
sigma_min = 0.002
rho = 7.0
sigma_max = 80.0
num_timesteps= timesteps_schedule(current_training_step=5000,total_training_steps=50000,final_timesteps=final_timesteps,initial_timesteps=initial_timesteps)
        # N or number of timesteps

print(num_timesteps)
boundaries = karras_schedule(num_timesteps, sigma_min, sigma_max, rho) # karras boundaries 

x= torch.zeros(size=(64,3,128,128))

timesteps = torch.randint(0, num_timesteps - 1, (x.shape[0],)) # uniform distribution
 
print("boundaries.shape: "+ str(boundaries.shape))
print("timesteps.shape: "+ str(timesteps.shape))

current_sigmas = boundaries[timesteps] 
next_sigmas = boundaries[timesteps + 1] 


print("current_sigmas.shape: "+ str(current_sigmas.shape))
print("next_sigmas.shape: "+ str(next_sigmas.shape))
 
