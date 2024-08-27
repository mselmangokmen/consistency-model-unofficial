import copy
import torch 
import torch.distributed
from torch.utils.data import   DataLoader  
from typing import Iterator
from torch import Tensor, nn 

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import math 
import numpy as np       

import random, os
from datetime import timedelta
from torch import nn
from architectures.UNET.unet import UNET   
from utils.common_functions import   create_output_folders, get_checkpoint, save_grid_no_norm, save_metrics, save_log, save_state_dict

from utils.datasetloader import ButterflyDatasetLoader, CelebAFIDLoader, CelebALoader, Cifar10FIDLoader, Cifar10Loader, ImageNetFidLoader, ImageNetLoader
from datetime import datetime 
from torcheval.metrics import FrechetInceptionDistance   
#from torchmetrics.image.fid import FrechetInceptionDistance
DIST_TYPE_IMP='imp'
DIST_TYPE_BETA='beta' 
DIST_TYPE_GOKMEN='gokmen'
CURRICULUM_TYPE_IMP='imp'
CURRICULUM_TYPE_CM='cm'
CURRICULUM_TYPE_GOKMEN='gokmen' 
def ddp_setup():  
    init_process_group(backend="nccl", timeout=timedelta(hours=1))
    torch.cuda.set_device(int(os.environ["RANK"]))  




class Trainer:
    def __init__(
        self,
        model_name, 
        dataset_name,
        model: torch.nn.Module,
        ema_model: torch.nn.Module,
        train_data: DataLoader, 
        fid_loader:DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        total_training_steps:int,
        world_size,
        rho: float = 7.0,    
        base_channels=128,
        batch_size=256,
        beta=5, 
        alpha=0.5, 
        final_timesteps: int = 250,
        initial_timesteps:int= 20,
        sigma_min: float = 0.002,
        sigma_data: float = 0.5,
        sigma_max: float = 80.0,   
        
        use_ema=False,   
        num_classes= 10,
        image_size=32,  
        curriculum_type=2,  
        lr=1e-5,  
        dist_type=DIST_TYPE_IMP, 
        ckpt_interval=20000,
        sample_interval=1000,
        fid_interval=250,
        #4 : 2
        #2 : 1
    ) -> None:
        self.beta=beta
        self.alpha=alpha  
        self.num_classes=num_classes
        self.model_name=model_name 
        self.lr = lr  
        self.use_ema=use_ema
        self.version='cfr 8.6'  
 
        self.fid_interval=fid_interval  
        self.sample_interval=sample_interval
        self.batch_size=batch_size 
        self.dist_type=dist_type
        self.ckpt_interval=ckpt_interval
        self.base_channels=base_channels 
        self.gpu_id = gpu_id     
        self.num_time_steps=initial_timesteps
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer 
        self.final_timesteps=final_timesteps
        self.sigma_min=sigma_min
        self.curriculum_type=curriculum_type 
        self.sigma_max=sigma_max
        self.rho=rho     
        self.dataset_name=dataset_name 
        self.sigma_data=sigma_data
        self.image_size=image_size
        self.total_training_steps=total_training_steps   
        if self.use_ema and self.gpu_id==0: 
            self.ema_model = ema_model 
            self.ema_model.eval()
        self.model = DDP(self.model, device_ids=[self.gpu_id],find_unused_parameters=True,output_device=0)

        self.model.train()
        self.epochs= 0    
        self.world_size=world_size 
        self.num_classes=num_classes 
        self.fid_fun = FrechetInceptionDistance(device=self.gpu_id)   
        self.fid_fun.reset()  
        self.sample_shape=(100,3,image_size,image_size)  
        #self.current_training_step= self.gpu_id  
        self.current_training_step= 0
        self.initial_timesteps=initial_timesteps  
        self.seed_everything(42)

        if self.gpu_id==0: 
            for x,_ in fid_loader:
                self.fid_fun.update(x.to(self.gpu_id), is_real=True) 
 
    def seed_everything(self,seed: int): 
        
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def ema_decay_rate_schedule(self, 
        num_timesteps: int, initial_ema_decay_rate: float = 0.95 
    ) -> float:
 
        return math.exp(
            (self.initial_timesteps * math.log(initial_ema_decay_rate)) / num_timesteps
        )

    def update_metrics(self, x ):
        if self.use_ema:
            generated_images = self.single_sample( model=self.ema_model,spec_shape=x.shape)
        else: 
            generated_images = self.single_sample( model=self.model,spec_shape=x.shape)
        
        generated_images= (generated_images * 0.5 + 0.5).clamp(0,1)  
 
        self.fid_fun.update(generated_images, is_real=False)
        fid_value= self.fid_fun.compute().item()    
        result = 'FID: '+ str(fid_value)  
        save_metrics(metrics=result,model_name=self.model_name,training_step=self.current_training_step) 
  
  
    def _run_batch(self, x):
        self.optimizer.zero_grad() 
 
        if self.curriculum_type==CURRICULUM_TYPE_GOKMEN: 
                self.num_time_steps =  self.gokmen_timesteps_schedule(current_training_step=self.current_training_step)   
        if self.curriculum_type==CURRICULUM_TYPE_IMP: 
                self.num_time_steps =  self.improved_timesteps_schedule(current_training_step=self.current_training_step)   
        if self.curriculum_type==CURRICULUM_TYPE_CM: 
                self.num_time_steps =  self.cm_timesteps_schedule(current_training_step=self.current_training_step)   
 
 
        boundaries = self.karras_boundaries(num_timesteps=self.num_time_steps) 
        
        if self.dist_type==DIST_TYPE_IMP:
            timesteps =  self.lognormal_timestep_distribution(sigmas=boundaries,num_samples=x.shape[0])   
          
        else:  
            #timesteps =  self.beta_timestep_distribution(sigmas=boundaries,num_samples=x.shape[0])   
            timesteps =  self.beta_timestep_distribution(num_time_steps=self.num_time_steps-1,num_samples=x.shape[0])   
        current_sigmas = boundaries[timesteps].to(device=self.gpu_id)
        next_sigmas = boundaries[timesteps + 1].to(device=self.gpu_id)
       #print(torch.sort(next_sigmas))
        #print('current_sigmas: ', current_sigmas.shape)
        current_noisy_data,next_noisy_data= self.add_noise(current_sigmas=current_sigmas,next_sigmas=next_sigmas,x=x) 
        loss = self.loss_fun_improved(current_noisy_data=current_noisy_data,current_sigmas=current_sigmas,next_noisy_data=next_noisy_data,next_sigmas=next_sigmas,
                                        sigmas=boundaries,timesteps=timesteps  )
            
 
        loss.backward()
        self.optimizer.step() 

        #torch.distributed.barrier()
        #self.scheduler.step() 
        return loss.item(),self.num_time_steps,timesteps + 1,  next_sigmas, timesteps, current_sigmas 
    
    def _run_epoch(self, epoch): 
        
        self.train_data.sampler.set_epoch(epoch)

        #pbar = tqdm(self.train_data)
        data_len= len(self.train_data)
        batch_step = 0 
        loss_list=[]

        for x, c in self.train_data:  
            if not self.model.training:
                self.model.train()  
            #print(c.dtype)
            x = x.to(self.gpu_id) 
            #print('train max: ', torch.amax(x))
            #print('train min: ', torch.amin(x)) 
            loss, num_timesteps,next_timesteps , next_sigmas, current_timesteps, current_sigmas= self._run_batch(x=x) 
            

            batch_step+=1
            now = datetime.now()
            
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S.%f")[:-3]
            result=  'Huber loss: {:.4f}\tTraining Step: {:7d}/{:7d}\tNumber of Time Steps: {:7d}\tMin noise index: {:5d}\tMin sigma: {:5f}\tMax noise index: {:5d}\tMax sigma: {:5f}\tBeta: {:2f}\tAlpha: {:2f}\tLR: {:2f} \tSigma Data: {:2f}\tCurriculum Type: {}\tBase Channels: {:4d}\tDist Type: {}\tBatch Step: {:4d}/{:4d}\tEpoch: {:5d}/{:5d}\tGpu ID: {:3d}\tVersion: {}\tTime: {}'.format(
                    loss,
                    int(self.current_training_step),
                    int(self.total_training_steps),
                    int(num_timesteps), 
                    torch.amin(current_timesteps ).item() ,
                    torch.amin(current_sigmas ).item(),
                    torch.amax(next_timesteps ).item() ,
                    torch.amax(next_sigmas ).item(), 
                    self.beta, 
                    self.alpha, 
                    self.optimizer.param_groups[0]["lr"], 
                    self.sigma_data,   
                    self.curriculum_type,
                    self.base_channels,
                    self.dist_type,
                    int(batch_step),
                    int(data_len),
                    int(epoch),
                    int(self.epochs),
                    int(self.gpu_id),
                    str(self.version),
                    dt_string
                )
            

            loss_list.append(loss)

            
            print(   result )

            if self.gpu_id==0 and  self.current_training_step%self.ckpt_interval==0 and self.current_training_step<self.total_training_steps:
                self.save_checkpoint(self.current_training_step)
            if self.gpu_id==0 and self.current_training_step%self.sample_interval==0:
                  
                self.sample_and_save(current_training_step=self.current_training_step,sample_steps=[1,4]) 
            
            if self.gpu_id==0 and self.use_ema:  
                self.ema_model= self.update_ema_model_(ema_model=self.ema_model, online_model=self.model, ema_decay_rate=  self.ema_decay_rate_schedule(num_timesteps=self.num_time_steps))
                
            if self.gpu_id == 0 and self.current_training_step% self.fid_interval==0: 
                self.update_metrics(x) 
            if self.current_training_step==self.total_training_steps:
                break
            self.current_training_step = min(self.current_training_step + 1, self.total_training_steps)
        return np.mean(loss_list), x 
     
 

    def load_model(self,ckpt_eopch,pre_trained_model_name):
            state_dict= get_checkpoint(epoch=ckpt_eopch,model_name=pre_trained_model_name)
            self.model.module.load_state_dict(state_dict)
            #ckpt_name= self.model_name+'_'+str(epoch)+'_ckpt.pt'
      
    

    def train(self):
        if self.gpu_id == 0 :
            
            create_output_folders(self.model_name)

        print('started.')
        #self.scheduler = torch.optim.lr_scheduler.LinearLR( self.optimizer,  start_factor=1e-5,  total_iters=1000 ) 
        print('model name: '+ self.model_name)  
        print('final_timesteps: '+ str(self.final_timesteps))
        print('rho: '+ str(self.rho))
        print('train_data len: '+ str(len(self.train_data))) 
        #self.epochs= math.ceil(self.total_training_steps / (len(self.train_data)*world_size))
        self.epochs= math.ceil(self.total_training_steps / len(self.train_data))
  
        for epoch in range(self.epochs): 
            avg_loss,x= self._run_epoch(epoch) 
            now = datetime.now() 
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S.%f")[:-3]
            epoch_result=  'Average Loss: {:.4f}\tEpoch: {:5d}/{:5d}\tNum Time Steps: {:5d}\tLR: {:5f}\tGpu ID: {:3d}\tTime: {}'.format(
                        avg_loss, 
                        epoch,
                        self.epochs,
                        int(self.num_time_steps),
                        self.optimizer.param_groups[0]["lr"],
                        self.gpu_id,
                        dt_string
                    )
            epoch_result= "*"*10 +'\t'+ epoch_result + '\t' +"*"*10 
            print(epoch_result)
            save_log(model_name=self.model_name,record=epoch_result) 
                    
        if self.gpu_id == 0:   
            self.save_checkpoint(self.current_training_step)  
            #self.update_metrics(x) 
            self.sample_and_save(current_training_step=self.total_training_steps,sample_steps=[1,4]) 

 

    def _update_ema_weights(self, 
        ema_weight_iter: Iterator[Tensor],
        online_weight_iter: Iterator[Tensor],
        ema_decay_rate: float,
    ) -> None:
        for ema_weight, online_weight in zip(ema_weight_iter, online_weight_iter):
            if ema_weight.data is None:
                ema_weight.data.copy_(online_weight.data)
            else:
                ema_weight.data.lerp_(online_weight.data, 1.0 - ema_decay_rate)

    def update_ema_model_(self, ema_model: nn.Module, online_model: nn.Module, ema_decay_rate: float) -> nn.Module:
        online_params = online_model.module.parameters() if isinstance(online_model, nn.parallel.DistributedDataParallel) else online_model.parameters()
        online_buffers = online_model.module.buffers() if isinstance(online_model, nn.parallel.DistributedDataParallel) else online_model.buffers()
         
        self._update_ema_weights(ema_model.parameters(), online_params, ema_decay_rate) 

        self._update_ema_weights(ema_model.buffers(), online_buffers, ema_decay_rate)
        return ema_model
    
    def single_sample(self,model, spec_shape=None): 
        model.eval()
        with torch.no_grad():
            if spec_shape==None:
                 spec_shape= self.sample_shape  
            x= torch.randn(size= spec_shape).to(device=self.gpu_id)
            #first_sigma=  (self.sigma_max**2 - self.sigma_min**2) ** 0.5
            sigma = torch.full((x.shape[0],), self.sigma_max, dtype=x.dtype, device=self.gpu_id)
            x=x* self.pad_dims_like(sigma,x) 
            x= self.model_forward_wrapper(model,x,sigma) 
            x= x.clamp(-1.0, 1.0)     
              
            return x 
    
    def sample(self, model,  ts, spec_shape=None): 
        model.eval()
        with torch.no_grad():
            first_sigma = ts[0]
            if spec_shape==None:
                 spec_shape= self.sample_shape 

            x= torch.randn(size= spec_shape).to(device=self.gpu_id)
            #first_sigma=  (first_sigma**2 - self.sigma_min**2) ** 0.5
            sigma = torch.full((x.shape[0],), first_sigma, dtype=x.dtype, device=self.gpu_id)
            x=x* self.pad_dims_like(sigma,x)
            
            
            x= self.model_forward_wrapper(model,x,sigma) 
            x= x.clamp(-1.0, 1.0)    
            for sigma in ts[1:]:
                sigma = torch.full((x.shape[0],), sigma, dtype=x.dtype, device=x.device)
                x = x + self.pad_dims_like(  (sigma**2 - self.sigma_min**2) ** 0.5, x  ) * torch.randn_like(x) 
                x= self.model_forward_wrapper(model,x,sigma)  
                x= x.clamp(-1.0, 1.0)   
            return x
  
    def get_sigmas_linear_reverse(self,n): 
        
        sigmas = np.linspace(start=self.sigma_max, stop=self.sigma_data , num=n+1 )  
        return sigmas
    
    def sample_and_save(self,current_training_step,sample_steps): 
 
            for sample_step in sample_steps:
                sigmas= self.get_sigmas_linear_reverse(sample_step) 
                if self.use_ema:
                    sample_results= self.sample(model=self.ema_model,ts=sigmas)
                else: 
                    sample_results= self.sample(model=self.model,ts=sigmas)
                sample_results= (sample_results * 0.5 + 0.5).clamp(0,1)  
                save_grid_no_norm(tensor=sample_results,epoch=int(current_training_step),model_name=self.model_name,sample_step=sample_step)
            #self.model.train() 
   


    def skip_scaling(self,sigma 
        ) :
            
            return self.sigma_data**2 / ((sigma - self.sigma_min) ** 2 + self.sigma_data**2)
    
  
    def output_scaling(self,sigma 
        )  : 
        return        (self.sigma_data * (sigma - self.sigma_min)) / (self.sigma_data**2 + sigma**2) ** 0.5
     
  
    def in_scaling(self,sigma 
        )  :

            return 1/(((sigma**2 + self.sigma_data**2))**0.5)
    
    
    def model_forward_wrapper( self,model ,x ,sigma)  :
        
            c_skip = self.skip_scaling(sigma )
            c_out = self.output_scaling(sigma) 
            c_in = self.in_scaling(sigma)  
              
            
            c_skip = self.pad_dims_like(c_skip, x)
            c_out = self.pad_dims_like(c_out, x)
            c_in = self.pad_dims_like(c_in,x)  
            return c_skip * x + c_out * model(x* c_in, 0.25 * torch.log(sigma) ) 
            #return c_skip  * x + c_out * model(x, sigma)
    
 

    def pseudo_huber_loss(self,input, target) :  
        c = 0.00054* math.sqrt(math.prod(input.shape[1:]))
        
        return torch.sqrt((input - target) ** 2 + c**2) - c
 
    def pad_dims_like(self,x, other) :

        ndim = other.ndim - x.ndim
        return x.view(*x.shape, *((1,) * ndim))
 
    def loss_fun_improved(self ,current_noisy_data,next_noisy_data,current_sigmas,next_sigmas, sigmas, timesteps  ): 
        
        next_x= self.model_forward_wrapper(self.model,next_noisy_data,next_sigmas )
        with torch.no_grad():
                current_x = self.model_forward_wrapper(self.model,current_noisy_data,current_sigmas )
                
        #loss_weights = self.pad_dims_like(self.improved_loss_weighting(sigmas)[timesteps], next_x)
        loss_weights = self.pad_dims_like(self.improved_loss_weighting_2(current_sigmas=current_sigmas,next_sigmas=next_sigmas)  , next_x) 
        
        ph_loss= self.pseudo_huber_loss(current_x, next_x)   
        
        loss = (loss_weights* ph_loss).mean()
            
        return loss 
     
    def improved_loss_weighting(self,sigmas)  :
        
        return 1 / (sigmas[1:] - sigmas[:-1]) 
  
    def improved_loss_weighting_2(self,current_sigmas,next_sigmas)  :
        
        return 1 / (next_sigmas - current_sigmas) 
     
  
    def add_noise(self,x,current_sigmas,next_sigmas):
        noise = torch.randn_like(x).to(device=self.gpu_id)  
        current_noisy_data = x + self.pad_dims_like(current_sigmas,x) * noise
        next_noisy_data = x + self.pad_dims_like(next_sigmas,x)  * noise 
 
        return current_noisy_data,next_noisy_data
 
    def cm_timesteps_schedule( self,   current_training_step
        ) -> int: 
            
        N = math.ceil(math.sqrt(current_training_step * ((self.final_timesteps +1)**2 - self.initial_timesteps**2) / self.total_training_steps + self.initial_timesteps**2) - 1) + 1
        return int(N) 

 

    def improved_timesteps_schedule( self,   current_training_step
        ) -> int: 
            
            total_training_steps_prime = math.floor(
                self.total_training_steps
                / (math.log2(math.floor( self.final_timesteps / self.initial_timesteps)) + 1)
            )
            num_timesteps =  self.initial_timesteps * math.pow(
                2, math.floor(current_training_step / total_training_steps_prime)
            )
            num_timesteps = min(num_timesteps, self.final_timesteps) + 1

            return  int(num_timesteps)

 
    def lognormal_timestep_distribution(self,
        num_samples: int,
        sigmas: torch.Tensor,
        mean: float = -1.1,
        std: float = 2.0,
    ) -> torch.Tensor: 
        pdf = torch.erf((torch.log(sigmas[1:]) - mean) / (std * math.sqrt(2))) - torch.erf(
            (torch.log(sigmas[:-1]) - mean) / (std * math.sqrt(2))
        )
        pdf = pdf / pdf.sum()

        timesteps = torch.multinomial(pdf, num_samples, replacement=True)

        return timesteps
    

    def gokmen_timesteps_schedule(self, current_training_step ):
        
        normalized_step = current_training_step   /  self.total_training_steps 
        normalized_step = math.floor((normalized_step * math.pi) * 3) /3
        result = (self.final_timesteps) * math.sin(normalized_step/2)  + self.initial_timesteps
        return min(math.ceil(abs(result)) +1, self.final_timesteps+1)
 

    def karras_boundaries(self,num_timesteps):
         
        rho_inv = 1.0 / self.rho 
        steps = torch.arange(num_timesteps, device=self.gpu_id) / max(num_timesteps - 1, 1)
        sigmas = self.sigma_min**rho_inv + steps * (
                self.sigma_max**rho_inv - self.sigma_min**rho_inv
            )
        sigmas = sigmas**self.rho

        return sigmas
 
   

    def beta_timestep_distribution(self,num_time_steps:int,
                                    num_samples: int ): 
        values = np.random.beta(self.alpha, self.beta, num_samples)  
        
        min_value = np.min(values)
        max_value = np.max(values) 
        normalized_values = (values - min_value) / (max_value - min_value)
 
        choices_scaled = normalized_values * (num_time_steps - 1)
        choices_scaled=np.sort(choices_scaled)
        return torch.tensor(choices_scaled, dtype=torch.int32)
    '''

    def beta_timestep_distribution( self, num_samples,sigmas ): 
        beta_dist = torch.distributions.Beta(alpha, beta)
    
        samples = beta_dist.sample((num_samples,))
        samples = samples.to(sigmas.device)

            # Sigmas tensörünün sınırlarını belirle
        max_index = len(sigmas) - 2
        max_sigma = sigmas[max_index]
    
        min_sigma = sigmas.min()
        normalized_samples = min_sigma + samples * (max_sigma - min_sigma)
    
        timesteps = torch.bucketize(normalized_samples, sigmas[:max_index+1])
    
        sorted_timesteps = torch.sort(timesteps).values

        return sorted_timesteps
 

    '''
 

    def save_checkpoint(self,epoch):
 
            if self.use_ema:
                model_state_dict = copy.deepcopy(self.ema_model.state_dict())
            else: 
                model_state_dict = copy.deepcopy(self.model.module.state_dict())
            save_state_dict(state_dict=model_state_dict,epoch=epoch,model_name=self.model_name) 
        
def numel(m: torch.nn.Module, only_trainable: bool = True):
        """
        Returns the total number of parameters used by `m` (only counting
        shared parameters once); if `only_trainable` is True, then only
        includes parameters with `requires_grad = True`
        """
        parameters = list(m.parameters())
        if only_trainable:
            parameters = [p for p in parameters if p.requires_grad]
        unique = {p.data_ptr(): p for p in parameters}.values()
        return sum(p.numel() for p in unique)
    

def main(world_size, dataset_name,batch_size , model_name ,total_training_steps, 
         curriculum    ,dist_type, use_ema,model_type
         ):    
    
    #ddp_setup(rank, world_size) 

     
    #print(parameters)
    ddp_setup() 
    gpu_id = int(os.environ["RANK"])
    batch_size= batch_size // world_size 
    fid_loader=None
    if dataset_name == 'cifar10':
        train_data = Cifar10Loader(batch_size=batch_size,rank=gpu_id, shuffle=True).train_dataloader  

        fid_loader = Cifar10FIDLoader(batch_size=batch_size).fid_loader  
        image_size=32
    if dataset_name == 'celeba':
        train_data = CelebALoader(batch_size=batch_size).dataloader 
        if gpu_id==0:
            fid_loader = CelebAFIDLoader(batch_size=batch_size).fid_loader  
        image_size=64
    elif dataset_name == 'imagenet':
        train_data = ImageNetLoader(batch_size=batch_size,rank=gpu_id,).train_dataloader 
        fid_loader = ImageNetFidLoader(batch_size=batch_size,rank=gpu_id,).test_dataloader  
        image_size=64
     
    elif dataset_name == 'butterfly':
        image_size=256

        train_data = ButterflyDatasetLoader(batch_size=batch_size, image_size=image_size, rank=gpu_id).dataloader 
    if model_type=='small':
         base_channels=128
         num_res_blocks=2
    elif model_type=='medium':
         base_channels=128
         num_res_blocks=4
    
    elif model_type=='small-HR':
         base_channels=192
         num_res_blocks=2
    
    elif model_type=='large':
         base_channels=128
         num_res_blocks=6
    
    elif model_type=='huge':
         base_channels=128
         num_res_blocks=8
    model = UNET(  device=gpu_id,img_channels=3, groupnorm=32,
     dropout=0.0,base_channels=base_channels, num_head_channels=32,
        num_res_blocks=num_res_blocks,    
    use_conv_up  =True, use_conv_down  =True).to(device=gpu_id)
 
 

    if use_ema:
        ema_model = UNET(  device=gpu_id,img_channels=3, groupnorm=32, 
        dropout=0.0,base_channels=base_channels, num_head_channels=32,
            num_res_blocks=num_res_blocks,    
        use_conv_up  =True, use_conv_down  =True).to(device=gpu_id)
        ema_model.load_state_dict(model.state_dict())
    else: 
         ema_model=None
    lr=1e-4

    print('learning rate: ', lr )
    print('use ema: ', use_ema)

    optimizer= torch.optim.RAdam(model.parameters(), lr=lr , betas=(0.9, 0.995)) 
    trainer = Trainer(model_name=model_name,model=model, train_data=train_data, optimizer=optimizer, gpu_id=gpu_id,rho = 7,   fid_loader=fid_loader,
                      dataset_name=dataset_name,ckpt_interval=20000,   use_ema=use_ema, ema_model=ema_model,
        batch_size=batch_size,  
    
          total_training_steps=total_training_steps , world_size=world_size, num_classes=10 
            ,    image_size=image_size, lr=lr ,  
            curriculum_type=curriculum,base_channels=base_channels,dist_type=dist_type  )
    trainer.train()

    destroy_process_group()    

def list_of_strings(arg):
    list_str= arg.split(',')
    res = [eval(i) for i in list_str]
    return res

#tmux new-session -d -s "myTempSession"  torchrun --nnodes=1 --nproc_per_node=2 train_hn_unconditional.py --model_name hn_small_cifar10_test --dataset_name cifar10  --batch_size 512  --total_training_steps 800000 --model_type small --curriculum gokmen  --dist_type beta --use_ema False 

if __name__ == "__main__":  
        import argparse
        import os 

        parser = argparse.ArgumentParser(description='simple distributed sampling job')  
        parser.add_argument('--model_name', type=str, dest='model_name', help='Model Name')    
        parser.add_argument('--dataset_name', type=str, dest='dataset_name', help='Dataset Name')   
        parser.add_argument('--batch_size', type=int, dest='batch_size', help='Batch size')     
        parser.add_argument('--total_training_steps', type=int, dest='total_training_steps', help='Total training steps')  
        parser.add_argument('--model_type', type=str, dest='model_type', help='model_type value')         
        parser.add_argument('--curriculum', type=str, dest='curriculum', help='curriculum type')       
        parser.add_argument('--dist_type', type=str, dest='dist_type', help='dist_type value')       
        parser.add_argument('--use_ema', type=str, dest='use_ema', help='use_ema value')    
  
        args = parser.parse_args() 
        model_name = args.model_name 
        dataset_name = args.dataset_name
        batch_size = args.batch_size 
        model_type = args.model_type 
        total_training_steps = args.total_training_steps   
        dist_type=args.dist_type
        use_ema_str=args.use_ema
        curriculum = args.curriculum    
   

        world_size =int(os.environ['WORLD_SIZE'])
        local_world_size =int(os.environ['LOCAL_WORLD_SIZE'])
        print('local_world_size :'+ str(local_world_size))
        print('world_size :'+ str(world_size))

 

        print('RANK  :'+ str(int(os.environ["RANK"])))  
        use_ema=False
        if 'rue' in use_ema_str: 
            use_ema=True
  
    
        main(world_size=world_size, model_name=model_name, dataset_name=dataset_name,batch_size=batch_size,  
             total_training_steps=total_training_steps, curriculum=curriculum  ,model_type=model_type
            ,  dist_type=dist_type,use_ema=use_ema  )
#sudo docker build -t train_cm:latest .
#sudo docker tag  train_cm:latest mselmangokmen/train_cm:latest
#sudo docker push mselmangokmen/train_cm:latest
#torchrun --nnodes=1 --nproc_per_node=2 train_hn_unconditional.py --model_name hn_small_cifar10_test --dataset_name cifar10  --batch_size 512  --total_training_steps 800000 --model_type small --curriculum gokmen  --dist_type beta --use_ema False 
