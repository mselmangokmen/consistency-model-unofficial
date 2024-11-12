import copy
import torch 
from torch.utils.data import   DataLoader  

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import math 
import numpy as np     
 
import random, os
from datetime import timedelta 
 
import torch.optim.lr_scheduler as lr_scheduler 
from architectures.UNET.unet import UNET
from utils.common_functions import   create_output_folders, get_checkpoint, save_grid_with_range, save_metrics, save_log, save_state_dict
from utils.datasetloader import  LDCTDatasetLoader
from datetime import datetime     
from torchmetrics.functional.image.psnr import peak_signal_noise_ratio
from torchmetrics.functional.image.ssim import structural_similarity_index_measure
def ddp_setup():  
    init_process_group(backend="nccl", timeout=timedelta(hours=1))
    torch.cuda.set_device(int(os.environ["RANK"]))  




class Trainer:
    def __init__(
        self,
        model_name,  
        model: torch.nn.Module,
        train_data: DataLoader,
        test_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int, 
        total_training_steps:int,
        world_size,
        rho: float = 7.0,    
        base_channels=128,
        batch_size=256,
        beta=5, 
        constant_N=False,
        alpha=0.5,

        final_timesteps: int = 1280,
        initial_timesteps:int= 10,
        sigma_min: float = 0.002,
        sigma_data: float = 0.5,
        sigma_max: float = 80.0,   
        num_time_steps=20,  
        num_classes= 10,
        image_size=32,   
        lr=1e-5,   
        ckpt_interval=20000,
        sample_interval=1000,
        fid_interval=250, 
    ) -> None:
        
        self.constant_N=constant_N
        self.beta=beta
        self.alpha=alpha 
        self.num_classes=num_classes
        self.model_name=model_name   
        self.lr = lr  
        self.version='12.5' 
        self.fid_interval=fid_interval  
        self.sample_interval=sample_interval
        self.batch_size=batch_size  
        self.ckpt_interval=ckpt_interval
        self.base_channels=base_channels 
        self.gpu_id = gpu_id    
        self.num_time_steps=num_time_steps



        self.model = model.to(gpu_id)
        
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer 
        self.final_timesteps=final_timesteps
        self.sigma_min=sigma_min  
        self.sigma_max=sigma_max
        self.rho=rho      
        self.sigma_data=sigma_data
        self.image_size=image_size
        self.total_training_steps=total_training_steps  
        self.model = DDP(self.model, device_ids=[self.gpu_id],find_unused_parameters=True,output_device=0)
        self.epochs= 0   
        self.world_size=world_size 
        self.num_classes=num_classes  
        self.trunc_min=-160.0
        self.trunc_max=240.0
        self.norm_range_max=3072.0
        self.norm_range_min=-1024.0 
        self.current_training_step= 0
        self.initial_timesteps=initial_timesteps  

        self.seed_everything(42)
         
        
    def seed_everything(self,seed: int): 
        
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)



    def denormalize_(self, image):
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min

        return image


    def trunc(self, mat):
        mat= mat.clamp(self.trunc_min, self.trunc_max) 
        return mat

        
    def update_metrics(self,current_training_step,save_images=False ):

        self.model.eval()   
         
         
        generated_images,f_img ,q_img = self.sample(model=self.model )
        if save_images:
             self.save_images(current_training_step=current_training_step,fd_image=f_img,qd_image=q_img,sample_results=generated_images) 
        f_img= (f_img * 0.5 + 0.5).clamp(0,1)   
        q_img= (q_img * 0.5 + 0.5).clamp(0,1)   
        generated_images= (generated_images * 0.5 + 0.5).clamp(0,1)   
        #lpip_score= learned_perceptual_image_patch_similarity(generated_images , f_img , net_type='vgg',normalize=True ).to(self.gpu_id)

        

        f_img =  self.denormalize_(f_img)
        q_img =  self.denormalize_(q_img)
        print('f img max: ', torch.amax(f_img))
        print('f img min: ', torch.amin(f_img))
        generated_images =  self.denormalize_(generated_images) 
        psnr_value=  peak_signal_noise_ratio(data_range=(self.norm_range_min,self.norm_range_max),target=f_img,preds=generated_images).to(self.gpu_id)
        ssim_value= structural_similarity_index_measure(data_range=(self.norm_range_min,self.norm_range_max),target=f_img,preds=generated_images) .to(self.gpu_id)
        #result = f'PSNR: {psnr_value.item():.3f} SSIM: {ssim_value.item():.3f} LPIPS: {lpip_score.item():.3f}' 
        result = f'PSNR: {psnr_value.item():.3f} SSIM: {ssim_value.item():.3f}' 
        
        save_metrics(metrics=result,model_name=self.model_name,training_step=self.current_training_step) 


  
    def _run_batch(self, y):
        self.optimizer.zero_grad() 
        if self.constant_N:
             self.num_time_steps=20 
        else:
            self.num_time_steps =  self.improved_timesteps_schedule(current_training_step=self.current_training_step)    



        boundaries = self.karras_boundaries(num_timesteps=self.num_time_steps) 
          
 
        timesteps =  self.lognormal_timestep_distribution(sigmas=boundaries,num_samples=y.shape[0])    
  
        current_sigmas = boundaries[timesteps].to(device=self.gpu_id)
        next_sigmas = boundaries[timesteps + 1].to(device=self.gpu_id)
        
        current_noisy_data,next_noisy_data= self.add_noise(current_sigmas=current_sigmas,next_sigmas=next_sigmas,y=y)  
    

        loss = self.loss_fun_improved(current_noisy_data=current_noisy_data,current_sigmas=current_sigmas,next_noisy_data=next_noisy_data,next_sigmas=next_sigmas )
        loss.backward() 
        self.optimizer.step()
        
        return loss.item(),self.num_time_steps,timesteps + 1,  next_sigmas, timesteps, current_sigmas 
    
    def _run_epoch(self, epoch): 
        
        self.train_data.sampler.set_epoch(epoch)
 
        data_len= len(self.train_data)
        batch_step = 0 
        loss_list=[]

        for x, y in self.train_data:  
            if not self.model.training:
                self.model.train()  
            #print(c.dtype)
            x = x.to(self.gpu_id ) 
            y = y.to(self.gpu_id ) 
            loss, num_timesteps,next_timesteps , next_sigmas, current_timesteps, current_sigmas= self._run_batch(y=y) 
            

            batch_step+=1
            now = datetime.now()
 
            gpu_id =self.gpu_id
            total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
            reserved_memory = torch.cuda.memory_reserved(gpu_id) 
 
            total_memory_gb = total_memory / (1024 ** 3) 
            reserved_memory_gb = reserved_memory / (1024 ** 3)  
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S.%f")[:-3]
            result=  'Huber Loss: {:.4f}\tTraining Step: {:7d}/{:7d}\tNumber of Time Steps: {:7d}\tMin noise index: {:5d}\tMin sigma: {:5f}\tMax noise index: {:5d}\tMax sigma: {:5f}\tBeta: {:2f}\tAlpha: {:2f}\tLR: {:2f} \tSigma Data: {:2f}\tBase Channels: {:4d} \tBatch Step: {:4d}/{:4d}\tEpoch: {:5d}/{:5d}\tGpu ID: {:3d}\tMemory: {:.2f}/{:.2f}\tVersion: {}\tTime: {}'.format(
                  
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
                    self.lr,    
                    self.sigma_data,   
                    self.base_channels, 
                    int(batch_step),
                    int(data_len),
                    int(epoch),
                    int(self.epochs),
                    int(self.gpu_id),
                     reserved_memory_gb, 
                    total_memory_gb, 
                    str(self.version),
                    dt_string
                )
            
            print(   result )

            loss_list.append(loss)

            
            
            if self.gpu_id==0 and  self.current_training_step%self.ckpt_interval==0 and self.current_training_step<self.total_training_steps:
                self.save_checkpoint(self.current_training_step) 
            
            if self.gpu_id == 0 and self.current_training_step% self.fid_interval==0:
                save_imgs= self.current_training_step%self.sample_interval==0
                with torch.no_grad():  
                    self.update_metrics(current_training_step=self.current_training_step, save_images=save_imgs) 
         
        
            self.current_training_step = min(self.current_training_step + 1, self.total_training_steps)
        return np.mean(loss_list)
     
 

    def load_model(self,ckpt_eopch,pre_trained_model_name):
            state_dict= get_checkpoint(epoch=ckpt_eopch,model_name=pre_trained_model_name)
            self.model.module.load_state_dict(state_dict)
            #ckpt_name= self.model_name+'_'+str(epoch)+'_ckpt.pt'
      
    

    def train(self):

        if self.gpu_id == 0 :
            
            create_output_folders(self.model_name)

        print('started.')

        self.scheduler = lr_scheduler.StepLR(self.optimizer,  step_size = 50000,  gamma = 0.5)  
        
        print('model name: '+ self.model_name)   
        print('rho: '+ str(self.rho))
        print('train_data len: '+ str(len(self.train_data))) 
        #self.epochs= math.ceil(self.total_training_steps / (len(self.train_data)*world_size))
        self.epochs= math.ceil(self.total_training_steps / len(self.train_data))
 
        #self.scheduler = CosineAnnealingLR(self.optimizer,   T_max = self.epochs,     eta_min = self.eta_min)  
        for epoch in range(self.epochs): 
            avg_loss= self._run_epoch(epoch)
                #self.scheduler.step()
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
            with torch.no_grad(): 
                self.save_checkpoint(self.current_training_step)  
                self.update_metrics(current_training_step=self.current_training_step, save_images=True)  


    def sample(self, model,  b_size=4): 
            
            data_iter = iter(self.test_data) 
            test_len= len(self.test_data)
            r = np.random.randint(1,test_len)
            for i in range(r):
                q_img, f_img= next(data_iter)  
            q_img =  q_img[:b_size,:,:,:] 
            f_img =  f_img[:b_size,:,:,:] 
            print('f_img amax: ', torch.amax(f_img))
            print('f_img amin: ', torch.amin(f_img))
            f_img = f_img.to(self.gpu_id)  
            q_img = q_img.to(self.gpu_id)  

            first_sigma = self.sigma_max
            # q_img ==> [-1 , 1 ] ==> 0.5 ==>  
            # q_img ==> [-1 , 1 ] ==> -1024, 3072 (16 bit) ==>  increase max noise level ==> 
            # torch.randn_like(q_img).to(device=self.gpu_id)  * 80) ==> -150, 150
            #y= torch.randn_like(q_img).to(device=self.gpu_id) * first_sigma + q_img
            y= (torch.randn_like(q_img).to(device=self.gpu_id)  * 20) + (q_img * 150)
            #y=   q_img
            
            sigma = torch.full((y.shape[0],), first_sigma, dtype=y.dtype, device=self.gpu_id) 

            y= self.model_forward_wrapper(model,y,sigma )
            print('y amax: ', torch.amax(y))
            print('y amin: ', torch.amin(y))
            y= y.clamp(-1.0, 1.0) 
            f_img= f_img.clamp(-1.0, 1.0) 
            q_img= q_img.clamp(-1.0, 1.0) 
            print('y amax after clamp: ', torch.amax(y))
            print('y amin after clamp: ', torch.amin(y)) 
            return y,f_img, q_img
  
  
      
    def save_images(self,fd_image,qd_image,sample_results,current_training_step): 
            
            sample_results= (sample_results * 0.5 + 0.5).clamp(0,1)   
            sample_results = self.trunc(self.denormalize_(sample_results))

            fd_image= (fd_image * 0.5 + 0.5).clamp(0,1)  
            fd_image = self.trunc(self.denormalize_(fd_image))

            qd_image= (qd_image * 0.5 + 0.5).clamp(0,1)  
            qd_image = self.trunc(self.denormalize_(qd_image))

            save_grid_with_range(tensor=sample_results,epoch=int(current_training_step),model_name=self.model_name,sample_step=1, filename='sample',min_range=self.trunc_min, max_range=self.trunc_max)
            save_grid_with_range(tensor=qd_image,epoch=int(current_training_step),model_name=self.model_name,sample_step=1, filename='quarter_dose',min_range=self.trunc_min, max_range=self.trunc_max)
            save_grid_with_range(tensor=fd_image,epoch=int(current_training_step),model_name=self.model_name,sample_step=1, filename='full_dose',min_range=self.trunc_min, max_range=self.trunc_max)
            #save_grid_with_range_val(tensor=val_img,epoch=int(current_training_step),model_name=self.model_name,sample_step=sample_step,min_range=self.trunc_min, max_range=self.trunc_max)
        #self.model.train() 

    def skip_scaling(self,sigma 
        ) :
            
            return self.sigma_data**2 / ((sigma - self.sigma_min) ** 2 + self.sigma_data**2)
    
  
    def output_scaling(self,sigma 
        )  :

            return self.sigma_data * (sigma - self.sigma_min) / (self.sigma_data**2 + sigma**2) ** 0.5
     
  
    def in_scaling(self,sigma 
        )  :

            return 1/(((sigma**2 + self.sigma_data**2))**0.5)
    

    def pad_dims_like(self,x, other) :

        ndim = other.ndim - x.ndim
        return x.view(*x.shape, *((1,) * ndim))
    

    def model_forward_wrapper( self,model ,x ,sigma)  :
        

            c_skip = self.skip_scaling(sigma )
            c_out = self.output_scaling(sigma) 
            c_in = self.in_scaling(sigma)  
              
            
            c_skip = self.pad_dims_like(c_skip, x)
            c_out = self.pad_dims_like(c_out, x)
            c_in = self.pad_dims_like(c_in,x)  
            
            return c_skip   * x + c_out  * model( x* c_in, 0.25 * torch.log(sigma) )

 
    def loss_fun_improved(self ,current_noisy_data,next_noisy_data,current_sigmas,next_sigmas  ): 
        
        next_y= self.model_forward_wrapper(self.model,next_noisy_data,next_sigmas )
        with torch.no_grad():
                current_y = self.model_forward_wrapper(self.model,current_noisy_data,current_sigmas )
                
        
        loss_weights= self.pad_dims_like( self.improved_loss_weighting_2(current_sigmas=current_sigmas,next_sigmas=next_sigmas) ,next_y )
         
        ph_loss= self.pseudo_huber_loss(current_y, next_y)   

        loss = (loss_weights* ph_loss).mean()
            
        return loss 
     
     
    
    def improved_loss_weighting(self,sigmas)  :
        
        return 1 / (sigmas[1:] - sigmas[:-1]) 
  
    def improved_loss_weighting_2(self,current_sigmas,next_sigmas)  :
        
        return 1 / (next_sigmas - current_sigmas) 
     
  

    def pseudo_huber_loss(self,input, target) :  
        c = 0.00054* math.sqrt(math.prod(input.shape[1:]))
        
        return torch.sqrt((input - target) ** 2 + c**2) - c
 
    def add_noise(self,y,current_sigmas,next_sigmas):
        noise = torch.randn_like(y).to(device=self.gpu_id)    
        current_noisy_data = y + current_sigmas[:,None,None,None] * noise
        next_noisy_data = y +  next_sigmas[:,None,None,None] * noise 
         
        return current_noisy_data,next_noisy_data
  

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
 

    def karras_boundaries(self,num_timesteps):
         
        
        ramp = torch.linspace(1, 0, num_timesteps)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 /self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho 
        return sigmas.to(device= self.gpu_id)
    
 
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
    
    def save_checkpoint(self,epoch):
        model_state_dict = copy.deepcopy(self.model.module.state_dict())
        save_state_dict(state_dict=model_state_dict,epoch=epoch,model_name=self.model_name) 
    

   

def main(world_size,batch_size,  num_res_blocks , model_name ,total_training_steps, 
          dropout  ,constant_N
         ):    
    
    base_channels=128
    ddp_setup() 
    gpu_id = int(os.environ["RANK"])
    batch_size= batch_size // world_size 
    dataset, max_val, min_val= LDCTDatasetLoader(batch_size=batch_size,rank=gpu_id ).getDataLoader()
    train_data = dataset['train']
    test_data = dataset['val'] 
    image_size=256 
    
    model = UNET(  device=gpu_id,img_channels=1, groupnorm=32,
     dropout=0.0,base_channels=base_channels, num_head_channels=32,
        num_res_blocks=num_res_blocks,    
    use_conv_up  =True, use_conv_down  =True).to(device=gpu_id).to(device=gpu_id)
    lr=1e-4
 

    
    optimizer= torch.optim.RAdam(model.parameters(), lr=lr , betas=(0.9, 0.995)) 
    trainer = Trainer(model_name=model_name,model=model, train_data=train_data,test_data=test_data, optimizer=optimizer, gpu_id=gpu_id,rho = 7,  
                      ckpt_interval=20000,  constant_N=constant_N,
       batch_size=batch_size,  
          total_training_steps=total_training_steps , world_size=world_size, num_classes=10 
            ,    image_size=image_size, lr=lr ,  
            base_channels=base_channels  )
    trainer.train()

    destroy_process_group()   

 
#tmux new-session -d -s "myTempSession"  torchrun --nnodes=1 --nproc_per_node=2 train_ldct_iCT.py --model_name cm_ldct_small_test_3 --batch_size 12 --num_res_blocks 2 --dropout 0.1 --total_training_steps 400000     

if __name__ == "__main__":  
        import argparse
        import os 

        parser = argparse.ArgumentParser(description='simple distributed sampling job')  
        parser.add_argument('--model_name', type=str, dest='model_name', help='Model Name')     
        parser.add_argument('--batch_size', type=int, dest='batch_size', help='Batch size')    
        parser.add_argument('--num_res_blocks', type=int, dest='num_res_blocks', help='Number of residual blocks')   
        parser.add_argument('--dropout', type=float, dest='dropout', help='dropout')    
        parser.add_argument('--total_training_steps', type=int, dest='total_training_steps', help='Total training steps')         
        parser.add_argument('--constant_N', type=str, dest='constant_N', help='constant_N_str value')     
 
        #args = parser.parse_args()

        args = parser.parse_args() 
        model_name = args.model_name  
        batch_size = args.batch_size
        num_res_blocks = args.num_res_blocks
        total_training_steps = args.total_training_steps    
        dropout = args.dropout    
     
        constant_N_str=args.constant_N  

        world_size =int(os.environ['WORLD_SIZE'])
        local_world_size =int(os.environ['LOCAL_WORLD_SIZE'])
        print('local_world_size :'+ str(local_world_size))
        print('world_size :'+ str(world_size))

  


        print('RANK  :'+ str(int(os.environ["RANK"])))  
         
  
 

        constant_N=False
        if 'rue' in constant_N_str:
             constant_N=True
        print('constant_N  :',constant_N)
 
        #mp.spawn(main, args=(world_size, args.model_name, args.batch_size, args.epochs), nprocs=world_size)
        main(world_size=world_size, model_name=model_name,  batch_size=batch_size,   
             num_res_blocks=num_res_blocks,   total_training_steps=total_training_steps 
            , dropout=dropout,constant_N=constant_N  )

 
 
# sudo docker build -t train_cm:latest . 
# sudo docker tag  train_cm:latest mselmangokmen/train_cm:latest
# sudo docker push mselmangokmen/train_cm:latest