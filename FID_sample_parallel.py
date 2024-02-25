import copy
import torch 
from torch.utils.data import   DataLoader
from architectures.UNET.unet import UNET 
 
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import math  
from pytorch_fid import fid_score
from architectures.openai.unet import UNetModel 
from utils.common_functions import check_evaluation_dataset, create_dataset_folder_in_evaluation, create_evaluation_folder, create_output_folders, get_checkpoint, get_dataset_folder_name_in_evaluation, save_generated_images, save_grid, save_image_list_in_dataset_dir, save_log, save_state_dict, write_to_evaluation_results
from utils.datasetloader import Cifar10Loader 

DEEP_MODEL='deep'
OPENAI_MODEL='oai'
IMAGE_SIZE=32
IMG_DIMENSION=3
def ddp_setup():  
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["RANK"]))  

class Sampler:
    def __init__(
        self,
        model_name, 
        pre_trained_model_name,
        dataset_name,
        model: torch.nn.Module,
        train_data: DataLoader, 
        ckpt_eopch,
        gpu_id: int,  
        sigma_min: float = 0.002,
        sigma_data: float = 0.5,
        sigma_max: float = 80.0,

    ) -> None:
        self.model_name=model_name
        self.gpu_id = gpu_id
        self.pre_trained_model_name=pre_trained_model_name
        self.model = model.to(gpu_id)
        self.train_data = train_data  
        self.sigma_min=sigma_min
        self.sigma_max=sigma_max 
        self.sigma_data=sigma_data 
        self.dataset_name=dataset_name
        self.model = DDP(self.model, device_ids=[self.gpu_id],find_unused_parameters=True)
        self.epochs= 0
        self.world_size=world_size 
        self.ckpt_eopch=ckpt_eopch
        self.current_training_step= self.gpu_id  
 


    def sample( self,sample_steps) : 
        
        self.load_model() 

        self.model.eval()
        #print('started.')
        #create_sampling_folder(model_name)
        #if self.check_evaluation_dataset()==False:
        #    evaluation_dataset_path= self.save_all_images_in_dataset()
        #else: 
        evaluation_dataset_path= get_dataset_folder_name_in_evaluation(dataset_name=self.dataset_name)
        
        _,training_sample_path_generated= create_evaluation_folder(self.model_name)
        fo_generated=0
        #fo_real=0
        print('train_data: '+ str(len(self.train_data)))
        self.train_data.sampler.set_epoch(0)
        for x,_ in self.train_data:
            _,generated_images = self.basic_sample_as_list( size=x.shape,sample_steps=sample_steps )
            fo_generated=  save_generated_images(image_list=generated_images,model_name=self.model_name,offset=fo_generated)
            #fo_real = save_real_images(image_list=real_images,model_name=model_name,offset=fo_real)
            print('generated images: '+ str(fo_generated))

        dim=2048
        fid_batch=1024
        fid_value = fid_score.calculate_fid_given_paths([evaluation_dataset_path,training_sample_path_generated],
                                                        batch_size=fid_batch,
                                                        device=self.gpu_id,
                                                        dims=dim,

                                                        num_workers=0) 


        result = f"\nEvaluation results: \n datasetname: {self.dataset_name} \nmodel name: {self.model_name} \nsampling steps: {sample_steps} \npre-trained model name: {self.pre_trained_model_name}\nfid result: {str(fid_value)} \n "

        write_to_evaluation_results(result=result)

    def skip_scaling(self,sigma 
        ) :
            
            return self.sigma_data**2 / ((sigma - self.sigma_min) ** 2 + self.sigma_data**2)
    

    def output_scaling(self,sigma 
        )  :

            return (self.sigma_data * (sigma - self.sigma_min)) / (self.sigma_data**2 + sigma**2) ** 0.5
    

    def model_forward_wrapper( self,model ,x ,sigma)  :
        
            c_skip = self.skip_scaling(sigma )
            c_out = self.output_scaling(sigma) 
            
            c_skip = self.pad_dims_like(c_skip, x)
            c_out = self.pad_dims_like(c_out, x)
            return c_skip  * x + c_out * model(x, sigma)


    def pad_dims_like(self,x, other) :

        ndim = other.ndim - x.ndim
        return x.view(*x.shape, *((1,) * ndim))
    

    def sample_with_size(self,size, model,  ts): 
            first_sigma = ts[0]
            x= torch.randn(size= size).to(device=self.gpu_id) * first_sigma
            
            sigma = torch.full((x.shape[0],), first_sigma, dtype=x.dtype, device=self.gpu_id)
            sigma= torch.squeeze(sigma,dim=-1)
            #x = model(x, sigma)

            x= self.model_forward_wrapper(model,x,sigma)
            for sigma in ts[1:]:
                z = torch.randn_like(x).to(device=self.gpu_id)
                x = x + math.sqrt(sigma**2 - self.sigma_min**2) * z
                sigma = torch.full((x.shape[0],1), sigma, dtype=x.dtype, device=self.gpu_id)
                sigma= torch.squeeze(sigma,dim=-1)
                x= self.model_forward_wrapper(model,x,sigma)
                #x = model(x, t)

            return x
    

    def get_sigmas_linear_reverse(self,n,sigma_min= 0.002,sigma_max=80): 
        sigmas = torch.linspace(sigma_max, sigma_min, n, dtype=torch.float16 ).to(device=self.gpu_id)
        return sigmas
    
    def basic_sample_as_list(self,size,sample_steps):

        with torch.no_grad():
            sigmas= self.get_sigmas_linear_reverse(sample_steps,self.sigma_min,self.sigma_max) 
            sample_results= self.sample_with_size( size=size, model=self.model,ts= sigmas )
            sample_results = (sample_results * 0.5 + 0.5).clamp(0, 1)    

            transform = transforms.ToPILImage()

            image_list = [transform(sample_results[i]) for i in range(sample_results.size(0))]

        return sample_results,image_list 
    
    def save_all_images_in_dataset(self):
        ix = 0
        evaluation_dataset_path= create_dataset_folder_in_evaluation(dataset_name=self.dataset_name)
        transform = transforms.ToPILImage()
        for x,_ in self.train_data:
            x = (x * 0.5 + 0.5).clamp(0, 1) 
            image_list = [transform(x[i]) for i in range(x.size(0))]
            save_image_list_in_dataset_dir(image_list=image_list,dataset_name=self.dataset_name,offset=ix)
            ix+=x.shape[0]
            print(ix)
        return evaluation_dataset_path
    
    def check_evaluation_dataset(self):
        isExist,cnt = check_evaluation_dataset(self.dataset_name)
        if isExist and cnt== len(self.train_data):
            return True
        elif isExist and cnt!= len(self.train_data):
            return False
        else:
            return False
        

    def load_model(self):
            state_dict= get_checkpoint(epoch=self.ckpt_eopch,model_name=self.pre_trained_model_name)
            self.model.module.load_state_dict(state_dict)
            #ckpt_name= self.model_name+'_'+str(epoch)+'_ckpt.pt'
     
     
    '''
    def gokmen_timesteps_schedule(self,current_training_step):
         
        frequency = (self.final_timesteps  ) **(1/self.rho) 

        normalized_step= (current_training_step /self.total_training_steps) 
        normalized_step= math.floor(normalized_step * math.pi**(self.rho**(1/2))) 
 
        result =  (self.final_timesteps  )   * math.cos(normalized_step*frequency + frequency/2 ) 

        return math.ceil(abs(result) ) 
    '''
    

   

def main( model_name ,pretrained_model_name,dataset_name, model_type,ckpt_epoch, 
        sampling_step,

        img_channels=3, 
    base_channels=192,
    num_res_blocks=6,
    groupnorm=16,
    num_heads=6,
    num_head_channels=64,
    use_scale_shift_norm=False,
    use_conv=True,
    attention_resolutions=[32,16,8],
        ):    
    
    #ddp_setup(rank, world_size) 
    ddp_setup() 
    train_data = Cifar10Loader(batch_size=128).dataloader 
    gpu_id = int(os.environ["RANK"])
    if model_type== DEEP_MODEL:
        attention_resolutions=[8,16]
        model = UNET( img_channels=img_channels,  device=gpu_id,groupnorm=groupnorm, attention_resolution=attention_resolutions, num_heads=num_heads, use_conv=use_conv,
                    base_channels=base_channels,num_res_blocks=num_res_blocks, num_head_channels=num_head_channels,use_scale_shift_norm=use_scale_shift_norm).to(device=gpu_id)
    else:
        model= UNetModel(attention_resolutions=attention_resolutions, use_scale_shift_norm=use_scale_shift_norm,model_channels=base_channels,num_head_channels=num_head_channels,
                         num_res_blocks=num_res_blocks,resblock_updown=True,image_size=IMAGE_SIZE,in_channels=IMG_DIMENSION,out_channels=IMG_DIMENSION)
         

    trainer = Sampler(model_name=model_name,ckpt_eopch=ckpt_epoch,dataset_name=dataset_name,gpu_id=gpu_id,model=model,pre_trained_model_name=pretrained_model_name,
                      train_data=train_data)
    trainer.sample(sample_steps=sampling_step)
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed sampling job')
    parser.add_argument('model_name',  default='cm_test',type=str, help='Model Name')
    parser.add_argument('pretrained_model_name',  default='cm_test',type=str, help='PreTrained Name')
    parser.add_argument('dataset_name',  default='cm_test',type=str, help='Dataset Name')
    parser.add_argument('model_type', default='deep', type=str, help='Batch Type') 
    parser.add_argument('ckpt_epoch', default=100, type=int, help='Batch Type') 
    parser.add_argument('sampling_step', default=4, type=int, help='Number of Inference Steps') 
    args = parser.parse_args()
    
    world_size =int(os.environ['WORLD_SIZE'])
    local_world_size =int(os.environ['LOCAL_WORLD_SIZE'])
    print('local_world_size :'+ str(local_world_size))
    print('world_size :'+ str(world_size))
    print('RANK  :'+ str(int(os.environ["RANK"]))) 

 
    main( model_name= args.model_name,pretrained_model_name= args.pretrained_model_name, 
        dataset_name=args.dataset_name ,model_type= args.model_type,ckpt_epoch=args.ckpt_epoch,sampling_step=args.sampling_step )
