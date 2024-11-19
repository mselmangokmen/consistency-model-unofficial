 
import torch  
  
import os
import math 
import numpy as np     
 
import pydicom
import random, os
from datetime import timedelta 
from pydicom.dataset import Dataset, FileDataset
from skimage.feature import graycomatrix, graycoprops
 
from architectures.UNET_CT.unet_ct import UNET_CT
from utils.common_functions import   MODEL_SAMPLES_FOLDER, create_output_folders, get_latest_checkpoint, save_grid_with_range_name, save_grid_with_range_path, save_metrics, save_test_metrics, save_test_metrics_path 
from utils.datasetloader import  LDCTDatasetLoader
from datetime import datetime     
from torchmetrics.functional.image.psnr import peak_signal_noise_ratio
from torchmetrics.functional.image.ssim import structural_similarity_index_measure
from torchmetrics.functional.image.lpips import learned_perceptual_image_patch_similarity
from skimage.feature import graycomatrix, graycoprops 
 


class Tester:
    def __init__(
        self,
        model_name,  
        model: torch.nn.Module,  
        device,  
        output_path, 
        quarter_dose_data_path, 
        full_dose_data_path,  
        log_file_name,
        base_channels=128,
        batch_size=256,
        beta=5, 
        alpha=0.5, 
        sigma_min: float = 0.002,
        sigma_data: float = 0.5,
        sigma_max: float = 80.0,     
    ) -> None:
        
        
        self.beta=beta
        self.alpha=alpha  
        self.log_file_name=log_file_name
        self.model_name=model_name    
        self.version='14.2'   
        self.batch_size=batch_size   
        self.base_channels=base_channels  
        self.device=device
        self.model = model.to(device)
        self.img_cnt=0
        self.output_path = output_path 
        self.quarter_dose_data_path = quarter_dose_data_path 
        self.full_dose_data_path = full_dose_data_path 
        self.sigma_min=sigma_min  
        self.sigma_max=sigma_max 
        self.sigma_data=sigma_data 
        self.model = self.model 
        self.trunc_min=-160.0
        self.trunc_max=240.0
        self.norm_range_max=3072.0
        self.norm_range_min=-1024.0  
        
        self.seed_everything(42)
         
        self.model.eval()
        self.model.requires_grad_(False)

        self.full_dose_original_slices= self.load_scan(full_dose_data_path)
        self.full_dose_data= self.get_pixels_hu(self.full_dose_original_slices)

        self.quarter_dose_original_slices= self.load_scan(quarter_dose_data_path)
        self.quarter_dose_data= self.get_pixels_hu(self.quarter_dose_original_slices)

        self.full_dose_data= self.normalize_(self.full_dose_data)
        self.quarter_dose_data= self.normalize_(self.quarter_dose_data) 

    def seed_everything(self,seed: int): 
        
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)



    def normalize_(self, image, MIN_B=-1024.0, MAX_B=3072.0):
        image = (image - MIN_B) / (MAX_B - MIN_B) 
        image = image * 2 - 1 
        return image
    def denormalize_(self, image):
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min

        return image


    def get_pixels_hu(self, slices):
        
        image = np.stack([s.pixel_array for s in slices])
        image = image.astype(np.int16)
        image[image == -2000] = 0
        for slice_number in range(len(slices)):
            intercept = slices[slice_number].RescaleIntercept
            slope = slices[slice_number].RescaleSlope
            if slope != 1:
                image[slice_number] = slope * image[slice_number].astype(np.float64)
                image[slice_number] = image[slice_number].astype(np.int16)
            image[slice_number] += np.int16(intercept)
        return np.array(image, dtype=np.int16)


    def save_dicom_series(self, tensor_data, original_slices, output_dir):
        """
        Save a tensor of medical images as a DICOM series.

        Parameters:
            tensor_data (torch.Tensor): Denoised tensor data of shape (slice_cnt, 1, 512, 512).
            original_slices (list): List of original DICOM slices (pydicom Dataset objects).
            output_dir (str): Directory where the DICOM series will be saved.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        tensor_data = tensor_data.squeeze(1).cpu().numpy()  # Remove channel dimension, shape becomes (slice_cnt, 512, 512)
        tensor_data=tensor_data+1024.0
        for i, slice_data in enumerate(tensor_data):
            slice_data=np.squeeze(slice_data )
            original_metadata = original_slices[i]
            filename = os.path.join(output_dir, f"slice_{i:04d}.dcm")

            # Create a new DICOM file
            ds = FileDataset(filename, {}, file_meta=original_metadata.file_meta, preamble=b"\0" * 128)

            # Copy original metadata
            for elem in original_metadata:
                if elem.tag.is_private:
                    continue
                ds.add(elem)

            # Update pixel data
            ds.PixelData = slice_data.astype(np.int16).tobytes()  
            ds.Rows, ds.Columns = slice_data.shape

            # Update instance UID and other identifiers
            ds.SOPInstanceUID = pydicom.uid.generate_uid()
            ds.InstanceNumber = i + 1

            # Save the file
            ds.save_as(filename)
    def load_scan(self, path):
        # referred from https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
        sorted_path= os.listdir(path)
        slices = [pydicom.read_file(os.path.join(path, s)) for s in sorted_path]
        
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        try:
            slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        for s in slices:
            s.SliceThickness = slice_thickness
        return slices

    def trunc(self, mat):
        mat= mat.clamp(self.trunc_min, self.trunc_max) 
        return mat
    
    
    
    def concordance_correlation_coefficient(self, y_true, y_pred):
        """Compute the Concordance Correlation Coefficient between two arrays."""
        mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
        var_true, var_pred = np.var(y_true), np.var(y_pred)
        covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
        ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
        return ccc

    def extract_glcm_features(self, image, distances=[1], angles=[0], levels=4096):
        """Extract GLCM features from a 128x128 center crop of an image tensor in the range [-1024, 3072]."""
        # Convert the image to numpy, squeeze extra dimensions, and offset the range
        image = image.cpu().numpy().squeeze().astype(int)
        image = np.clip(image + 1024, 0, 4095)
        
        # Get center crop of size 128x128
        center_x, center_y = image.shape[0] // 2, image.shape[1] // 2
        start_x, start_y = center_x - 64, center_y - 64
        cropped_image = image[start_x:start_x + 128, start_y:start_y + 128]
        
        # Calculate GLCM and extract texture properties
        glcm = graycomatrix(cropped_image, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast').flatten()
        dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
        homogeneity = graycoprops(glcm, 'homogeneity').flatten()
        energy = graycoprops(glcm, 'energy').flatten()
        correlation = graycoprops(glcm, 'correlation').flatten()
        asm = graycoprops(glcm, 'ASM').flatten()
        
        # Combine all GLCM features into a single array
        features = np.hstack([contrast, dissimilarity, homogeneity, energy, correlation, asm])
        return features

    def compare_glcm_ccc(self, image1, image2):   
        features1 = self.extract_glcm_features(image1)
        features2 = self.extract_glcm_features(image2)
         
        ccc_value = self.concordance_correlation_coefficient(features1, features2)
        return ccc_value
    

 

    def update_metrics(self,generated_images_main,f_img_main,  q_img_main ):
        
        result=''
        bs = generated_images_main.shape[0] 
        for i in range(bs): 
            generated_images= generated_images_main[i,:,:,:]
            f_img= f_img_main[i,:,:,:]
            q_img= q_img_main[i,:,:,:]
            generated_images=torch.unsqueeze(generated_images, dim=0)
            f_img=torch.unsqueeze(f_img, dim=0)
            q_img=torch.unsqueeze(q_img, dim=0)
             
            lpips_generated_images=generated_images
            lpips_f_img=f_img

            f_img= (f_img * 0.5 + 0.5).clamp(0,1)   
            q_img= (q_img * 0.5 + 0.5).clamp(0,1)   
            generated_images= (generated_images * 0.5 + 0.5).clamp(0,1)   
            f_img =  self.denormalize_(f_img)
            q_img =  self.denormalize_(q_img) 
            generated_images =  self.denormalize_(generated_images) 
            ccc= self.compare_glcm_ccc(f_img, generated_images)
 
            save_grid_with_range_path(tensor=self.trunc(generated_images),file_name=f'sample_{self.img_cnt}.png',min_range=self.trunc_min, max_range=self.trunc_max,output_path=os.path.join(self.output_path,'png'))
            save_grid_with_range_path(tensor=self.trunc(q_img), file_name=f'quarter_dose_{self.img_cnt}.png',min_range=self.trunc_min, max_range=self.trunc_max,output_path=os.path.join(self.output_path,'png'))
            save_grid_with_range_path(tensor=self.trunc(f_img), file_name=f'full_dose_{self.img_cnt}.png',min_range=self.trunc_min, max_range=self.trunc_max,output_path=os.path.join(self.output_path,'png'))
            
            lpip_score= learned_perceptual_image_patch_similarity(lpips_generated_images.repeat(1,3,1,1) , lpips_f_img.repeat(1,3,1,1) ,normalize=False ).to(self.device) 


            psnr_value=  peak_signal_noise_ratio(data_range=(self.norm_range_min,self.norm_range_max),target=f_img,preds=generated_images).to(self.device)
            ssim_value= structural_similarity_index_measure(data_range=(self.norm_range_min,self.norm_range_max),target=f_img,preds=generated_images) .to(self.device)
            result =   result + f'Image Index: {self.img_cnt} PSNR: {psnr_value.item():.3f} SSIM: {ssim_value.item():.3f} LPIPS: {lpip_score.item():.3f} CCC: {ccc:.3f}  ' + '\n'  
            #result = f'PSNR: {psnr_value.item():.3f} SSIM: {ssim_value.item():.3f} ' 
            
            self.img_cnt+=1 
        return result,generated_images


   
    def _run_test(self):  
  
            
        create_output_folders(self.model_name)
        data_len = len(self.quarter_dose_data)
        batch_size = self.batch_size  
        total_batches = (data_len + batch_size - 1) // batch_size  
        batch_step = 0   
        denoised_images=[]
        for i in range(0, data_len, batch_size):  
            x = torch.tensor(self.quarter_dose_data[i:i+batch_size,:,:], dtype=torch.float32, device=self.device)
            x = torch.unsqueeze(x, dim=1)
            y = torch.tensor(self.full_dose_data[i:i+batch_size,:,:], dtype=torch.float32, device=self.device)
            y = torch.unsqueeze(y, dim=1)   
            batch_step += 1 
            generated_samples = self.sample(model=self.model, q_img=x) 
            result,_ = self.update_metrics(generated_images_main=generated_samples, f_img_main=y, q_img_main=x)

            denorm_samples= (generated_samples * 0.5 + 0.5).clamp(0,1)    
            denorm_samples =  self.denormalize_(denorm_samples)  
            denoised_images.extend(denorm_samples.cpu().numpy())
            # Use total_batches instead of data_len for accurate progress
            progress = f"Progress: Batch {batch_step}/{total_batches} \n" + result
            print(progress)
            save_test_metrics_path(metrics=result, file_path=self.log_file_name)
        denoised_images = torch.tensor(denoised_images, dtype=torch.float32).to(self.device)
        print('denoised_images shape: ', denoised_images.shape)
        self.save_dicom_series(tensor_data=denoised_images,original_slices=self.full_dose_original_slices,output_dir=os.path.join(self.output_path,'dicom'))
    def sample(self, model,q_img ): 
              
            q_img = q_img.to(self.device)  

            first_sigma = self.sigma_max
            
            y= torch.randn_like(q_img).to(device=self.device) * first_sigma
            
            sigma = torch.full((y.shape[0],), first_sigma, dtype=y.dtype, device=self.device) 

            y= self.model_forward_wrapper(model,y,sigma, cond=q_img) 
            y= y.clamp(-1.0, 1.0)   
            return y 
  
  
      
      

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
    

    def model_forward_wrapper( self,model ,x ,sigma, cond)  :
        

            c_skip = self.skip_scaling(sigma )
            c_out = self.output_scaling(sigma) 
            c_in = self.in_scaling(sigma)  
              
            
            c_skip = self.pad_dims_like(c_skip, x)
            c_out = self.pad_dims_like(c_out, x)
            c_in = self.pad_dims_like(c_in,x)  
            
            return c_skip   * x + c_out  * model( x* c_in, 0.25 * torch.log(sigma) , cond) 
 
     
     
  
   

def main( batch_size,  num_res_blocks, model_name, pretrained_model_name, output_path, quarter_dose_data_path, full_dose_data_path, log_file_name):    
    
    base_channels=64 
    device = torch.device("cuda:1") 
    model = UNET_CT(  device=device,img_channels=1, groupnorm=16, base_channels=base_channels, num_head_channels=32,
        num_res_blocks=num_res_blocks ).to(device=device) 
    ckpt_path,current_training_step= get_latest_checkpoint(pretrained_model_name)
    state_dict=torch.load(ckpt_path)
    model.load_state_dict(state_dict) 

    print('Model loaded from : ',ckpt_path)
  
    tester = Tester(model_name=model_name,model=model, output_path=output_path,quarter_dose_data_path=quarter_dose_data_path, full_dose_data_path=full_dose_data_path ,  device=device,   log_file_name=log_file_name, batch_size=batch_size, base_channels=base_channels  )
    tester._run_test()
 

 
if __name__ == "__main__":  
        import argparse
        import os 

        parser = argparse.ArgumentParser(description='simple distributed sampling job')  
        parser.add_argument('--model_name', type=str, dest='model_name', help='Model Name')     
        parser.add_argument('--pretrained_model_name', type=str, dest='pretrained_model_name', help='pretrained Model Name')     
        parser.add_argument('--batch_size', type=int, dest='batch_size', help='Batch size')    
        parser.add_argument('--num_res_blocks', type=int, dest='num_res_blocks', help='Number of residual blocks')  
        parser.add_argument('--quarter_dose_data_path', type=str, dest='quarter_dose_data_path',  default='dataset/LDCT_raw/test/L506/quarter_dose')   
        parser.add_argument('--full_dose_data_path', type=str, dest='full_dose_data_path',  default='dataset/LDCT_raw/test/L506/full_dose')     
        parser.add_argument('--output_path', type=str, dest='output_path',  default='outputs/model_samples/L506')     
        parser.add_argument('--log_file_name', type=str, dest='log_file_name',  default='outputs/model_samples/L506.txt'  ) 
  

        args = parser.parse_args() 
        model_name = args.model_name  
        pretrained_model_name = args.pretrained_model_name  
        batch_size = args.batch_size
        num_res_blocks = args.num_res_blocks   
        quarter_dose_data_path=args.quarter_dose_data_path
        full_dose_data_path=args.full_dose_data_path
        output_path=args.output_path  
        log_file_name=args.log_file_name  
  

 
        
  
  
 
  
        main( model_name=model_name,  batch_size=batch_size, full_dose_data_path=full_dose_data_path, 
             quarter_dose_data_path=quarter_dose_data_path, log_file_name=log_file_name,
             num_res_blocks=num_res_blocks,    pretrained_model_name=pretrained_model_name,output_path=output_path )
  