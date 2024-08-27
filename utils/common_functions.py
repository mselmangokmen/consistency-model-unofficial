
import copy
from torchvision.utils import save_image

import torchvision
import os 
import shutil  
import torch
import zipfile
import math 
OUTPUT_FOLDER='outputs'
TRAINING_SAMPLE_FOLDER='training_samples'
EVALUATION_FOLDER='evaluation'
CHECKPOINT_FOLDER='checkpoints'
TRAINING_RESULTS_FOLDER='training_results'  
ZIP_FOLDER='zip_files'  
MODEL_SAMPLES_FOLDER='model_samples'  

 

def save_zip(source_folder, file_name):
    out_path= os.path.join(OUTPUT_FOLDER,ZIP_FOLDER,file_name)
    zip_folder_with_zipfile(output_path=out_path,source_folder=source_folder)

def zip_folder_with_zipfile(source_folder, output_path):
   
   with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
      for root, dirs, files in os.walk(source_folder):
         for file in files:
            file_path = os.path.join(root, file)
            archive_path = os.path.relpath(file_path, source_folder)
            zipf.write(file_path, archive_path)

def save_tensor_as_grid(tensor: torch.Tensor, filename: str, nrow: int = 4 ) -> None:

    #grid = torchvision.utils.make_grid( tensor.clamp(-1.0, 1.0), value_range=(-1.0, 1.0), normalize=True , nrow=nrow ) 
    grid = torchvision.utils.make_grid((tensor * 0.5 + 0.5).clamp(0,1) , nrow=nrow ) 
    save_image(grid, filename)


def save_tensor_no_norm(tensor: torch.Tensor, filename: str, nrow: int = 4 ) -> None:

    #grid = torchvision.utils.make_grid( tensor.clamp(-1.0, 1.0), value_range=(-1.0, 1.0), normalize=True , nrow=nrow ) 
    grid = torchvision.utils.make_grid(tensor , nrow=nrow, normalize=True  ) 
    save_image(grid, filename)


def get_checkpoints_path(model_name):
 
    ckpt_path=  os.path.join(OUTPUT_FOLDER,CHECKPOINT_FOLDER,model_name)
    file_names= os.listdir(ckpt_path)
    ckpt_list= [ os.path.join(ckpt_path, fn) for fn in file_names] 
    
    file_names_splitted= [fn.split('.')[0] for fn in file_names]
    return ckpt_list,file_names_splitted


def write_to_evaluation_results(result, model_name):

    EVALUATION_RESULT_FILE= os.path.join(OUTPUT_FOLDER,EVALUATION_FOLDER,str(model_name)+'.txt')
    if os.path.isfile(EVALUATION_RESULT_FILE):
        file1 = open(EVALUATION_RESULT_FILE, "a") 
    else:
        file1 = open(EVALUATION_RESULT_FILE, "w+") 

    file1.write(result)
    file1.close()

def save_grid_no_norm(tensor,model_name,sample_step,epoch):
    model_sample_path= os.path.join(OUTPUT_FOLDER,TRAINING_SAMPLE_FOLDER,model_name)
    
    file_name= 'sample_step_'+ str(sample_step)+ '_epoch_'+str(epoch)+'.png'
    full_path= os.path.join(model_sample_path,file_name)
    save_tensor_no_norm(tensor,full_path, int(math.sqrt(tensor.shape[0])))


def save_grid_no_norm_with_name(tensor,model_name,name,epoch):
    model_sample_path= os.path.join(OUTPUT_FOLDER,TRAINING_SAMPLE_FOLDER,model_name)
    
    file_name= str(name)+ '_epoch_'+str(epoch)+'.png'
    full_path= os.path.join(model_sample_path,file_name)
    save_tensor_no_norm(tensor,full_path, int(math.sqrt(tensor.shape[0])))


def save_grid_no_norm_fd(tensor,model_name,sample_step,epoch):
    model_sample_path= os.path.join(OUTPUT_FOLDER,TRAINING_SAMPLE_FOLDER,model_name)
    
    file_name= 'full_dose_'+ str(sample_step)+ '_epoch_'+str(epoch)+'.png'
    full_path= os.path.join(model_sample_path,file_name)
    save_tensor_no_norm(tensor,full_path, int(math.sqrt(tensor.shape[0])))


def save_grid_no_norm_qd(tensor,model_name,sample_step,epoch):
    model_sample_path= os.path.join(OUTPUT_FOLDER,TRAINING_SAMPLE_FOLDER,model_name)
    
    file_name= 'quarter_dose_'+ str(sample_step)+ '_epoch_'+str(epoch)+'.png'
    full_path= os.path.join(model_sample_path,file_name)
    save_tensor_no_norm(tensor,full_path, int(math.sqrt(tensor.shape[0])))


def save_grid_with_range(tensor,model_name,sample_step,epoch,min_range,max_range, filename):
    model_sample_path= os.path.join(OUTPUT_FOLDER,TRAINING_SAMPLE_FOLDER,model_name)
    
    file_name= filename+'_step_'+ str(sample_step)+ '_epoch_'+str(epoch)+'.png'
    full_path= os.path.join(model_sample_path,file_name)

    grid = torchvision.utils.make_grid( tensor, value_range=(min_range, max_range) , nrow= int(math.sqrt(tensor.shape[0])), normalize=True ) 

    save_image(grid, full_path) 



def save_grid_with_range_val(tensor,model_name,sample_step,epoch,min_range,max_range):
    model_sample_path= os.path.join(OUTPUT_FOLDER,TRAINING_SAMPLE_FOLDER,model_name)
    
    file_name= 'val_'+ str(sample_step)+ '_epoch_'+str(epoch)+'.png'
    full_path= os.path.join(model_sample_path,file_name)

    grid = torchvision.utils.make_grid( tensor, value_range=(min_range, max_range) , nrow= int(math.sqrt(tensor.shape[0])), normalize=True ) 

    save_image(grid, full_path) 

def save_grid(tensor,model_name,sample_step,epoch):
    model_sample_path= os.path.join(OUTPUT_FOLDER,TRAINING_SAMPLE_FOLDER,model_name)
    
    file_name= 'sample_step_'+ str(sample_step)+ '_epoch_'+str(epoch)+'.png'
    full_path= os.path.join(model_sample_path,file_name)
    save_tensor_as_grid(tensor,full_path, int(math.sqrt(tensor.shape[0])))

def save_grid_with_name(tensor,img_name,model_name,epoch):
    model_sample_path= os.path.join(OUTPUT_FOLDER,TRAINING_SAMPLE_FOLDER,model_name)
    
    file_name= str(img_name)+ '_epoch_'+str(epoch) +'.png'

    #file_name= 'epoch_'+str(epoch) + '_img_'+str(img_name)+'.png'
    full_path= os.path.join(model_sample_path,file_name)
    save_tensor_as_grid(tensor,full_path, int(math.sqrt(tensor.shape[0])))


def save_grid_to_model_samples(tensor,img_name,model_name,sample_steps):
    model_sample_path= os.path.join(OUTPUT_FOLDER,MODEL_SAMPLES_FOLDER,model_name)
    
    file_name=  'img_'+str(img_name)+'_inf_step_'+str(sample_steps)+'.png'
    full_path= os.path.join(model_sample_path,file_name)
    save_tensor_as_grid(tensor,full_path, int(math.sqrt(tensor.shape[0])))




def check_evaluation_dataset(dataset_name):

    evaluation_sample_path= os.path.join(OUTPUT_FOLDER,EVALUATION_FOLDER,dataset_name)
    isExist = os.path.exists(evaluation_sample_path)
    cnt=0
    if isExist:
        cnt = len(os.listdir(evaluation_sample_path))
    return isExist,cnt

 
def save_image_list_in_dataset_dir(image_list,dataset_name,offset=0):
    for idx,img in enumerate(image_list):
        idx= offset + idx
        #print(idx)
        training_sample_path_generated= os.path.join(OUTPUT_FOLDER,EVALUATION_FOLDER,dataset_name,'img_'+str(idx)+ '.png')
        
        grid = torchvision.utils.make_grid( img.clamp(-1.0, 1.0), value_range=(-1.0, 1.0), normalize=True )
        save_image(grid,training_sample_path_generated)

def save_generated_images(image_list,model_name,offset=0):
    final_offset=0
    for idx,img in enumerate(image_list):
        idx= offset + idx
        #print(idx)
        training_sample_path_generated= os.path.join(OUTPUT_FOLDER,EVALUATION_FOLDER,model_name,'generated','img_'+str(idx)+ '.png')
        grid = torchvision.utils.make_grid( img.clamp(-1.0, 1.0), value_range=(-1.0, 1.0), normalize=True  )
        save_image(grid,training_sample_path_generated)
        #img.save(training_sample_path_generated)
        final_offset=idx+1
    return final_offset

def save_metrics(metrics, model_name, training_step):

    fidFilePath =  os.path.join(OUTPUT_FOLDER,TRAINING_RESULTS_FOLDER,model_name+'_metrics.txt')  
    file1 = open(fidFilePath, "a")  
    file1.write('Training Step: '+str(training_step)+ ' '+str(metrics)+" \n")
    file1.close()

def save_real_images(image_list,model_name,offset=0):
    final_offset=0
    for idx,img in enumerate(image_list):
        idx= offset + idx
        #print(idx)
        training_sample_path_generated= os.path.join(OUTPUT_FOLDER,EVALUATION_FOLDER,model_name,'real','img_'+str(idx)+ '.png')
        img.save(training_sample_path_generated)
        final_offset=idx+1
    return final_offset

def save_log(model_name,record):
    
    resultFilePath =  os.path.join(OUTPUT_FOLDER,TRAINING_RESULTS_FOLDER,model_name+'.txt')  
    file1 = open(resultFilePath, "a")  # append mode
    file1.write(  record+" \n")
    file1.close()

def save_checkpoint(model,model_name,epoch):
    
    ckpt_name= model_name+'_'+str(epoch)+'_ckpt.pt'
    state_dict= copy.deepcopy(model.state_dict())

    torch.save(state_dict, os.path.join(OUTPUT_FOLDER,CHECKPOINT_FOLDER,model_name, ckpt_name))

def get_checkpoint(model_name,epoch):
    ckpt_name= model_name+'_'+str(epoch)+'_ckpt.pt'
    ckpt_path=  os.path.join(OUTPUT_FOLDER,CHECKPOINT_FOLDER,model_name, ckpt_name)
    return torch.load(ckpt_path)

def get_checkpoint_by_path(ckpt_path):  
    return torch.load(ckpt_path)



def save_state_dict(state_dict,model_name,epoch):
    
    ckpt_name= model_name+'_'+str(epoch)+'_ckpt.pt'
    state_dict= copy.deepcopy(state_dict)
    if not os.path.exists(os.path.join(OUTPUT_FOLDER,CHECKPOINT_FOLDER,model_name, ckpt_name)):
        torch.save(state_dict, os.path.join(OUTPUT_FOLDER,CHECKPOINT_FOLDER,model_name, ckpt_name))
 

def create_training_sampling_folder(model_name):

    training_sample_path= os.path.join(OUTPUT_FOLDER,TRAINING_SAMPLE_FOLDER,model_name)
    isExist = os.path.exists(training_sample_path)
    if not isExist:
        os.mkdir(training_sample_path)
    else:
        shutil.rmtree(training_sample_path, ignore_errors=True)
        os.mkdir(training_sample_path)


def create_dataset_folder_in_evaluation(dataset_name):

    evaluation_dataset_path= os.path.join(OUTPUT_FOLDER,EVALUATION_FOLDER,dataset_name)
    isExist = os.path.exists(evaluation_dataset_path)
    if not isExist:
        os.mkdir(evaluation_dataset_path)
    else:
        shutil.rmtree(evaluation_dataset_path, ignore_errors=True)
        os.mkdir(evaluation_dataset_path)
    return evaluation_dataset_path

def get_dataset_folder_name_in_evaluation(dataset_name):
    return os.path.join(OUTPUT_FOLDER,EVALUATION_FOLDER,dataset_name)



def check_evaluation_folder(model_name):
    
    evaluation_sample_path= os.path.join(OUTPUT_FOLDER,EVALUATION_FOLDER,model_name)
    evalFolderExist = os.path.exists(evaluation_sample_path)
    return evalFolderExist


def create_evaluation_folder(model_name):
    
    evaluation_sample_path= os.path.join(OUTPUT_FOLDER,EVALUATION_FOLDER,model_name)
    evalFolderExist = os.path.exists(evaluation_sample_path)
    if not evalFolderExist:
        os.mkdir(evaluation_sample_path)

    training_sample_path_real= os.path.join(OUTPUT_FOLDER,EVALUATION_FOLDER,model_name,'real')
    training_sample_path_generated= os.path.join(OUTPUT_FOLDER,EVALUATION_FOLDER,model_name,'generated')

    isExist = os.path.exists(evaluation_sample_path)

    if not isExist:
        os.mkdir(evaluation_sample_path)
        os.mkdir(training_sample_path_real)
        os.mkdir(training_sample_path_generated)
        #os.mkdir(evaluation_sample_path)
        
    else:
        shutil.rmtree(evaluation_sample_path, ignore_errors=True)
        os.mkdir(evaluation_sample_path)
        os.mkdir(training_sample_path_real)
        os.mkdir(training_sample_path_generated)
    return training_sample_path_real,training_sample_path_generated


def create_model_samples_folder(model_name):
    sample_dir= os.path.join(OUTPUT_FOLDER,MODEL_SAMPLES_FOLDER)
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    model_sample_path= os.path.join(OUTPUT_FOLDER,MODEL_SAMPLES_FOLDER,model_name)
    modelSampleExist = os.path.exists(model_sample_path)
    if not modelSampleExist:
        os.mkdir(model_sample_path)

    model_sample_path= os.path.join(OUTPUT_FOLDER,MODEL_SAMPLES_FOLDER,model_name) 

    isExist = os.path.exists(model_sample_path)

    if not isExist:
        os.mkdir(model_sample_path) 
        #os.mkdir(evaluation_sample_path)
        
    else:
        shutil.rmtree(model_sample_path, ignore_errors=True)
        os.mkdir(model_sample_path) 
    return model_sample_path



def create_output_folders(model_name):
    
    outFolder = os.path.exists(OUTPUT_FOLDER) 
    if not outFolder:
        os.mkdir(OUTPUT_FOLDER)


    zipFolder= os.path.join(OUTPUT_FOLDER,ZIP_FOLDER)
    isZipFolderExist = os.path.exists(zipFolder) 
    if not isZipFolderExist:
        os.mkdir(zipFolder)


    ckptFolder= os.path.join(OUTPUT_FOLDER,CHECKPOINT_FOLDER)
    isCkptExists = os.path.exists(ckptFolder) 
    if not isCkptExists:
        os.mkdir(ckptFolder)


    ckptFolderModel=os.path.join(OUTPUT_FOLDER,CHECKPOINT_FOLDER,model_name)
    ckptFolderModelexist = os.path.exists(ckptFolderModel)

    if not ckptFolderModelexist:
        os.mkdir(ckptFolderModel)
    else:
        #print('file deleted')
        shutil.rmtree(ckptFolderModel, ignore_errors=True)
        os.mkdir(ckptFolderModel)

    trFolder= os.path.join(OUTPUT_FOLDER,TRAINING_RESULTS_FOLDER)
    trFolderExist = os.path.exists(trFolder)
    if not trFolderExist:
        os.mkdir(trFolder)

    
    evalFolder= os.path.join(OUTPUT_FOLDER,EVALUATION_FOLDER)
    evalFolderExist = os.path.exists(evalFolder)
    if not evalFolderExist:
        os.mkdir(evalFolder)
    

    resultFilePath =  os.path.join(trFolder,model_name+'.txt')  
    resultFileExist= os.path.isfile(resultFilePath)
    if resultFileExist:
        os.remove(resultFilePath)

    with open(resultFilePath, 'w') as fp:
        pass
    
    
    tsFolder= os.path.join(OUTPUT_FOLDER,TRAINING_SAMPLE_FOLDER)
    tsFolderExist = os.path.exists(tsFolder)
    if not tsFolderExist:
        os.mkdir(tsFolder)

    training_sample_path= os.path.join(tsFolder,model_name)
    isExist = os.path.exists(training_sample_path)
    if not isExist:
        os.mkdir(training_sample_path)
    else:
        shutil.rmtree(training_sample_path, ignore_errors=True)
        os.mkdir(training_sample_path)


