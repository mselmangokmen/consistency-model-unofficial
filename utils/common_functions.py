
import copy
from torchvision.utils import save_image

import torchvision
import os 
import shutil  
import torch
import math 

TRAINING_SAMPLE_FOLDER='training_samples'
EVALUATION_FOLDER='evaluation'
CHECKPOINT_FOLDER='checkpoints'
TRAINING_RESULTS_FOLDER='training_results'
SCHEDULE_IMPROVED='improved'
SCHEDULE_GOKMEN='gokmen'
EVALUATION_RESULT_FILE= os.path.join(EVALUATION_FOLDER,'evaluation_results.txt')
def save_tensor_as_grid(tensor: torch.Tensor, filename: str, nrow: int = 4) -> None:
    
    grid = torchvision.utils.make_grid(tensor, nrow=nrow)
 
    save_image(grid, filename)


def write_to_evaluation_results(result):
    if os.path.isfile(EVALUATION_RESULT_FILE):
        file1 = open(EVALUATION_RESULT_FILE, "a") 

    else:
        file1 = open(EVALUATION_RESULT_FILE, "w+") 

    file1.write(result)
    file1.close()

def save_grid(tensor,model_name,sample_step,epoch):
    model_sample_path= os.path.join(TRAINING_SAMPLE_FOLDER,model_name)

    file_name= 'sample_step_'+ str(sample_step)+ '_epoch_'+str(epoch)+'.png'
    full_path= os.path.join(model_sample_path,file_name)
    save_tensor_as_grid(tensor,full_path, int(math.sqrt(tensor.shape[0])))



def check_evaluation_dataset(dataset_name):

    evaluation_sample_path= os.path.join(EVALUATION_FOLDER,dataset_name)
    isExist = os.path.exists(evaluation_sample_path)
    cnt=0
    if isExist:
        cnt = len(os.listdir(evaluation_sample_path))
    return isExist,cnt

 
def save_image_list_in_dataset_dir(image_list,dataset_name,offset=0):
    for idx,img in enumerate(image_list):
        idx= offset + idx
        #print(idx)
        training_sample_path_generated= os.path.join(EVALUATION_FOLDER,dataset_name,'img_'+str(idx)+ '.png')
        img.save(training_sample_path_generated)

def save_generated_images(image_list,model_name,offset=0):
    final_offset=0
    for idx,img in enumerate(image_list):
        idx= offset + idx
        #print(idx)
        training_sample_path_generated= os.path.join(EVALUATION_FOLDER,model_name,'generated','img_'+str(idx)+ '.png')
        img.save(training_sample_path_generated)
        final_offset=idx+1
    return final_offset

def save_real_images(image_list,model_name,offset=0):
    final_offset=0
    for idx,img in enumerate(image_list):
        idx= offset + idx
        #print(idx)
        training_sample_path_generated= os.path.join(EVALUATION_FOLDER,model_name,'real','img_'+str(idx)+ '.png')
        img.save(training_sample_path_generated)
        final_offset=idx+1
    return final_offset

def save_log(model_name,record):
    
    resultFilePath =  os.path.join(TRAINING_RESULTS_FOLDER,model_name+'.txt')  
    file1 = open(resultFilePath, "a")  # append mode
    file1.write(  record+" \n")
    file1.close()

def save_checkpoint(model,model_name,epoch):
    
    ckpt_name= model_name+'_'+str(epoch)+'_ckpt.pt'
    state_dict= copy.deepcopy(model.state_dict())

    torch.save(state_dict, os.path.join(CHECKPOINT_FOLDER,model_name, ckpt_name))

def get_checkpoint(model_name,epoch):
    ckpt_name= model_name+'_'+str(epoch)+'_ckpt.pt'
    ckpt_path=  os.path.join(CHECKPOINT_FOLDER,model_name, ckpt_name)
    return torch.load(ckpt_path)


def save_state_dict(state_dict,model_name,epoch):
    
    ckpt_name= model_name+'_'+str(epoch)+'_ckpt.pt'
    state_dict= copy.deepcopy(state_dict)
    
    torch.save(state_dict, os.path.join(CHECKPOINT_FOLDER,model_name, ckpt_name))
 

def create_sampling_folder(model_name):

    training_sample_path= os.path.join(TRAINING_SAMPLE_FOLDER,model_name)
    isExist = os.path.exists(training_sample_path)
    if not isExist:
        os.mkdir(training_sample_path)
    else:
        shutil.rmtree(training_sample_path, ignore_errors=True)
        os.mkdir(training_sample_path)


def create_dataset_folder_in_evaluation(dataset_name):

    evaluation_dataset_path= os.path.join(EVALUATION_FOLDER,dataset_name)
    isExist = os.path.exists(evaluation_dataset_path)
    if not isExist:
        os.mkdir(evaluation_dataset_path)
    else:
        shutil.rmtree(evaluation_dataset_path, ignore_errors=True)
        os.mkdir(evaluation_dataset_path)
    return evaluation_dataset_path

def get_dataset_folder_name_in_evaluation(dataset_name):
    return os.path.join(EVALUATION_FOLDER,dataset_name)

def create_evaluation_folder(model_name):

    evalFolder = os.path.exists(EVALUATION_FOLDER)
    if not evalFolder:
        os.mkdir(EVALUATION_FOLDER)

    evaluation_sample_path= os.path.join(EVALUATION_FOLDER,model_name)
    training_sample_path_real= os.path.join(EVALUATION_FOLDER,model_name,'real')
    training_sample_path_generated= os.path.join(EVALUATION_FOLDER,model_name,'generated')

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


def create_output_folders(model_name):
     
    ckptFolder = os.path.exists(CHECKPOINT_FOLDER) 
    if not ckptFolder:
        os.mkdir(CHECKPOINT_FOLDER)


    ckptFolderModel=os.path.join(CHECKPOINT_FOLDER,model_name)
    ckptFolderModelexist = os.path.exists(ckptFolderModel)

    if not ckptFolderModelexist:
        os.mkdir(ckptFolderModel)
    else:
        #print('file deleted')
        shutil.rmtree(ckptFolderModel, ignore_errors=True)
        os.mkdir(ckptFolderModel)

    trFolderExist = os.path.exists(TRAINING_RESULTS_FOLDER)
    if not trFolderExist:
        os.mkdir(TRAINING_RESULTS_FOLDER)

    resultFilePath =  os.path.join(TRAINING_RESULTS_FOLDER,model_name+'.txt')  
    resultFileExist= os.path.isfile(resultFilePath)
    if resultFileExist:
        os.remove(resultFilePath)

    with open(resultFilePath, 'w') as fp:
        pass
    
    
    tsFolder = os.path.exists(TRAINING_SAMPLE_FOLDER)

    if not tsFolder:
        os.mkdir(TRAINING_SAMPLE_FOLDER)

    training_sample_path= os.path.join(TRAINING_SAMPLE_FOLDER,model_name)
    isExist = os.path.exists(training_sample_path)
    if not isExist:
        os.mkdir(training_sample_path)
    else:
        shutil.rmtree(training_sample_path, ignore_errors=True)
        os.mkdir(training_sample_path)
    