


#from consistency_model.cm_training import cm_main
from consistency_model.cm_functions import trainCM_Improved_epoch
from consistency_model.cm_training import cm_main, cm_train_improved_500k
from ddpm_model.ddim_sample import ddim_main, ddim_sample_full
from ddpm_model.ddpm_training import ddpm_main 

#cm_main()
#ddpm_main()
#ddim_main()


# Import the library
import argparse
# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--type', type=str, required=True)
parser.add_argument('--gpu', type=str, required=False)
parser.add_argument('--schedule', type=str, required=False)
parser.add_argument('--model_name', type=str, required=False)
parser.add_argument('--epochs', type=int, required=False)
parser.add_argument('--tsteps', type=int, required=False)
# Parse the argument
args = parser.parse_args()

# Print "Hello" + the user input argument

if args.type=='ddim-sample':
    ddim_main()

elif args.type=='ddim-fid-full':
    ddim_sample_full()

elif args.type=='ddpm-train':
    ddpm_main()
elif args.type=='cm-train':
    gpu = str(args.gpu) 
    model_name= str(args.model_name)
    epochs= int(args.epochs)

    cm_main(cuda_device=gpu,model_name=model_name,epochs=epochs) 

elif args.type=='cm-train-improved-epoch': 
    schedule= str(args.schedule)
    model_name= str(args.model_name)
    epochs= int(args.epochs)
    gpu = str(args.gpu) 

    trainCM_Improved_epoch(schedule=schedule,model_name=model_name,total_training_steps=epochs,device=gpu) 

elif args.type=='cm-train-improved-no-epoch': 
    schedule= str(args.schedule)
    model_name= str(args.model_name)
    tsteps= int(args.tsteps)
    cm_train_improved_500k( schedule=schedule,
                        model_name=model_name,      
                        total_training_steps=400000)  


