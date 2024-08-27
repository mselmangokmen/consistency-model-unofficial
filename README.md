# Unofficial repository for training consistency models

You can train and generate samples by running the main.py file.
For hyperparameter adjustments, just edit parameters.yaml file.

# For running unconditional image generation on multiple gpus or a single gpu :

**set nproc_per_node as [number of nodes] and nnodes as [desired number of GPUs].**
```
torchrun --nnodes=1 --nproc_per_node=2 train_hn_unconditional.py --model_name your_model_name --dataset_name cifar10  --batch_size 512  --total_training_steps 800000 --model_type small --curriculum gokmen  --dist_type beta --use_ema False 

```


# For running conditional image generation on multiple gpus or a single gpu :
```
torchrun --nnodes=1 --nproc_per_node=2 train_ldct_CM.py --model_name your_model_name --batch_size 12 --num_res_blocks 2 --dropout 0.1 --total_training_steps 400000     --constant_N False   

```
 