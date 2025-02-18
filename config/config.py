import torch
# 15 Million Parameters (155,651,712)
VOCAB_SIZE = 50304           
CONTEXT_LENGTH = 512         
N_EMBED = 768              
N_HEAD = 12               
N_BLOCKS = 12      
 
TRAIN_PATH = "/home/pooja/Prince/13M_LLM/data/train/pile_train.h5"   
DEV_PATH = "/home/pooja/Prince/13M_LLM/data/val/pile_dev.h5"       
 
T_BATCH_SIZE = 16        
T_CONTEXT_LENGTH = 16       
T_TRAIN_STEPS = 200000      
T_EVAL_STEPS = 1000         
T_EVAL_ITERS = 250          
T_LR_DECAY_STEP = 50000     
T_LR = 5e-4                
T_LR_DECAYED = 5e-5         
T_OUT_PATH = "models/transformer_13M.pt"   
 
DEVICE = 'cuda'
 
default_config = {
    'vocab_size': VOCAB_SIZE,
    'context_length': CONTEXT_LENGTH,
    'n_embed': N_EMBED,
    'n_head': N_HEAD,
    'n_blocks': N_BLOCKS,
    'train_path': TRAIN_PATH,
    'dev_path': DEV_PATH,
    't_batch_size': T_BATCH_SIZE,
    't_context_length': T_CONTEXT_LENGTH,
    't_train_steps': T_TRAIN_STEPS,
    't_eval_steps': T_EVAL_STEPS,
    't_eval_iters': T_EVAL_ITERS,
    't_lr_decay_step': T_LR_DECAY_STEP,
    't_lr': T_LR,
    't_lr_decayed': T_LR_DECAYED,
    't_out_path': T_OUT_PATH,
    'device': DEVICE,
}