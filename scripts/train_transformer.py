import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
import numpy as np
import sys
from debug_utils import TransformerDebugger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import default_config as config
from src.models.transformer import Transformer
from data_loader.data_loader import get_batch_iterator

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main(): 
    model = Transformer(
        n_head=config['n_head'],
        n_embed=config['n_embed'],
        context_length=config['context_length'],
        vocab_size=config['vocab_size'],
        N_BLOCKS=config['n_blocks']
    ).to(config['device'])
 
    debugger = TransformerDebugger(model, config)
 
    valid_config, config_error = debugger.validate_model_config()
    if not valid_config:
        print(f"Invalid model configuration: {config_error}")
        return
 
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters in the model: {total_params:,}")
 
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['t_lr'])
     
    batch_iterator = get_batch_iterator(
        config['train_path'],
        config['t_batch_size'],
        config['t_context_length'],
        device=config['device']
    )
 
    losses = []
    pbar = tqdm(range(config['t_train_steps']))
    
    try:
        for step in pbar:
            try: 
                xb, yb = next(batch_iterator)
                 
                if step == 0 or step % 1000 == 0:
                    stats = debugger.validate_batch(xb, yb)
                    debugger.print_batch_stats(stats)
                    
                    if stats['max_input_token'] >= config['vocab_size']:
                        raise ValueError("Input tokens exceed vocabulary size")
                 
                _, loss = model(xb, yb)
                 
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
                avg_loss = np.mean(losses[-64:]) if losses else 0
                pbar.set_description(f"Loss: {avg_loss:.4f}")
                
                if step == config['t_lr_decay_step']:
                    print('Decaying learning rate')
                    for g in optimizer.param_groups:
                        g['lr'] = config['t_lr_decayed']
                        
            except StopIteration:
                print("Training data iterator finished")
                break
                
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
   
    try:
        os.makedirs(os.path.dirname(config['t_out_path']), exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': losses,
        }, config['t_out_path'])
        print(f"Model saved to {config['t_out_path']}")
    except Exception as e:
        print(f"Error saving model: {str(e)}")

if __name__ == "__main__":
    main()