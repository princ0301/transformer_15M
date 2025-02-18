import torch
from typing import Tuple, Dict, Optional

class TransformerDebugger:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
    def validate_model_config(self) -> Tuple[bool, Optional[str]]:
        """Validate model configuration parameters."""
        checks = [
            (self.config['vocab_size'] > 0, "vocab_size must be positive"),
            (self.config['context_length'] > 0, "context_length must be positive"),
            (self.config['n_embed'] > 0, "n_embed must be positive"),
            (self.config['n_head'] > 0, "n_head must be positive"),
            (self.config['n_blocks'] > 0, "n_blocks must be positive"),
            (self.config['n_embed'] % self.config['n_head'] == 0, 
             f"n_embed ({self.config['n_embed']}) must be divisible by n_head ({self.config['n_head']})")
        ]
        
        for check, message in checks:
            if not check:
                return False, message
        return True, None

    def validate_batch(self, xb: torch.Tensor, yb: torch.Tensor) -> Dict[str, any]:
        """Validate input and target tensors."""
        stats = {
            'input_shape': xb.shape,
            'target_shape': yb.shape,
            'input_device': xb.device,
            'target_device': yb.device,
            'input_dtype': xb.dtype,
            'target_dtype': yb.dtype,
            'max_input_token': torch.max(xb).item(),
            'min_input_token': torch.min(xb).item(),
            'max_target_token': torch.max(yb).item(),
            'min_target_token': torch.min(yb).item(),
            'input_unique_tokens': torch.unique(xb).shape[0],
            'target_unique_tokens': torch.unique(yb).shape[0]
        }
        
        return stats

    def print_batch_stats(self, stats: Dict[str, any]) -> None:
        """Print batch statistics in a readable format."""
        print("\n=== Batch Statistics ===")
        print(f"Input shape: {stats['input_shape']}")
        print(f"Target shape: {stats['target_shape']}")
        print(f"Input device: {stats['input_device']}")
        print(f"Target device: {stats['target_device']}")
        print(f"\nToken Statistics:")
        print(f"Input token range: [{stats['min_input_token']}, {stats['max_input_token']}]")
        print(f"Target token range: [{stats['min_target_token']}, {stats['max_target_token']}]")
        print(f"Unique tokens in input: {stats['input_unique_tokens']}")
        print(f"Unique tokens in target: {stats['target_unique_tokens']}")
        print(f"\nVocab size: {self.config['vocab_size']}")
        print(f"Context length: {self.config['context_length']}")
        
        # Check for potential issues
        if stats['max_input_token'] >= self.config['vocab_size']:
            print(f"\n⚠️  WARNING: Input contains token indices >= vocab_size")
        if stats['max_target_token'] >= self.config['vocab_size']:
            print(f"\n⚠️  WARNING: Target contains token indices >= vocab_size")
        if stats['input_shape'][1] > self.config['context_length']:
            print(f"\n⚠️  WARNING: Input sequence length exceeds context_length")