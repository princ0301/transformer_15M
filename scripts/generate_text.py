import torch
import tiktoken
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import default_config as config
from src.models.transformer import Transformer   

def load_model(model_path: str, device: str = 'cuda'):
     
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    checkpoint = torch.load(model_path, map_location=torch.device(device))

    model = Transformer(
        n_head=config['n_head'],
        n_embed=config['n_embed'],
        context_length=config['context_length'],
        vocab_size=config['vocab_size'],
        N_BLOCKS=config['n_blocks']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model

def generate_text(model, tokenizer, input_text: str, max_new_tokens: int = 100, device: str = 'cuda') -> str:
    start_ids = tokenizer.encode_ordinary(input_text)
    context = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        generated_tokens = model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()

    return tokenizer.decode(generated_tokens)

def main():
    model_path = "/home/pooja/Prince/13M_LLM/scripts/models/transformer_13M.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        model = load_model(model_path, device)
        tokenizer = tiktoken.get_encoding("r50k_base")
    except Exception as e:
        print(f"Error: {e}")
        return

    print("\n🚀 Model loaded! Type a prompt and press Enter to generate text.")
    print("🔴 Type 'exit' to quit.\n")

    while True:
        input_text = input("You: ").strip()
        if input_text.lower() == "exit":
            print("👋 Exiting. Have a great day!")
            break

        output_text = generate_text(model, tokenizer, input_text, max_new_tokens=100, device=device)
        print(f"🤖 AI: {output_text}\n")

if __name__ == "__main__":
    main()
