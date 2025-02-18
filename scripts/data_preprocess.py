import os
import json
import zstandard as zstd
import tiktoken
import h5py
from tqdm import tqdm
import argparse
from typing import Optional

def process_files(input_dir: str, output_file: str, tokenizer_name: str, max_data: Optional[int] = None) -> None:
    if max_data is not None:
        print(f"You have choosen max_data = {max_data}. Processing only the top {max_data} JSON objects from each file.")
    else:
        print("Processing all available JSON objects from each file.")

    enc = tiktoken.get_encoding(tokenizer_name)

    with h5py.File(output_file, 'w') as out_f:
        dataset = out_f.create_dataset('tokens', (0,), maxshape=(None,), dtype='i')
        start_index = 0

        for filename in sorted(os.listdir(input_dir)):
            if filename.endswith(".jsonl.zst"):
                in_file = os.path.join(input_dir, filename)
                print(f"Processing: {in_file}")

                processed_lines = 0

                with zstd.open(in_file, 'rt', encoding='utf-8') as in_f:
                    for line in tqdm(in_f, desc=f"Processing {filename}", total=max_data if max_data is not None else None):
                        try:
                            data = json.loads(line)
                            text = data.get('text')

                            if text:
                                encoded = enc.encode(text + "<|endoftext|>", allowed_special={'<|endoftext|>'})
                                encoded_len = len(encoded)

                                end_index = start_index + encoded_len
                                dataset.resize(dataset.shape[0] + encoded_len, axis=0)

                                dataset[start_index:end_index] = encoded
                                start_index = end_index
                            else:
                                print(f"Warning: 'text' key missing in line from {filename}" )
                        except json.JSONDecodeError:
                            print(f"Warning: Could not decode JSON from line in {filename}")
                        except Exception as e:
                            print(f"An error occurred while processing line in {filename}: {e}")

                        processed_lines += 1

                        if max_data is not None and processed_lines >= max_data:
                            break

def main():

    parser = argparse.ArgumentParser(description="Preprocess PILE dataset files and save tokens to HDF5.")
    parser.add_argument("--train_dir", type=str, default="data/train", help="Directory containing training .jsonl.zst files.")
    parser.add_argument("--val_dir", type=str, default="data/val", help="Directory containing validation .jsonl.zst files.")
    parser.add_argument("--out_train_file", type=str, default="data/train/pile_train.h5", help="Path to the output training HDF5 file.")
    parser.add_argument("--out_val_file", type=str, default="data/val/pile_dev.h5", help="Path to the output validation HDF5 file.")
    parser.add_argument("--tokenizer_name", type=str, default="r50k_base", help="Name of the tiktoken tokenizer to use.")
    parser.add_argument("--max_data", type=int, default=1000, help="Maximum number of json objects to process from each file in both train and val datasets (default: 1000).")

    args = parser.parse_args()

    if not os.path.isdir(args.train_dir):
        print(f"Error: Training directory not found: {args.train_dir}")
        return
    if not os.path.isdir(args.val_dir):
        print(f"Error: Validation directory not found: {args.val_dir}")
        return
    
    print("Starting training data preprocessing...")
    process_files(args.train_dir, args.out_train_file, args.tokenizer_name, args.max_data)
    print("Training data processing complete.")

    print("Starting validation data preprocessing...")
    process_files(args.val_dir, args.out_val_file, args.tokenizer_name, args.max_data)
    print("Validation data preprocessing complete.")

if __name__=="__main__":
    main()

                        