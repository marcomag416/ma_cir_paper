# metadata files can be downloaded with: curl -L -o ./data/metadata/laion  https://www.kaggle.com/api/v1/datasets/download/romainbeaumont/laion400m

#run example: python utils/datasets/laion/sample_dataset.py --input_dir ./data/metadata/laion400m --output_dir ./data/metadata/laion_sampled --num_samples 2000000 --success_rate 0.6

import pandas as pd
import argparse
import os
from tqdm.auto import tqdm

def main(input_dir, output_dir, num_samples, random_seed):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    num_files = len([f for f in os.listdir(input_dir) if f.endswith('.parquet')])
    print(f"Found {num_files} parquet files in {input_dir}")
    samples_per_file = int(num_samples // num_files)

    print(f"Sampling {samples_per_file} samples from each file to get a total of {samples_per_file * num_files} samples.")

    for file in tqdm(os.listdir(input_dir), desc="Processing files"):
        if file.endswith(".parquet"):
            file_path = os.path.join(input_dir, file)
            df = pd.read_parquet(file_path)

            #keep only images with width and height >= 256
            df = df[(df['WIDTH'] >= 256) & (df['HEIGHT'] >= 256)]

            # Sample the dataframe
            sampled_df = df.sample(n=samples_per_file, random_state=random_seed)

            # Save the sampled dataframe to a new parquet file
            output_file_path = os.path.join(output_dir, file)
            sampled_df.to_parquet(output_file_path)
            print(f"Saved sampled data to {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="Path to the input parquet file")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory")
    parser.add_argument("--num_samples", type=int, default=2000000, help="Number of samples to extract")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--success_rate", type=float, default=1.0, help="Expected success rate for downloading images")
    args = parser.parse_args()

    num_samples = args.num_samples / args.success_rate # Adjust number of samples based on expected success rate

    main(args.input_dir, args.output_dir, num_samples, args.random_seed)
