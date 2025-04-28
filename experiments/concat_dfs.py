import os
import pandas as pd
import argparse
from src.project_dirs import output_dir_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='concatenate all csv files in a directory')
    parser.add_argument('--op_name', type=str, required=True, help='Path to the directory containing csv files')
    args = parser.parse_args()
    op_name = args.op_name
    filepath = str(output_dir_name(op_name))

    print(f"Concatenating CSV files in directory: {filepath}")
    files = os.listdir(filepath)
    csv_files = [file for file in files if file.endswith('.csv')]
    print(f"Number of .csv files: {len(csv_files)}")

    df_all = pd.concat([pd.read_csv(os.path.join(filepath, file)) for file in csv_files], ignore_index=True)
    #remove all csv files in directory
    for file in csv_files:
        os.remove(os.path.join(filepath, file))
    #save concatenated dataframe to csv
    output_file = os.path.join(filepath, 'concatenated_data.csv')
    df_all.to_csv(output_file, index=False)