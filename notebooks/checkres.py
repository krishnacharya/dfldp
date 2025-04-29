import os
import pandas as pd
from src.project_dirs import output_dir_name


#CHECK res of sbatch

if __name__ == "__main__":
    filepath = str(output_dir_name('vsb_100'))
    print(filepath)
    files = os.listdir(filepath)
    csv_files = [file for file in files if file.endswith('.csv')]
    if len(csv_files) != 1:
        raise ValueError("There should be exactly one CSV file in the directory, the concatenated one.")
    file = os.path.join(filepath, csv_files[0])
    df_all = pd.read_csv(file)
    print((df_all['testdq_dfl'] > df_all['testdq_lr']).sum() / df_all.shape[0] * 100)
    # percent = []
    # for k in df_all['key'].unique(): # check case by case for each key in which dfl is better
    #     df_k = df_all[df_all['key'] == k]
    #     percent.append((k, (df_k['testdq_dfl'] >= df_k['testdq_lr']).sum() / df_k.shape[0] * 100))

    # sorted_k = sorted(percent, key=lambda x: x[1], reverse=True)