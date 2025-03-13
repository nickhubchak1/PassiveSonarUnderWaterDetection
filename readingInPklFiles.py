
#-----------------------------------------------------
# Deep Learning Final Project 2025
# Under Water Passive Acoustic Source Localization
# Author: Nick Hubchak
# All Rights Reserved 2025-2030
#----------------------------------------------------


import os
import joblib

folder_path = 'swellex_real_vla_ts_data'

pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]

loaded_data = {}

for pkl_file in pkl_files:
    file_path = os.path.join(folder_path, pkl_file)
    try:
        data = joblib.load(file_path)
        loaded_data[pkl_file] = data
        print(f"Successfully loaded {pkl_file}")
    except Exception as e:
        print(f"Failed to load {pkl_file}: {e}")


for file_name in loaded_data.keys():
    data_item = loaded_data[file_name]
    print(f"Data from {file_name}: ")
    print(data_item)
   