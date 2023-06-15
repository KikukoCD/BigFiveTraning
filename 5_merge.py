import numpy as np
import os

# Folder path
folder_path = 'C:/Users/zhang/PycharmProjects/final_version/fv_400_50'

# Get the paths of all csv files in the folder
csv_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

# Read all csv files
csv_data = [np.genfromtxt(f, delimiter=',').T for f in csv_paths]

#  Stacking all data
stacked_data = np.vstack(csv_data)

# # Save as npy file
np.save('C:/Users/zhang/PycharmProjects/final_version/train.npy', stacked_data)