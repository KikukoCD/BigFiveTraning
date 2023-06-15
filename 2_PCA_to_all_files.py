import os
import pandas as pd
import pickle

# Load the trained PCA model
with open('pca_model_768to400.pkl', 'rb') as f:
    pca = pickle.load(f)

# Set the input and output folder paths
input_folder_path = 'D:/sample/train'
output_folder_path = './train_to_400/'

# Iterate through all .csv files in the input folder
for file_name in os.listdir(input_folder_path):
    if file_name.endswith('.csv') and not file_name.endswith('_400.csv'):
        # Read raw csv file
        data = pd.read_csv(os.path.join(input_folder_path, file_name), header=None)
        # Extracting feature data
        X = data.iloc[:, :-1].values
        # Perform PCA downscaling to 400 dimensions
        X_pca = pca.transform(X)
        # Save the descended csv file
        out_file_name = file_name[:-4] + '_400.csv'
        out_path = os.path.join(output_folder_path, out_file_name)
        pd.DataFrame(X_pca).to_csv(out_path, header=False, index=False)