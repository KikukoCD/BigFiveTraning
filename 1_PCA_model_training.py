import pandas as pd
from sklearn.decomposition import PCA
import pickle

# Read csv files
data = pd.read_csv('merged_subsampled_wav2vec.csv', header=None)
# Extracting feature data
X = data.iloc[:, :-1].values
# Perform PCA downscaling to 400 dimensions
pca = PCA(n_components=400)
X_pca = pca.fit_transform(X)

# Save the downscaled data as a csv file
df = pd.DataFrame(X_pca)
df.to_csv('pca_400_training_data.csv', index=False)

 # Save the trained PCA model as a pkl file
with open('pca_model.pkl', 'wb') as f:
     pickle.dump(pca, f)











