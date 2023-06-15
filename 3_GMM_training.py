import numpy as np
from sklearn.mixture import GaussianMixture
import joblib
import csv

# Load data from csv files and concatenate into a 2D array
data = []
with open('pca_400_training_data.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        data.append([float(x) for x in row])
data = np.array(data)

# Train the Gaussian Mixture Model using the reduced data
gmm = GaussianMixture(n_components=50,covariance_type='diag', verbose=1, random_state=42, max_iter=300)

gmm.fit(data)

# Save the trained GMM models to local files
joblib.dump(gmm, 'gmm50_pca400.pkl')