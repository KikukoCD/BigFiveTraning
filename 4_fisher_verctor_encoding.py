import numpy as np
import pdb
import pickle
from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture as GMM
import pandas as pd
import joblib
import os

def load_feature_vectors(csv_file_path):
    # Load feature vectors from CSV file.
    df = pd.read_csv(csv_file_path, header=None)
    X = df.to_numpy()
    X = X.T
    return X


def fisher_vector(xx, gmm):



    xx = np.atleast_2d(xx)
    N = xx.shape[0]

    # Compute posterior probabilities.
    Q = gmm.predict_proba(xx)  # NxK

    # Compute the sufficient statistics of descriptors.
    Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
    Q_xx = np.dot(Q.T, xx) / N
    Q_xx_2 = np.dot(Q.T, xx ** 2) / N


    # Compute derivatives with respect to mixing weights, means and variances.
    d_pi = Q_sum.squeeze() - gmm.weights_
    d_mu = Q_xx - Q_sum * gmm.means_

    d_sigma = (
        - Q_xx_2
        - Q_sum * gmm.means_ ** 2
        + Q_sum * gmm.covariances_
        + 2 * Q_xx * gmm.means_)



    # Merge derivatives into a vector.
    ##return np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))


    return np.hstack((d_mu.flatten(), d_sigma.flatten()))


def main():
    # Path to directory containing input CSV files.
    input_dir = './train_to_400/'

    # Path to directory to save output CSV files.
    output_dir = 'D:/train_GMM50PCA400'

    # Load pre-trained GMM.
    gmm = joblib.load('gmm50_pca400.pkl')

    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            # Load CSV file containing feature vectors.
            input_file = os.path.join(input_dir, filename)
            X = load_feature_vectors(input_file)
            X = X.T

            # Generate Fisher vectors.
            fv = fisher_vector(X, gmm)

            # Save Fisher vectors to CSV file.
            output_file = os.path.join(output_dir, filename.replace('_outputs_pca.csv', '.csv'))
            np.savetxt(output_file, fv, delimiter=",")



if __name__ == '__main__':
    main()