from src.preprocessing.pca import PCA
from src.utilities.datasets import load_fingerprint_spoofing_detection_train_set

if __name__ == '__main__':
    # import fingerprint train set
    data_matrix, labels = load_fingerprint_spoofing_detection_train_set()

    # plot the explained variance after performing PCA for n = 2
    pca = PCA(n_components=2)
    pca.fit(data_matrix)
    pca.plot_explained_variance_pca()
