from src.features_analysis.plots import plot_2d_pca_features_plot
from src.utilities.datasets import load_fingerprint_spoofing_detection_train_set

if __name__ == '__main__':
    # import fingerprint train set
    data_matrix, labels = load_fingerprint_spoofing_detection_train_set()

    # plot histograms and scatter plots after performing PCA for n = 2
    plot_2d_pca_features_plot(data_matrix, labels)
