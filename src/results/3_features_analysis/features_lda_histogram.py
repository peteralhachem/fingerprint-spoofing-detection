from src.features_analysis.plots import plot_lda_histogram
from src.utilities.datasets import load_fingerprint_spoofing_detection_train_set

if __name__ == '__main__':
    # import fingerprint train set
    data_matrix, labels = load_fingerprint_spoofing_detection_train_set()

    # plot histogram after performing lda with 1-direction
    plot_lda_histogram(data_matrix, labels)
