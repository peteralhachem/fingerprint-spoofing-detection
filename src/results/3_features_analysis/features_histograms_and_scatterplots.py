from src.features_analysis.plots import plot_centered_attributes_pairs_scatter_plots
from src.utilities.datasets import load_fingerprint_spoofing_detection_train_set

if __name__ == '__main__':
    # import fingerprint train set
    data_matrix, labels = load_fingerprint_spoofing_detection_train_set()

    # plot graphs after centering dataset
    plot_centered_attributes_pairs_scatter_plots(data_matrix, labels)
