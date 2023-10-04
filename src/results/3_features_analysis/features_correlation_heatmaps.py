from src.features_analysis.plots import plot_centered_dataset_heatmaps
from src.utilities.datasets import load_fingerprint_spoofing_detection_train_set

if __name__ == '__main__':
    # import fingerprint train set
    data_matrix, labels = load_fingerprint_spoofing_detection_train_set()
    # plot heatmaps after centering dataset
    plot_centered_dataset_heatmaps(data_matrix, labels)
