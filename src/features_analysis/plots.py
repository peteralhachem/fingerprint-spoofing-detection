import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.utilities.arrays import vcol
from src.utilities.statistics import mean_and_covariance_of

from src.preprocessing.pca import PCA
from src.preprocessing.z_norm import ZNorm
from src.preprocessing.lda import LDA

classes = {"colors": ["#0000ff", "#ff0000"],
           "labels": ["Spoofed Fingerprint", "Authentic Fingerprint"]}


def plot_centered_attributes_pairs_scatter_plots(dataset, labels, filepath=None):
    """
    Center the dataset, removing its mean from all samples, and plot the related features
    pairs' scatter plots.
    :param dataset: 2-D numpy array of the dataset (10, 2325)
    :param labels: 1-D numpy array of labels (2325)
    :param filepath: (optional) path of the file to save the figure
    """
    _plot_centered(_plot_pairs_features_scatter_plots, dataset, labels, filepath=filepath)


def _plot_centered(plot_func, dataset, labels, filepath=None):
    """
    Center the IRIS dataset and plot it with the specified function.
    :param plot_func: a plot function that takes the dataset and labels as parameters.
    :param dataset: 2-D numpy array of the dataset (10, 2325)
    :param labels: 1-D numpy array of labels (2325)
    :param filepath: (optional) path of the file to save the figure
    """

    # center dataset computing the attributes' means
    features_means = vcol(dataset.mean(axis=1))  # 2-D column vector
    centered_dataset = dataset - features_means  # leveraging broadcasting

    # plot the centered dataset
    plot_func(centered_dataset, labels, filepath=filepath)


def _plot_pairs_features_scatter_plots(dataset, labels, filepath=None):
    """
    Plot in a 10x10 matrix a figure for each couple of dataset features, showing a scatter-plot of the samples
    for each class on each pair of attributes. In the diagonal subplots the attributes histograms are shown.
    :param dataset: 2-D numpy array of the dataset (D, 2325) where D is the number of features.
    :param labels: 1-D numpy array of labels (2325)
    :param filepath: (optional) path of the file to save the figure
    """

    num_attributes = dataset.shape[0]
    fig, axs = plt.subplots(num_attributes, num_attributes, figsize=(15, 15))

    for feature_1 in range(num_attributes):
        for feature_2 in range(num_attributes):
            subplot = axs[feature_1, feature_2]
            subplot.set_xlabel('')
            subplot.set_ylabel('')

            if feature_1 == feature_2:
                _plot_feature_histograms(subplot, dataset, labels, feature_1)
            else:
                _plot_scatter_feature_pairs(subplot, dataset, labels, feature_1, feature_2)

    plt.subplots_adjust(wspace=0.65, hspace=0.55, left=0.05, top=0.95, right=0.95, bottom=0.05)
    if filepath is not None:
        plt.savefig(filepath, bbox_inches='tight')
    plt.show()


# plot features histograms and scatter plots

def _plot_feature_histograms(subplot, data_matrix, labels, feature):
    """
    Plot a histogram for each class for the specified feature
    :param subplot: matplotlib subplot where to draw the histogram
    :param data_matrix: 2-D numpy array of the dataset (10, 2325) with one sample per column
    :param labels: 1-D numpy array of labels (2325)
    :param feature: index of the interested feature

    """

    for class_label in np.unique(labels):
        class_dataset = data_matrix[feature, labels == class_label]
        subplot.hist(
            class_dataset,
            bins=25,
            density=True,
            color="%s77" % classes["colors"][class_label],
            histtype="barstacked",
            label=classes["labels"][class_label]
        )


def _plot_scatter_feature_pairs(subplot, dataset, labels, feature_1, feature_2):
    """
    Plot in the specified subplot a scatter plot for each class,
    showing the values for two different specified attributes

    :param subplot: matplotlib subplot where to draw
    :param dataset: 2-D numpy array of the dataset (10, 2325) with one sample per column
    :param labels: 1-D numpy array of labels (2325)
    :param feature_1: value of the attribute from the range [0,9]
    :param feature_2: value of the attribute from the range [0,9] **different** from feature_1
    """

    for class_label in np.unique(labels):
        # take all the attributes values related to that class
        class_feature1_values = dataset[feature_1, labels == class_label]
        class_feature2_values = dataset[feature_2, labels == class_label]

        # plot class scatter plot
        subplot.scatter(
            class_feature1_values,
            class_feature2_values,
            color="%s" % classes["colors"][class_label],
            label=classes["labels"][class_label],
            alpha=0.5,
            marker="."
        )

        subplot.set_xlim(-25, 25)
        subplot.set_ylim(-25, 25)


# Heatmap

def plot_centered_dataset_heatmaps(dataset, labels, filepath=None):
    """
    Plot the heatmap of the full dataset, the Spoofed Fingerprint dataset only and the authenticated dataset only.
    The axis of the heatmaps represents the features of the dataset.
    :param dataset: 2-D numpy array of the dataset (10, 2325)
    :param labels: 1-D numpy array of the labels (2325)
    :param filepath: (optional) filename to save the image
    """

    arguments = {
        "dataset": [dataset, dataset[:, labels == 0], dataset[:, labels == 1]],
        "color": ["Greys", "Blues", "Reds"],
        "title": ["Full Dataset Heatmap", "Spoofed Fingerprint Dataset Heatmap",
                  "Authenticated Fingerprint Dataset Heatmap"]
    }

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    for index in range(len(arguments["color"])):
        # center the dataset using z-normalization
        z_norm = ZNorm()
        z_norm.fit(arguments["dataset"][index])
        preprocessed_data = z_norm.preprocess(arguments["dataset"][index])

        sns.heatmap(
            mean_and_covariance_of(preprocessed_data)[1],
            cmap=arguments["color"][index],
            annot=False,
            ax=ax[index],
            square=True
        )

        ax[index].set_title(arguments["title"][index], pad=15)
        ax[index].set_xlabel("Features")
        ax[index].set_ylabel("Features")

    # plt.tight_layout()
    plt.subplots_adjust(left=0.05, top=0.92, right=0.98, bottom=0.08)

    if filepath is not None:
        plt.savefig(filepath)

    plt.show()


# Dataset PCA/LDA analysis

def plot_2d_pca_features_plot(dataset, labels, filename=None):
    pca = PCA(n_components=2)

    pca.fit(dataset)
    reduced_data = pca.preprocess(dataset)

    fig, axs = plt.subplots(2, 2)

    for index, ax in enumerate(axs.flat):
        if index == 0:
            for class_label in np.unique(labels):
                class_dataset = reduced_data[0, labels == class_label]
                ax.hist(
                    class_dataset,
                    bins=25,
                    density=True,
                    color="%s77" % classes["colors"][class_label],
                    histtype="barstacked",
                    label=classes["labels"][class_label]
                )

        elif index == 1:
            for class_label in np.unique(labels):
                # take all the attributes values related to that class
                class_values_x = reduced_data[0, labels == class_label]
                class_values_y = reduced_data[1, labels == class_label]

                # plot class scatter plot
                ax.scatter(
                    class_values_x,
                    class_values_y,
                    color="%s" % classes["colors"][class_label],
                    label=classes["labels"][class_label],
                    alpha=0.5,
                    marker="."
                )

        elif index == 2:
            ax.axis("off")
            continue

        elif index == 3:
            for class_label in np.unique(labels):
                class_dataset = reduced_data[1, labels == class_label]
                ax.hist(
                    class_dataset,
                    bins=25,
                    density=True,
                    color="%s77" % classes["colors"][class_label],
                    histtype="barstacked",
                    label=classes["labels"][class_label]
                )

    if filename is not None:
        plt.savefig(filename)

    plt.show()


def plot_lda_histogram(dataset, labels, filename=None):
    lda = LDA(n_components=1)
    reduced_data = lda.preprocess(dataset, labels)

    for class_label in np.unique(labels):

        class_dataset = reduced_data[0, labels == class_label]
        plt.hist(
            class_dataset,
            bins=25,
            density=True,
            color="%s77" % classes["colors"][class_label],
            histtype="barstacked",
            label=classes["labels"][class_label]
        )

    plt.legend()

    if filename is not None:
        plt.savefig(filename)

    plt.show()
