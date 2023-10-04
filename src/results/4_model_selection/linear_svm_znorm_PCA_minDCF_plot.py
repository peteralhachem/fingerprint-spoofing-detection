import math
import matplotlib.pyplot as plt

from src.classifiers.discriminative_models.svm_classifier import SvmClassifier
from src.cross_validation.binary_cross_validator import BinaryCrossValidator
from src.cross_validation.cross_validation_results import CrossValidationResults
from src.preprocessing.pca import PCA
from src.preprocessing.z_norm import ZNorm
from src.utilities.arrays import flat_map
from src.utilities.datasets import load_fingerprint_spoofing_detection_train_set

regularization_param_values = [0] + [
    math.pow(10.0, i)
    for i in range(-6, 3, 1)
]

pca_dimensions = [9]

def __compute_and_save_results_to(_filename):
    data_matrix, labels = load_fingerprint_spoofing_detection_train_set()

    # no PCA

    linear_svm_classifiers = [
        SvmClassifier(
            c
        )
        for c in regularization_param_values
    ]

    znorm_linear_svm_classifiers = [
        SvmClassifier(
            c,
            preprocessors=[ZNorm()]
        )
        for c in regularization_param_values
    ]

    # PCA

    def create_linear_svm_pca_and_znorm_linear_svm_pca_classifiers(pca_dim):
        linear_svm_pca_classifiers = [
            SvmClassifier(
                c,
                preprocessors=[PCA(pca_dim)]
            )
            for c in regularization_param_values
        ]

        znorm_linear_svm_pca_classifiers = [
            SvmClassifier(
                c,
                preprocessors=[ZNorm(), PCA(pca_dim)]
            )
            for c in regularization_param_values
        ]

        return linear_svm_pca_classifiers + znorm_linear_svm_pca_classifiers

    linear_svm_pca_and_znorm_classifiers = flat_map(
        lambda x: x,
        [
            create_linear_svm_pca_and_znorm_linear_svm_pca_classifiers(pca_dim)
            for pca_dim in pca_dimensions
        ]
    )

    classifiers = linear_svm_classifiers + znorm_linear_svm_classifiers + linear_svm_pca_and_znorm_classifiers

    working_points = [
        (0.5, 1.0, 10.0)
    ]

    # perform Cross Validation
    cv = BinaryCrossValidator()
    k = 5
    cv.k_folds(data_matrix, labels, k, seed=0)

    cv.cross_validation(
        classifiers,
        working_points=working_points,
    )

    cv_results = cv.results(
        error_rates=True,
        min_DCFs=True,
        actual_DCFs=True
    )

    return cv_results.save(_filename)

def __load_and_plot_results_from(_filename, save_plot_into=None):
    # load results from file
    cv_results = CrossValidationResults.load(_filename)
    cv_results.print()

    # plot results
    minDCFs = cv_results.models_min_DCFs
    plt.figure(figsize=(10,7))

    linear_svm_minDCFs = list(map(lambda x: x[0], minDCFs[0:len(regularization_param_values)]))
    znorm_linear_svm_minDCFs = list(map(
        lambda x: x[0],
        minDCFs[len(regularization_param_values):2*len(regularization_param_values)])
    )
    plt.plot(regularization_param_values, linear_svm_minDCFs, label="SVM")
    plt.plot(regularization_param_values, znorm_linear_svm_minDCFs, label="SVM (Z-norm)")

    for i,pca_dim in enumerate(pca_dimensions):
        i*=2
        # retrieve minDCFs
        linear_svm_pca_minDCFs = list(map(
            lambda x: x[0],
            minDCFs[(i+2)*len(regularization_param_values):(i+3)*len(regularization_param_values)])
        )
        znorm_linear_svm_pca_minDCFs = list(map(
            lambda x: x[0],
            minDCFs[(i+3)*len(regularization_param_values):(i+4)*len(regularization_param_values)])
        )

        # plot curves
        plt.plot(regularization_param_values, linear_svm_pca_minDCFs, label="SVM (PCA=%d)" % pca_dim)
        plt.plot(regularization_param_values, znorm_linear_svm_pca_minDCFs, label="SVM (PCA=%d, Z-norm)" % pca_dim)

    plt.xscale("log")
    plt.xlabel("C")
    plt.ylabel("min DCF")
    plt.title("5-fold CV - Linear SVM - PCA + Z-norm effect")
    plt.grid()
    plt.legend(loc="upper left")

    if save_plot_into is not None:
        plt.savefig(save_plot_into)

    plt.show()



if __name__ == "__main__":
    # * Linear Support Vector Machine (SVM) classifiers with different C hyperparameter values,
    #   with (or without) z-normalization preprocessing, for different values of PCA *

    # specify the name of the file which will contain the results inside output/
    filename = "4_model_selection/linear_svm_znorm_PCA_minDCF"

    # compute results and save them into the file
    # filename = __compute_and_save_results_to(filename)  # <-- run it just once, then comment it

    # print results
    __load_and_plot_results_from(filename, save_plot_into="linear_svm_znorm_PCA_minDCF.pdf")
