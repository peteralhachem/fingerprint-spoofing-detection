import math
import matplotlib.pyplot as plt

from src.classifiers.discriminative_models.svm_classifier import SvmClassifier
from src.cross_validation.binary_cross_validator import BinaryCrossValidator
from src.cross_validation.cross_validation_results import CrossValidationResults
from src.preprocessing.pca import PCA
from src.preprocessing.z_norm import ZNorm
from src.utilities.datasets import load_fingerprint_spoofing_detection_train_set
from src.utilities.kernels import PolynomialKernel

regularization_param_values = [0] + [
    math.pow(10.0, i)
    for i in range(-3, 1, 1)
]

pca_dimension = 8

def __compute_and_save_results_to(_filename):
    data_matrix, labels = load_fingerprint_spoofing_detection_train_set()

    poly_3_svm_pca_8_classifiers = [
        SvmClassifier(
            c,
            kernel=PolynomialKernel(3),
            preprocessors=[PCA(pca_dimension)]
        )
        for c in regularization_param_values
    ]

    znorm_poly_3_svm_pca_8_classifiers = [
        SvmClassifier(
            c,
            kernel=PolynomialKernel(3),
            preprocessors=[ZNorm(), PCA(pca_dimension)]
        )
        for c in regularization_param_values
    ]
    classifiers = poly_3_svm_pca_8_classifiers + znorm_poly_3_svm_pca_8_classifiers

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

    poly_3_svm_PCA_8_minDCFs = list(map(lambda x: x[0], minDCFs[0:len(regularization_param_values)]))
    znorm_poly_3_svm_PCA_8_minDCFs = list(map(
        lambda x: x[0],
        minDCFs[len(regularization_param_values): 2*len(regularization_param_values)])
    )
    plt.plot(regularization_param_values, poly_3_svm_PCA_8_minDCFs,
             label="SVM - Poly(3) (PCA=%d)" % pca_dimension)
    plt.plot(regularization_param_values, znorm_poly_3_svm_PCA_8_minDCFs,
             label="SVM - Poly(3) (PCA=%d, Z-norm)" % pca_dimension)

    plt.xscale("log")
    plt.xlabel("C")
    plt.ylabel("min DCF")
    plt.title("5-fold CV - 3D Polynomial SVM - PCA=8 + Z-norm effect")
    plt.grid()
    plt.legend(loc="upper left")

    if save_plot_into is not None:
        plt.savefig(save_plot_into)

    plt.show()



if __name__ == "__main__":
    # * SVM classifiers with 3D Polynomial kernel, with different C hyperparameter values,
    #   with (or without) z-normalization preprocessing, for different values of PCA *

    # specify the name of the file which will contain the results inside output/
    filename = "4_model_selection/poly_3_svm_znorm_PCA_minDCF"

    # compute results and save them into the file
    # filename = __compute_and_save_results_to(filename)  # <-- run it just once, then comment it

    # print results
    __load_and_plot_results_from(filename, save_plot_into="poly_3_svm_znorm_PCA_minDCF.pdf")
