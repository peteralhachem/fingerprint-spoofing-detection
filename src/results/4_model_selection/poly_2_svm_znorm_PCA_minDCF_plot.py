import math
import matplotlib.pyplot as plt

from src.classifiers.discriminative_models.svm_classifier import SvmClassifier
from src.cross_validation.binary_cross_validator import BinaryCrossValidator
from src.cross_validation.cross_validation_results import CrossValidationResults
from src.preprocessing.pca import PCA
from src.preprocessing.z_norm import ZNorm
from src.utilities.arrays import flat_map
from src.utilities.datasets import load_fingerprint_spoofing_detection_train_set
from src.utilities.kernels import PolynomialKernel

regularization_param_values = [0] + [
    math.pow(10.0, i)
    for i in range(-6, 3, 1)
]

pca_dimensions = [9,8,7,6]

def __compute_and_save_results_to(_filename):
    data_matrix, labels = load_fingerprint_spoofing_detection_train_set()

    # no PCA

    poly_2_svm_classifiers = [
        SvmClassifier(
            c,
            kernel=PolynomialKernel(2)
        )
        for c in regularization_param_values
    ]

    znorm_poly_2_svm_classifiers = [
        SvmClassifier(
            c,
            kernel=PolynomialKernel(2),
            preprocessors=[ZNorm()]
        )
        for c in regularization_param_values
    ]

    # PCA

    def create_poly_2_svm_pca_and_znorm_poly_2_svm_pca_classifiers(pca_dim):
        poly_2_svm_pca_classifiers = [
            SvmClassifier(
                c,
                kernel=PolynomialKernel(2),
                preprocessors=[PCA(pca_dim)]
            )
            for c in regularization_param_values
        ]

        znorm_poly_2_svm_pca_classifiers = [
            SvmClassifier(
                c,
                kernel=PolynomialKernel(2),
                preprocessors=[ZNorm(), PCA(pca_dim)]
            )
            for c in regularization_param_values
        ]

        return poly_2_svm_pca_classifiers + znorm_poly_2_svm_pca_classifiers

    poly_2_svm_pca_and_znorm_classifiers = flat_map(
        lambda x: x,
        [
            create_poly_2_svm_pca_and_znorm_poly_2_svm_pca_classifiers(pca_dim)
            for pca_dim in pca_dimensions
        ]
    )

    classifiers = poly_2_svm_classifiers + znorm_poly_2_svm_classifiers + poly_2_svm_pca_and_znorm_classifiers

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

    # no PCA
    poly_2_svm_minDCFs = list(map(lambda x: x[0], minDCFs[0:len(regularization_param_values)]))
    znorm_poly_2_svm_minDCFs = list(map(
        lambda x: x[0],
        minDCFs[len(regularization_param_values):2*len(regularization_param_values)])
    )
    plt.plot(regularization_param_values, poly_2_svm_minDCFs, label="SVM - Poly(2)")
    plt.plot(regularization_param_values, znorm_poly_2_svm_minDCFs, label="SVM - Poly(2) (Z-norm)")

    for i,pca_dim in enumerate(pca_dimensions):
        i*=2
        # retrieve minDCFs
        poly_2_svm_pca_minDCFs = list(map(
            lambda x: x[0],
            minDCFs[(i+2)*len(regularization_param_values):(i+3)*len(regularization_param_values)])
        )
        znorm_poly_2_svm_pca_minDCFs = list(map(
            lambda x: x[0],
            minDCFs[(i+3)*len(regularization_param_values):(i+4)*len(regularization_param_values)])
        )

        # plot curves
        plt.plot(regularization_param_values, poly_2_svm_pca_minDCFs,
                 label="SVM - Poly(2) (PCA=%d)" % pca_dim)
        plt.plot(regularization_param_values, znorm_poly_2_svm_pca_minDCFs,
                 label="SVM - Poly(2) (PCA=%d, Z-norm)" % pca_dim)

    plt.xscale("log")
    plt.xlabel("C")
    plt.ylabel("min DCF")
    plt.title("5-fold CV - 2D Polynomial SVM - PCA + Z-norm effect")
    plt.grid()
    plt.legend(loc="upper left")

    if save_plot_into is not None:
        plt.savefig(save_plot_into)

    plt.show()



if __name__ == "__main__":
    # * SVM classifiers with 2D Polynomial kernel, with different C hyperparameter values,
    #   with (or without) z-normalization preprocessing, for different values of PCA *

    # specify the name of the file which will contain the results inside output/
    filename = "4_model_selection/poly_2_svm_znorm_PCA_minDCF"

    # compute results and save them into the file
    # filename = __compute_and_save_results_to(filename)  # <-- run it just once, then comment it

    # print results
    __load_and_plot_results_from(filename, save_plot_into="poly_2_svm_znorm_PCA_minDCF.pdf")
