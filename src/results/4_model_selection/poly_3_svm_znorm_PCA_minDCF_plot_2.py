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
    for i in range(-5, -1, 1)
]

pca_dimensions = [8,7,6]

def __compute_and_save_results_to(_filename):
    data_matrix, labels = load_fingerprint_spoofing_detection_train_set()

    # PCA

    def create_znorm_poly_3_svm_pca_classifiers(pca_dim):
        znorm_poly_3_svm_pca_classifiers = [
            SvmClassifier(
                c,
                kernel=PolynomialKernel(3),
                preprocessors=[ZNorm(), PCA(pca_dim)]
            )
            for c in regularization_param_values
        ]

        return znorm_poly_3_svm_pca_classifiers

    poly_3_svm_classifiers = flat_map(
        lambda x: x,
        [
            create_znorm_poly_3_svm_pca_classifiers(pca_dim)
            for pca_dim in pca_dimensions
        ]
    )

    classifiers = poly_3_svm_classifiers

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


    for i,pca_dim in enumerate(pca_dimensions):
        # retrieve minDCFs
        znorm_poly_3_svm_pca_minDCFs = list(map(
            lambda x: x[0],
            minDCFs[i*len(regularization_param_values):(i+1)*len(regularization_param_values)])
        )

        # plot curves
        plt.plot(regularization_param_values, znorm_poly_3_svm_pca_minDCFs,
                 label="SVM - Poly(3) (PCA=%d, Z-norm)" % pca_dim)


    plt.xscale("log")
    plt.xlabel("C")
    plt.ylabel("min DCF")
    plt.title("5-fold CV - 3D Polynomial SVM - PCA + Z-norm")
    plt.grid()
    plt.legend(loc="upper left")

    if save_plot_into is not None:
        plt.savefig(save_plot_into)

    plt.show()



if __name__ == "__main__":
    # * SVM classifiers with 3D Polynomial kernel, with different C hyperparameter values,
    #   with z-norm preprocessing, for different values of PCA *

    # specify the name of the file which will contain the results inside output/
    filename = "4_model_selection/poly_3_svm_znorm_PCA_minDCF_2"

    # compute results and save them into the file
    # filename = __compute_and_save_results_to(filename)  # <-- run it just once, then comment it

    # print results
    __load_and_plot_results_from(filename, save_plot_into="poly_3_svm_znorm_PCA_minDCF_2.pdf")
