import math
import matplotlib.pyplot as plt

from src.classifiers.discriminative_models.svm_classifier import SvmClassifier
from src.cross_validation.binary_cross_validator import BinaryCrossValidator
from src.cross_validation.cross_validation_results import CrossValidationResults
from src.preprocessing.pca import PCA
from src.utilities.arrays import flat_map
from src.utilities.datasets import load_fingerprint_spoofing_detection_train_set
from src.utilities.kernels import RadialBasisFunctionKernel

regularization_param_values = [
    math.pow(10.0, i)
    for i in range(-1, 2, 1)
]

rbf_gamma_param_values = [
    math.pow(10.0, i)
    for i in range(-3, -2, 1)
]

pca_dimensions = [9,8,7]


def __compute_and_save_results_to(_filename):
    data_matrix, labels = load_fingerprint_spoofing_detection_train_set()

    # no PCA

    rbf_svm_classifiers = [
        SvmClassifier(
            c,
            kernel=RadialBasisFunctionKernel(gamma)
        )
        for gamma in rbf_gamma_param_values for c in regularization_param_values
    ]

    # PCA

    def create_rbf_svm_pca_classifiers(pca_dim):
        rbf_svm_pca_classifiers = [
            SvmClassifier(
                c,
                kernel=RadialBasisFunctionKernel(gamma),
                preprocessors=[PCA(pca_dim)]
            )
            for gamma in rbf_gamma_param_values for c in regularization_param_values
        ]

        return rbf_svm_pca_classifiers

    rbf_svm_pca_and_znorm_classifiers = flat_map(
        lambda x: x,
        [
            create_rbf_svm_pca_classifiers(pca_dim)
            for pca_dim in pca_dimensions
        ]
    )

    classifiers = rbf_svm_classifiers + rbf_svm_pca_and_znorm_classifiers

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
    for i,gamma in enumerate(rbf_gamma_param_values):
        rbf_svm_minDCFs = list(map(
            lambda x: x[0],
            minDCFs[i*len(regularization_param_values):(i+1)*len(regularization_param_values)])
        )
        plt.plot(regularization_param_values, rbf_svm_minDCFs,
                 label="SVM - RBF ($\\log{\\gamma}$ = %d)" % (math.log10(gamma)))

    for i,pca_dim in enumerate(pca_dimensions):
        l = len(regularization_param_values) * len(rbf_gamma_param_values)

        for j,gamma in enumerate(rbf_gamma_param_values):
            rbf_svm_pca_minDCFs = list(map(
                lambda x: x[0],
                minDCFs[((i+1)*l) + j*len(regularization_param_values):((i+1)*l) + (j+1)*len(regularization_param_values)])
            )
            plt.plot(regularization_param_values, rbf_svm_pca_minDCFs,
                     label="SVM - RBF ($\\log{\\gamma}$ = %d) (PCA=%d)" % (math.log10(gamma), pca_dim))

    plt.xscale("log")
    plt.xlabel("C")
    plt.ylabel("min DCF")
    plt.title("5-fold CV - RBF($\\gamma=10^{-3}$) SVM - PCA")
    plt.grid()
    plt.legend(loc="upper left")

    if save_plot_into is not None:
        plt.savefig(save_plot_into)

    plt.show()



if __name__ == "__main__":
    # * SVM classifiers with Radial Basis Function (RBF) kernel, with different C and gamma=10^-3
    #   without z-normalization preprocessing, for different values of PCA *

    # specify the name of the file which will contain the results inside output/
    filename = "4_model_selection/rbf_3_svm_PCA_minDCF"

    # compute results and save them into the file
    # filename = __compute_and_save_results_to(filename)  # <-- run it just once, then comment it

    # print results
    __load_and_plot_results_from(filename, save_plot_into="rbf_3_svm_PCA_minDCF.pdf")
