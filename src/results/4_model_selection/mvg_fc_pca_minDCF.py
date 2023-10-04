from src.cross_validation.binary_cross_validator import BinaryCrossValidator
from src.classifiers.generative_models.binary_gaussian_classifier import BinaryGaussianClassifier
from src.cross_validation.cross_validation_results import CrossValidationResults
from src.preprocessing.pca import PCA
from src.utilities.datasets import load_fingerprint_spoofing_detection_train_set


def __compute_and_save_results_to(_filename):
    data_matrix, labels = load_fingerprint_spoofing_detection_train_set()

    classifiers = [
        BinaryGaussianClassifier(),
        BinaryGaussianClassifier(
            preprocessors=[PCA(10)]
        ),
        BinaryGaussianClassifier(
            preprocessors=[PCA(9)]
        ),
        BinaryGaussianClassifier(
            preprocessors=[PCA(8)]
        ),
        BinaryGaussianClassifier(
            preprocessors=[PCA(7)]
        ),
        BinaryGaussianClassifier(
            preprocessors=[PCA(6)]
        ),
        BinaryGaussianClassifier(
            preprocessors=[PCA(5)]
        )
    ]

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
        # error_rates=True,
        min_DCFs=True,
        # actual_DCFs=True
    )

    return cv_results.save(_filename)

def __load_and_print_results_from(_filename):
    # load results from file
    cv_results = CrossValidationResults.load(_filename)
    cv_results.print()


if __name__ == "__main__":
    # * Multivariate Gaussian Classifiers (MVGs) with Full Covariance and different PCA dimensions *

    # specify the name of the file which will contain the results inside output/
    filename = "4_model_selection/mvg_fc_pca_minDCF"

    # compute results and save them into the file
    # filename = __compute_and_save_results_to(filename)  # <-- run it just once, then comment it

    # print results
    __load_and_print_results_from(filename)
