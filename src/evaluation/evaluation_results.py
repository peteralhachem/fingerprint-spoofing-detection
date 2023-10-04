import json

from src.utilities.classifiers_results import ClassifiersResults

class EvaluationResults(ClassifiersResults):
    __default_output_file_name = "results"
    __array_metrics_names = ["models_error_rates", "models_min_DCFs", "models_actual_DCFs"]
    __all_metrics_names = __array_metrics_names + ["bayes_error_plot_values"]

    def print(self, models_error_rates=False, models_min_DCFs=False, models_actual_DCFs=False, phase=None):
        return super().print(
            models_error_rates=models_error_rates,
            models_min_DCFs=models_min_DCFs,
            models_actual_DCFs=models_actual_DCFs,
            phase="Evaluation" if phase is None else phase
        )

    @staticmethod
    def load(filename=__default_output_file_name):
        """
        Load the results from a json file inside the /output folder. Json file must comply with the format
        used by the EvaluationResults.save() method
        :param filename: the relative path of the file, with respect to the /output folder, where
         the results are stored
        :return: an instance of EvaluationResults containing the results in the specified file
        """
        filename = "../../../output/%s.json" % filename

        with open(filename, "r") as input_file:
            decoded_results = json.load(input_file)

        for metric_name in EvaluationResults.__all_metrics_names:
            if metric_name not in decoded_results:
                decoded_results[metric_name] = None

        return EvaluationResults(
            decoded_results,
            decoded_results["classifiers"],
            decoded_results["working_points"]
        )
