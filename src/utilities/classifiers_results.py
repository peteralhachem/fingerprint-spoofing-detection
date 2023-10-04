import sys
import json
from matplotlib import pyplot
from src.utilities.print import print_metrics_per_working_point
from os.path import exists

class ClassifiersResults:
    """
    Note: .bayes_error_plot_values is a list of dictionaries with keys: "prior_log_odds", "actual_DCFs", "min_DCFs"
    """
    __default_output_file_name = "results"
    __max_files_with_same_name = 1000
    __array_metrics_names = ["models_error_rates", "models_min_DCFs", "models_actual_DCFs"]
    __all_metrics_names = __array_metrics_names + ["bayes_error_plot_values"]
    __default_colors = ["red", "blue", "green", "black", "cyan", "purple", "slategray", "gold", "darkorange"]
    __vertical_line_color = "gold"

    def __init__(self, results, classifiers, working_points):
        """
        Create a new instance of ClassifiersResults specifying the results values, the (ordered)
        names of the used classifiers and the (ordered) working points analyzed
        :param results: an object containing all the evaluation results (models_error_rates,
         models_min_DCFs, models_actual_DCFs, bayes_error_plot_values).
        :param classifiers: list of the classifiers' names used in the cross validation
        :param working_points: list of working points (pi_T, C_fn, C_fp) analyzed in the cross validation
        """
        self.models_error_rates = results["models_error_rates"]
        self.models_min_DCFs = results["models_min_DCFs"]
        self.models_actual_DCFs = results["models_actual_DCFs"]
        self.bayes_error_plot_values = results["bayes_error_plot_values"]

        self.classifiers = classifiers
        self.working_points = working_points

    def print(self, models_error_rates=False, models_min_DCFs=False, models_actual_DCFs=False, phase=None):
        """
        Print the results on standard output in a table format. Optionally specify only the
        metrics you want to print
        :param models_error_rates: (optional) if printing error rates
        :param models_min_DCFs: (optional) if printing minimum DCFs
        :param models_actual_DCFs: (optional) if printing actual DCFs
        :param phase: (optional) the name of the analysis phase the results are for (e.g. validation,
         cross validation, etc.)
        """

        # if nothing provided, enabled all three metrics
        if not models_error_rates and not models_min_DCFs and not models_actual_DCFs:
            models_error_rates = True
            models_min_DCFs = True
            models_actual_DCFs = True

        metrics = {}
        ordered_metrics_names = []

        if models_min_DCFs and self.models_min_DCFs is not None:
            metric_name = "minimum DCF"
            metrics[metric_name] = self.models_min_DCFs
            ordered_metrics_names.append(metric_name)

        if models_actual_DCFs and self.models_actual_DCFs is not None:
            metric_name = "actual DCF"
            metrics[metric_name] = self.models_actual_DCFs
            ordered_metrics_names.append(metric_name)

        if models_error_rates and self.models_error_rates is not None:
            metric_name = "error rate"
            metrics[metric_name] = self.models_error_rates
            ordered_metrics_names.append(metric_name)

        print_metrics_per_working_point(
            metrics,
            self.classifiers,
            self.working_points,
            ordered_metrics_names=ordered_metrics_names,
            phase=phase
        )

    def save(self, filename=__default_output_file_name):
        """
        Save the results in a json file at the specified path, which will be inside the /output folder.
        :param filename: (optional) the relative path **without extension** of the file in which the
         results will be saved, with respect to the /output_path (default: "results.json")
        :return: the actual filename where results have been saved
        """
        # encode results in a dictionary
        encoded_results = self.encode()

        # now look for the path where to save the json file

        path = "../../../output/%s" % filename
        path_is_free = False

        for i in range(ClassifiersResults.__max_files_with_same_name):
            path_to_check = "%s.json" % path if i == 0 else ("%s_%d.json" % (path,i+1))

            if not exists(path_to_check):
                path_is_free = True
                path = path_to_check
                break

        if not path_is_free:
            print("Error: you have too many files (>%d) saved with that name. "
                  "Change the destination file to save the results" %
                  ClassifiersResults.__max_files_with_same_name, file=sys.stderr)
            return

        # open the file
        with open(path, "w") as save_file:
            # save data
            json.dump(encoded_results, save_file, indent=4)

        actual_filename = path.split("/")[-1].split(".")[0]
        actual_path = "".join(map(
            lambda path_piece: "%s/" % path_piece,
            filename.split("/")[:-1])
        ) + actual_filename
        return actual_path

    def encode(self):
        res = {}

        if self.models_error_rates is not None:
            res["models_error_rates"] = self.models_error_rates

        if self.models_actual_DCFs is not None:
            res["models_actual_DCFs"] = self.models_actual_DCFs

        if self.models_min_DCFs is not None:
            res["models_min_DCFs"] = self.models_min_DCFs

        if self.bayes_error_plot_values is not None:
            res["bayes_error_plot_values"] = self.bayes_error_plot_values

        res["classifiers"] = self.classifiers
        res["working_points"] = self.working_points

        return res

    @staticmethod
    def load(filename=__default_output_file_name):
        """
        Load the results from a json file inside the /output folder. Json file must comply with the format
        used by the ClassifiersResults.save() method
        :param filename: the relative path of the file, with respect to the /output folder, where
         the results are stored
        :return: an instance of ClassifiersResults containing the results in the specified file
        """
        filename = "../../../output/%s.json" % filename

        with open(filename, "r") as input_file:
            decoded_results = json.load(input_file)

        for metric_name in ClassifiersResults.__all_metrics_names:
            if metric_name not in decoded_results:
                decoded_results[metric_name] = None

        return ClassifiersResults(
            decoded_results,
            decoded_results["classifiers"],
            decoded_results["working_points"]
        )

    def merge(self, other_results):
        """
        Merge the current results with new results over the same working points
        :param other_results: another instance of ClassifiersResults
        :return: a new instance of ClassifiersResults containing (in order) the
         results of both the previous and the new instance
        """
        if self.working_points != other_results.working_points:
            print("Error: you cannot merge two kind results over different working points", file=sys.stderr)
            return None

        num_classifiers_first = len(self.classifiers)
        num_classifiers_second = len(other_results.classifiers)

        encoded_first = self.encode()
        encoded_second = other_results.encode()

        # fill missing values with None
        for metric in ClassifiersResults.__array_metrics_names:
            if metric in encoded_first and metric not in encoded_second:
                encoded_second[metric] = [
                    [None for _ in range(len(self.working_points))]
                    for _ in range(num_classifiers_second)
                ]
            if metric not in encoded_first and metric in encoded_second:
                encoded_first[metric] = [
                    [None for _ in range(len(self.working_points))]
                    for _ in range(num_classifiers_first)
                ]

        # merge results
        for metric in ClassifiersResults.__all_metrics_names:
            if metric in encoded_first:
                encoded_first[metric] += encoded_second[metric]
            else:
                encoded_first[metric] = None

        encoded_first["classifiers"] += encoded_second["classifiers"]

        return ClassifiersResults(encoded_first, encoded_first["classifiers"], encoded_first["working_points"])

    def bayes_error_plot(self, labels=None, save_to=None,
                         ylim=None, xlabel=None, ylabel=None,
                         application_prior=None):
        """
        Print a Bayes Error plot with the results already computed
        """

        if self.bayes_error_plot_values is None:
            print("Error: you cannot show a Bayes Error plot because there aren't"
                  " Bayes error plot values in these results", file=sys.stderr)
            return

        pyplot.figure(figsize=(9,6))

        colors = ClassifiersResults.__default_colors[0:len(self.classifiers)]

        max_y = 0

        for i, (model_values, color) in enumerate(zip(self.bayes_error_plot_values, colors)):
            # model min DCF
            pyplot.plot(
                model_values["prior_log_odds"],
                model_values["min_DCFs"],
                linestyle="dashed",
                color=color,
                label="%s - min DCF" % labels[i] if labels is not None else self.classifiers[i]
            )

            max_y = max(max_y, max(model_values["min_DCFs"]))

            # model actual DCF
            pyplot.plot(
                model_values["prior_log_odds"],
                model_values["actual_DCFs"],
                color=color,
                label="%s - DCF" % labels[i] if labels is not None else self.classifiers[i]
            )

            max_y = max(max_y, max(model_values["actual_DCFs"]))

        if application_prior is not None:
            # vertical line
            pyplot.plot(
                [application_prior, application_prior],
                [0, max_y],
                color=ClassifiersResults.__vertical_line_color
            )

        pyplot.legend(loc="upper left")

        if ylim is not None:
            pyplot.ylim(ylim)

        if xlabel is not None:
            pyplot.xlabel(xlabel)

        if ylabel is not None:
            pyplot.ylabel(ylabel)

        if save_to is not None:
            pyplot.savefig(save_to)

        pyplot.show()

    def delete_results_of_classifiers(self, indexes_list):
        models_error_rates = []
        models_min_DCFs = []
        models_actual_DCFs = []
        bayes_error_plot_values = []
        classifiers = []

        for i in range(len(self.classifiers)):
            if i in indexes_list:
                continue  # skip classifiers to delete

            if self.models_error_rates is not None:
                models_error_rates.append(self.models_error_rates[i])

            if self.models_min_DCFs is not None:
                models_min_DCFs.append(self.models_min_DCFs[i])

            if self.models_actual_DCFs is not None:
                models_actual_DCFs.append(self.models_actual_DCFs[i])

            if self.bayes_error_plot_values is not None:
                bayes_error_plot_values.append(self.bayes_error_plot_values[i])

            if self.classifiers is not None:
                classifiers.append(self.classifiers[i])

        # replace results
        if self.models_error_rates is not None:
            self.models_error_rates = models_error_rates

        if self.models_min_DCFs is not None:
            self.models_min_DCFs = models_min_DCFs

        if self.models_actual_DCFs is not None:
            self.models_actual_DCFs = models_actual_DCFs

        if self.bayes_error_plot_values is not None:
            self.bayes_error_plot_values = bayes_error_plot_values

        if self.classifiers is not None:
            self.classifiers = classifiers
