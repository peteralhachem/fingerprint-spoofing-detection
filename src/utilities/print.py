
def print_metrics_per_type_of_metric(metrics_values, classifiers, working_points, ordered_metrics_names=None):
    print("\n* Cross Validation Results *")

    if ordered_metrics_names is None:
        ordered_metrics_names = list(sorted(metrics_values.keys()))

    value_length = 5
    working_point_length = max(value_length, max(map(lambda working_point: len(str(working_point)), working_points)))
    label_format_str = "{:^%d}" % ((working_point_length + 1) * len(working_points) - 1)

    print("   ".join(
        [label_format_str.format(metric_label) for metric_label in ordered_metrics_names]
    ))

    print("%s      Classifier" %
        "   ".join([
            " ".join([
                ("{:^%d}" % working_point_length).format(str(w))
                for w in working_points
            ])
            for _ in range(len(ordered_metrics_names))
        ])
    )

    print("-" * (
            ((working_point_length + 1) * len(working_points) - 1 + 3) * len(ordered_metrics_names)  # metrics length
            + 3
            + max([len(classifier) for classifier in classifiers])  # classifier length
    ))

    for i, classifier in enumerate(classifiers):
        print("%s      %s" % (
            "   ".join([
                " ".join(
                    [("{:^%d.3f}" % working_point_length).format(value) for value in metric[i]]
                )
                for metric in [metrics_values[x] for x in ordered_metrics_names]
            ]),
            classifier
        ))

def print_metrics_per_working_point(
        metrics_values,
        classifiers,
        working_points,
        ordered_metrics_names=None,
        phase=None
):
    print("\n*%s Results *" % (" %s" % phase) if phase is not None else "")

    if ordered_metrics_names is None:
        ordered_metrics_names = list(sorted(metrics_values.keys()))

    value_length = 5
    metric_length = max(value_length, max(map(lambda metric: len(metric), ordered_metrics_names)))
    working_point_format_str = "{:^%d}" % ((metric_length + 1) * len(ordered_metrics_names) - 1)

    print("   ".join(
        [working_point_format_str.format(str(working_point)) for working_point in working_points]
    ))

    print("%s      Classifier" %
        "   ".join([
            " ".join([
                ("{:^%d}" % metric_length).format(m)
                for m in ordered_metrics_names
            ])
            for _ in range(len(working_points))
        ])
    )

    print("-" * (
            ((metric_length + 1) * len(ordered_metrics_names) - 1 + 3) * len(working_points)  # metrics length
            + 3
            + max([len(classifier) for classifier in classifiers])  # classifier length
    ))

    for i, classifier in enumerate(classifiers):
        print("%s      %s" % (
            "   ".join([
                " ".join([
                    (("{:^%d.3f}" % metric_length).format(metric[i][w])
                     if metric[i][w] is not None else ("{:^%d}" % metric_length).format("-"))
                    for metric in [metrics_values[x] for x in ordered_metrics_names]
                ])
                for w in range(len(working_points))
            ]),
            classifier
        ))
