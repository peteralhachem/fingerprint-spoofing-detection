import numpy as np

from src.utilities.target_application import TargetApplication

if __name__ == "__main__":
    # * show target application working point parameters *

    # working point
    pi_T = TargetApplication.true_class_prior
    C_fn = TargetApplication.false_negative_cost
    C_fp = TargetApplication.false_positive_cost

    # effective prior
    pi_tilde = TargetApplication.effective_prior()

    # corresponding prior log odd
    prior_log_odd = TargetApplication.application_prior_log_odd()

    # * print results *
    print("* Target application *\n")
    print("   true class prior (pi_T):  %.1f" % pi_T)
    print("false negative cost (C_fn):  %.1f" % C_fn)
    print("false positive cost (C_fp):  %.1f" % C_fp)
    print("="*40)
    print("  working point: (%.1f, %.1f, %.1f)" % (pi_T, C_fn, C_fp))
    print("effective prior: %s" % pi_tilde)
    print("\ncorresponding prior log odd: %s" % prior_log_odd)
