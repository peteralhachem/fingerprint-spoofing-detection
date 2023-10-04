import numpy as np


class TargetApplication:
    true_class_prior = 0.5
    false_negative_cost = 1.0
    false_positive_cost = 10.0

    @staticmethod
    def pi_T():
        return TargetApplication.true_class_prior

    @staticmethod
    def C_fn():
        return TargetApplication.false_negative_cost

    @staticmethod
    def C_fp():
        return TargetApplication.false_positive_cost

    @staticmethod
    def effective_prior():
        pi_T = TargetApplication.pi_T()
        C_fn = TargetApplication.C_fn()
        C_fp = TargetApplication.C_fp()

        return (pi_T*C_fn) / (pi_T*C_fn + (1.0 - pi_T)*C_fp)

    @staticmethod
    def application_prior_log_odd():
        pi = TargetApplication.effective_prior()
        return np.log(pi) - np.log(1.0 - pi)
