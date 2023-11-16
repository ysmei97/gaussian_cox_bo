import numpy as np


def online_changepoint_detection(data, hazard_function, log_likelihood_class):
    """
    Use online bayesian changepoint detection
    :param data: the time series data
    :param hazard_function:
    :param log_likelihood_class:
    :return:
        mat_r: the probability at time step t that the last sequence is already s time steps long
        cmax_mat_r: the argmax on column axis of matrix mat_r (growth probability value) for each time step
    """
    cmax_mat_r = np.zeros(len(data) + 1)
    mat_r = np.zeros((len(data) + 1, len(data) + 1))
    mat_r[0, 0] = 1

    for t, x in enumerate(data):
        # Evaluate the predictive distribution for the new datum under each of the parameters
        predict_probs = log_likelihood_class.pdf(x)

        # Evaluate the hazard function for this interval
        hazard = hazard_function(np.array(range(t + 1)))

        # Evaluate the growth probabilities: shift the probabilities down and to the right, scaled by the hazard function and the predictive probabilities.
        mat_r[1:(t + 2), t + 1] = mat_r[0:(t + 1), t] * predict_probs * (1 - hazard)

        # Evaluate the probability that there *was* a change point, and we're accumulating the mass back down at r = 0.
        mat_r[0, t + 1] = np.sum(mat_r[0:(t + 1), t] * predict_probs * hazard)

        # Re-normalize the run length probabilities for improved numerical stability.
        mat_r[:, t + 1] = mat_r[:, t + 1] / np.sum(mat_r[:, t + 1])

        # Update the parameter sets for each possible run length.
        log_likelihood_class.update_theta(x, t=t)

        cmax_mat_r[t] = mat_r[:, t].argmax()

    return mat_r, cmax_mat_r
