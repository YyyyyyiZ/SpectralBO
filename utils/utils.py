import plotly.graph_objects as go
import torch
import numpy as np
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.priors.torch_priors import GammaPrior

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound, qKnowledgeGradient
from botorch.optim import optimize_acqf


def get_next_points(acq, kernel, n_mixture, init_x, init_y, best_init_y, bounds, n_points=1):
    noise_prior = GammaPrior(1.1, 0.5)
    noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
    likelihood = GaussianLikelihood(
        noise_prior=noise_prior,
        batch_shape=[],
        noise_constraint=GreaterThan(
            # 0.000005,  # minimum observation noise assumed in the GP model
            0.0001,
            transform=None,
            initial_value=noise_prior_mode,
        ),
    )
    single_model = SingleTaskGP(init_x, init_y, likelihood=likelihood, covar_module=kernel, n_mixture=n_mixture)
    mll = ExactMarginalLogLikelihood(single_model.likelihood, single_model)
    fit_gpytorch_mll(mll)
    if acq == 'ei':
        acq_function = ExpectedImprovement(model=single_model, best_f=best_init_y, maximize=False)
    elif acq == 'pi':
        acq_function = ProbabilityOfImprovement(model=single_model, best_f=best_init_y, maximize=False)
    elif acq == 'ucb':
        acq_function = UpperConfidenceBound(model=single_model, beta=0.5, maximize=False)
    # elif acq == 'kg':
    #     acq_function = qKnowledgeGradient(model=single_model)
    else:
        raise ValueError('Acquisition function not identified')
    candidates, _ = optimize_acqf(acq_function=acq_function, bounds=bounds, q=n_points, num_restarts=100,
                                  raw_samples=512, options={"batch_limit": 5, "maxiter": 100})
    return candidates


def compute_acquisition_function(single_model, best_init_y, l_bound=-2., h_bound=10., resolution=1000):
    linspace = torch.linspace(l_bound, h_bound, steps=resolution)
    x_test = torch.tensor([linspace[0]]).unsqueeze(-1)
    EI = ExpectedImprovement(model=single_model, best_f=best_init_y)
    result = []
    for x in linspace:
        x_test = torch.tensor([x]).unsqueeze(-1)
        result.append(EI(x_test))
    return torch.tensor(result)


def print_acquisition_function(acq_fun, iteration, l_bound=-2., h_bound=10., resolution=1000, suggested=None):
    x = torch.linspace(l_bound, h_bound, steps=resolution).detach().numpy()
    x_new = x.reshape((resolution, -1))
    z = acq_fun
    max_acq_fun = x[((acq_fun == acq_fun.max().item()).nonzero(as_tuple=True)[0])]
    data = go.Scatter(x=x, y=z, line_color="yellow")

    fig = go.Figure(data=data)
    fig.update_layout(title="Expected Improvement acquisition function. Iteration " + str(iteration),
                      xaxis_title="input", yaxis_title="output")
    if (suggested == None):
        fig.add_vline(x=max_acq_fun, line_width=3, line_color="red")
    else:
        fig.add_vline(x=float(suggested[0][0]), line_width=3, line_color="red")
    fig.show()


def compute_predictive_distribution(single_model, best_init_y, l_bound=-2., h_bound=10., resolution=1000):
    linspace = torch.linspace(l_bound, h_bound, steps=resolution)
    x_test = torch.tensor([linspace[0]]).unsqueeze(-1)
    result = []
    variances = []
    for x in linspace:
        x_test = torch.tensor([x]).unsqueeze(-1)
        result.append(single_model.posterior(x_test).mean)
        variances.append(single_model.posterior(x_test).variance)
    return torch.tensor(result), torch.tensor(variances)


def print_predictive_mean(predictive_mean, predictive_variance, iteration, l_bound=-2., h_bound=10., resolution=1000,
                          suggested=None, old_obs=[], old_values=[]):
    x = torch.linspace(l_bound, h_bound, steps=resolution).detach().numpy()
    x_new = x.reshape((resolution, -1))
    z = predictive_mean
    max_predictive_mean = x[((predictive_mean == predictive_mean.max().item()).nonzero(as_tuple=True)[0])]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x, y=predictive_mean + torch.sqrt(predictive_variance),
                             mode='lines',
                             line=dict(color="#19D3F3", width=0.1),
                             name='upper bound'))
    fig.add_trace(go.Scatter(x=x, y=predictive_mean,
                             mode='lines',
                             line=dict(color="blue"),
                             fill='tonexty',
                             name='predictive mean'))
    fig.add_trace(go.Scatter(x=x, y=predictive_mean - torch.sqrt(predictive_variance),
                             mode='lines',
                             line=dict(color="blue", width=0.1),
                             fill='tonexty',
                             name='lower bound'))

    fig.update_layout(title="GP Predictive distribution. Iteration " + str(iteration), xaxis_title="input",
                      yaxis_title="output", showlegend=False)

    if (suggested == None):
        fig.add_vline(x=max_predictive_mean, line_width=3, line_color="red")
    else:
        fig.add_vline(x=float(suggested[0][0]), line_width=3, line_color="red")

    if (len(old_obs) > 0):
        fig.add_trace(go.Scatter(x=old_obs, y=old_values, mode='markers', marker_color="black", marker_size=10))

    fig.show()


def print_objective_function(target_function, best_candidate, iteration):
    x = np.linspace(-2., 10., 100)
    x_new = x.reshape((100, -1))
    z = target_function(x_new)
    data = go.Scatter(x=x, y=z, line_color="#FE73FF")
    fig = go.Figure(data=data)
    fig.update_layout(title="Objective function. Iteration " + str(iteration), xaxis_title="input",
                      yaxis_title="output")
    fig.add_vline(x=best_candidate, line_width=3, line_color="red")
    fig.show()


def visualize_functions(target_function, single_model, best_init_y, best_candidate, candidate_acq_fun, iteration,
                        previous_observations,
                        previous_values):
    predictive_mean, predictive_variance = compute_predictive_distribution(single_model, best_init_y)
    print_predictive_mean(predictive_mean, predictive_variance, iteration, suggested=candidate_acq_fun,
                          old_obs=previous_observations, old_values=previous_values)
    acq_fun = compute_acquisition_function(single_model, best_init_y)
    print_acquisition_function(acq_fun, iteration, suggested=candidate_acq_fun)
    print_objective_function(target_function, best_candidate, iteration)


def get_next_points_and_visualize(target_function, init_x, init_y, best_init_y, bounds, iteration,
                                  previous_observations,
                                  previous_values, n_points=1):
    single_model = SingleTaskGP(init_x, init_y)
    mll = ExactMarginalLogLikelihood(single_model.likelihood, single_model)
    fit_gpytorch_mll(mll)

    EI = ExpectedImprovement(model=single_model, best_f=best_init_y)

    candidates, _ = optimize_acqf(acq_function=EI, bounds=bounds, q=n_points, num_restarts=200, raw_samples=512,
                                  options={"batch_limit": 5, "maxiter": 200})
    best_candidate = init_x[((init_y == best_init_y).nonzero(as_tuple=True)[0])][0][0]

    visualize_functions(target_function, single_model, best_init_y, best_candidate, candidates, iteration,
                        previous_observations,
                        previous_values)

    return candidates
