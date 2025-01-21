import os
import torch
import matplotlib.pyplot as plt

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(0)


def generate_data(x):
    return torch.sin(x) + torch.sin((10.0 / 3.0) * x)

def plot_acq(X, y, X_pred, kernel, save_dir):
    gp = SingleTaskGP(X, y, covar_module=kernel, n_mixture=5)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    posterior = gp.posterior(X_pred)
    mean = posterior.mean.detach().numpy().squeeze()
    lower, upper = posterior.mvn.confidence_region()
    lower, upper = lower.detach().numpy().squeeze(), upper.detach().numpy().squeeze()

    acq_func = UpperConfidenceBound(gp, beta=1.96, maximize=False)  # UCB acquisition function
    next_point, _ = optimize_acqf(
        acq_func,
        bounds=torch.tensor([[0.0], [6.0]]),
        q=1,
        num_restarts=10,
        raw_samples=100
    )

    acq_values = acq_func(X_pred.unsqueeze(1)).detach().numpy().squeeze()

    X_new = torch.cat([X, next_point])
    y_new = generate_data(X_new).view(-1, 1)
    gp_new = SingleTaskGP(X_new, y_new, covar_module=kernel, n_mixture=5)

    mll_new = ExactMarginalLogLikelihood(gp_new.likelihood, gp_new)
    fit_gpytorch_mll(mll_new)
    posterior_new = gp_new.posterior(X_pred)
    mean_new = posterior_new.mean.detach().numpy().squeeze()
    lower_new, upper_new = posterior_new.mvn.confidence_region()
    lower_new, upper_new = lower_new.detach().numpy().squeeze(), upper_new.detach().numpy().squeeze()

    # Step 6: Plot the results
    fig, ax = plt.subplots(2, 1, figsize=(5, 4))

    # Top plot: Posterior and next sample
    ax[0].plot(X_pred.numpy(), generate_data(X_pred).numpy(), color="#786D5F", label='Objective')
    ax[0].plot(X_pred.numpy(), mean, '#368BC1', label='Posterior (before)')
    ax[0].fill_between(X_pred.numpy().squeeze(), lower, upper, alpha=0.2, color='#368BC1')
    ax[0].scatter(X.numpy(), y.numpy(), c='k')  # Existing data points
    ax[0].scatter(next_point.numpy(), generate_data(next_point).numpy(), c='r', marker='v', label='Next sample', zorder=5)  # New point
    ax[0].plot(X_pred.numpy(), mean_new, 'g--', label='Posterior (after)')
    ax[0].fill_between(X_pred.numpy().squeeze(), lower_new, upper_new, alpha=0.1, color='green')
    ax[0].axis('off')

    # Bottom plot: Acquisition function
    ax[1].plot(X_pred.numpy(), acq_values, '#008080', label='Acquisition function')
    ax[1].scatter(next_point.numpy(), acq_func(next_point).detach().numpy(), c='red', marker='v')
    ax[1].axis('off')

    handles, labels = [], []
    for a in ax:  # Loop through each subplot to collect handles and labels
        for h, l in zip(*a.get_legend_handles_labels()):
            handles.append(h)
            labels.append(l)
    save_path = os.path.join(save_dir, f'compare_acq_{kernel}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()

    fig_legend = plt.figure(figsize=(12, 1))
    fig_legend.legend(handles, labels, loc="center", fontsize=12, frameon=False, ncol=5)
    plt.tight_layout()
    legend_path = os.path.join(save_dir, f'_legend_acq.png')
    plt.savefig(legend_path, dpi=300)

if __name__ == '__main__':
    n = 6
    a, b = 3.0, 6.0
    data = torch.distributions.Beta(a, b).sample((n,))
    data_min = data.min()
    data_max = data.max()
    # train_x = torch.linspace(0, 1, 15).view(-1, 1)*6
    X = ((data - data_min) / (data_max - data_min)).view(-1, 1) * 6
    y = generate_data(X).view(-1, 1)
    X_pred = torch.linspace(0, 6, 100).unsqueeze(-1)
    data_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(data_dir, f'exp_res_best')
    save_dir = os.path.join(data_dir, 'plot_acq')
    kernel_list = ['gsm', 'csm', 'matern', 'rbf', 'rq']
    # num_mixtures = [1, 3, 5, 7]
    for kernel in kernel_list:
        # for n_mixture in num_mixtures:
            plot_acq(X, y, X_pred, kernel, save_dir)
