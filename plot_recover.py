import os
import torch
from gpytorch.kernels import ScaleKernel, MaternKernel, RBFKernel, PeriodicKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
import gpytorch
import matplotlib.pyplot as plt
import GPy
import numpy as np
import random

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(0)
random.seed(0)

def normalize(arr):
    max_val = np.max(arr)
    min_val = np.min(arr)
    normalized_arr = (arr - min_val) / (max_val - min_val)
    normalized_arr = arr
    return normalized_arr


def gen_data(option, x_sample, dir):
    if option == "MA52":
        kernel = GPy.kern.Matern32(input_dim=1, lengthscale=10.0)
    elif option == "MA32":
        kernel = GPy.kern.Matern32(input_dim=1, lengthscale=10.0)
    elif option == "Add":
        kernel = GPy.kern.Matern52(input_dim=1, lengthscale=10.0) + GPy.kern.PeriodicExponential(input_dim=1, lengthscale=20, period=20.0)
    elif option == "Mul":
        kernel = GPy.kern.Matern52(input_dim=1, lengthscale=10.0) * GPy.kern.PeriodicExponential(input_dim=1, lengthscale=20, period=20.0)
    else:
        raise ValueError

    gp = GPy.models.GPRegression(x_sample, np.zeros_like(x_sample), kernel)
    y_sample = gp.posterior_samples_f(x_sample, size=1)
    covariances_obj = normalize(gp.kern.K(x_sample, x_sample).flatten()[:len(x_sample)])
    
    # model.eval()
    # with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #     prior_dist = model(x_sample)
    #     y_sample = prior_dist.sample().detach()

    x_sample = torch.from_numpy(x_sample)
    y_sample = torch.from_numpy(y_sample).squeeze(-1)

    dist_matrix = torch.cdist(x_sample, x_sample, p=2)
    dist_matrix_np = dist_matrix.detach().numpy()
    distances = dist_matrix_np.flatten()[:len(x_sample)]

    # covar_matrix_obj = model.covar_module(x_sample).evaluate()
    # covar_matrix_np_obj = covar_matrix_obj.detach().numpy()
    # covariances_obj = covar_matrix_np_obj.flatten()[:len(x_sample)]

    appro(x_sample, y_sample, distances, covariances_obj, dir, option)


def appro(train_x, train_y, distances, covariances_obj, save_dir, option):
    model_rbf = SingleTaskGP(train_x, train_y, covar_module='rbf')
    mll_rbf = ExactMarginalLogLikelihood(model_rbf.likelihood, model_rbf)
    fit_gpytorch_mll(mll_rbf)
    marginal_likelihood_rbf = mll_rbf(model_rbf(train_x.squeeze()), train_y.squeeze()).item()
    posterior_rbf = model_rbf.posterior(train_x)
    mean_rbf = posterior_rbf.mean.detach().numpy().squeeze()

    covar_matrix_rbf = model_rbf.covar_module(train_x).evaluate()
    covar_matrix_np_rbf = covar_matrix_rbf.detach().numpy()
    covariances_rbf = normalize(covar_matrix_np_rbf.flatten()[:len(train_x)])

    model_rq = SingleTaskGP(train_x, train_y, covar_module='rq')
    mll_rq = ExactMarginalLogLikelihood(model_rq.likelihood, model_rq)
    fit_gpytorch_mll(mll_rq)
    marginal_likelihood_rq = mll_rq(model_rq(train_x.squeeze()), train_y.squeeze()).item()
    posterior_rq = model_rq.posterior(train_x)
    mean_rq = posterior_rq.mean.detach().numpy().squeeze()

    covar_matrix_rq = model_rq.covar_module(train_x).evaluate()
    covar_matrix_np_rq = covar_matrix_rq.detach().numpy()
    covariances_rq = normalize(covar_matrix_np_rq.flatten()[:len(train_x)])

    model_gsm = SingleTaskGP(train_x, train_y, covar_module='gsm', n_mixture=3)
    mll_gsm = ExactMarginalLogLikelihood(model_gsm.likelihood, model_gsm)
    fit_gpytorch_mll(mll_gsm)
    marginal_likelihood_gsm = mll_gsm(model_gsm(train_x.squeeze()), train_y.squeeze()).item()
    posterior_gsm = model_gsm.posterior(train_x)
    mean_gsm = posterior_gsm.mean.detach().numpy().squeeze()

    covar_matrix_gsm = model_gsm.covar_module(train_x).evaluate()
    covar_matrix_np_gsm = covar_matrix_gsm.detach().numpy()
    covariances_gsm = normalize(covar_matrix_np_gsm.flatten()[:len(train_x)])

    model_csm = SingleTaskGP(train_x, train_y, covar_module='csm', n_mixture=3)
    mll_csm = ExactMarginalLogLikelihood(model_csm.likelihood, model_csm)
    fit_gpytorch_mll(mll_csm)
    marginal_likelihood_csm = mll_csm(model_csm(train_x.squeeze()), train_y.squeeze()).item()
    posterior_csm = model_csm.posterior(train_x)
    mean_csm = posterior_csm.mean.detach().numpy().squeeze()

    covar_matrix_csm = model_csm.covar_module(train_x).evaluate()
    covar_matrix_np_csm = covar_matrix_csm.detach().numpy()
    covariances_csm = normalize(covar_matrix_np_csm.flatten()[:len(train_x)])

    print(f"Marginal likelihood of GP with PE kernel: {marginal_likelihood_rbf}")
    print(f"Marginal likelihood of GP with RQ kernel: {marginal_likelihood_rq}")
    print(f"Marginal likelihood of GP with GSM kernel: {marginal_likelihood_gsm}")
    print(f"Marginal likelihood of GP with CSM kernel: {marginal_likelihood_csm}")

    # plt.figure(figsize=(5, 5))
    # plt.plot(train_x, train_y, '#786D5F', label='Target')
    # plt.plot(train_x, mean_rbf, 'g--', label='PE')
    # plt.plot(train_x, mean_rq, 'k--', label='RQ')
    # plt.plot(train_x, mean_gsm, '#008080', label='GSM')
    # plt.plot(train_x, mean_csm, '#368BC1', label='CSM')
    # save_dir1 = os.path.join(save_dir, f'recover_val_{option}.png')
    # plt.savefig(save_dir1, dpi=300)
    # plt.close()

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.plot(distances, covariances_obj, '#786D5F', label='Target')
    ax.plot(distances, covariances_rbf, 'g--', label='SE')
    ax.plot(distances, covariances_rq, 'k--', label='RQ')
    ax.plot(distances, covariances_gsm, '#008080', label='GSM')
    ax.plot(distances, covariances_csm, '#368BC1', label='CSM')
    ax.set_ylabel('Correlation')
    save_dir2 = os.path.join(save_dir, f'recover_corr_{option}.png')
    plt.savefig(save_dir2, dpi=300)
    plt.close()

    fig_legend = plt.figure(figsize=(10, 2))
    handles, labels = ax.get_legend_handles_labels()
    fig_legend.legend(handles, labels, loc="center", fontsize=12, frameon=False, ncol=5)
    save_legend = os.path.join(save_dir, f'_legend.png')
    fig_legend.savefig(save_legend, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == '__main__':
    init = np.linspace(0, 80, 1000)
    n_sample = 200
    x_sample = np.sort(np.concatenate((np.array([0]), np.random.choice(init, n_sample, replace=False)))).reshape(-1, 1)
    data_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(data_dir, 'exp_res_best', 'plot_recover')
    option_list = ["MA52", "MA32", "Add", "Mul"]
    for item in option_list:
        print(f"Sampling form {item}......:")
        gen_data(item,x_sample, save_dir)

