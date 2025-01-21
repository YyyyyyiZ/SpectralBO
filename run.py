import torch
import argparse
import pickle
import warnings
import os
import random
import time

warnings.filterwarnings("ignore")

from test_functions import *
from utils.utils import get_next_points

parser = argparse.ArgumentParser('Run BayesOpt Experiments')
parser.add_argument('--function_name', type=str, default='rosen14', help='objective function')
parser.add_argument('--n_iter', type=int, default=10, help='number of trials')
parser.add_argument('--n_init', type=int, default=5, help='number of initial random points')
parser.add_argument('--kernel', type=str, default='csm', help='choice of kernel')
parser.add_argument('--acq', type=str, default='ei', help='choice of the acquisition function')
parser.add_argument('--n_mixture', type=int, default=0, help='number of mixtures for a SM kernel')
parser.add_argument('--seed', type=int, default=10, help='random seed')

args = parser.parse_args()
options = vars(args)
print(options)

seed_list = range(0, args.seed, 1)
query_all = []
best_all = []
time_start = time.time()
for seed in seed_list:
    random.seed(seed)
    print("----------------------------Running seed {}----------------------------".format(seed))
    if args.function_name == 'branin2':
        func = Branin()
    elif args.function_name == 'hartmann3':
        func = Hartmann3()
    elif args.function_name == 'hartmann6':
        func = Hartmann6()
    elif args.function_name == 'exp10':
        func = Exponential(dim=10)
    elif args.function_name == 'robot3':
        gpos = 10 * torch.randn(1, 2) - 5
        func = Robot3(gpos[0][0], gpos[0][1])
    elif args.function_name == 'robot4':
        gpos = 10 * torch.randn(1, 2) - 5
        func = Robot4(gpos[0][0], gpos[0][1])
    else:
        raise ValueError('Unrecognised problem %s' % args.function_name)

    d = func.dim
    lb = func.lb
    ub = func.ub
    bounds = torch.stack((lb, ub))
    optimum = func.min

    init_x = torch.rand(args.n_init, d, dtype=torch.float32)
    init_x = bounds[0] + (bounds[1] - bounds[0]) * init_x
    if args.function_name == 'robot3' or args.function_name == 'robot4':
        init_x = torch.mean(init_x, dim=0, keepdim=True)
    init_y = func.eval(init_x)
    best_init_y = init_y.min().item()

    n_iterations = args.n_iter
    candidates = []
    results = []
    best_result = []
    best_result.append(best_init_y)
    try:
        for i in range(n_iterations):
            print(f"Number of iterations done: {i}")
            try:
                new_candidates = get_next_points(args.acq, args.kernel, args.n_mixture, init_x, init_y, best_init_y, bounds, 1)
                # new_candidates = get_next_points_and_visualize(init_x, init_y, best_init_y, bounds, i, candidates, results, 1)
                new_results = func.eval(new_candidates)

                print(f"New candidates are: {new_candidates}, {new_results}")
                init_x = torch.cat([init_x, new_candidates])
                init_y = torch.cat([init_y, new_results])

                best_init_y = init_y.min().item()
                print(f"f_min: {best_init_y}")
                best_result.append(best_init_y)
                candidates.append(float(new_candidates[0][0]))
                results.append(float(new_results[0][0]))
            except:
                print(f"Iteration {i} failed")
                best_init_y = init_y.min().item()
                best_result.append(best_init_y)
                candidates.append(candidates[-1])
                results.append(results[-1])
        query_all.append(results)
        best_all.append(best_result)
    except:
        print(f"Seed {seed} failed")

time_end = time.time()
running_time = (time_end - time_start)/len(seed_list)
print(f"Running time for {args.function_name}: {running_time} seconds")

current_dir = os.path.dirname(os.path.abspath(__file__))
if args.n_mixture != 0:
    filename = args.function_name + '_' + args.kernel + str(args.n_mixture) + '_' + args.acq + '.pkl'
else:
    filename = args.function_name + '_' + args.kernel + '_' + args.acq + '.pkl'

save_dir = os.path.join(current_dir, 'exp_res_query', 'pkl')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
file_path = os.path.join(save_dir, filename)
with open(file_path, 'wb') as f:
    pickle.dump(query_all, f)

save_dir = os.path.join(current_dir, 'exp_res_best', 'pkl')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
file_path = os.path.join(save_dir, filename)
with open(file_path, 'wb') as f:
    pickle.dump(best_all, f)