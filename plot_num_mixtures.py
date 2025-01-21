import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from utils.viz import *


# Spectral kernels
GSM_KERNELS = ['gsm3', 'gsm5', 'gsm7', 'gsm9']
CSM_KERNELS = ['csm3', 'csm5', 'csm7', 'csm9']


# Function to group GSM and CSM kernels
def group_special_kernels(file_list):
    gsm_files = []
    csm_files = []
    for file in file_list:
        if file.endswith('.pkl'):
            _, kernel, _ = parse_filename(file)
            if kernel in GSM_KERNELS:
                gsm_files.append(file)
            elif kernel in CSM_KERNELS:
                csm_files.append(file)
    return gsm_files, csm_files


# Plotting function
def plot_gsm_csm_results(gsm_files, csm_files, data_dir, save_dir, name, acq, global_min):
    fig, axes = plt.subplots(2, 1, figsize=(6, 9))  # Two subplots: GSM and CSM
    gsm_ax, csm_ax = axes  # Split axes

    # Plot GSM kernels
    for file in gsm_files:
        kernel = parse_filename(file)[1]
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            continue
        results = load_pickle(file_path)
        if not len(results):
            continue

        # Optimality Gap
        mean_values = np.mean(results, axis=0)  # Mean over seeds
        std_values = np.std(results, axis=0)  # Standard deviation over seeds
        optimality_gap = np.log(np.abs(mean_values - global_min))
        optimality_gap_std = std_values
        iterations = np.arange(0, len(results[0]))  # Iterations 1 to n

        markers, caps, bars = gsm_ax.errorbar(iterations, optimality_gap, yerr=BAR_DICT[name]["beta"] * optimality_gap_std,
                                           **PARAMS_DICT[kernel], errorevery=BAR_DICT[name]["errorevery"], ms=5)
        [bar.set_alpha(0.5) for bar in bars]
        # gsm_ax.plot(iterations, mean_values, label=f'{kernel}')
        # gsm_ax.fill_between(iterations,
        #                     mean_values - 0.2*std_values,
        #                     mean_values + 0.2*std_values,
        #                     alpha=0.2)

    # gsm_ax.axhline(y=global_max, color='r', linestyle='--', label='Global Minimum')
    # gsm_ax.set_title(f'{name} - {acq} - GSM Kernels')
    gsm_ax.set_xlabel('# Iterations')
    gsm_ax.set_ylabel('Log Optimality Gap')
    gsm_ax.grid(True)
    # gsm_ax.legend()

    # Plot CSM kernels
    for file in csm_files:
        kernel = parse_filename(file)[1]
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            continue
        results = load_pickle(file_path)
        if not len(results):
            continue

        mean_values = np.mean(results, axis=0)  # Mean over seeds
        std_values = np.std(results, axis=0)  # Standard deviation over seeds
        optimality_gap = np.log(np.abs(mean_values - global_min))
        optimality_gap_std = std_values
        iterations = np.arange(0, len(results[0]))  # Iterations 1 to n
        markers, caps, bars = csm_ax.errorbar(iterations, optimality_gap, yerr=BAR_DICT[name]["beta"] * optimality_gap_std,
                                           **PARAMS_DICT[kernel], errorevery=BAR_DICT[name]["errorevery"], ms=5)
        [bar.set_alpha(0.5) for bar in bars]

        # csm_ax.plot(iterations, mean_values, label=f'{kernel}')
        # csm_ax.fill_between(iterations,
        #                     mean_values - 0.2*std_values,
        #                     mean_values + 0.2*std_values,
        #                     alpha=0.2)

    # csm_ax.axhline(y=global_max, color='r', linestyle='--', label='Global Minimum')
    # csm_ax.set_title(f'{name} - {acq} - CSM Kernels')
    csm_ax.set_xlabel('# Iterations')
    csm_ax.set_ylabel('Log Optimality Gap')
    csm_ax.grid(True)
    # csm_ax.legend()

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{name}_sm_{acq}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f'Saved GSM and CSM plot: {save_path}')

    fig_legend = plt.figure(figsize=(8, 3))
    handles = []
    labels = []
    for ax in axes:
        temp_handles, temp_labels = ax.get_legend_handles_labels()
        handles.extend(temp_handles)
        labels.extend(temp_labels)

    fig_legend.legend(handles, labels, loc="center", fontsize=12, frameon=False, ncol=8)
    save_path = os.path.join(save_dir, f'_legend_vertical.png')
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == '__main__':
    option = 'best'
    cur_dir = os.path.dirname(os.path.abspath(__file__))  # Directory containing the .pkl files
    data_dir = os.path.join(cur_dir, f'exp_res_{option}', 'pkl')
    save_dir = os.path.join(cur_dir, f'exp_res_{option}', 'plot_sm')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # List all .pkl files
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]

    # Group files by GSM and CSM kernels
    gsm_files, csm_files = group_special_kernels(file_list)

    # Process each name and acquisition function
    grouped_by_name_acq = defaultdict(lambda: defaultdict(list))
    for file in file_list:
        name, _, acq = parse_filename(file)
        grouped_by_name_acq[name][acq].append(file)

    for name, acq_dict in grouped_by_name_acq.items():
        global_min = GLOBAL_MINIMUM.get(name, None)
        if global_min is None:
            print(f"Warning: No global minimum defined for '{name}'")
            continue

        for acq, files in acq_dict.items():
            gsm_files_acq = [file for file in gsm_files if file in files]
            csm_files_acq = [file for file in csm_files if file in files]
            if gsm_files_acq or csm_files_acq:
                plot_gsm_csm_results(gsm_files_acq, csm_files_acq, data_dir, save_dir, name, acq, global_min)
