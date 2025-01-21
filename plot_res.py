import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from utils.viz import *


# Function to group files by "name" and "acq"
def group_files(file_list):
    grouped_files = defaultdict(lambda: defaultdict(list))
    for file in file_list:
        if file.endswith('.pkl'):
            name, kernel, acq = parse_filename(file)
            grouped_files[name][acq].append((kernel, file))
    return grouped_files


# Plotting function
def plot_results(grouped_files, data_dir, save_dir):
    for name, acq_dict in grouped_files.items():
        if name not in ['exp10', 'hartmann6']:
            continue
        global_min = GLOBAL_MINIMUM.get(name, None)  # Get the global maximum for the current name
        if global_min is None:
            print(f"Warning: No global minimum defined for '{name}'")
            continue

        for acq, kernel_files in acq_dict.items():
            # Prepare data for each kernel
            all_mean_values = {}
            all_std_values = {}
            optimality_gaps = {}
            relative_improvements = {}

            for kernel, file in kernel_files:
                if kernel in ['csm', 'csm3', 'csm5', 'csm9', 'gsm', 'gsm3', 'gsm5', 'gsm9']:
                    continue
                file_path = os.path.join(data_dir, file)
                if not os.path.exists(file_path):
                    continue
                results = load_pickle(file_path)
                if not results:
                    continue

                mean_values = np.mean(results, axis=0)  # Mean over seeds
                std_values = np.std(results, axis=0)  # Standard deviation over seeds

                iterations = np.arange(0, len(results[0]))  # Iterations 0 to n

                # # Calculate metrics
                # f_best = np.maximum.accumulate(mean_values)  # Cumulative maximum (f_best)
                # f_first = mean_values[0]  # First value (f_first)

                # Log Optimality Gap
                optimality_gap = np.log(np.abs(mean_values - global_min))
                optimality_gap_std = np.log(np.abs(std_values+1))

                # # Relative Improvement
                # if global_min != f_first:  # Prevent division by zero
                #     relative_improvement = (mean_values - f_first) / (global_min - f_first)
                #     relative_improvement_std = std_values / abs(global_min - f_first)
                # else:
                #     relative_improvement = np.zeros_like(f_best)
                #     relative_improvement_std = np.zeros_like(f_best)

                # Store data for plotting
                all_mean_values[kernel] = (iterations, mean_values, std_values)
                optimality_gaps[kernel] = (iterations, optimality_gap, optimality_gap_std)
                # relative_improvements[kernel] = (iterations, relative_improvement, relative_improvement_std)

            # # Plot 1: Mean with ±1 std deviation
            # plt.figure(figsize=(10, 5))
            # for kernel, (iterations, mean_values, std_values) in all_mean_values.items():
            #     plt.plot(iterations, mean_values, label=f'{kernel}')
            #     plt.fill_between(iterations,
            #                      mean_values - 0.2*std_values,
            #                      mean_values + 0.2*std_values,
            #                      alpha=0.2)
            # plt.axhline(y=global_min, color='r', linestyle='--', label='Global Minimum')
            # plt.title(f'{name} - Acquisition Function: {acq} (Objective Value)')
            # plt.xlabel('Iterations')
            # plt.ylabel('Objective Value')
            # plt.legend()
            # save_path = os.path.join(save_dir, f'{name}_{acq}_mean.png')
            # plt.savefig(save_path)
            # plt.close()
            # print(f'Saved mean curve plot: {save_path}')

            # Plot 2: Optimality Gap with ±1 std deviation
            fig = plt.figure(figsize=(6, 4.5))
            ax = fig.add_subplot(111)
            for kernel, (iterations, optimality_gap, optimality_gap_std) in optimality_gaps.items():
                markers, caps, bars = plt.errorbar(iterations, optimality_gap, yerr=BAR_DICT[name]["beta"] * optimality_gap_std,
                                                   **PARAMS_DICT[kernel], errorevery=BAR_DICT[name]["errorevery"], ms=5)
                [bar.set_alpha(0.5) for bar in bars]
                # plt.plot(iterations, optimality_gap, label=param['label'], marker=param['marker'], linestyle=param['linestyle'], color=param['color'])
                #
                # plt.fill_between(iterations,
                #                  optimality_gap - 0.2*optimality_gap_std,
                #                  optimality_gap + 0.2*optimality_gap_std,
                #                  alpha=0.2)
            # plt.axhline(y=0, color='r', linestyle='--', label='Gap=0')
            # plt.title(f'{name} - Acquisition Function: {acq} (Optimality Gap)')
            plt.xlabel('# Iterations')
            plt.ylabel('Log Optimality Gap')
            plt.grid(True)
            # plt.legend()
            save_path = os.path.join(save_dir, f'{name}_{acq}_optgap.png')
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f'Saved optimality gap plot: {save_path}')

            fig_legend = plt.figure(figsize=(4, 2))
            handles, labels = ax.get_legend_handles_labels()
            fig_legend.legend(handles, labels, loc="center", fontsize=12, frameon=False,  ncol=6)
            save_legend = os.path.join(save_dir, f'_legend.png')
            fig_legend.savefig(save_legend, dpi=300, bbox_inches="tight")
            plt.close()

            # # Plot 3: Relative Improvement with ±1 std deviation
            # plt.figure(figsize=(10, 5))
            # for kernel, (iterations, relative_improvement, relative_improvement_std) in relative_improvements.items():
            #     plt.plot(iterations, relative_improvement, label=f'{kernel}')
            #     # plt.fill_between(iterations,
            #     #                  relative_improvement - relative_improvement_std,
            #     #                  relative_improvement + relative_improvement_std,
            #     #                  alpha=0.2)
            # plt.axhline(y=1, color='r', linestyle='--', label='Improve=100%')
            # plt.title(f'{name} - Acquisition Function: {acq} (Relative Improvement)')
            # plt.xlabel('Iterations')
            # plt.ylabel('Relative Improvement')
            # plt.legend()
            # save_path = os.path.join(save_dir, f'{name}_{acq}_improve.png')
            # plt.savefig(save_path)
            # plt.close()
            # print(f'Saved relative improvement plot: {save_path}')


if __name__ == '__main__':
    option = 'best'
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(cur_dir, f'exp_res_{option}', 'pkl')
    save_dir = os.path.join(cur_dir, f'exp_res_{option}', 'plot_res')
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    grouped_files = group_files(file_list)
    plot_results(grouped_files, data_dir, save_dir)
