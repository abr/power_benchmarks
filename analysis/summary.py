import argparse
import os
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from analysis import load_running_samples, load_idle_samples, average_idle_power
sns.set()

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str)
parser.add_argument("--baseline_dir", type=str)
args = parser.parse_args()

# the general to approach to building plots is as follows: each log entry
# corresponding to a power reading gets matched to the json metadata of the
# benchmarking experiment that gave rise to the log. A 'sample' consists of
# one such unit of log information. We load all samples from both idling and
# runtime logs into a single Pandas dataframe so the we can do arbitrary
# queries to selectively plot information of interest. The queries have become
# a little complicated and unintuitive as further experimental configurations
# and plots were added over time (i.e. they were initially unanticipated).

def set_legend_title(ax):
    '''Convenience function for repeated plotting code'''
    legend = ax[0].legend(loc='upper left')
    legend.set_title('')
    legend = ax[1].legend(loc='upper left')
    legend.set_title('')


# collect regular idling samples from logs in baseline_dir
idle_samples = []

idle_filenames = [f for f in os.listdir(args.baseline_dir) if f.endswith('.csv')]

for fname in idle_filenames:
    path = os.path.join(args.baseline_dir, fname)
    samples = load_idle_samples(path)
    idle_samples.extend(samples)

# collect running samples, metadata from logs in log_dir
# we use prefixes because we want both the json and csv for each prefix
run_file_prefixes = list(set([f.split('.')[0] for f in os.listdir(args.log_dir)]))
run_file_prefixes = [p for p in run_file_prefixes if p != '']

run_samples = []

for prefix in run_file_prefixes:
    path = os.path.join(args.log_dir, prefix)

    # idle power average for hardware indicated by path
    idle_power = average_idle_power(idle_samples, prefix)

    # load_samples collates metadata and log entries, hence use of prefix
    # idle power is subtracted from running power in each entry
    samples = load_running_samples(path, idle_power=idle_power)

    run_samples.extend(samples)


# load all samples into pandas to support arbitrary qeueries
all_samples = idle_samples + run_samples

dframe = pandas.DataFrame(all_samples)
order = ['CPU', 'GPU', 'JETSON', 'NCS2', 'MOVIDIUS', 'LOIHI']

# TABLE 1: Power, energy cost for all hardware devices

# this selects out the data for functional versions of the model
joules_dframe = dframe.loc[(dframe['batchsize'] == 1) &
                           (dframe['n_copies'] == 1) &
                           (dframe['n_layers'].isnull()) &
                           (dframe['nx_neurons'] == 1)]

# compute means over all samples for each hardware device
mean_loihi = joules_dframe.loc[joules_dframe['hardware'] == 'LOIHI'].mean()
mean_ncs2 = joules_dframe.loc[joules_dframe['hardware'] == 'NCS2'].mean()
mean_movidius = joules_dframe.loc[joules_dframe['hardware'] == 'MOVIDIUS'].mean()
mean_jetson = joules_dframe.loc[joules_dframe['hardware'] == 'JETSON'].mean()
mean_cpu = joules_dframe.loc[joules_dframe['hardware'] == 'CPU'].mean()
mean_gpu = joules_dframe.loc[joules_dframe['hardware'] == 'GPU'].mean()

# print out mean values for populating table in summary document
prefixes = ['loihi', 'movidius', 'ncs2', 'jetson', 'cpu', 'gpu']

print('Idle Power')
for prefix in prefixes:
    print(prefix + ': %4f' % average_idle_power(idle_samples, prefix))

print('')
means = [mean_loihi, mean_movidius, mean_ncs2, mean_jetson, mean_cpu, mean_gpu]
for data, prefix in zip(means, prefixes):
    print(prefix + ':')
    print('Total Power: %4f' % data['total_power'])
    print('Dynamice Power: %4f'  %  data['dynamic_power'])
    print('Inf/Sec: %4f' % data['inf_per_second'])
    print('Joules/Inf: %4f' % data['dynamic_joules_per_inf'])
    print('')

# compute ratios for energy costs for plotting numbers alongside bars
movidius_x = mean_movidius['dynamic_joules_per_inf'] / mean_loihi['dynamic_joules_per_inf']
ncs_x = mean_ncs2['dynamic_joules_per_inf'] / mean_loihi['dynamic_joules_per_inf']
jetson_x = mean_jetson['dynamic_joules_per_inf'] / mean_loihi['dynamic_joules_per_inf']
cpu_x = mean_cpu['dynamic_joules_per_inf'] / mean_loihi['dynamic_joules_per_inf']
gpu_x = mean_gpu['dynamic_joules_per_inf'] / mean_loihi['dynamic_joules_per_inf']


# PLOT 1. Dynamic joules per inference comparison
plt.figure(figsize=(6, 6))
plot = sns.barplot(
    'hardware', 'dynamic_joules_per_inf',
    data=joules_dframe,
    order=['LOIHI', 'MOVIDIUS', 'NCS2', 'JETSON', 'CPU', 'GPU'])

# add ratios of power consumption to plot
plt.gcf().text(
    0.18, 0.17, str(1) + 'x', fontsize=15, fontweight='bold')
plt.gcf().text(
    0.28, 0.17, str(round(movidius_x, 1)) + 'x', fontsize=15, fontweight='bold')
plt.gcf().text(
    0.41, 0.17, str(round(ncs_x, 1)) + 'x', fontsize=15, fontweight='bold')
plt.gcf().text(
    0.52, 0.17, str(round(jetson_x, 1)) + 'x', fontsize=15, fontweight='bold')
plt.gcf().text(
    0.65, 0.17, str(round(cpu_x, 1)) + 'x', fontsize=15, fontweight='bold')
plt.gcf().text(
    0.77, 0.17, str(round(gpu_x, 1)) + 'x', fontsize=15, fontweight='bold')

plot.set_title('Dynamic Energy Cost Per Inference (batchsize = 1)', fontsize=14)
plot.set_xlabel('', labelpad=10)
plot.set_ylabel('Joules', fontsize=14)

plot.figure.savefig("./paper/figures/per_inf_comparison.png")
plt.show()

# PLOT 2. Batchsize comparison
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
fig.subplots_adjust(hspace=0.9)

# pick out dframe subste corresponding to the batchable hardware devices
bsize_dframe = dframe.loc[(dframe['nx_neurons'] == 1) &
                          (dframe['hardware'].isin(['JETSON', 'CPU', 'GPU'])) &
                          (dframe['n_copies'] == 1)]

bsize_order = ['JETSON', 'CPU', 'GPU']

sns.barplot(
    'batchsize', 'inf_per_second',
    hue='hardware', ax=ax[0],
    data=bsize_dframe, hue_order=bsize_order)

sns.barplot(
    'batchsize', 'dynamic_joules_per_inf',
    hue='hardware', ax=ax[1],
    data=bsize_dframe, hue_order=bsize_order)

ax[0].set_title('Inferences per Second', fontsize=14)
ax[0].set_xlabel('Batchsize', fontsize=14)
ax[0].set_ylabel('Inferences', fontsize=14)

ax[1].set_title('Dynamic Energy Cost Per Inference', fontsize=14)
ax[1].set_xlabel('Batchsize', fontsize=14)
ax[1].set_ylabel('Joules', fontsize=14)

plt.gcf().text(0.91, 0.14, 'Movidius', fontsize=11)
plt.gcf().text(0.91, 0.11, 'Loihi', fontsize=11)

# add lines for single batch movidius, loihi energy cost
ax[1].axhline(mean_movidius['dynamic_joules_per_inf'], ls='--', c='k', linewidth=0.7)
ax[1].axhline(mean_loihi['dynamic_joules_per_inf'], ls='--', c='k', linewidth=0.7)

set_legend_title(ax)

fig.savefig("./paper/figures/batch_comparison.png")
plt.show()


# PLOT 3. Movidius/Loihi Comparison Details
# this is for scaling with multiple copies or branches, with 10 layers/branch
fig, ax = plt.subplots(1, 3, figsize=(16, 7))
fig.subplots_adjust(hspace=0.9, wspace=0.3)

comp_dframe = dframe.loc[(dframe['batchsize'] == 1) &
                         (dframe['hardware'].isin(['MOVIDIUS', 'LOIHI'])) &
                         ((dframe['n_layers'] == 10) | (dframe['n_layers'] == 0)) &
                         (dframe['n_copies'] != 1) &
                         (dframe['nx_neurons'] == 1)]

sns.barplot(
    'n_copies', 'dynamic_power', ax=ax[0], hue='hardware',
    data=comp_dframe, hue_order=['MOVIDIUS', 'LOIHI'])

speed = sns.barplot(
    'n_copies', 'inf_per_second', ax=ax[1], hue='hardware',
    data=comp_dframe, hue_order=['MOVIDIUS', 'LOIHI'])

cost = sns.barplot(
    'n_copies', 'dynamic_joules_per_inf', hue='hardware', ax=ax[2],
    data=comp_dframe, hue_order=['MOVIDIUS', 'LOIHI'])


# get inf cost, speed ratios for plot
costs = np.array([d.get_height() for d in cost.patches])
costs = costs.reshape(2, -1)
cost_ratios = costs[0, :] / costs[1, :]

# print('Cost multipliers: ', cost_ratios)
# print('')
speeds = np.array([d.get_height() for d in speed.patches])
speeds = speeds.reshape(2, -1)
speed_ratios = speeds[1, :] / speeds[0, :]
# print('Speed multipliers: ', speed_ratios)
# print('')

ax[0].set_title('Average Dynamic Power', fontsize=14)
ax[0].set_xlabel('N (# neurons = N*10*256 + 512)', fontsize=14)
ax[0].set_ylabel('Watts', fontsize=14)

ax[1].set_title('Average Inference Speed', fontsize=14)
ax[1].set_xlabel('N (# neurons = N*10*256 + 512)', fontsize=14)
ax[1].set_ylabel('Inferences Per Second', fontsize=14)
ax[1].axhline(100, ls='--', c='k', linewidth=0.7)

ax[2].set_title('Average Cost Per Inference', fontsize=14)
ax[2].set_xlabel('N (# neurons = N*10*256 + 512)', fontsize=14)
ax[2].set_ylabel('Joules')

# add energy cost ratios for comparison of scaled Movidius, Loihi models
plt.gcf().text(
    0.704, 0.2, str(round(cost_ratios[0], 1)) + 'x', fontsize=10, fontweight='bold')
plt.gcf().text(
    0.74, 0.2, str(round(cost_ratios[1], 1)) + 'x', fontsize=10, fontweight='bold')
plt.gcf().text(
    0.775, 0.2, str(round(cost_ratios[2], 1)) + 'x', fontsize=10, fontweight='bold')
plt.gcf().text(
    0.811, 0.2, str(round(cost_ratios[3], 1)) + 'x', fontsize=10, fontweight='bold')
plt.gcf().text(
    0.846, 0.2, str(round(cost_ratios[4], 1)) + 'x', fontsize=10, fontweight='bold')
plt.gcf().text(
    0.882, 0.2, str(round(cost_ratios[5], 1)) + 'x', fontsize=10, fontweight='bold')

legend = ax[0].legend()
legend.set_title('')
legend = ax[1].legend()
legend.set_title('')
legend = ax[2].legend()
legend.set_title('')

fig.savefig("./paper/figures/movidius_summary.png")
plt.show()


# PLOT 4: Loihi Scaling Details
fig, ax = plt.subplots(1, 3, figsize=(15, 7))
fig.subplots_adjust(hspace=0.9, wspace=0.3)

# pull out Loihi samples, excluding the original keyword spotter
loihi_dframe = dframe.loc[(dframe['batchsize'] == 1) &
                          (dframe['hardware'].isin(['LOIHI'])) &
                          (dframe['n_copies'] != 1) &
                          ((dframe['n_layers'] == 10) | (dframe['n_layers'] == 0)) &
                          (dframe['nx_neurons'] == 1)]

sns.barplot(
    'n_copies', 'dynamic_power', ax=ax[0], hue='hardware', data=loihi_dframe)

sns.barplot(
    'n_copies', 'inf_per_second', ax=ax[1], hue='hardware', data=loihi_dframe)

sns.barplot(
    'n_copies', 'dynamic_joules_per_inf', hue='hardware', ax=ax[2], data=loihi_dframe)


ax[0].set_title('Average Dynamic Power', fontsize=14)
ax[0].set_xlabel('N (# neurons = N*10*256 + 512)', fontsize=14)
ax[0].set_ylabel('Watts', fontsize=14)

ax[1].set_title('Average Inference Speed', fontsize=14)
ax[1].set_xlabel('N (# neurons = N*10*256 + 512)', fontsize=14)
ax[1].set_ylabel('Inferences Per Second', fontsize=14)

ax[2].set_title('Average Cost Per Inference', fontsize=14)
ax[2].set_xlabel('N (# neurons = N*10*256 + 512)', fontsize=14)
ax[2].set_ylabel('Joules', fontsize=14)

legend = ax[0].legend()
legend.set_title('')

legend = ax[1].legend()
legend.set_title('')

legend = ax[2].legend()
legend.set_title('')

fig.savefig("./paper/figures/loihi_summary.png")
plt.show()


# PLOT 5. CPU/GPU/MOVIDIUS Comparison
# this is for scaling with multiple copies or branches, with 10 layers/branch
fig, ax = plt.subplots(1, 3, figsize=(16, 7))
fig.subplots_adjust(hspace=0.9, wspace=0.3)

scale_dframe = dframe.loc[(dframe['batchsize'] == 1) &
                          (dframe['hardware'].isin(['MOVIDIUS', 'CPU', 'GPU'])) &
                          ((dframe['n_layers'] == 10) | (dframe['n_layers'] == 0)) &
                          (dframe['n_copies'] != 1) &
                          (dframe['nx_neurons'] == 1)]

sns.barplot(
    'n_copies', 'dynamic_power', ax=ax[0], hue='hardware',
    data=scale_dframe, hue_order=['GPU', 'CPU', 'MOVIDIUS'])

speed = sns.barplot(
    'n_copies', 'inf_per_second', ax=ax[1], hue='hardware',
    data=scale_dframe, hue_order=['GPU', 'CPU', 'MOVIDIUS'])

cost = sns.barplot(
    'n_copies', 'dynamic_joules_per_inf', hue='hardware', ax=ax[2],
    data=scale_dframe, hue_order=['GPU', 'CPU', 'MOVIDIUS'])


ax[0].set_title('Average Dynamic Power', fontsize=14)
ax[0].set_xlabel('N (# neurons = N*10*256 + 512)', fontsize=14)
ax[0].set_ylabel('Watts', fontsize=14)

ax[1].set_title('Average Inference Speed', fontsize=14)
ax[1].set_xlabel('N (# neurons = N*10*256 + 512)', fontsize=14)
ax[1].set_ylabel('Inferences Per Second', fontsize=14)

ax[2].set_title('Average Cost Per Inference', fontsize=14)
ax[2].set_xlabel('N (# neurons = N*10*256 + 512)', fontsize=14)
ax[2].set_ylabel('Joules')

legend = ax[0].legend()
legend.set_title('')
legend = ax[1].legend()
legend.set_title('')
legend = ax[2].legend()
legend.set_title('')

fig.savefig("./paper/figures/comp_summary.png")
plt.show()
