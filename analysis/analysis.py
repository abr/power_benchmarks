import csv
import json
import string

from bootstrap import mean
from datetime import datetime 


def format(timestamp, loihi=False):
    '''Convert timestamp to format used by nvidia-smi, s-tui for matching'''
    timestamp = timestamp.replace('/', '-')
    timestamp = timestamp.replace(' ', '_')

    if not loihi:
        timestamp = timestamp.split('.')[0]

    return timestamp


def filestring_to_hardware(filestring):
    '''Get hardware type from filestring (note file naming requirements)'''
    # get only filename, not directory name
    if '/' in filestring:
        filestring = filestring.split('/')[-1]

    if 'cpu' in filestring:
        hardware = 'CPU'
    elif 'gpu' in filestring:
        hardware = 'GPU'
    elif 'ncs2' in filestring:
        hardware = 'NCS2'
    elif 'movidius' in filestring:
        hardware = 'MOVIDIUS'
    elif 'loihi' in filestring:
        hardware = 'LOIHI'
    elif 'jetson' in filestring:
        hardware = 'JETSON'
    else:
        raise Exception('Hardware unspecified in filestring!')

    return hardware


def load_running_samples(path_prefix, idle_power=0):
    '''Load log, summary to make data samples, w/ idle power to subtract'''
    with open(path_prefix + '.json', 'r') as jfile:
        summary = json.load(jfile)

    with open(path_prefix + '.csv', 'r') as cfile:
        rows = list(csv.reader(cfile))

    loihi = 1 if 'loihi' in path_prefix else 0

    start = summary['start_time']
    end = summary['end_time']

    if loihi:
        # compute delta between timestamps at millisecond level
        fmt = '%Y-%m-%d_%H:%M:%S.%f'
        delta = datetime.strptime(end, fmt) - datetime.strptime(start, fmt)

        summary['n_seconds'] = delta.total_seconds()
        summary['inf_per_second'] = summary['n_inferences'] / summary['n_seconds']

    # pull out start and end indices based on time stamps in summary 
    # we drop the last entry its timestamp doesn't align with logging times 
    # (this happens on loihi if execution ending falls between the log interval)
    start_ids = [i for i, row in enumerate(rows) if format(row[0], loihi) == start]
    end_ids = [i for i, row in enumerate(rows) if format(row[0], loihi) <= end]

    start_idx = start_ids[0]
    end_idx = end_ids[-1]

    power = []
    times = []

    for idx, row in enumerate(rows):
        if idx >= start_idx and idx <= end_idx:
            # for handling different logging formats
            if len(row) > 5 and not loihi:
                watts = row[7]  # s-tui output
            else:
                watts = row[1]  # nvidia-smi, jetson, movidius, loihi output

            # screen out letters (e.g. W as unit of measure)
            watts = [i for i in watts if i not in string.ascii_letters]
            watts = ''.join(watts).strip()

            if len(watts) > 0:
                power.append(float(watts))
                times.append(row[0])  # timestamp is always first column
 
    assert len(power) == len(times)

    log_data = list(zip(times, power))

    dt = summary['n_seconds'] / len(log_data)  # avg. dt between readings
    inf_per_dt = (summary['n_inferences'] / summary['n_seconds']) * dt

    samples = []

    for time, watts in log_data:
        sample = summary.copy()
        sample['timestamp'] = time
        sample['total_power'] = watts
        sample['dynamic_power'] = watts - idle_power
        sample['total_joules'] = dt * watts
        sample['dynamic_joules'] = dt * sample['dynamic_power']
        sample['total_joules_per_inf'] = sample['total_joules'] / inf_per_dt
        sample['dynamic_joules_per_inf'] = sample['dynamic_joules'] / inf_per_dt

        samples.append(sample)

    return samples


def load_idle_samples(logfile, index_buffer=20):
    '''Average idle power readings to get a baseline consumption value'''
    with open(logfile, 'r') as cfile:
        rows = list(csv.reader(cfile))
    
    # we set a buffer to exclude start and end of the idling log
    start_idx = index_buffer
    end_idx = len(rows) - index_buffer

    power = []

    for idx, row in enumerate(rows):
        if idx >= start_idx and idx <= end_idx:
            # for handling different logging formats
            if len(row) > 5 and 'loihi' not in logfile:
                watts = row[7]  # s-tui output
            else:
                watts = row[1]  # nvidia-smi, loihi output

            # screen out letters (e.g. W as unit of measure)
            watts = [i for i in watts if i not in string.ascii_letters]
            watts = ''.join(watts).strip()

            if len(watts) > 0:
                power.append(float(watts))

    hardware = filestring_to_hardware(logfile)

    samples = []
    for reading in power:
        sample = {}
        sample['status'] = 'Idle'
        sample['total_power'] = reading
        sample['hardware'] = hardware
        samples.append(sample)

    return samples


def average_idle_power(samples, file_prefix):
    '''Compute average idle power to subtract from entries in log file'''
    hardware = filestring_to_hardware(file_prefix)
    hardware_samples = [s for s in samples if s['hardware'] == hardware]
    average_power = mean([s['total_power'] for s in hardware_samples])

    return average_power
