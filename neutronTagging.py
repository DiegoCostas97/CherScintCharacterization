import sys

import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt

from tqdm import tqdm

path_to_first    = "/Users/diiego/Library/Mobile Documents/com~apple~CloudDocs/Desktop/DIEGO_cloud/USC/PHD/HK/HK SOURCES/code/ambe_source/npz_ana/cher_scint_characterization"
path_to_paquetes = "/Users/diiego/Library/Mobile Documents/com~apple~CloudDocs/Desktop/DIEGO_cloud/USC/PHD/HK/HK SOURCES/code/ambe_source/npz_ana/paquetes"
sys.path.append(path_to_paquetes)
sys.path.append(path_to_first)

from first import plot_light

verbose = True
path_to_data = str(sys.argv[1])
threshold    = int(sys.argv[2])

full_df = pd.read_csv(path_to_data)

# Function that checks if there are values in list separated more than 20 units
def sep_20ns(lista):
    for i in range(len(lista)):
        for j in range(i + 1, len(lista)):
            if abs(lista[i] - lista[j]) < 20:
                return True
    return False

def nCandidateSearch(df, thresh):
    # Select the events of the DataFrame
    events = np.unique(full_df['event_id'])
    thresh = 5

    # Create an empty list that will store the list of clusters and list of t0s in every event
    hits_in_cluster_per_event = []

    clusters = []
    times    = []
    event    = []

    # Loop over all events in DataFrame
    for ev in tqdm(events, desc="10 ns sliding window", unit=" event", dynamic_ncols=False):
        # Create a temporary DataFrame with just the hits from the current processed event
        temp_df = full_df[full_df['event_id'].values == ev].sort_values('digi_hit_time')

        # Create the 10 ns sliding window in the temporary DataFrame and store the length of the DataFrame in that 10 ns window,
        # i.e. how many hits the window has. Maybe this can be speed up somehow.
        # Now we are just storing the sliding windows with 5 or more hits
        hits_in_cluster = [len(temp_df[(temp_df['digi_hit_time'].values >= t) &
                                       (temp_df['digi_hit_time'].values <= t+10)]) for t in temp_df['digi_hit_time'].values if len(temp_df[(temp_df['digi_hit_time'].values >= t) &
                                                                                                                                  (temp_df['digi_hit_time'].values <= t+10)]) >= thresh]

        t0 = [np.min(temp_df[(temp_df['digi_hit_time'].values >= t) &
                    (temp_df['digi_hit_time'].values <= t+10)]['digi_hit_time']) for t in temp_df['digi_hit_time'].values if len(temp_df[(temp_df['digi_hit_time'].values >= t) &
                                                                                                                                (temp_df['digi_hit_time'].values <= t+10)]) >= thresh]

        # Append the list of clusters and t0s to the list of lists
        clusters.append(hits_in_cluster)
        times.append(t0)
        event.append(ev)

    # Create the output array
    hits_in_cluster_per_event = np.zeros(len(times), dtype=object)

    # Loop over all times. If the clusters are separated by less than 20 ns, the cluster with the more hits is the neutron candidate
    for t, i in tqdm(zip(times,range(len(times))), total=len(times), desc="20 ns between candidates", unit=" event", dynamic_ncols=False):
        if sep_20ns(t):
            hits_in_cluster_per_event[i] = [event[i], [clusters[i][0]], [times[i][0]]]
        else:
            hits_in_cluster_per_event[i] = [event[i], clusters[i], times[i]]

    # Count Valid Candidates
    zero_candidates     = 0
    unique_candidates   = 0
    multiple_candidates = 0

    for i in hits_in_cluster_per_event:
        if len(i[1]) == 0:
            zero_candidates+=1
        elif len(i[1]) == 1:
            unique_candidates+=1
        elif len(i[1]) > 1:
            multiple_candidates+=1
        else:
            print("You have a cluster (a list) with length different from 0, 1 or greater than 1, that's weird")

    if verbose:
        print(f'We are processing {len(event)} events.')
        print(f'We are finding a valid neutron candidate in {unique_candidates} events')
        print(f'This represent the {unique_candidates/len(event)*100:.2f}% of the total.\n')
        print(f'We have {zero_candidates} events in which we cannot find any candidate.')
        print(f'We have {multiple_candidates} events in which we have more than one valid candidate.')

nCandidateSearch(full_df, threshold)
