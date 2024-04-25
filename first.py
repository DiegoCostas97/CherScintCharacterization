import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt

import sys
import pickle

# Import Additional Tools
from tqdm                            import tqdm
from matplotlib.backends.backend_pdf import PdfPages

##################################################################################################
##################################################################################################
##################################################################################################
# First, we need to run some checks in order to know that our simulation is doing what we expect
# it to do.
# - Neutron Energy Spectrum Plot
# - Gamma Energy Spectrum Plot
# - Number of nCaptures in the simulation (expected 1 per event)
# - Number of DigiHits per event. Related to the trigger threshold
# - Number of events with DigiHits (i.e. meeting the threshold), also related to the trigger threshold

def neutronEnergySpectrum(track_df, path, plot=False, save=True):
    fig = plt.figure(figsize=(8,6))
    neutron_mass = 939.56542052
    df = pd.read_csv("/Users/diiego/Library/Mobile Documents/com~apple~CloudDocs/Desktop/DIEGO_cloud/USC/PHD/HK/HK SOURCES/code/ambe_source/npz_ana/copy_alnspectra_A.dat", 
                     sep=" ")

    energy = (track_df[track_df['track_id'] == 1]['track_energy'] - neutron_mass)
    counts, bins = np.histogram(energy, 50)

    counts
    normalized_counts = counts / np.max(counts) * np.max(df['CountsSum'].to_numpy())

    valuesQ0 = df['CountsQ0'].to_numpy()
    valuesQ0[valuesQ0==0] = np.nan
    valuesQ1 = df['CountsQ1'].to_numpy()
    valuesQ1[valuesQ1==0] = np.nan
    valuesQ2 = df['CountsQ2'].to_numpy()
    valuesQ2[valuesQ2==0] = np.nan

    plt.bar(bins[0:-1], normalized_counts, width=0.5, color='lightseagreen', label='MC Simulated Data');
    plt.plot(df['Energy'], valuesQ0, linestyle='-.', color='red', label='Ground State Contribution');
    plt.plot(df['Energy'], valuesQ1, linestyle='--', color='yellow', label='First Excited State Contribution');
    plt.plot(df['Energy'], valuesQ2, linestyle=':', color='blue', label='Second Excited State Contribution');

    plt.xlabel('Neutron Energy [MeV]');

    plt.legend();

    if plot:
        plt.show()

    if save:
        plt.savefig(path)

    plt.close

    return 0

def gammaEnergySpectrum(track_df, path, plot=False, save=True):
    fig = plt.figure(figsize=(8,6))
    plt.hist(track_df[track_df['track_id'] == 2]['track_energy'], bins=50);
    plt.xlabel("Tag Gamma Energy [MeV]");

    if plot:
        plt.show()

    if save:
        plt.savefig(path)

    plt.close

    return 0


def nCaptureNumber(track_df, nevents):
    data = len(track_df[(track_df['track_creator_process'].values == 'nCapture') & (track_df['track_pid'].values == 22)])
    print("In {} events the neutron is captured in the water, this represents a {:.2f}% of the total".format(data, data/float(nevents)*100))

def digiHitsNumber(digihit_df, path, threshold, lim, plot=False, save=True):
    fig = plt.figure(figsize=(8,6))
    plt.hist(digihit_df.groupby('event_id').count()['digi_hit_pmt'].values);
    plt.vlines(threshold,0,300, color='r');
    plt.xlim(0, lim);

    plt.xlabel("Number of DigiHits per Event")

    if plot:
        plt.show()

    if save:
        plt.savefig(path)

    plt.close

    return 0

def eventsWithDigihits(digihits_df, nevents):
    data = len(np.unique(digihits_df['event_id']))
    print("{} events meet the DigiHits threshold. This represent a {:.2f}% of the total".format(data, data/float(nevents)*100))

###################################################################################################
###################################################################################################
###################################################################################################
# Selection of the DigiHits for reconstruction
# This is done in two epochs:
# - Firts: We select the Scintillation Photons that come from the Tag Gamma-ray
# - Second: We select the Cherenkov Photons that come from the nCapture Gamma-ray
##################################################################################################
###################################################################################################
###################################################################################################
# Selection of the Scintillation Photons from the Tag Gamma-ray Functions

def electrons_from_Scintillation(df, nevents):

    # Take into account not only electrons may produce Scintillation, but for simplicity reasons we leave this like that.

    # Create a dictionary where the key is the event number and the info are the electrons that produced 
    # Scintillation light that produced hits in our PMTs.

    # Initialize the dictionary that's gonna record the info
    electrons_that_produce_scintillation = {}

    # Loop over all events
    for i in nevents:

        # If the event only has one track, append the track parent in a special way for the code not to break
        if len(df[df['event_id'].values == i]) == 1:
            electrons_that_produce_scintillation[i] = [df['true_hit_parent'][i]]

        # In every other scenario, append the track parents unique appearances
        else:
            electrons_that_produce_scintillation[i] = list(np.unique(df['true_hit_parent'][i].values))

    return electrons_that_produce_scintillation

def real_scintillation_electrons(df, e, nevents):

    # Create a dictionary where the key is the event number and the info are the electron which produced
    # Scintillation light (that produced hits) that comes from the tag gamma

    # Initialize the dictionary
    el = {}

    # Loop over the events in which we have scintillation
    for i in tqdm(nevents, desc="Running real_scintillation_electrons", unit="iter", dynamic_ncols=True):

        # Create a sub-DataFrame with the data of the current event.
        # This speeds up the process 
        temp_df = df[df['event_id'].values == i]

        # Fill the dictionary with the track ID of those electrons that come from track 2, 
        # which is the tag gamma. Again, creating a temp DF is faster.
        el_temp = temp_df[temp_df['track_id'].isin(e[i])]
        el[i]   = sorted(el_temp[el_temp['track_parent'].values == 2]['track_id'].values)

    # Finally, if the dictionary is empty for one of these events, drop that index
    re = {}
    for i in nevents:
        if len(el[i]) == 0:
            pass
        else:
            re[i] = el[i]

    return re

###################################################################################################
###################################################################################################
###################################################################################################
# Storing the important data in order to plot the Scintillation light main info
def scintillation_info(e, df):
    events_scint  = []
    counts_scint  = []
    indices_scint = []

    # Loop over the events in which we have electrons from tag gamma
    for i in tqdm(e.keys(), desc="Running scintillation_info", unit="iter", dynamic_ncols=True):
        count = 0

        # Create a sub-DataFrame with the info of this event
        temp_df = df[df['event_id'] == i]
        indices = temp_df['digi_hit_truehit_parent_trackID'].index
        values  = temp_df['digi_hit_truehit_parent_trackID']

        # Loop over the hit parent of the digihits in the event and the index in the DF
        for j,l in zip(values, indices):
            # Loop over the electrons that come from the tag gamma in this event
            for k in e[i]:
                # If we have DigiHits for the current event and the electron appears in the
                # DigiHits DataFrame, count and append the index
                if temp_df.notna().any()[1] and k in j:
                    count += 1
                    indices_scint.append(l)

        # If we are in a event in which some of the DigiHits are created by the tag gamma,
        # append the events and the counts
        if count != 0:
            events_scint.append(i)
            counts_scint.append(count)

    return events_scint, counts_scint, indices_scint

###################################################################################################
###################################################################################################
###################################################################################################
# Plot Light DigiHits
def plot_light(data, bins, gamma_type, light_type, events, threshold, sfm, path, plot=False, save=True):
        fig = plt.figure(figsize=(8,6))

        plt.hist(data, bins=bins, color='green', alpha=0.4)
        plt.xlabel('# of DigiHits produced by {} Gamma {} Hits'.format(gamma_type, light_type))
        plt.yscale('log')
        plt.title('TriggerNDigits/Threshold == {} \n /DAQ/SaveFailures/Mode {}'.format(threshold, sfm))

        x_limits = plt.gca().get_xlim()
        y_limits = plt.gca().get_ylim()

        plt.annotate("In {} events we have DigiHits produced by".format(len(events)), xy=(0.10, 0.95), xycoords='axes fraction')
        plt.annotate("the {} light from the {} Gamma".format(light_type, gamma_type), xy=(0.10, 0.90), xycoords='axes fraction')
        plt.annotate("Average number of DigiHits per event is {:.0f}".format(np.mean(data)), xy=(0.10, 0.80), xycoords='axes fraction')
        plt.annotate("Max number of DigiHits per event is {:.0f}".format(np.max(data)), xy=(0.10, 0.75), xycoords='axes fraction')

        if plot:
            plt.show()

        if save:
            plt.savefig(path)

        plt.close

        return 0

###################################################################################################
###################################################################################################
###################################################################################################
# Selection of the Cherenkov Photons from the nCapture Gamma-ray functions

def electrons_from_Cherenkov(df, nevents):

    # Create a dictionary where the key is the event number and the info are the electrons that produced 
    # Cherenkov light that produced hits in our PMTs.

    # Initialize the dictionary that's gonna record the info
    electrons_that_produce_cherenkov = {}

    # Loop over all events
    for i in nevents:

        # If the event only has one track, append the track parent in a special way for the code not to break
        if len(df[df['event_id'].values == i]) == 1:
            electrons_that_produce_cherenkov[i] = [df['true_hit_parent'][i]]

        # In every other scenario, append the track parents unique appearances
        else:
            electrons_that_produce_cherenkov[i] = list(np.unique(df['true_hit_parent'][i].values))

    return electrons_that_produce_cherenkov

def real_cherenkov_electrons(df, e, nevents):

    # Create a dictionary where the key is the event number and the info are the electron which produced
    # Cherenkov light (that produced hits) that comes from the nCapture

    # Initialize the dictionary
    el = {}

    # Loop over the events in which we have nCapture and Cherenkov
    for i in tqdm(nevents, desc="Running real_cherenkov_electrons", unit="iter", dynamic_ncols=True):
        # Create a sub-DataFrame with the data of the current event.
        # This incredibly speeds up the process 
        temp_df = df[df['event_id'].values == i]

        # Look for the track ID of the nCapture Gamma in this event
        temp_ncID_df = temp_df[(temp_df['track_pid'].values == 22)]
        ncID         = temp_ncID_df[(temp_ncID_df['track_creator_process'].values == "nCapture")]['track_id'].values[0]

        # Take only the electrons which have been produced by the nCapture gamma
        temp_df = temp_df[(temp_df['track_id'].isin(e[i]))]
        el[i] = sorted(temp_df[(temp_df['track_parent'].values == ncID)]['track_id'].values)

    re = {}
    for i in nevents:
        if len(el[i]) == 0:
            pass
        else:
            re[i] = el[i]

    return re

###################################################################################################
###################################################################################################
###################################################################################################
# Storing the important data in order to plot the Cherenkov light main info

def anyCherenkov_info(e, df):
    counts_anyCher = []
    events_anyCher = []

    for i in tqdm(e.keys(), desc="Running anyCherenkov info", unit="iter", dynamic_ncols=True):
        count = 0

        temp_df = df[df['event_id'].values == i]

        for j in temp_df['digi_hit_truehit_parent_trackID']:
            for k in e[i]:
                if temp_df.notna().any()[1] and k in j:
                    count += 1

        if count != 0:
            events_anyCher.append(i)
            counts_anyCher.append(count)

    return events_anyCher, counts_anyCher

def nCapture_Cherenkov_info(e, df):
    events_nCCher  = []
    counts_nCCher  = []
    indices_nCCher = []

    # Loop over the events in which we have electrons from the nCapture gamma
    for i in tqdm(e.keys(), desc="Running nCapture_Cherenkov_info", unit="iter", dynamic_ncols=True):
        count = 0
        # Create a sub-DataFrame of the current event
        temp_df = df[df['event_id'].values == i]

        # Loop over the hit parents that created the DigiHits and the index
        for j, l in zip(temp_df['digi_hit_truehit_parent_trackID'], temp_df['digi_hit_truehit_parent_trackID'].index):
            # Loop over the electrons created by the nCapture gamma in this event
            for k in e[i]:
                # If this event actually has DigiHits and the electron is in the DigiHits DF, append the index and count
                if temp_df.notna().any()[1] and k in j:
                    indices_nCCher.append(l)
                    count += 1

        # If we are in an event we want to append, do it and print info           
        if count != 0:
            events_nCCher.append(i)
            counts_nCCher.append(count)

    return events_nCCher, counts_nCCher, indices_nCCher

###################################################################################################
###################################################################################################
###################################################################################################

# DataFrame with the Cherenkov light from the events that we can actually tag
# We can do this with the df_final_cher, which is the DF of those events which we can actually tag
# Or we can do it simply with df_cher, which is the DF of every event with cher light from the nCapture gamma 
# (for the sake of more training data)
def output_reconstruction_variables(light_df, track_df):
    # Input
    charge   = []
    time     = []
    position = []
    x        = []
    y        = []
    z        = []

    # Output
    gamma_int_vertex = []
    gamma_int_time   = []
    gamma_cre_vertex = []
    neutr_int_vertex = []
    neutr_int_time   = []

    gamma_df = track_df[(track_df['track_creator_process'].values == 'nCapture') &
                            (track_df['track_pid'].values == 22)]

    for i in tqdm(np.unique(light_df['event_id']), desc="Outputting Reconstruction Variables", unit="iter", dynamic_ncols=True):
        # Input
        temp_df = light_df[light_df['event_id'].values == i]

        charge.append(list(temp_df['digi_hit_charge']))
        time.append(list(temp_df['digi_hit_time'].values))
        position.append(list(temp_df['digi_hit_r'].values))
        x.append(list(temp_df['digi_hit_x'].values))
        y.append(list(temp_df['digi_hit_y'].values))
        z.append(list(temp_df['digi_hit_z'].values))

        # Output
        temp_df_gamma = gamma_df[(gamma_df['event_id'].values == i)]

        gamma_int_vertex.append(temp_df_gamma['track_rf'].values[0])
        gamma_int_time.append(temp_df_gamma['track_ti'].values[0]) # Please note this is not correct, but WCSim does not return a stop time for the tracks
        gamma_cre_vertex.append(temp_df_gamma['track_ri'].values[0])

        temp_df_neutr = track_df[track_df['event_id'].values == i]

        neutr_int_vertex.append(temp_df_neutr[(temp_df_neutr['track_id'].values == 1)]['track_rf'].values[0])
        neutr_int_time.append(temp_df_gamma['track_ti'].values[0])

    return charge, time, position, x, y, z, gamma_int_vertex, gamma_int_time, gamma_cre_vertex, neutr_int_vertex, neutr_int_time

def output_background_variables(light_df):
    # Input
    charge   = []
    time     = []
    position = []
    x        = []
    y        = []
    z        = []

    for i in tqdm(np.unique(light_df['event_id']), desc="Outputting Background Variables", unit="iter", dynamic_ncols=True):
        # Input
        temp_df = light_df[light_df['event_id'].values == i]

        charge.append(list(temp_df['digi_hit_charge']))
        time.append(list(temp_df['digi_hit_time'].values))
        position.append(list(temp_df['digi_hit_r'].values))
        x.append(list(temp_df['digi_hit_x'].values))
        y.append(list(temp_df['digi_hit_y'].values))
        z.append(list(temp_df['digi_hit_z'].values))

    return charge, time, position, x, y, z

###################################################################################################
###################################################################################################
###################################################################################################
# Trigger Times Check

def writeTriggerTimesPDF(nevents, df, path):
    print("Plotting and saving plots at {}".format(path))
    with PdfPages('./trigger_times.pdf') as pdf:
        for j in tqdm(range(nevents), desc="Writing Trigger Times PDF", unit="iter", dynamic_ncols=True):
            event    = j
            triggers = np.unique(df[(df['event_id'].values == event)]['digi_hit_trigger'])

            dhS = []
            thS = []
            for i in triggers:
                dhS.append(df[(df['event_id'].values == event) &
                                    (df['digi_hit_trigger'].values == i)]['digi_hit_time'].values)
                thS.append([i[0] for i in df[(df['event_id'].values == event) &
                                    (df['digi_hit_trigger'].values == i)]['digi_hit_truehit_times'].values])

            trigger_list = ["Trigger_"+str(i) for i in triggers]

            fig = plt.figure(figsize=(8,6))

            ax1 = fig.add_subplot(211)

            ax1.hist(dhS, alpha=0.7, label=trigger_list);
            ax1.set_xlabel('DigiHit Time [ns]')
            ax1.set_title('Event {}'.format(event));
            ax1.set_yscale('log')
            ax1.legend();

            ax2 = fig.add_subplot(212)
            ax2.hist(thS, alpha=0.7, label=trigger_list);
            ax2.set_xlabel('TrueHit Time [ns]')
            ax2.set_yscale('log')
            ax2.legend();

            pdf.savefig(fig)
            plt.close()

    return 0
