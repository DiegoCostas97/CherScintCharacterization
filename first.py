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
    plt.plot(df['Energy'].to_numpy(), valuesQ0, linestyle='-.', color='red', label='Ground State Contribution');
    plt.plot(df['Energy'].to_numpy(), valuesQ1, linestyle='--', color='yellow', label='First Excited State Contribution');
    plt.plot(df['Energy'].to_numpy(), valuesQ2, linestyle=':', color='blue', label='Second Excited State Contribution');

    plt.xlabel('Neutron Energy [MeV]');

    plt.legend();

    if plot:
        plt.show()

    if save:
        plt.savefig(path)

    plt.close

    return

def gammaEnergySpectrum(track_df, path, plot=False, save=True):
    fig = plt.figure(figsize=(8,6))
    plt.hist(track_df[track_df['track_id'] == 2]['track_energy'], bins=50);
    plt.xlabel("Tag Gamma Energy [MeV]");

    if plot:
        plt.show()

    if save:
        plt.savefig(path)

    plt.close

    return


def nCaptureNumber(track_df, nevents):
    data = len(track_df[(track_df['track_creator_process'].values == 'nCapture') & (track_df['track_pid'].values == 22)])
    print("In {} events the neutron is captured in the water, this represents a {:.2f}% of the total".format(data, data/float(nevents)*100))

def nCaptureInEveryIsotope(track_df, nevents):
    deuteron = np.unique(track_df[(track_df['track_pid'].values == 1000010020)]['event_id'])
    gd156    = np.unique(track_df[(track_df['track_pid'].values == 1000641560)]['event_id'])
    gd158    = np.unique(track_df[(track_df['track_pid'].values == 1000641580)]['event_id'])

    print(f"In {len(deuteron)} events the neutron is captured by a Hidrogen nucleus, this represents a {len(deuteron)/nevents*100:.1f}%")
    print(f"In {len(gd156)} events the neutron is captured by a 156Gd nucleus, this represents a {len(gd156)/nevents*100:.1f}%")
    print(f"In {len(gd158)} events the neutron is captured by a 158Gd nucleus, this represents a {len(gd158)/nevents*100:.1f}%")
    print(f"Neutron is captured in the {(len(deuteron)+len(gd156)+len(gd158))/nevents*100}% of the events\n")
    print(f"The rest of the events, in which we see other different isotopes, are not nCapture events, those isotopes are produced in scattering processes")

def digiHitsNumber(digihit_df, path, threshold, plot=False, save=True):
    fig = plt.figure(figsize=(8,6))
    plt.hist(digihit_df.groupby('event_id').count()['digi_hit_pmt'].values);
    plt.vlines(threshold,0,300, color='r');

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
        creators = temp_df['digi_hit_truehit_creator']

        # Loop over the hit parent of the digihits in the event and the index in the DF
        for j,l,c in zip(values, indices, creators):
            # Loop over the electrons that come from the tag gamma in this event
            for k in e[i]:
                # If we have DigiHits for the current event and the electron appears in the
                # DigiHits DataFrame, count and append the index
                if k in j and 'Scintillation' in c:
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
def plot_light(data, bins, gamma_type, light_type, events, threshold, sfm, path, xlabel, color='green', plot=False, save=True, logY=True, logX=False, title=True, different_label=False, legend='right'):
        fig = plt.figure(figsize=(8,6))

        plt.hist(data, bins=bins, color=color, alpha=0.4)
        plt.xlabel('# of DigiHits produced by {} Gamma {} Hits'.format(gamma_type, light_type), fontsize=15)

        if different_label:
            plt.xlabel(xlabel)
        if logY:
            plt.yscale('log')
        if logX:
            plt.xscale('log')

        plt.tick_params(axis='both', which='major', labelsize=15)

        if title:
            plt.title('TriggerNDigits/Threshold == {} \n /DAQ/SaveFailures/Mode {}'.format(threshold, sfm))

        x_limits = plt.gca().get_xlim()
        y_limits = plt.gca().get_ylim()

        if legend == 'left':
            plt.annotate("n = {} events".format(len(events)), xy=(0.05, 0.95), xycoords="axes fraction", fontsize=15)
            plt.annotate("$\mu$ = {:.0f} DigiHits ".format(np.mean(data)), xy=(0.05, 0.90), xycoords="axes fraction", fontsize=15)
            plt.annotate("max = {:.0f} DigiHits".format(np.max(data)), xy=(0.05, 0.85), xycoords="axes fraction", fontsize=15)
        elif legend == 'right':
            plt.annotate("n = {} events".format(len(events)), xy=(0.65, 0.95), xycoords="axes fraction", fontsize=15)
            plt.annotate("$\mu$ = {:.0f} DigiHits ".format(np.mean(data)), xy=(0.65, 0.90), xycoords="axes fraction", fontsize=15)
            plt.annotate("max = {:.0f} DigiHits".format(np.max(data)), xy=(0.65, 0.85), xycoords="axes fraction", fontsize=15)

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
                if k in j:
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
        indices = temp_df['digi_hit_truehit_parent_trackID'].index
        values  = temp_df['digi_hit_truehit_parent_trackID']
        creators = temp_df['digi_hit_truehit_creator']

        # Loop over the hit parents that created the DigiHits and the index
        for j, l, c in zip(values, indices, creators):
            # Loop over the electrons created by the nCapture gamma in this event
            for k in e[i]:
                # If this event actually has DigiHits and the electron is in the DigiHits DF, append the index and count
                if k in j and 'Cerenkov' in c:
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

###################################################################################################
###################################################################################################
###################################################################################################

# SAVE DATAFRAME FOR NEUTRON CANDIDATE SEARCH
# ################### FIRST EPOCH ##########################################################################
# Creating a new DigiHits DataFrame just with the Tag and nCapture light
def save_data_for_nc_search(indices_cherenkov, indices_scintillation, df_digi, path):
    df_cher  = df_digi.loc[indices_cherenkov]
    df_scint = df_digi.loc[indices_scintillation]

    cher_events  = np.unique(df_cher['event_id'])
    scint_events = np.unique(df_scint['event_id'])

    common_evts  = [i for i in np.unique(df_digi['event_id']) if i in cher_events and i in scint_events]

    print("There are {} events with both Tag and nCapture light".format(len(common_evts)))
    print(" ")

    common_events_cher_indices  = df_cher[df_cher['event_id'].isin(common_evts)].index.to_numpy()
    common_events_scint_indices = df_scint[df_scint['event_id'].isin(common_evts)].index.to_numpy()

    common_events_indices = np.sort(np.concatenate([common_events_cher_indices, common_events_scint_indices]))

    # This are the final DataFrames that we will use:
    # - df_final has the info of all Scintillation and Cherenkov Light that comes from tagGamma and nCapture
    # - df_final_cher only has the info of the Cherenkov Light produced by nCapture
    # - df_final_scint only has the info of the Scintillation Light produced by the tagGamma
    df_final       = df_digi.loc[common_events_indices]
    df_final_cher  = df_cher[df_cher['event_id'].isin(common_evts)]
    df_final_scint = df_scint[df_scint['event_id'].isin(common_evts)]

    ################### SECOND EPOCH ##########################################################################
    # What is the time of the last Scintllation DigiHit?
    last_tag_hit_time = df_final_scint.groupby('event_id')['digi_hit_time'].max()
    repeated_last_tag_hit_time = df_final_cher['event_id'].map(last_tag_hit_time)

    # Add the column "last_tag_hit_time" to df_final_cher
    df_final_cher.loc[:, 'last_tag_hit_time'] = repeated_last_tag_hit_time

    ################### THIRD EPOCH ###########################################################################
    # Filter the DataFrame so now we just have the Cherenkov Light in the selected window
    filter_df_final_cher = df_final_cher.loc[(df_final_cher['digi_hit_time'] >= df_final_cher['last_tag_hit_time']) & (df_final_cher['digi_hit_time'] < df_final_cher['last_tag_hit_time']+250000)]

    ################### FOURTH EPOCH ##########################################################################
    # Now we need to add the DarkNoise to our DataFrame since it represents the background we want the algorithm to discard
    indices = []

    for i in tqdm(np.unique(filter_df_final_cher['event_id']), desc="Creating DataFrame for Neutron Candidate Algorithm"):
        temp_df = df_digi[df_digi['event_id'] == i]
        tag_t   = filter_df_final_cher[filter_df_final_cher['event_id'].values == i]['last_tag_hit_time'].values[0]

        for p, ind, cher_t in zip(temp_df['digi_hit_truehit_parent_trackID'], temp_df['digi_hit_truehit_parent_trackID'].index, temp_df['digi_hit_time']):
            if -1 in p and cher_t >= tag_t and cher_t < tag_t + 250000:
                indices.append(ind)

    bkg_df = df_digi.loc[indices]

    # Output Reconstruction Variables
    # This is not necessary anymore (for the moment)
    # This allows to save the variables of the CherenkovLight and DarkNoise so we can use them in a NN or something 
    # charge, time, position, x, y, z = output_background_variables(bkg_df)

    # Save the variables in a file
    # path = "./background_variables.pkl"
    # print("Saving Background Varibales at                         {}".format(path))

    # with open(path, 'wb') as file:
    #     pickle.dump([charge, time, position, x, y, z], file);

    ################### FIFTH EPOCH ###########################################################################
    # There are events with no DarkNoise which we do not want
    events_with_no_dr           = [i for i in np.unique(filter_df_final_cher['event_id']) if i not in np.unique(bkg_df['event_id'])]
    filter_df_final_cher_events = np.unique(filter_df_final_cher['event_id'])

    final_events = [i for i in filter_df_final_cher_events if i not in events_with_no_dr]

    ################### SIXTH EPOCH ###########################################################################
    # Same as before, this is only needed within a possible and future NN/ML analysis
    # Output Reconstruction Variables
    # final_data = filter_df_final_cher[filter_df_final_cher['event_id'].isin(final_events)]
    # charge, time, position, x, y, z, gamma_int_vertex, gamma_int_time, gamma_cre_vertex, neutr_int_vertex, neutr_int_time = output_reconstruction_variables(final_data, df_simple_track)

    # # Save the variables in a file
    # path = "./reconstruction_variables.pkl"
    # print("Saving Reconstruction Varibales at                     {}".format(path))

    # with open(path, 'wb') as file:
        # pickle.dump([charge, time, position, x, y, z, gamma_int_vertex, gamma_int_time, gamma_cre_vertex, neutr_int_vertex, neutr_int_time], file);

    ################### SEVENTH EPOCH ###########################################################################
    merged_df = pd.concat([filter_df_final_cher, bkg_df])
    # WE ARE JUST STORING THE FIRST CREATOR PROCESS OF THE SMEARED DIGIHIT FOR EASIER ANALYSIS IN THE NEUTRON CANDIDATE SECTION
    # PLEASE NOTE THIS IS NOT AS BAD AS IT SEEMS SINCE THE HOLE ANALYSIS IS BEING MADE WITH THE ASUMPTION THAT EVERY DIGIHIT IS CREATED BY JUST ONE HIT
    # MAYBE I SHOULD QUANTIFY THIS (WHAT % OF THE DIGIHITS ARE CREATED BY MORE THAN ONE HIT)
    merged_df['digi_hit_truehit_creator'] = [i[0] for i in merged_df['digi_hit_truehit_creator']]
    merged_df.sort_values(by='event_id', inplace=True)
    merged_df.to_csv(path, index=False)

    print("Saving Data for Neutron Candidate Search at {}".format(path))


def scintillation_profile(e, df):
    events_scint  = []
    counts_scint  = []
    indices_scint = []

    # Loop over the events in which we have electrons from tag gamma
    for i in tqdm(e.keys(), desc="Running scintillation_profile", unit="iter", dynamic_ncols=True):
        count = 0

        # Create a sub-DataFrame with the info of this event
        df.reset_index(drop=True, inplace=True) # Need to properly set the tracks df index, just in this case 
                                                     # cause now I don't remember if this would affect any other part
        temp_df = df[df['event_id'].values == i]
        indices = temp_df['track_parent'].index
        values  = temp_df['track_parent'].to_numpy()
        creators = temp_df['track_creator_process']

        # Loop over the hit parent of the digihits in the event and the index in the DF
        # Loop over the electrons that come from the tag gamma in this event
        for v,l,c in zip(values, indices, creators):
            for k in e[i]:
            # If we have DigiHits for the current event and the electron appears in the
            # DigiHits DataFrame, count and append the index
            # if temp_df.notna().any()[1] and k in j: NO ES NECESARIO (CASI SEGURO)
                if k == v and 'Scintillation' in c:
                    count += 1
                    indices_scint.append(l)

        # If we are in a event in which some of the DigiHits are created by the tag gamma,
        # append the events and the counts
        if count != 0:
            events_scint.append(i)
            counts_scint.append(count)

    return events_scint, indices_scint

def cherenkov_profile(e, df):
    events_nCCher  = []
    counts_nCCher  = []
    indices_nCCher = []

    # Loop over the events in which we have electrons from the nCapture gamma
    for i in tqdm(e.keys(), desc="Running cherenkov_profile", unit="iter", dynamic_ncols=True):
        count = 0
        # Create a sub-DataFrame of the current event
        df.reset_index(drop=True, inplace=True) # Need to properly set the tracks df index, just in this case
                                                     # cause now I don't remember if this would affect any other part
        temp_df = df[df['event_id'].values == i]
        indices = temp_df['track_parent'].index
        values  = temp_df['track_parent'].to_numpy()
        creators = temp_df['track_creator_process']

        # Loop over the hit parents that created the DigiHits and the index
        for j, l, c in zip(values, indices, creators):
            # Loop over the electrons created by the nCapture gamma in this event
            for k in e[i]:
                # If this event actually has DigiHits and the electron is in the DigiHits DF, append the index and count
                # if temp_df.notna().any()[1] and k in j: NO ES NECESARIO (CASI SEGURO)
                if k == j and 'Cerenkov' in c:
                    indices_nCCher.append(l)
                    count += 1

        # If we are in an event we want to append, do it and print info
        if count != 0:
            events_nCCher.append(i)
            counts_nCCher.append(count)

    return events_nCCher, indices_nCCher
