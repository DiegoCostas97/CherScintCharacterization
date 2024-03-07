import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt

import sys
import pickle

path_to_paquetes = "/Users/diiego/Library/Mobile Documents/com~apple~CloudDocs/Desktop/DIEGO_cloud/USC/PHD/HK/HK SOURCES/code/ambe_source/npz_ana/paquetes"
sys.path.append(path_to_paquetes)

# Import Additional Tools
from npz_to_df                       import truehits_info_to_df
from npz_to_df                       import simple_track_info_to_df
from npz_to_df                       import digihits_info_to_df
from tqdm                            import tqdm
from matplotlib.backends.backend_pdf import PdfPages

##################################################################################################
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
        fig = plt.figure(figsize=(7,7))

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
                # If the electron is in the DigiHits DF, append the index and count
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

    # Output
    gamma_int_vertex = []
    gamma_cre_vertex = []
    neutr_int_vertex = []

    gamma_df = track_df[(track_df['track_creator_process'].values == 'nCapture') & 
                            (track_df['track_pid'].values == 22)]

    for i in tqdm(np.unique(light_df['event_id']), desc="Outputting Reconstruction Variables", unit="iter", dynamic_ncols=True):
        # Input
        temp_df = light_df[light_df['event_id'].values == i]
        
        charge.append(list(temp_df['digi_hit_charge']))
        time.append(list(temp_df['digi_hit_time'].values))
        position.append(list(temp_df['digi_hit_r'].values))
    
        # Output
        temp_df_gamma = gamma_df[(gamma_df['event_id'].values == i)]
        
        gamma_int_vertex.append(temp_df_gamma['track_rf'].values[0])
        gamma_cre_vertex.append(temp_df_gamma['track_ri'].values[0])
        
        temp_df_neutr = track_df[track_df['event_id'].values == i]
        
        neutr_int_vertex.append(temp_df_neutr[(temp_df_neutr['track_id'].values == 1)]['track_rf'].values[0])

    return charge, time, position, gamma_int_vertex, gamma_cre_vertex, neutr_int_vertex

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
            
            fig = plt.figure(figsize=(7,7))

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

""" # Path to the npz file in our machine
npz = '/Users/diiego/Desktop/install/wcsim.npz'

threshold = 2   #/DAQ/TriggerNDigits/Threshold set in macros/daq.mac. Needed for the plots.
sfm       = 0   #/DAQ/TriggerSaveFailures/Mode set in macros/daq.mac. Needed for the plots.
nevents   = 100 # Number of events in the .npz

# Creation of the three main DataFrames: trueHits, digiHits and tracks
df_trueHits       = truehits_info_to_df(npz)
df_digiHits       = digihits_info_to_df(npz, nevents)
df_simple_track   = simple_track_info_to_df(npz) # This is simpler and less memory consuming version
                                                 # of the tracks_df. Please note there is a more complex
                                                 # version in "paquetes" under the name of track_info_to_df

# Selection of every Scintillation Photon
events_with_Scintillation = np.unique(df_simple_track[df_simple_track['track_creator_process'] == 'Scintillation']['event_id'].to_numpy())
sHits_Tag_temp            = df_trueHits[(df_trueHits['event_id'].isin(events_with_Scintillation))]
sHits_Tag                 = df_trueHits[(df_trueHits['true_hit_creatorProcess'].values == 'Scintillation')]

# Select only those events in which we have Scintillation so we don't loop over the hole dataset
scint_nevents = list(sHits_Tag.groupby('event_id').count().index.values)

# Call the functions and store the results in the main variable electrons_from_tagGamma
electrons_that_produce_scintillation = electrons_from_Scintillation(sHits_Tag, scint_nevents)
electrons_from_tagGamma              = real_scintillation_electrons(df_simple_track, electrons_that_produce_scintillation, scint_nevents)
print(" ")


events_scint, counts_scint, indices_scint = scintillation_info(electrons_from_tagGamma, df_digiHits)
print(" ")

print("In {} events we have DigiHits produced by the Scintillation light from the Tag gamma".format(len(events_scint)))
print("Maximum number of DigiHits per event is {}".format(np.max(counts_scint)))
print("Average number of DigiHits per event is {:.0f}".format(np.mean(counts_scint)))
print(" ")

# Selection of every Cherenkov Photon
events_with_nCapture = np.unique(df_simple_track[df_simple_track['track_creator_process'] == 'nCapture']['event_id'].to_numpy())
temp_cHits_nCapture = df_trueHits[df_trueHits['event_id'].isin(events_with_nCapture)]
cHits_nCapture      = temp_cHits_nCapture[temp_cHits_nCapture['true_hit_creatorProcess'].values == 'Cerenkov']

# Select only those events in which we have Cherenkov so we don't loop over the hole dataset
cher_nevents = cHits_nCapture.groupby('event_id').count().index.values

# Call the functions and store the results in the main variable electrons_from_nCapture
electrons_that_produce_cherenkov = electrons_from_Cherenkov(cHits_nCapture, cher_nevents)
electrons_from_nCapture          = real_cherenkov_electrons(df_simple_track, electrons_that_produce_cherenkov, cher_nevents)
print(" ")

events_anyCher, counts_anyCher = anyCherenkov_info(electrons_that_produce_cherenkov, df_digiHits)
print(" ")
print("In {} events we have DigiHits produced by the Cherenkov light from any gamma".format(len(events_anyCher)))
print("Maximum number of DigiHits per event is {}".format(np.max(counts_anyCher)))
print(" ")

events_nCCher, counts_nCCher, indices_nCCher = nCapture_Cherenkov_info(electrons_from_nCapture, df_digiHits)

print("In {} events we have DigiHits produced by the Cherenkov light from the nCapture gamma".format(len(events_nCCher)))
print("Maximum number of DigiHits per event is {}".format(np.max(counts_nCCher)))

# Plot
plot_light(counts_scint, "Tag", "Scintillation", events_scint)
plot_light(counts_nCCher, "nCapture", "Cherenkov", events_nCCher)

# Creating a new DigiHits DataFrame just with the Tag and nCapture light
df_cher  = df_digiHits.loc[indices_nCCher]
df_scint = df_digiHits.loc[indices_scint]

cher_events  = np.unique(df_cher['event_id'])
scint_events = np.unique(df_scint['event_id'])

common_evts = [i for i in np.unique(df_digiHits['event_id']) if i in cher_events and i in scint_events]
print("There are {} events with both Tag and nCapture light".format(len(common_evts)))
print(" ")
cher_indices  = df_cher[df_cher['event_id'].isin(common_evts)].index.to_numpy()
scint_indices = df_scint[df_scint['event_id'].isin(common_evts)].index.to_numpy()

final_indices = np.sort(np.concatenate([cher_indices, scint_indices]))

df_final = df_digiHits.loc[final_indices]
df_final_cher = df_cher[df_cher['event_id'].isin(common_evts)]

# Output Reconstruction Variables
charge, time, position, gamma_int_vertex, gamma_cre_vertex, neutr_int_vertex = output_reconstruction_variables(df_cher, df_simple_track)

# Save the variables in a file
with open('./reconstruction_variables.pkl', 'wb') as file:
    pickle.dump([charge, time, position, gamma_int_vertex, gamma_cre_vertex, neutr_int_vertex], file)

# Just write that many events as you want to inspect, default is 30
writeTriggerTimesPDF(30) """
