import sys
import pickle

import numpy as np
import pandas as pd

path_to_first    = "/mnt/netapp2/Store_uni/home/usc/ie/dcr/software/hk/CherScintCharacterization"
path_to_paquetes = "/mnt/netapp2/Store_uni/home/usc/ie/dcr/software/hk/WCSimFilePackages"
sys.path.append(path_to_paquetes)
sys.path.append(path_to_first)

from npz_to_df import truehits_info_to_df
from npz_to_df import simple_track_info_to_df
from npz_to_df import digihits_info_to_df

from first import *

from tqdm import tqdm


# Path to the npz file in our machine
npz = str(sys.argv[1])

threshold = int(sys.argv[2])   #/DAQ/TriggerNDigits/Threshold set in macros/daq.mac. Needed for the plots.
sfm       = int(sys.argv[3])   #/DAQ/TriggerSaveFailures/Mode set in macros/daq.mac. Needed for the plots.
nevents   = int(sys.argv[4])   # Number of events in the .npz
neutron_candidate_data_fileName = str(sys.argv[5])
verbose   = False

# Creation of the three main DataFrames: trueHits, digiHits and tracks
print("Creating DataFrames...")
df_trueHits       = truehits_info_to_df(npz)
df_digiHits       = digihits_info_to_df(npz, nevents)
df_simple_track   = simple_track_info_to_df(npz) # This is simpler and less memory consuming version
                                                 # of the tracks_df. Please note there is a more complex
                                                 # version in "paquetes" under the name of track_info_to_df
print("DataFrames Created!")
print(" ")

# Simulation Check Plots and prints
print("Running Simulation Checks...")
neutronEnergySpectrum(df_simple_track, "./neutronEnergySpectrum.png")
gammaEnergySpectrum(df_simple_track, "./gammaEnergySpectrum.png")
nCaptureInEveryIsotope(df_simple_track, nevents)
digiHitsNumber(df_digiHits, "./digiHitsNumber.png", threshold)
eventsWithDigihits(df_digiHits, nevents)
print(" ")

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

if verbose:
    print("In {} events we have DigiHits produced by the Scintillation light from the Tag gamma".format(len(events_scint)))
    print("Maximum number of DigiHits per event is {}".format(np.max(counts_scint)))
    print("Average number of DigiHits per event is {:.0f}".format(np.mean(counts_scint)))

# Selection of every Cherenkov Photon
events_with_nCapture = np.unique(df_simple_track[df_simple_track['track_creator_process'] == 'nCapture']['event_id'].to_numpy())
temp_cHits_nCapture  = df_trueHits[df_trueHits['event_id'].isin(events_with_nCapture)]
cHits_nCapture       = temp_cHits_nCapture[temp_cHits_nCapture['true_hit_creatorProcess'].values == 'Cerenkov']

# Select only those events in which we have Cherenkov so we don't loop over the hole dataset
cher_nevents = cHits_nCapture.groupby('event_id').count().index.values

# Call the functions and store the results in the main variable electrons_from_nCapture
electrons_that_produce_cherenkov = electrons_from_Cherenkov(cHits_nCapture, cher_nevents)
electrons_from_nCapture          = real_cherenkov_electrons(df_simple_track, electrons_that_produce_cherenkov, cher_nevents)


events_nCCher, counts_nCCher, indices_nCCher = nCapture_Cherenkov_info(electrons_from_nCapture, df_digiHits)

if verbose:
    print("In {} events we have DigiHits produced by the Cherenkov light from the nCapture gamma".format(len(events_nCCher)))
    print("Maximum number of DigiHits per event is {}".format(np.max(counts_nCCher)))
    print("Average number of DigiHits per event is {:.0f}".format(np.mean(counts_nCCher)))

plot_light(counts_scint, 50, "Tag", "Scintillation", events_scint, threshold, sfm, "./scint_light.pdf", xlabel="", plot=False, save=True, logY=True, title=True, different_label=False);
plot_light(counts_nCCher, 20, "nCapture", "Cherenkov", events_nCCher, threshold, sfm, "./cher_light.pdf", xlabel="", color='purple', plot=False, save=True, logY=True, title=True, different_label=False);

print("Starting Neutron Tagging Data Preparation...")
save_data_for_nc_search(indices_nCCher, indices_scint, df_digiHits, neutron_candidate_data_fileName)
print(" ")
print("Process Finished!")

# Energy and Time distribution plots
# scintillation_profile_events, scintillation_profile_indices = scintillation_profile(electrons_from_tagGamma, df_simple_track)
# cherenkov_profile_events, cherenkov_profile_indices = cherenkov_profile(electrons_from_nCapture, df_simple_track)

# dfS = df_simple_track.loc[scintillation_profile_indices]
# dfC = df_simple_track.loc[cherenkov_profile_indices]

# plot_light(dfS['track_energy']*1e6, 10, "nCapture", "Cherenkov", scintillation_profile_events, threshold, sfm, "./cher_light.pdf", xlabel="Scintillation Light From tagGamma E Distribution [eV]", plot=True, save=False, logY=False, title=False, different_label=True, legend='left');
# plot_light(dfC['track_energy']*1e6, 10, "nCapture", "Cherenkov", cherenkov_profile_events, threshold, sfm, "./cher_light.pdf", xlabel="Cherenkov Light From nCapture E Distribution [eV]", color='purple', plot=True, save=False, logY=False, title=False, different_label=True, legend='left');

# plot_light(dfS['track_ti'], 10, "nCapture", "Cherenkov", scintillation_profile_events, threshold, sfm, "./cher_light.pdf", xlabel="Scintillation Light From tagGamma T [ns]", plot=True, save=False, logY=True, logX=True, title=False, different_label=True, legend='right');
# plot_light(dfC['track_ti'], 10, "nCapture", "Cherenkov", cherenkov_profile_events, threshold, sfm, "./cher_light.pdf", xlabel="Cherenkov Light From nCapture T [ns]", color='purple', plot=True, save=False, logY=True, logX=True, title=False, different_label=True, legend='right');
