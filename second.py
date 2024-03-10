import sys
import pickle

import numpy as np

path_to_first    = "/Users/diiego/Library/Mobile Documents/com~apple~CloudDocs/Desktop/DIEGO_cloud/USC/PHD/HK/HK SOURCES/code/ambe_source/npz_ana/cher_scint_characterization"
path_to_paquetes = "/Users/diiego/Library/Mobile Documents/com~apple~CloudDocs/Desktop/DIEGO_cloud/USC/PHD/HK/HK SOURCES/code/ambe_source/npz_ana/paquetes"
sys.path.append(path_to_paquetes)
sys.path.append(path_to_first)

from npz_to_df import truehits_info_to_df
from npz_to_df import simple_track_info_to_df
from npz_to_df import digihits_info_to_df

from first import neutronEnergySpectrum
from first import gammaEnergySpectrum
from first import nCaptureNumber
from first import digiHitsNumber
from first import eventsWithDigihits
from first import electrons_from_Scintillation
from first import real_scintillation_electrons
from first import scintillation_info
from first import electrons_from_Cherenkov
from first import real_cherenkov_electrons
from first import anyCherenkov_info
from first import nCapture_Cherenkov_info
from first import plot_light
from first import output_reconstruction_variables
from first import writeTriggerTimesPDF


# Path to the npz file in our machine
npz = str(sys.argv[1])

threshold = int(sys.argv[2])   #/DAQ/TriggerNDigits/Threshold set in macros/daq.mac. Needed for the plots.
sfm       = int(sys.argv[3])   #/DAQ/TriggerSaveFailures/Mode set in macros/daq.mac. Needed for the plots.
nevents   = int(sys.argv[4])   # Number of events in the .npz

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
nCaptureNumber(df_simple_track, nevents)
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
print("Average number of DigiHits per event is {:.0f}".format(np.mean(counts_anyCher)))
print(" ")

events_nCCher, counts_nCCher, indices_nCCher = nCapture_Cherenkov_info(electrons_from_nCapture, df_digiHits)

print("In {} events we have DigiHits produced by the Cherenkov light from the nCapture gamma".format(len(events_nCCher)))
print("Maximum number of DigiHits per event is {}".format(np.max(counts_nCCher)))
print("Average number of DigiHits per event is {:.0f}".format(np.mean(counts_nCCher)))

# Plot
plot_light(counts_scint, 50, "Tag", "Scintillation", events_scint, threshold, sfm, "./scint_light.pdf", plot=False, save=True)
plot_light(counts_nCCher, 30, "nCapture", "Cherenkov", events_nCCher, threshold, sfm, "./cher_light.pdf", plot=False, save=True)
print(" ")

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

df_final       = df_digiHits.loc[final_indices]
df_final_cher  = df_cher[df_cher['event_id'].isin(common_evts)]
df_final_scint = df_scint[df_scint['event_id'].isin(common_evts)]

# Output Reconstruction Variables
charge, time, position, x, y, z, gamma_int_vertex, gamma_int_time, gamma_cre_vertex, neutr_int_vertex, neutr_int_time = output_reconstruction_variables(df_cher, df_simple_track)

# Save the variables in a file
path = "./reconstruction_variables.pkl"
print("Saving Reconstruction Varibales at {}".format(path))
with open(path, 'wb') as file:
    pickle.dump([charge, time, position, x, y, z, gamma_int_vertex, gamma_int_time, gamma_cre_vertex, neutr_int_vertex, neutr_int_time], file)

# Just write that many events as you want to inspect, default is 5
inspected_files = 5
writeTriggerTimesPDF(inspected_files, df_digiHits, "./trigger_times.pdf")
