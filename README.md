# CherScintCharacterization

The code performs an early Cherenkov and Scintillation light analysis for a WCSim simulation output file.
Note that this code processes an `.npz` file produced by the [WatChMal DataTools](https://github.com/WatChMaL/DataTools).

## Overall Information
- first.py
This file has all the needed functions for the analysis.

- second.py
Runs the functions inside first.py and performs the actual analysis. This will return
    - Checks
        - Neutron energy distribution plot.
        - Gamma energy distribution plot.
        - Number of events in which the neutron actually gets captured.
        - Trigger times for DigiHits and TrueHits.
    - Analysis
        - Scintillation light coming from the Tag Gamma-Ray plot + info.
        - Cherenkov light coming from the nCapture Gamma-Ray plot + info.
        - Reconstruction variables (inputs + possible targets) stored in file.

Please note that this relies on a previous set of functions, contained inside the [WCSimFilePackages](https://github.com/DiegoCostas97/WCSimFilePackages) repository.

## How to run the code
In order to run the analysis, just 
```
python second.py int(/DAQ/TriggerNDigits/Threshold) int(/DAQ/TriggerSaveFailures/Mode) int(NumberOfEvents)
```

- `/DAQ/TriggerNDigits/Threshold` is the one you set in your `daq.mac` file.
- `/DAQ/TriggerSaveFailures/Mode` is the one you set in your `daq.mac` file.
- `NumberOfEvents` is the number of events processed by `event_dump.py`.

Please note that you'll need to modify the path for the "paquetes" library (i.e. [WCSimFilePackages](https://github.com/DiegoCostas97/WCSimFilePackages))
as well as the path for `first.py`.

## Required libraries
```
numpy
matplotlib
sys
pickle
pandas
tqdm 
PdfPages
```
