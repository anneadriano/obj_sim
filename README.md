tle_propagator.py
-Writes key positions to positions file in sample_pos_angles directory
    -First line is camera position
    -Second line is sun position
    -Following lines represent positions of moving object of interest

lc_processing directory is for 
    -tracking lc generation
    -Generating plots

labels .npy file header:
    Track Number, Object Name, Material Attitude Regime, x, y, z, %padded
    -Contains only most important info - more metadata needs to be looked un in the corresponding track folders

Ligh curve pre-processing
    1. collect_lcs.py - Stacks and z-normalizes all available light curves of the specified length
    2. wst.py - Extracts wavelet scattering coefficients, option also to plot spectrograms of a sample
    3. xgb.py - Performs xgboost shape classfication on wavelet scattering coefficients
    3. xgb_raw.py - Performs shape classification on raw light curve stack
