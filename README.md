3D Data Processing: deal_3D:

First, in nmrpipe, proc_dirct is used to convert the original .fid data into .ft1 data (the original .fid 3D data is in hypercomplex form; .ft1 is essentially performing a Fourier transform on the direct dimension (likely with phase modulation), and then only the real part is retained).

Then, the hyper2complex function in the hyper_complex_data_proc code is used to convert the hypercomplex data into a real part and an imaginary part (both complex numbers). Then, find_peak_3D_without_limit is used to slice the 3D data along the direct dimension, obtaining the corresponding 2D slice data for undersampling reconstruction.

The reconstructed results are then aggregated along the direct dimension to form 3D data. Finally, complex2hyper in hyper_complex_data_proc, combined with proc_indirct in nmrpipe, is used to obtain the corresponding reconstructed, zero-padded, windowed .mat data.
