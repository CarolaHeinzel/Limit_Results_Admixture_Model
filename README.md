# Distribution of the Maximum Likelihood Estimator in the Admixture Model

This repository contains code to calculate the distirbution of Maximum Likelihood Estimator for the ancestry and the allele frequencies in the Admixture Model. Therefore, we consider different settings:

1) The true parameters are in the interior of the parameter space.
2) The true parameters are on the boundary of the parameter space.

The code to calculate the asymptotic distribution can be found in the file Asymptotic_Distribution.py. 

The repository also contains example output of ADMIXTURE (example_data.Q and example_data.P). 

To apply the code to new data, just replace the path at data_p and data_q with the output of ADMIXTURE for your data.
