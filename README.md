# ExonetNet-PyTorch
Exonet is a modified version of [Astronet](https://github.com/tensorflow/models/tree/master/research/astronet), with added scientific "domain knowledge" and translated from TensorFlow to PyTorch.
This work is a direct result of the 2018, [NASA's Frontier Development Lab](https://frontierdevelopmentlab.org/) (FDL) 
If you use this work please cite: 2018 NASA FDL Exoplanet Team (2018), submitted.

The 2018 NASA FDL Exoplanet Team included:
[Megan Ansdell](https://www.meganansdell.com),
[Yani Ioannou](https://yani.io/annou/),
[Hugh Osborn](https://www.hughosborn.co.uk/),
[Michele Sasdelli](https://uk.linkedin.com/in/michelesasdelli)

There will soon also be a code for downloading the required *Kepler* light curves and generating the input views and labels. 
For now, you can follow the instructions on the Astronet GitHub webpage to generate the light curve inputs, and then modify them accordingly to create the centroid inputs.
The required info files with the stellar parameter inputs and labels can be found in the "input" directory in this repository.
Alternatively, you can contact one of the FDL Exoplanet Team members for these files and any help setting up the code.
