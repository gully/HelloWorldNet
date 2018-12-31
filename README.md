# ExonetNet-PyTorch
Exonet is a modified version of [Astronet](https://github.com/tensorflow/models/tree/master/research/astronet), with added scientific "domain knowledge" and translated from TensorFlow to PyTorch.
This work is a direct result of the 2018, [NASA's Frontier Development Lab](https://frontierdevelopmentlab.org/) (FDL) 
If you use this work please cite: [2018 NASA FDL Exoplanet Team (2018), ApJ Letters, 869, L7](http://adsabs.harvard.edu/abs/2018ApJ...869L...7A).

The 2018 NASA FDL Exoplanet Team included:
[Megan Ansdell](https://www.meganansdell.com),
[Yani Ioannou](https://yani.io/annou/),
[Hugh Osborn](https://www.hughosborn.co.uk/),
[Michele Sasdelli](https://uk.linkedin.com/in/michelesasdelli)

There will soon also be a code for downloading the required *Kepler* light curves and generating the input views and labels. 
For now, you can download the required input files from [this DropBox link](https://www.dropbox.com/sh/sxj7r30thd66nij/AACptMysLyaMhXe817e4z7Sya?dl=0) 
(you must divide them into train, val, and test folders for the code to work).