# HelloWorldNet



HelloWorldnet is a modified version of [Exonet](https://gitlab.com/frontierdevelopmentlab/exoplanets/exonet-pytorch), which is in turn a modified version of [Astronet](https://github.com/tensorflow/models/tree/master/research/astronet)

This work is a direct result of the [2019 PyTorch Summer Hackathon](https://info.devpost.com/pytorchmpkrules), hosted at Facebook HQ, with team members:

- [Gully](https://github.com/gully)
- [Grant](https://github.com/GrantRVD) ([twitter](https://twitter.com/usethespacebar))
- [Humayun](https://github.com/humayun)


Our goal is to apply PyTorch to improve the speed and reliability of detecting exoplanets in [lightcurve](https://imagine.gsfc.nasa.gov/features/yba/M31_velocity/lightcurve/lightcurve_more.html) data. Specifically, we're attempting to

- extend Exonet and Astronet for better precision and recall
- creating dataloaders for various data sources, such as Kepler, TESS, and K2
- exploring model architectures to improve transfer learning between exoplanet monitoring and detection tasks


#### Original README for [ExonetNet-PyTorch](https://gitlab.com/frontierdevelopmentlab/exoplanets/exonet-pytorch):

> Exonet is a modified version of [Astronet](https://github.com/tensorflow/models/tree/master/research/astronet), with added scientific "domain knowledge" and translated from TensorFlow to PyTorch.
This work is a direct result of the 2018, [NASA's Frontier Development Lab](https://frontierdevelopmentlab.org/) (FDL)
If you use this work please cite: [2018 NASA FDL Exoplanet Team (2018), ApJ Letters, 869, L7](http://adsabs.harvard.edu/abs/2018ApJ...869L...7A).

> The 2018 NASA FDL Exoplanet Team included:
[Megan Ansdell](https://www.meganansdell.com),
[Yani Ioannou](https://yani.io/annou/),
[Hugh Osborn](https://www.hughosborn.co.uk/),
[Michele Sasdelli](https://uk.linkedin.com/in/michelesasdelli)

> ßThere will soon also be a code for downloading the required *Kepler* light curves and generating the input views and labels.
For now, you can download the required input files from [this DropBox link](https://www.dropbox.com/sh/sxj7r30thd66nij/AACptMysLyaMhXe817e4z7Sya?dl=0)
(you must divide them into train, val, and test folders for the code to work).
