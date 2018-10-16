# channels
## Code to reproduce the paper *Machine learning-guided channelrhodopsin engineering enables minimally-invasive optogenetics*. Gaussian process models for optimizing channelrhodopsin properties. 

### Computing Environment:

This was originally developed using Anaconda Python 3.6 and the following packages and versions:

1. numpy 1.13.3
2. pandas 0.20.3
3. scipy 0.19.1
4. sklearn 0.19.0
5. gpmodel (https://github.com/yangkky/gpmodel)

### File structure

The repository is divided into two self-contained directories containing all the code and inputs for the regression and classification models, respectively. For regression, the GP code is here. For classification, the GP code is in the gpmodel repository (https://github.com/yangkky/gpmodel)