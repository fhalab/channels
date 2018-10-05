# channels
## Code to reproduce the paper *Machine learning-guided channelrhodopsin engineering enables minimally-invasive optogenetics*. Gaussian process models for optimizing channelrhodopsin properties. 

### Computing Environment:

This was originally developed using Anaconda Python 3.5 and the following packages and versions:

1. numpy 1.13.1
2. pandas 0.20.3
3. scipy 0.19.1
4. sklearn 0.19.0

### File structure

The repository is divided into code and inputs. Inputs contains all labeled sequences used to build GP regression & classification models and the labeled sequences used to build Gaussian process models. Code contains Python implementations of Gaussian process regression with the Matérn kernel. 
