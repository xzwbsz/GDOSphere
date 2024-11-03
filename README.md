# GDOSphere
Accurate Physics-informed Data-driven Weather Prediction Model

Accurate weather forecasting is essential for a variety of sectors, including agriculture, transportation, and emergency management. 
The existing approaches often incur high training costs, suffer from non-convergence and persistent errors, making it challenging to achieve both high fidelity and low computational costs.
In this paper, we propose a novel model architecture, GDOSphere, for short-term weather prediction that leverages oriented differentiation of the multi-scale icosahedral spaces. 
GDOSphere enhances prediction accuracy, and meanwhile significantly reduces computation time compared to traditional models. 
Through extensive experiments, we demonstrate the effectiveness of our approach, paving the way for its deployment in operational settings.

## Dependency & Installation
All dependent files are kept in `setup.py` and can be installed by simply running `pip install -e .`.

## GDOSphere Implementation
GDOSphere is implemented in `train.py`.

## Multi-scale Icosahedral Mesh
You must firstly generate the icosahedral mesh.

Enter the `data_process/meshcnn` and run `genmesh.py`, and you can change the variable `level` for target sub-division level.

If your system can not support `pyigl`, please use docker environment that need to run `docker pull gilureta/pyigl`.
Then use `exec -i -t gilureta/pyigl /bin/bash` to enter the docker in order to use operate the code.

## Train
After generating multi-scale icosahedral mesh, you can operate `bash run.sh` to execute the training.

The `run.sh` contains several training configuration including prediction leading time, batch size and decay, etc.
You may adjust the number for your convenient.
