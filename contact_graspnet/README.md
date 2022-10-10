# Contract GraspNet

This folder provides a Contact GraspNet baseline to solve the PickSingleYCB task

## Getting Started

### Installation

There are two parts to install, Contact GraspNet, and ManiSkill2

### Contact GraspNet Installation

First install maniskill 2 in a new environment
```
conda env create -n ms2cgn -f environment.yml
conda activate ms2cgn
python setup.py develop
```


First `cd` into the `contact_graspnet` folder, then run

```
conda env update -f contact_graspnet_env.yml
```

You can also upgrade Tensorflow and recompile ops as follows if using a RTX 30xx
```
conda install -c conda-forge cudatoolkit=11.2
conda install -c conda-forge cudnn=8.2
pip install tensorflow==2.5 tensorflow-gpu==2.5

sh compile_pointnet_tfops.sh
```

```
pip install pyyaml==5.4.1
```

If mlab visualization fails, you may need this:
```
export ETS_TOOLKIT=qt4
export QT_API=pyqt5
```

## Models

To run inference, you can download one of the pretrained models [here](https://drive.google.com/drive/folders/1tBHKf60K8DLM5arm-Chyf7jxkzOr5zGl?usp=sharing). We recommend downloading the one called
`scene_test_2048_bs3_hor_sigma_001`


## Test
```
python test.py
```
Will first generate some pointcloud data beforehand. Once prompted, generate the contact grasps and save the results to a `results` folder with the following

```
python contact_graspnet/contact_graspnet/inference.py --np_path=test_data/*  --forward_passes=5
```

which will store predictions in the `results/` folder. Then press enter to continue running and store results in a `results.pkl` file.