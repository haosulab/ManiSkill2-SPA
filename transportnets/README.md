# Transporter Networks

This repo provides a transporter networks baseline to solve the AssemblingKits task

## Getting Started

### Installation

First create a new maniskill2 conda environment if you have not already.

Then follow the installation instructions on the [transporter networks github repo (called ravens]()


You can upgrade tensorflow to 2.10.0 if 2.3.0 does not work
```
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/ # might be necessary to get code to run
python3 -m pip install tensorflow
```

Then install pymp, a suite of motion planning tools.
```
conda install pinocchio=2.6.8 -c jigu -c defaults -c conda-forge
pip install --upgrade git+https://github.com/Jiayuan-Gu/pymp.git
```

After installing everything, make sure to `cd` into this folder.

### Evaluation

### Training 

As our AssemblingKits environment looks different to the ravens environment and is much stricter with success metrics, we provide tools to create a dataset and train a transporter network model from scratch.


#### Generate data
To generate the dataset, run

```
python create_dataset.py --env-name AssemblingKits-v0 --num-procs 10 \
    --traj-name demos/AssemblingKits-v0/trajectory.h5 \
    --json-name demos/AssemblingKits-v0/trajectory.json \
    --output-name demos/AssemblingKits-v0/trajectory.pd_joint_delta_pos_rgbd_train_1000.h5 \
    --control-mode pd_joint_delta_pos --max-num-traj 1000 --obs-mode rgbd \
    --n-points 1200 --obs-frame base --reward-mode dense --render

python create_dataset.py --env-name AssemblingKits-v0 --num-procs 10 \
    --traj-name demos/AssemblingKits-v0/trajectory.h5 \
    --json-name demos/AssemblingKits-v0/trajectory.json \
    --output-name demos/AssemblingKits-v0/trajectory.pd_joint_delta_pos_rgbd_test_600.h5 \
    --control-mode pd_joint_delta_pos --max-num-traj 600 --obs-mode rgbd \
    --n-points 1200 --obs-frame base --reward-mode dense --render --test-split
```

#### Run training

To run the traning code, simply run the following

```
python train.py --task=assembly144 --agent=transporter --n_demos=1000 --n_rotations=144
```


# Train/Test


Train

```
python train.py --task=assembly144 --agent=transporter --n_demos=1000 --n_rotations=144
```


Test


```
python test.py --assets_root=./ravens/environments/assets/ --disp=True --task=assembly144 --agent=transporter --n_demos=1000 --n_steps=100000 --n_rotations=144
```