# Transporter Networks

This folder provides a transporter networks baseline to solve the AssemblingKits task

## Getting Started

### Installation

First create a new maniskill2 conda environment if you have not already.


```
git clone https://github.com/google-research/ravens.git
```

Then follow the installation instructions on the [transporter networks github repo (called ravens)](github.com/google-research/ravens)


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

We have a pretrained model on google drive, download it here [TODO]
<!-- 
To then test the model, run
```
python test_gripper.py \
    --model="assembly144-transporter-1000-0" --n-steps=100000 --n-rotations=144 --json-name="train_episodes.json"
```

You can also use `test_suction.py` to test with the suction gripper instead of a two-finger gripper. The pretrained models should get around 15-20% success rate of slotting in the piece into the assembly kit. -->

To evaluate the model run

```
python -m mani_skill2.evaluation.run_evaluation -e "AssemblingKits-v0" -o out --cfg train_episodes.json
```

### Training 

As our AssemblingKits environment looks different to the ravens environment and is much stricter with success metrics, we provide tools to create a dataset and train a transporter network model from scratch.


#### Generate data
To generate the dataset, first download the demonstrations for AssemblingKits from [Maniskill2](https://github.com/haosulab/Maniskill2). These demonstrations are simply used to generate the initial RGBD images of the assembly kit and the initial and goal poses.

Once the demos are saved to a local `demos` folder, run the following

```
python create_dataset.py --env-name AssemblingKits-v0 --num-procs 10 \
    --traj-name demos/AssemblingKits-v0/trajectory.h5 \
    --json-name demos/AssemblingKits-v0/trajectory.json \
    --output-name train_1000.h5 \
    --control-mode pd_joint_delta_pos --max-num-traj 1000 --obs-mode rgbd \
    --n-points 1200 --obs-frame base --reward-mode dense --render

python create_dataset.py --env-name AssemblingKits-v0 --num-procs 10 \
    --traj-name demos/AssemblingKits-v0/trajectory.h5 \
    --json-name demos/AssemblingKits-v0/trajectory.json \
    --output-name test_600.h5 \
    --control-mode pd_joint_delta_pos --max-num-traj 600 --obs-mode rgbd \
    --n-points 1200 --obs-frame base --reward-mode dense --render --test-split
```

#### Run training

To run the training code, simply run the following

```
python train.py --task=assembly144 --agent=transporter --n_demos=1000 --n_rotations=144 
```

which will save models to `checkpoints/assembly144-transporter-1000-0`.