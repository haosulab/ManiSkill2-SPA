# Transporter Networks

This folder provides a transporter networks baseline to solve the AssemblingKits task.

The original Transporter Networks method has some imprecision that won't be able to solve the more strict ManiSkill2 AssemblingKits environment. As a result, as a more engineered solution this particular baseline does the following

1. Perform an initial scan over the environment, capturing 10 images from the hand-view camera. This is all fused into a single bird-eye view camera and height map
2. Train the Transporter Network on this scanned data, while also predicting a bin of 144 rotations instead of the default 36.
3. During evaluation (e.g in the ManiSkill2 challenge), perform the same initial scan and then predict the pose of the target object and the pose of the goal object (the actions of the TransporterNetwork). A motion planning solution is then used to pick and place according to the predicted poses.

## Getting Started

### Installation

First create a new conda environment and install mani-skill2 and TransporterNetworks 
```
conda create --name ms2tpn python=3.8
conda activate ms2tpn

# clone transporternetworks (called ravens) and install it
git clone https://github.com/google-research/ravens.git
cd ravens
pip install -r requirements.txt
python setup.py install --user

# install an appropriate tensorflow version. 2.11 works
conda install -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
pip install --upgrade pip
pip install tensorflow==2.11.*
pip install tensorflow-addons==0.19.0

pip install mani-skill2
```

Then install pymp, a suite of motion planning tools.
```
conda install pinocchio -c defaults -c conda-forge
pip install --upgrade git+https://github.com/Jiayuan-Gu/pymp.git
```

After installing everything, make sure to `cd` into this folder.

### Evaluation

We have a pretrained model on google drive, download it here https://drive.google.com/file/d/1QoelH5swqgUUPOT7IQePcL14dWRHazrB/view?usp=sharing
<!-- 
To then test the model, run
```
python test_gripper.py \
    --model="assembly144-transporter-1000-0" --n-steps=100000 --n-rotations=144 --json-name="train_episodes.json"

python test_gripper.py \
    --model="assembly2-transporter-1000-0" --n-steps=100000 --n-rotations=144 --json-name="train_episodes.json"
```

You can also use `test_suction.py` to test with the suction gripper instead of a two-finger gripper. The pretrained models should get around 15-20% success rate of slotting in the piece into the assembly kit. -->

To evaluate the model run

```
python -m mani_skill2.evaluation.run_evaluation -e "AssemblingKits-v0" -o out --cfg train_episodes.json
```

### Training 

As our AssemblingKits environment looks different to the ravens environment and is much stricter with success metrics, we provide tools to create a dataset and train a transporter network model from scratch.


#### Generate data
To generate the dataset, first download the demonstrations for AssemblingKits from [Maniskill2](https://github.com/haosulab/Maniskill2). You can run this CLI tool to do so:
```
python -m mani_skill2.utils.download_demo AssemblingKits-v0 -o demos
```

These demonstrations are simply used to generate the initial RGBD images of the assembly kit and the initial and goal poses.

Once the demos are saved to a local `demos` folder, run the following to generate a training and test dataset

```
python gen_dataset.py --num-procs 8 \
    --traj-name demos/rigid_body/AssemblingKits-v0/trajectory.h5 \
    --json-name demos/rigid_body/AssemblingKits-v0/trajectory.json \
    --output-name data/train_1000.h5 \
    --max-num-traj 1000
python gen_dataset.py --num-procs 8 \
    --traj-name demos/rigid_body/AssemblingKits-v0/trajectory.h5 \
    --json-name demos/rigid_body/AssemblingKits-v0/trajectory.json \
    --output-name data/test_600.h5 \
    --max-num-traj 600 --test-split
```

#### Run training

To run the training code, simply run the following

```
python train.py --task=assemblyscan --agent=transporter --n_demos=1000 --n_rotations=144 
```

which will save models to `checkpoints/assembly144-transporter-1000-0`.












```
conda create --name ms2tpn python=3.8
conda activate ms2tpnfix

git clone https://github.com/google-research/ravens.git
cd ravens
pip install -r requirements.txt
python setup.py install --user


conda install -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
pip install --upgrade pip
pip install tensorflow==2.11.*
pip install tensorflow-addons==0.19.0

pip install mani-skill2
```