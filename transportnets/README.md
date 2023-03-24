# Transporter Networks

This folder provides a transporter networks baseline to solve the AssemblingKits task

## Getting Started

### Installation

First create a new conda environment and install mani-skill2 and TransporterNetworks 
```
conda create -n ms2tpn python==3.8
conda activate ms2tpn
pip install mani-skill2

# clone transporternetworks (called ravens) and install it
git clone https://github.com/google-research/ravens.git
cd ravens
pip install -r requirements.txt
python setup.py install --user
```

For some systems you may need a specific version of cudatoolkit, cudnn, and tensorflow.
```
pip install --upgrade numpy
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/ # might be necessary to get code to run
python3 -m pip install --upgrade tensorflow # You can upgrade tensorflow to 2.10.0 if 2.3.0 does not work
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
python train.py --task=assembly --agent=transporter --n_demos=1000 --n_rotations=144 
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