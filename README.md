# VIPO: Value Function Inconsistency Penalized Offline Policy Optimization


## Requirements

To install all the required dependencies:

1. Install MuJoCo engine, which can be downloaded from [here](https://mujoco.org/download).
2. Install Python packages listed in `requirements.yml` using `conda env create -f requirement.yml`. You should specify the version of `mujoco-py` in `requirements.ymal` depending on the version of MuJoCo engine you have installed.
3. Manually download and install `d4rl` package from [here](https://github.com/rail-berkeley/d4rl).
4. Manually download and install `neorl` package from [here](https://github.com/polixir/NeoRL).

## Usage

Just run `train.py` with specifying the task name. Other hyperparameters are automatically loaded from `config`.

```bash
python train.py --task [TASKNAME]
```
