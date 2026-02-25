# Amortized Safe Active Learning (ASAL)

[![arXiv](https://img.shields.io/badge/arXiv-2501.15458-b31b1b.svg)](https://arxiv.org/abs/2501.15458)
[![AISTATS 2026](https://img.shields.io/badge/AISTATS-2026-blue.svg)](https://openreview.net/forum?id=WWrdo1tfdw)

Pretrained models and reproduction scripts for the paper:

**[Amortized Safe Active Learning for Real-Time Data Acquisition: Pretrained Neural Policies from Simulated Nonparametric Functions](https://arxiv.org/abs/2501.15458)**  
Cen-You Li, Marc Toussaint, Barbara Rakitsch*, Christoph Zimmer*  
*AISTATS 2026*

The core implementation is hosted at [active-learning-framework](https://github.com/cenyou/active-learning-framework).

## Citation

```bibtex
@inproceedings{li2026amortized,
    title={Amortized Safe Active Learning for Real-Time Data Acquisition: Pretrained Neural Policies from Simulated Nonparametric Functions},
    author={Cen-You Li and Marc Toussaint and Barbara Rakitsch and Christoph Zimmer},
    booktitle={International Conference on Artificial Intelligence and Statistics},
    year={2026},
    url={https://openreview.net/forum?id=WWrdo1tfdw}
}
```

---

# Installation

We use conda for environment management ([conda docs](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html)).

### 1. Clone and Install

```bash
git clone https://github.com/cenyou/active-learning-framework
cd active-learning-framework
conda env create --file environment.yml
conda activate alef
pip install -e .
```

### 2. Verify Installation

```bash
conda activate alef
pytest tests
```

### 3. Configure Paths

Edit `alef/configs/paths.py`:
- Set `EXPERIMENT_PATH` to your experiment folder (datasets will be saved to `EXPERIMENT_PATH/data`)
- Set `PYTHON_EXECUTOR` to your `alef` environment's Python path

### Troubleshooting

<details>
<summary><b>OMP Error / pytest fails loading packages</b></summary>

If you see `OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized`, remove the duplicate DLL:
```
path\to\user\.conda\envs\alef\Lib\site-packages\torch\lib\libiomp5md.dll
```
</details>

<details>
<summary><b>PyTorch and CUDA compatibility issues</b></summary>

Try one of the following:
```bash
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```
or
```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
</details>

<details>
<summary><b>JAX errors</b></summary>

```bash
conda uninstall jax jaxlib
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
</details>

---

# Pretraining Phase

## Training Unconstrained AL

```bash
python ./alef/active_learners/amortized_policies/training/main_train.py --loss_config GPMI2LossConfig --dimension D --num_experiments_initial N_init N_init --num_experiments 1 T --batch_random_shorter_sequences True --policy_knows_budget True --device cuda --seed 0 1 2 --lr 0.0001
```

**Parameters:**
- `D`: input dimension
- `N_init`: number of initial data points
- `T`: maximum number of queries

See Appendix Table S.F.4 for recommended settings.

**Loss function options:**
| Loss Config | Description |
|-------------|-------------|
| `GPMI2LossConfig` | $\mathcal{I}_{\text{mean}}$ (default) |
| `GPEntropy2LossConfig` | $\mathcal{H}$ |
| `GPEntropy1LossConfig` | $\mathcal{H}_{\text{mean}}$ |

## Training DAD Baseline

```bash
python ./alef/active_learners/amortized_policies/training/main_train.py --loss_config DADLossConfig --dimension D --num_experiments_initial N_init N_init --num_experiments T T --batch_random_shorter_sequences False --policy_knows_budget False --device cuda --seed 0 --lr 0.0001
```

## Training ALINE Baseline

Follow the instructions at [cenyou/ALINE (asal branch)](https://github.com/cenyou/ALINE/tree/asal).

```bash
git clone https://github.com/cenyou/ALINE
cd ALINE
git checkout asal
conda create ...
...
```

## Training Safe AL (ASAL)

```bash
python ./alef/active_learners/amortized_policies/training/main_train.py --loss_config MinUnsafeGPEntropy2LossConfig --alpha alpha --dimension D --num_experiments_initial N_init N_init --num_experiments 1 T --batch_random_shorter_sequences True --policy_knows_budget True --device cuda --seed 0 1 2 --lr 0.0001
```

> **Note:** The safety likelihood tolerance $\gamma$ from the paper is denoted as `alpha` in the code (in percentage). For example, `alpha=5` corresponds to $\gamma=0.05$.

See Appendix Table S.F.4 for recommended `(D, N_init, T)` settings.

**Ablation loss configs:**
| Loss Config | Description |
|-------------|-------------|
| `MinUnsafeGPEntropy1LossConfig` | $\mathcal{S}_{\mathcal{H}_{\text{mean}}}$ |
| `SafeGPEntropy2LossConfig` | $\mathcal{S}_{\mathcal{H}, \text{division}}$ |
| `SafeGPEntropy1LossConfig` | $\mathcal{S}_{\mathcal{H}_{\text{mean}}, \text{division}}$ |

---

# Experiments (Deployment Phase)

## Prerequisites

### 1. Install TabPFN/ALINE Baseline

From the `active-learning-framework` folder:
```bash
git checkout tabpfn
conda env create --file environment_tabpfn.yml
conda activate asal-tabpfn
pip install -e .
git checkout main
```

### 2. Install Amortized GP Baseline

```bash
git clone https://github.com/cenyou/Amor-Struct-GP/
cd Amor-Struct-GP
git checkout asal
conda activate alef
pip install -e .
```

Download the pretrained weights:
```bash
cd $EXPERIMENT_PATH
git clone https://github.com/boschresearch/Amor-Struct-GP-pretrained-weights
```
Model path: `EXPERIMENT_PATH/Amor-Struct-GP-pretrained-weights/main_state_dict_paper.pth`

### 3. Download Pretrained Policies and PFNs

Clone this repository and copy the models to `EXPERIMENT_PATH`:

| Source | Destination |
|--------|-------------|
| [./amortized_AL](./amortized_AL) | `EXPERIMENT_PATH/amortized_AL` |
| [./pfns](./pfns) | `EXPERIMENT_PATH/pfns` |
| [./ALINE](./ALINE) | `EXPERIMENT_PATH/ALINE` |

> **Note:** This repository uses [Git LFS](https://github.com/git-lfs/git-lfs) to store large model files. Make sure to install Git LFS and run `git lfs pull` after cloning to download the pretrained weights.

### 4. Download Datasets

| Dataset | Source | Destination |
|---------|--------|-------------|
| Airline | [Download](https://github.com/jbrownlee/Datasets/blob/master/airline-passengers.csv) | `EXPERIMENT_PATH/data/airlines/` |
| LGBB | [Download](https://bobby.gramacy.com/surrogates/lgbb.tar.gz) | `EXPERIMENT_PATH/data/lgbb/` |
| Airfoil | [Download](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise) | `EXPERIMENT_PATH/data/airfoil/` |
| Engine | [Download](https://github.com/boschresearch/Bosch-Engine-Datasets/tree/master/pengines/engine2_normalized.xlsx) | `EXPERIMENT_PATH/data/engine/` |

## Running Experiments

### Using `alef` environment

```bash
conda activate alef
```

**Unconstrained AL:**
| Experiment | Command |
|------------|----------|
| AAL on Sin, Branin | `python ./alef/experiments/oracle_active_learning/run_main_policy_oal.py` |
| AAL on Airline, LGBB, Airfoil | `python ./alef/experiments/pool_active_learning/run_main_policy_dataset_al.py` |
| Baselines on Sin, Branin | `python ./alef/experiments/oracle_active_learning/run_main_oal.py` |
| Baselines on Airline, LGBB, Airfoil | `python ./alef/experiments/pool_active_learning/run_main_dataset_al.py` |

**Safe AL:**
| Experiment | Command |
|------------|----------|
| ASAL on Simionescu, Townsend | `python ./alef/experiments/safe_learning/run_main_policy_safe_oal.py` |
| ASAL on LGBB, Engine, Fluid System | `python ./alef/experiments/safe_learning/run_main_policy_safe_al.py` |
| Baselines | `python ./alef/experiments/safe_learning/run_main_safe_al.py` |

### Using `asal-tabpfn` environment

```bash
git checkout tabpfn
conda activate asal-tabpfn
```

**Unconstrained AL:**
| Experiment | Command |
|------------|----------|
| ALINE on Sin, Branin | `python ./alef/experiments/oracle_active_learning/run_main_aline_oal.py` |
| ALINE on Airline, LGBB | `python ./alef/experiments/pool_active_learning/run_main_aline_dataset_al.py` |
| TabPFN on Sin, Branin | `python ./alef/experiments/oracle_active_learning/run_main_oal.py` |
| TabPFN on Airline, LGBB, Airfoil | `python ./alef/experiments/pool_active_learning/run_main_dataset_al.py` |

**Safe AL:**
| Experiment | Command |
|------------|----------|
| TabPFN Baseline | `python ./alef/experiments/safe_learning/run_main_safe_al.py` |

---

## License

See [LICENSE](./LICENSE) for details.