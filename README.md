<div align="center">

# Brightening

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.8+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.10+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 1.5+-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
<a href="https://lightning-flash.readthedocs.io/"><img alt="Flash" src="https://img.shields.io/badge/-FLASH 0.7.5+-D582FF?style=for-the-badge&logo=lightningflash&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra 1.1-89b8cd?style=for-the-badge&labelColor=gray"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>

</div>

## 📌&nbsp;&nbsp;Task lists
- [x] Lightning
- [x] Flash
- [x] Hydra config
- [x] Base training pipeline
- [x] Linters - code/docstring/yaml formatting, sorting, code analysis, cell output clear(jupyter)
- [x] W&B
- [ ] Hyper parameter tuning pipeline (model backbone/head, lr, batch size..etc)
- [ ] Target Model Training / Test
- [ ] Dataset Versioning (DVC)
- [ ] Model Versioning (mlflow)
- [ ] Model Serving (kubernates, bentoml)
- [ ] Monitoring (prometheus & grafana)

## ⚡&nbsp;&nbsp;Introduction

The Baseline Lightning-flash template to deep learning project.




## 🚀&nbsp;&nbsp;Quickstart

```bash
# clone project
git clone https://github.com/tiktakdad/Brightening
cd Brightening

# [OPTIONAL] create conda environment
conda create -n Brightening python=3.8
conda activate Brightening

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```


Log in W&B:
```bash
wandb login
```

install hooks from .pre-commit-config.yaml:
```bash
pre-commit install
pre-commit run -a
```


Train model with default configuration

```bash
# train on CPU
python train.py trainer.gpus=0

# train on GPU
python train.py trainer.gpus=1
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python train.py trainer.max_epochs=20 datamodule.batch_size=64
```

## ❤️&nbsp;&nbsp;Other Repositories

<details>
<summary><b>Inspirations</b></summary>

This template was inspired by:
[Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template),
[PyTorchLightning/deep-learninig-project-template](https://github.com/PyTorchLightning/deep-learning-project-template),
[drivendata/cookiecutter-data-science](https://github.com/drivendata/cookiecutter-data-science),
[tchaton/lightning-hydra-seed](https://github.com/tchaton/lightning-hydra-seed),
[Erlemar/pytorch_tempest](https://github.com/Erlemar/pytorch_tempest),
[lucmos/nn-template](https://github.com/lucmos/nn-template).

</details>
