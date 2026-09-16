# Correlational Training of Morphological Neural Networks

Code for ICASSP 2027 submission "CORRELATIONAL TRAINING OF MORPHOLOGICAL NEURAL
NETWORKS": Training and evaluation of MLP, Min-max-plus, ViT, and MorphViT models with either standard back-propagation or the proposed correlational/MWU-inspired update.

## Abstract

Neural networks are typically trained using first-order methods and back-propagation. It is unclear whether this approach is optimal for morphological layers whose weight Jacobians are sparse and whose resulting parameter gradients can be poor. In this work, we propose a novel weight update method for morphological neural networks inspired from the Multiplicative Weights Update (MWU) scheme. We view each morphological perceptron as an instance of the learning from experts' advice problem in logarithmic space, and use a correlation-based reward that favors inputs aligned with the desired output change, regardless of whether a strong gradient signal has reached their weight. We empirically evaluate our approach by training fully connected layers both as stand-alone models and as parts of larger transformer networks. Across nine benchmarks, correlational training yields improvements on eight, by up to 32.84 percentage points, while substantially reducing run-to-run variability. 

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

The provided `requirements.txt` reflects the CUDA/PyTorch environment used for the experiments. If necessary, install a PyTorch build appropriate for your own CUDA/CPU setup.

## Running experiments

Experiments are launched through `entry_point.py`. For example:

```bash
python entry_point.py \
    --model min-max-plus \
    --dataset mnist \
    --hidden_dims 256 \
    --num_layers 2 \
    --temperature 1.0 \
    --lr 0.01 \
    --optim adam \
    --weight_decay 0 \
    --epochs 50 \
    --batch_size 64 \
    --use_mwu \
    --seed 42
```

Useful model names are:

```text
mlp
min-max-plus
ViT
MorphViT
```

Supported dataset names are:

```text
mnist
fashion-mnist
cifar-10
pmlb
```

For PMLB, specify the dataset separately, for example:

```bash
python entry_point.py \
    --model min-max-plus \
    --dataset pmlb \
    --pmlb_name adult \
    --temperature 0 \
    --train_metric balanced_accuracy \
    --test_metric balanced_accuracy \
    --use_mwu
```

Use `--use_mwu` to enable the proposed correlational update. Without it, morphological models are trained using ordinary back-propagation.

## Reproducing the experiments

The `Makefile` contains the experiment configurations used in the paper. Seeds are fixed to `42 43 44 45 46`.

Run all experiments with:

```bash
make all
```

Individual experiment families can also be run separately, for example:

```bash
make mlp_mnist
make min-max-plus_mnist_NO-MWU
make min-max-plus_mnist_MWU

make mlp_pmlb
make min-max-plus_pmlb_NO-MWU
make min-max-plus_pmlb_MWU

make vit_cifar
make morphvit_cifar_NO-MWU
make morphvit_cifar_MWU
```

A single seed/run can be launched through its corresponding Make target, e.g.

```bash
make min-max-plus_mnist_MWU_seed42
make min-max-plus_pmlb-adult_MWU_seed42
make morphvit_cifar-10_MWU_seed42
```

## Output structure

Each run is written under `runs/` using a configuration-dependent name:

```text
runs/
  <run-name>/
    config.yaml
    checkpoints/
      best.pt
    results/
      training_history.txt
      test_metric.txt
```

`config.yaml` stores the full run configuration. `best.pt` is the checkpoint selected using validation performance.

## Aggregating results

To aggregate results across seeds:

```bash
python aggregate_results.py
```

This writes the mean and sample standard deviation of the test metric to:

```text
results_summary.txt
```

Results can be filtered, e.g.

```bash
python aggregate_results.py --include pmlb MWU
```

Training curves can be plotted by specifying experiment-name prefixes:

```bash
python aggregate_results.py \
    --plot \
    min-max-plus_mnist_NO-MWU_T1.0 \
    min-max-plus_mnist_MWU_T1.0 \
    --plot-output figures/mnist_training.pdf
```

The plotting script averages matching runs over seeds and shows the validation mean with a standard-deviation band.

## Main files

- `entry_point.py`: command-line interface and experiment setup.
- `models.py`: MLP, Min-max-plus, ViT, and MorphViT implementations.
- `training.py`: standard and correlational/MWU-inspired training rules.
- `dataloading.py`: MNIST, Fashion-MNIST, CIFAR-10, and PMLB loading/splitting.
- `metrics.py`, `losses.py`, `optimizer.py`: training utilities.
- `aggregate_results.py`: result aggregation and plotting.
- `Makefile`: paper experiment configurations.
