PYTHON := python
ENTRY := entry_point.py

SEEDS := 42 43 44 45 46

MNIST_DATASETS := mnist fashion-mnist

CIFAR_DATASETS := cifar-10

PMLB_DATASETS := \
	adult \
	mushroom \
	ionosphere \
	phoneme \
	spambase \
	chess

# -------------------------------------------------------------------
# Common hyperparameters
# -------------------------------------------------------------------

MNIST_ARGS := \
	--hidden_dims 256 \
	--num_layers 2 \
	--epochs 50 \
	--batch_size 64 \

CIFAR_ARGS := \
	--hidden_dims 256 \
	--patch_size 4 \
	--embed_dim 128 \
	--num_layers 6 \
	--epochs 50 \
	--batch_size 64 \
	--lr 0.001

PMLB_ARGS := \
	--hidden_dims 128 \
	--num_layers 1 \
	--epochs 100 \
	--batch_size 32 \
	--lr 0.01

# -------------------------------------------------------------------
# Convenience targets
# -------------------------------------------------------------------

.PHONY: all mlp_mnist min-max-plus_mnist_NO-MWU min-max-plus_mnist_MWU mlp_pmlb min-max-plus_pmlb_NO-MWU min-max-plus_pmlb_MWU vit_cifar morphvit_cifar_NO-MWU morphvit_cifar_MWU verbose clean

all: mlp_mnist min-max-plus_mnist_NO-MWU min-max-plus_mnist_MWU mlp_pmlb min-max-plus_pmlb_NO-MWU min-max-plus_pmlb_MWU vit_cifar morphvit_cifar_NO-MWU morphvit_cifar_MWU verbose

mlp_mnist: \
	$(foreach dataset,$(MNIST_DATASETS),\
		$(foreach seed,$(SEEDS),\
			mlp_$(dataset)_seed$(seed)))

mlp_pmlb: \
	$(foreach dataset,$(PMLB_DATASETS),\
		$(foreach seed,$(SEEDS),\
			mlp_pmlb-$(dataset)_seed$(seed)))

min-max-plus_mnist_NO-MWU: \
	$(foreach dataset,$(MNIST_DATASETS),\
		$(foreach seed,$(SEEDS),\
			min-max-plus_$(dataset)_NO-MWU_seed$(seed)))

min-max-plus_mnist_MWU: \
	$(foreach dataset,$(MNIST_DATASETS),\
		$(foreach seed,$(SEEDS),\
			min-max-plus_$(dataset)_MWU_seed$(seed)))

min-max-plus_pmlb_NO-MWU: \
	$(foreach dataset,$(PMLB_DATASETS),\
		$(foreach seed,$(SEEDS),\
			min-max-plus_pmlb-$(dataset)_NO-MWU_seed$(seed)))

min-max-plus_pmlb_MWU: \
	$(foreach dataset,$(PMLB_DATASETS),\
		$(foreach seed,$(SEEDS),\
			min-max-plus_pmlb-$(dataset)_MWU_seed$(seed)))

vit_cifar: \
	$(foreach dataset,$(CIFAR_DATASETS),\
		$(foreach seed,$(SEEDS),\
			vit_$(dataset)_seed$(seed)))

morphvit_cifar_NO-MWU: \
	$(foreach dataset,$(CIFAR_DATASETS),\
		$(foreach seed,$(SEEDS),\
			morphvit_$(dataset)_NO-MWU_seed$(seed)))

morphvit_cifar_MWU: \
	$(foreach dataset,$(CIFAR_DATASETS),\
		$(foreach seed,$(SEEDS),\
			morphvit_$(dataset)_MWU_seed$(seed)))

verbose:
	$(PYTHON) $(ENTRY) \
		--model min-max-plus \
		--dataset mnist \
		$(MNIST_ARGS) \
		--lr 0.001 \
		--optim adam \
		--weight_decay 0 \
		--verbose \
		--runs_root runs/_verbose \
		--epochs 10 \
		--seed 42

	$(PYTHON) $(ENTRY) \
		--model min-max-plus \
		--dataset mnist \
		$(MNIST_ARGS) \
		--lr 0.01 \
		--use_mwu \
		--optim adam \
		--weight_decay 0 \
		--verbose \
		--runs_root runs/_verbose \
		--epochs 10 \
		--seed 42

# -------------------------------------------------------------------
# MNIST
# -------------------------------------------------------------------

define MAKE_MNIST_RULE
.PHONY: mlp_mnist_seed$(1)
mlp_mnist_seed$(1):
	$(PYTHON) $(ENTRY) \
		--model mlp \
		--dataset mnist \
		$(MNIST_ARGS) \
		--lr 0.001 \
		--optim adam \
		--weight_decay 0 \
		--seed $(1)

.PHONY: min-max-plus_mnist_NO-MWU_seed$(1)
min-max-plus_mnist_NO-MWU_seed$(1):
	$(PYTHON) $(ENTRY) \
		--model min-max-plus \
		--dataset mnist \
		$(MNIST_ARGS) \
		--lr 0.001 \
		--optim adam \
		--weight_decay 0 \
		--seed $(1)

.PHONY: min-max-plus_mnist_MWU_seed$(1)
min-max-plus_mnist_MWU_seed$(1):
	$(PYTHON) $(ENTRY) \
		--model min-max-plus \
		--dataset mnist \
		$(MNIST_ARGS) \
		--lr 0.01 \
		--use_mwu \
		--optim adam \
		--weight_decay 0 \
		--seed $(1)

.PHONY: mlp_fashion-mnist_seed$(1)
mlp_fashion-mnist_seed$(1):
	$(PYTHON) $(ENTRY) \
		--model mlp \
		--dataset fashion-mnist \
		$(MNIST_ARGS) \
		--lr 0.001 \
		--optim adam \
		--weight_decay 0 \
		--seed $(1)

.PHONY: min-max-plus_fashion-mnist_NO-MWU_seed$(1)
min-max-plus_fashion-mnist_NO-MWU_seed$(1):
	$(PYTHON) $(ENTRY) \
		--model min-max-plus \
		--dataset fashion-mnist \
		$(MNIST_ARGS) \
		--lr 0.001 \
		--optim adam \
		--weight_decay 0 \
		--seed $(1)

.PHONY: min-max-plus_fashion-mnist_MWU_seed$(1)
min-max-plus_fashion-mnist_MWU_seed$(1):
	$(PYTHON) $(ENTRY) \
		--model min-max-plus \
		--dataset fashion-mnist \
		$(MNIST_ARGS) \
		--lr 0.01 \
		--use_mwu \
		--optim adam \
		--weight_decay 0 \
		--seed $(1)
endef

$(foreach seed,$(SEEDS),\
	$(eval $(call MAKE_MNIST_RULE,$(seed))))

# -------------------------------------------------------------------
# PMLB
# -------------------------------------------------------------------

define MAKE_PMLB_RULE
.PHONY: mlp_pmlb-$(1)_seed$(2)
mlp_pmlb-$(1)_seed$(2):
	$(PYTHON) $(ENTRY) \
		--model mlp \
		$(PMLB_ARGS) \
		--dataset pmlb \
		--pmlb_name $(1) \
		--optim adam \
		--weight_decay 0 \
		--train_metric balanced_accuracy \
		--test_metric balanced_accuracy \
		--seed $(2)

.PHONY: min-max-plus_pmlb-$(1)_NO-MWU_seed$(2)
min-max-plus_pmlb-$(1)_NO-MWU_seed$(2):
	$(PYTHON) $(ENTRY) \
		--model min-max-plus \
		$(PMLB_ARGS) \
		--dataset pmlb \
		--pmlb_name $(1) \
		--optim adam \
		--weight_decay 0 \
		--temperature 0 \
		--train_metric balanced_accuracy \
		--test_metric balanced_accuracy \
		--seed $(2)

.PHONY: min-max-plus_pmlb-$(1)_MWU_seed$(2)
min-max-plus_pmlb-$(1)_MWU_seed$(2):
	$(PYTHON) $(ENTRY) \
		--model min-max-plus \
		$(PMLB_ARGS) \
		--dataset pmlb \
		--pmlb_name $(1) \
		--optim adam \
		--use_mwu \
		--weight_decay 0 \
		--temperature 0 \
		--train_metric balanced_accuracy \
		--test_metric balanced_accuracy \
		--seed $(2)
endef

$(foreach dataset,$(PMLB_DATASETS),\
	$(foreach seed,$(SEEDS),\
		$(eval $(call MAKE_PMLB_RULE,$(dataset),$(seed)))))

# -------------------------------------------------------------------
# CIFAR
# -------------------------------------------------------------------

define MAKE_CIFAR_RULE
.PHONY: vit_cifar-10_seed$(1)
vit_cifar-10_seed$(1):
	$(PYTHON) $(ENTRY) \
		--model ViT \
		--dataset cifar-10 \
		--dataset_root ../data \
		$(CIFAR_ARGS) \
		--optim adam \
		--weight_decay 0 \
		--seed $(1)

.PHONY: morphvit_cifar-10_NO-MWU_seed$(1)
morphvit_cifar-10_NO-MWU_seed$(1):
	$(PYTHON) $(ENTRY) \
		--model MorphViT \
		--dataset cifar-10 \
		--dataset_root ../data \
		$(CIFAR_ARGS) \
		--optim adam \
		--weight_decay 0 \
		--use_cheap \
		--seed $(1)

.PHONY: morphvit_cifar-10_MWU_seed$(1)
morphvit_cifar-10_MWU_seed$(1):
	$(PYTHON) $(ENTRY) \
		--model MorphViT \
		--dataset cifar-10 \
		--dataset_root ../data \
		$(CIFAR_ARGS) \
		--optim adam \
		--use_mwu \
		--weight_decay 0 \
		--use_cheap \
		--seed $(1)
endef

$(foreach seed,$(SEEDS),\
	$(eval $(call MAKE_CIFAR_RULE,$(seed))))

# -------------------------------------------------------------------
# Cleanup
# -------------------------------------------------------------------

clean:
	rm -rf ./runs/*