PY := python
AMK := amk

.PHONY: help install validate-holdout validate-loo classify-knn classify-euclidean run-holdout-knn run-loo-1nn run-kfold-knn clean

help:
	@echo "Targets:"
	@echo "  install                Instala en editable (pip install -e .)"
	@echo "  validate-holdout        Corre SOLO validación holdout"
	@echo "  validate-loo            Corre SOLO validación LOO"
	@echo "  validate-kfold         Corre SOLO validación K-Fold"
	@echo "  classify-knn            Corre SOLO clasificación KNN (dataset+holdout)"
	@echo "  classify-euclidean      Corre SOLO clasificación Euclidiano (train/test separados)"
	@echo "  run-holdout-knn         Corre experimento completo holdout+KNN"
	@echo "  run-loo-1nn             Corre experimento completo LOO+1NN"
	@echo "  run-kfold-knn          Corre experimento completo K-Fold+KNN"
	@echo "  clean                   Borra outputs/runs"

install:
	pip install -e .

validate-holdout:
	$(AMK) validate --config configs/validate_holdout.yaml

validate-loo:
	$(AMK) validate --config configs/validate_loo.yaml

validate-kfold:
	$(AMK) validate --config configs/validate_kfold.yaml

classify-knn:
	$(AMK) classify --config configs/classify_knn_within_dataset_holdout.yaml

classify-euclidean:
	$(AMK) classify --config configs/classify_euclidean_train_test.yaml

run-holdout-knn:
	$(AMK) run --config configs/run_holdout_knn.yaml

run-loo-1nn:
	$(AMK) run --config configs/run_loo_1nn.yaml

run-kfold-knn:
	$(AMK) run --config configs/run_kfold_knn.yaml

clean:
	rm -rf outputs/runs
	@mkdir -p outputs/runs
