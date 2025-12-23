.PHONY: help run-hgb run-objective run-catboost run-lightgbm

PY ?= python
DATA_DIR ?= $(CURDIR)/data/raw

help:
	@echo "Targets:"
	@echo "  make run-hgb        # run HGB baseline"
	@echo "  make run-objective  # run objective-weight model"
	@echo "  make run-catboost   # run CatBoost baseline (needs catboost)"
	@echo "  make run-lightgbm   # run LightGBM baseline (needs lightgbm)"
	@echo ""
	@echo "Env vars:"
	@echo "  DATA_DIR=/path/to/data/raw (default: $(DATA_DIR))"

run-hgb:
	@DATA_DIR="$(DATA_DIR)" $(PY) src/baseline_hgb.py

run-objective:
	@DATA_DIR="$(DATA_DIR)" $(PY) src/objective_weight_model.py

run-catboost:
	@DATA_DIR="$(DATA_DIR)" $(PY) src/cat_lgbm_models.py --model catboost

run-lightgbm:
	@DATA_DIR="$(DATA_DIR)" $(PY) src/cat_lgbm_models.py --model lightgbm

