OK_MSG = \x1b[32m ✔\x1b[0m

clean: ## Clean autogenerated files
	rm -rf ./*.log
	rm -rf dist
	sudo rm -rf ./outputs/*
	# sudo rm -rf ./data
	sudo rm -rf ./logs
	sudo rm -rf ./remotefs
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -f .coverage

format:
	@echo -n "==> Checking that code is autoformatted with black..."
	@black --check --quiet .
	@echo -e "$(OK_MSG)"

train:
	git init .
	dvc init
	dlearn_train -m hydra/launcher=joblib hydra.launcher.n_jobs=1 experiment=s6 model.patch_size=1,2,4,8,16
	git add .
	dvc add data logs outputs
	git add *.dvc
	dvc remote add -d local ./remotefs
	dvc push -r local
	cd logs; aim up --host 0.0.0.0