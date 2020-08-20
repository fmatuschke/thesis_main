default: install

VENV := $(if $(venv),$(venv),env)
PYTHON=$(VENV)/bin/python3
PIP=$(VENV)/bin/python3 -m pip
BUILD=thesis

FLAG.install=--system-site-packages
FLAG.install-sc=--system-site-packages

.PHONY: install
install: env env-update requirements git-submodules clean-fastpli fastpli jupyter

.PHONY: install-sc
install-sc: env-sc env-update requirements-sc git-submodules clean-fastpli fastpli

.PHONY: env
env:
	python3 -m venv $(VENV)

.PHONY: env-sc
env-sc:
	python3 -m venv --system-site-packages $(VENV)

.PHONY: env-update
env-update:
	$(PIP) install --upgrade pip -q

.PHONY: requirements
requirements:
	$(PIP) install -r requirements.txt -q
	$(PIP) install 0_core/. -q

.PHONY: requirements-sc
requirements-sc:
	$(PIP) install -r requirements-sc.txt -q
	$(PIP) install 0_core/. -q	

.PHONY: git-submodules
git-submodules:
	git submodule update --init --recursive

.PHONY: git-submodules-pull
git-submodules-pull:
	cd fastpli
	git checkout development
	git pull
	cd ..
	cd fastpli_paper
	git checkout master
	git pull
	cd ..
	cd fastpli_wiki
	git checkout master
	git pull
	cd ..
	cd thesis
	git checkout master
	git pull

.PHONY: fastpli/setup
.ONESHELL:
fastpli/setup: fastpli/
	cd fastpli
	# rm -r build/
	make BUILD=$(BUILD) fastpli

.PHONY: fastpli
fastpli: fastpli/setup
	$(PIP) uninstall fastpli -y -q
	$(PIP) install fastpli/.

.PHONY: fastpli-
fastpli-: clean-fastpli fastpli

.PHONY: jupyter
jupyter:
	@if ! $(PIP) freeze | grep jupyterlab= -q; then \
		echo "Install Jupyter Notebook"; \
		$(PIP) install jupyter -q; \
		$(PIP) install jupyterthemes -q; \
		$(PIP) install jupyter_contrib_nbextensions -q; \
		$(VENV)/bin/jupyter contrib nbextension install --user; \
		$(VENV)/bin/jt -t onedork -cellw 99% -T -nf ptsans -lineh 125; \
		cp custom.css ~/.jupyter/custom/custom.css; \
	fi

# CLEANING
.PHONY: clean
clean: clean-fastpli
	rm -rf env-*

.PHONY: clean-fastpli
.ONESHELL:
clean-fastpli:
	cd fastpli
	make clean
