default: install

HOST=$(shell hostname)
PYTHON=env-$(HOST)/bin/python3
PIP=env-$(HOST)/bin/pip3

.PHONY: install
install: env fastpli

# ENV
env: env-$(HOST)/bin/python3
	$(PIP) install -r requirements.txt -q

env-$(HOST)/bin/python3:
	python3 -m venv env-$(HOST)/

.PHONY: env-update
env-update: env
	$(PIP) install --upgrade pip -q

# FASTPLI
fastpli/:
	git submodule add git@jugit.fz-juelich.de:f.matuschke/fastpli.git fastpli
	git submodule init

.PHONY: git-submodules
git-submodules:
	git submodule update --init --recursive

.PHONY: fastpli-pull
.Oneshell:
fastpli-pull:
	cd fastpli
	git pull origin development
	cd ..
	git add fastpli

.PHONY: fastpli/setup.py
.ONESHELL:
fastpli/setup.py: fastpli/
	cd fastpli
	make fastpli

.PHONY: fastpli
fastpli: env fastpli/ git-submodules fastpli/setup.py #clean-fastpli
	$(PIP) uninstall fastpli -y
	$(PIP) install fastpli/. -q
	@echo "Done"

# CLEANING
.PHONY: clean
clean: clean-env clean-fastpli

.PHONY: clean-env
clean-env:
	rm -rf env-*

.PHONY: clean-fastpli
.ONESHELL:
clean-fastpli:
	cd fastpli
	make clean
