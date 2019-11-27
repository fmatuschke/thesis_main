default: install

HOST=$(shell hostname)
PYTHON=env-$(HOST)/bin/python3
PIP=env-$(HOST)/bin/pip3

.PHONY: install
install: env env-update fastpli

# ENV
env: env-$(HOST)/bin/python3

env-$(HOST)/bin/python3:
	python3 -m venv env-$(HOST)/

.PHONY: env-update
env-update: env
	$(PIP) install --upgrade pip -q
	$(PIP) install -r requirements.txt -q

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
	rm -r build/
	git pull
	cd ..
	git add fastpli

.PHONY: fastpli/build
.ONESHELL:
fastpli/build: fastpli/
	cd fastpli
	# make build
	make BUILD=info build

.PHONY: fastpli
fastpli: env fastpli/ git-submodules clean-build fastpli/build
	$(PIP) uninstall fastpli -y
	$(PIP) install fastpli/build/. -q

# CLEANING
.PHONY: clean
clean: clean-env clean-build

.PHONY: clean-env
clean-env:
	rm -rf env-*

.PHONY: clean-build
clean-build:
	rm -rf fastpli/build
