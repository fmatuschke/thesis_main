default: install

VENV=env-$(shell hostname)
PYTHON=$(VENV)/bin/python3
PIP=$(VENV)/bin/pip3

.PHONY: install
install: env env-update fastpli

# ENV
env: $(VENV)/bin/python3

$(VENV)/bin/python3:
	python3 -m venv $(VENV)/

.PHONY: env-update
env-update: env
	$(PIP) install --upgrade pip -q
	$(PIP) install -r requirements.txt -q
	$(PIP) install 0_core/. -q

# FASTPLI
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

.PHONY: fastpli/setup
.ONESHELL:
fastpli/setup: fastpli/
	cd fastpli
	make fastpli

.PHONY: fastpli
fastpli: env git-submodules fastpli/setup
	$(PIP) uninstall fastpli -y -q
	$(PIP) install fastpli/.
	@echo "Done"

# CLEANING
.PHONY: clean
clean:
	rm -rf env-*

#.PHONY: clean-fastpli
#.ONESHELL:
#clean-fastpli:
#	cd fastpli
#	make clean
