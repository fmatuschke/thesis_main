default: install

VENV := $(if $(venv),$(venv),env)
PYTHON=$(VENV)/bin/python3
PIP=$(VENV)/bin/python3 -m pip

.PHONY: install
install: env env-update requirements git-submodules fastpli

# ENV
env: $(VENV)/bin/python3

$(VENV)/bin/python3:
	python3 -m venv $(VENV)

.PHONY: env-update
env-update: env
	$(PIP) install --upgrade pip -q

.PHONY: requirements
requirements: env	
	$(PIP) install -r requirements.txt -q
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
	rm -r build/
	make fastpli

.PHONY: fastpli
fastpli: env env-update fastpli/setup
	$(PIP) uninstall fastpli -y -q
	$(PIP) install fastpli/.

.PHONY: jupyter
jupyter: env
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
clean:
	rm -rf env-*

#.PHONY: clean-fastpli
#.ONESHELL:
#clean-fastpli:
#	cd fastpli
#	make clean
