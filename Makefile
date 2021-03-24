default: install

VENV := $(if $(venv),$(venv),env)
PYTHON=python3.8
PIP=$(VENV)/bin/python3 -m pip
BUILD=release

FLAG.install=--system-site-packages
FLAG.install-sc=--system-site-packages

.PHONY: install
install: $(VENV) env-update requirements clean-fastpli fastpli

.PHONY: install-sc
install-sc: clean env-sc env-update requirements-sc clean-fastpli fastpli

$(VENV):
	$(PYTHON) -m venv $(VENV)

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

# .PHONY: git-submodules
# git-submodules:
# 	cd fastpli
# 	git checkout development
# 	git pull
# 	cd ..
# 	cd fastpli_paper
# 	git checkout master
# 	git pull
# 	cd ..
# 	cd fastpli_wiki
# 	git checkout master
# 	git pull
# 	cd ..
# 	cd thesis
# 	git checkout master
# 	git pull

.PHONY: fastpli/setup
.ONESHELL:
fastpli/setup: clean-fastpli
	cd fastpli
	make BUILD=$(BUILD) fastpli

.PHONY: fastpli
fastpli: fastpli/setup
	$(PIP) uninstall fastpli -y -q
	$(PIP) install fastpli/.

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
	rm -rf $(VENV)

.PHONY: clean-fastpli
.ONESHELL:
clean-fastpli:
	cd fastpli
	make clean

# SYNC
.PHONY: rsync
rsync:
	rsync -au --no-owner --no-group --info=progress2 --filter="- .git" /data/PLI-Group/felix/data/thesis /p/fastdata/pli/Projects/Felix/
