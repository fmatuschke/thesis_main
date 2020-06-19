default: install

VENV=env
PYTHON=$(VENV)/bin/python3
PIP=$(VENV)/bin/pip3

.PHONY: install
install: env env-update fastpli jupyter

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
