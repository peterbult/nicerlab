.PHONY: install upgrade test docs

help:
	@echo "make options are:"
	@echo " * install  [install package]"
	@echo " * update   [update package installation]"
	@echo " * test     [run tests]"

install:
	pip install -r requirements.txt
	pip install .

upgrade:
	pip install . --upgrade

test:
	pytest

docs:
	make -C docs/ html
