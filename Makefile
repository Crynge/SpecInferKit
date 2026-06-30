.PHONY: install dev test lint clean build

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=specinferkit

lint:
	ruff check src/
	ruff format --check src/

format:
	ruff format src/

mypy:
	mypy src/ --ignore-missing-imports

clean:
	rm -rf build/ dist/ *.egg-info/ __pycache__/ .pytest_cache/
	find . -name "*.pyc" -delete

build:
	python -m build

docker:
	docker build -t specinferkit -f docker/Dockerfile .

run:
	specinferkit serve --checkpoint ./checkpoints/latest --port 8080
