.PHONY: help test fixture weights run run-gpu-image clean validate

SERVICE_NAME ?= pose-estimate
PYTHON ?= python3
VALIDATE_PYTHON ?= $(PYTHON)
export DOCKER_DEFAULT_PLATFORM = linux/amd64

.DEFAULT: help

help:
	@echo "Make Help for $(SERVICE_NAME)"
	@echo ""
	@echo "make weights   - download the SuperAnimal weights (checksummed) into the local DeepLabCut install"
	@echo "make test      - pytest: synthetic blob video + 2 s of real open-field video through the processor (no Docker, ~3 min on CPU)"
	@echo "make fixture   - write the test input to data/fixture/{linked,flat}/"
	@echo "make run       - build the cpu image and run it in Docker over data/fixture/linked/, writing to data/output/"
	@echo "make run-gpu-image - build the gpu (CUDA) image and run it without a GPU (checks the CPU fallback)"
	@echo "make validate  - app.yml against the manifest schema, CITATION.cff with cffconvert"
	@echo "make clean     - remove data/output/"
	@echo ""
	@echo "PYTHON=$(PYTHON) (set PYTHON=.venv/bin/python after pip install -r requirements-dev.txt)"

weights:
	$(PYTHON) -m processor.weights download

test:
	MPLBACKEND=Agg $(PYTHON) -m pytest -q tests

fixture:
	rm -rf data/fixture
	$(PYTHON) -m tests.make_fixture data/fixture --layout both

run: clean fixture
	mkdir -p data/output && chmod 777 data/output
	docker compose -f docker-compose.yml down --remove-orphans
	docker compose -f docker-compose.yml build
	docker compose -f docker-compose.yml up --exit-code-from processor
	docker compose -f docker-compose.yml down --remove-orphans
	@echo "--- outputs:"
	@find data/output -type f | sort

run-gpu-image: clean fixture
	mkdir -p data/output && chmod 777 data/output
	docker build --platform linux/amd64 --build-arg TORCH_FLAVOUR=cuda -t $(SERVICE_NAME):gpu .
	docker run --rm --platform linux/amd64 --network none --memory 4g --user 65534:65534 --read-only --tmpfs /tmp:rw,size=2g,mode=1777 \
		-e HOME=/tmp/w/home -e TMPDIR=/tmp/w/tmp -e INPUT_DIR=/tmp/s/input -e OUTPUT_DIR=/tmp/s/output \
		--env-file dev.env -e USE_GPU=auto -v $(CURDIR)/data/fixture/linked:/tmp/s/input:ro -v $(CURDIR)/data/output:/tmp/s/output \
		$(SERVICE_NAME):gpu

validate:
	$(VALIDATE_PYTHON) -c "import json, yaml, jsonschema; s = json.load(open('../workflow-systems-neuroscience/docs/app-manifest.v1.schema.json')); jsonschema.validate(yaml.safe_load(open('app.yml')), s); print('app.yml ok')"
	$(dir $(VALIDATE_PYTHON))cffconvert --validate && echo "CITATION.cff ok"

clean:
	rm -rf data/output
