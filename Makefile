PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
.PHONY: test eval eval-sync eval-check run docker-build docker-run
test:
	$(PYTHON) -m pytest -q
eval:
	$(PYTHON) -m eval.run_eval
eval-sync:
	$(PYTHON) -m eval.sync_readme
eval-check:
	$(PYTHON) -m eval.sync_readme --check
run:
	$(PYTHON) -m uvicorn finpilot.api:app --reload
docker-build:
	docker build -t finpilot .
docker-run:
	docker run --rm -p 8000:8000 -e FINPILOT_AUTOSEED=1 $(if $(MESH_API_KEY),-e MESH_API_KEY=$(MESH_API_KEY)) finpilot
