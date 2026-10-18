PYTHON ?= python3
.PHONY: test eval run
test:
	$(PYTHON) -m pytest -q
eval:
	$(PYTHON) -m eval.run_eval
run:
	$(PYTHON) -m uvicorn finpilot.api:app --reload
