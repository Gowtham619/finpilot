.PHONY: test eval run
test:
	pytest -q
eval:
	python -m eval.run_eval
run:
	uvicorn finpilot.api:app --reload
