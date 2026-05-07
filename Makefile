.PHONY: install dev test lint typecheck ingest docker-up docker-down clean

install:
	pip install -r requirements.txt

dev:
	uvicorn mcp_server.server:create_server --factory --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run dashboard/app.py

test:
	pytest evaluation/ -v --tb=short

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy agents/ mcp_server/ --ignore-missing-imports

ingest:
	python data/ingest.py --source data/sample_filings --collection finsight

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
