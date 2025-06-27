.PHONY: dirs scrape persist run test

# 1. Ensure data/processed directory exists
dirs:
	python -c "import os; os.makedirs('data/processed', exist_ok=True)"

# 2. Scrape SEP into data/sep.db
scrape: dirs
	python -m philo_topic_modeling.scraper

# 3. Fit & save TF–IDF + topic models
persist: dirs
	python scripts/persist.py

# 4. Run the Streamlit app
run:
	streamlit run philo_topic_modeling/app.py

# 5. Run tests
test:
	pytest --maxfail=1 --disable-warnings -q
