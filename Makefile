.PHONY: scrape run test

scrape:
	python -m philo_topic_modeling.scraper

run:
	streamlit run philo_topic_modeling/app.py

test:
	pytest --maxfail=1 --disable-warnings -q
