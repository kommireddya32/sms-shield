# SMS Shield

An experimental SMS spam detection project exploring character-level TF-IDF features, URL and message-structure features, text normalization, and adversarial evaluation.

## Project Structure

- `notebooks/` — Research notebooks for baseline modeling, URL features, normalization, and adversarial robustness.
- `sms-shield-fullstack/` — React frontend and FastAPI backend for the demonstration application.

## Research Workflow

1. Prepare and inspect the SMS dataset.
2. Train and evaluate baseline text-classification models.
3. Explore URL/structure features and obfuscation normalization.
4. Evaluate reviewed adversarial examples and compare model evasion rates.

## Running the Demo

Follow the frontend and backend setup instructions in `sms-shield-fullstack/`. Configure the required model paths and environment variables before starting the application.

## Research Limitations

The dataset used for the main classification experiments does not contain timestamps, so those experiments do not establish temporal performance. Adversarial results are based on a small, curated evaluation set and should be treated as exploratory.

## Technologies

Python, scikit-learn, Jupyter Notebook, React, and FastAPI.
