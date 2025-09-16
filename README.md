# Sentiment analyzer: Movie reviews

## Repo structure
`train-and-export-models.py` contains code that trains 3 different models on the dataset and exports them as `.joblib` files.

`stmUIForAllModel.py` is the frontend code (using Streamlit). It loads 3 NLP models from the 3 `.joblib` files located in the same directory. `requirements.txt` contains version information of all dependencies, and is required by Streamlit to run the frontend.

## Run
First, run `train-and-export-models.py` to export the `.joblib` files for all three models. (Note: `train-and-export-models.ipynb` is exactly the same as `train-and-export-models.py` except it's in Interactive Python Notebook format.)

- sentiment_analysis_pipeline.joblib (Multinomial Naives-Bayes)
- logistic_regression_pipeline.joblib (Logistic Regression)
- linear_svm_pipeline.joblib (Linear SVM)

Now that all three `.joblib` files have been exported and in the same directory as `stmUIForAllModel.py`, which is our frontend code, run it with:

```sh
streamlit run stmUIForAllModel.py
```

(Note: make sure you have `streamlit` installed in your current Python environment)

The frontend should now be served locally and the Movie Review Semantic Analyzer web app automatically opened in your default browser.