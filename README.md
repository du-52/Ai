# Sentiment analyzer: Movie reviews

## Repo structure
The dataset is located in the [`aclImdb/`](aclImdb/) directory. Check `aclImdb/README.md` for more info regarding the dataset.

`train-and-export-models.py` contains code that trains 3 different models on the dataset and exports them as `.joblib` files.

`stmUIForAllModel.py` is the frontend code (using Streamlit). It loads 3 NLP models from the 3 `.joblib` files located in the same directory. `requirements.txt` contains version information of all dependencies, and is required by Streamlit to run the frontend.

## Run
Run with `streamlit run stmUIForAllModel.py`. Make sure all 3 `.joblib` files and the `requirements.txt` are in the same directory as `stmUIForAllModel.py`.