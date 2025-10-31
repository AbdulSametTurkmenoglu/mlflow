# End-to-End Text Classification Pipeline with MLflow

A complete MLOps workflow demonstrating how to train a text classification model, select the best model, register it in MLflow Model Registry, and serve it as a REST API using MLflow.

The project trains a model (TF-IDF + Logistic Regression) on 42,000 Turkish news articles and simulates CI/CD processes by moving the model through "Staging" and "Production" stages.

##  Key Features

- **Data Preparation**: Text cleaning (lowercasing, removing punctuation and numbers)
- **Custom `pyfunc` Model**: Packages `TfidfVectorizer` and `LogisticRegression` in a single `mlflow.pyfunc.PythonModel` class for automatic preprocessing and prediction
- **Hyperparameter Optimization**: Grid search with nested MLflow runs using `ParameterGrid` to find optimal parameters (C, max_features, ngram_range)
- **Model Registry**: Automatically registers the best model (highest F1 score) to MLflow Model Registry
- **CI/CD Simulation**:
  - **Staging (CI)**: Automatically promotes best model to "Staging" and validates against quality threshold (F1 > 0.60)
  - **Production (CD)**: After successful staging validation, moves model to "Production" with manual approval simulation
- **Model Server**: Serves the "Production" model as a local REST API (port 5001)
- **API Testing**: Live predictions via `requests` library

##  Technology Stack

- **MLflow**: Experiment tracking, model management, registry, and serving
- **Scikit-learn**: `TfidfVectorizer` and `LogisticRegression`
- **Pandas**: Data processing
- **SHAP**: Model explainability (integration included)

##  Prerequisites

- Python 3.8+
- Dataset: 42K Turkish news articles in `.txt` format

##  Installation
```bash
pip install mlflow pandas scikit-learn numpy shap plotly matplotlib requests
```

Ensure your data is in `veri/42bin_haber/news` directory structure with `.txt` files accessible to the notebook.

##  Workflow Pipeline

The project runs the following steps sequentially via `ml_floww.ipynb` notebook:

1. **Data Loading & Preparation**: Reads `.txt` files from `veri/42bin_haber/news`, cleans text, splits into train/test sets

2. **Custom Model Class**: Defines `TextClassifierPyfunc` class that takes `metin` field as input and performs cleaning, TF-IDF transformation, and classification

3. **Grid Search (Parent Run)**: Starts a parent run under `Haber_Siniflandirma_Custom_Pyfunc_GridSearch_v1` experiment
   - **Training (Child Runs)**: Creates a child run for each parameter combination in `param_grid`
   - Logs parameters, metrics, and `pyfunc` model for each run
   - Saves best run ID (`best_run_id`) with highest F1 score

4. **Model Registration & Staging**:
   - Registers best model as `Haber_Siniflandirici_Pyfunc_Prod` in Model Registry
   - Automatically transitions to "Staging" stage

5. **Staging Validation (CI Simulation)**:
   - Loads model from `models:/Haber_Siniflandirici_Pyfunc_Prod/Staging`
   - Recalculates F1 score on test set
   - If F1 > 0.60 threshold, adds "CI/CD Validation PASSED" comment

6. **Production Promotion (CD Simulation)**:
   - Simulates "Team Lead Review"
   - Checks CI comments
   - With `human_approval = True`, promotes model to "Production"

7. **Model Server Launch**: Displays command to serve "Production" model

8. **API Testing (Inference)**: Sends test data to local server (port 5001) and retrieves predictions

##  Usage

### Step 1: Start MLflow UI (Recommended)

Open a terminal in the directory where `mlruns` will be created:
```bash
mlflow ui
```

Access the UI at `http://127.0.0.1:5000`

### Step 2: Run the Notebook

Execute all cells in `ml_floww.ipynb` **sequentially**. Cells 1-5 will train, select, and promote the model to "Production".

### Step 3: Start Model Server

Following instructions in cell 6, open a **separate terminal** and run:
```bash
mlflow models serve -m "models:/Haber_Siniflandirici_Pyfunc_Prod/Production" --port 5001 --no-conda
```

Wait for `Listening at: http://127.0.0.1:5001` message.

### Step 4: Test the API

Run the final cell (Section 2) in the notebook to send requests and get predictions.

##  API Request Example
```python
import requests
import json

test_data_json = {
    "dataframe_split": {
        "columns": ["metin"],
        "data": [
            ["Galatasaray şampiyonluk maçına çıkıyor, tüm biletler satıldı"],
            ["Enflasyon rakamları açıklandı, merkez bankası faiz kararı bekleniyor"],
            ["Yapay zeka modelleri metin özetlemede çığır açtı"]
        ]
    }
}

url = "http://127.0.0.1:5001/invocations"

response = requests.post(
    url,
    data=json.dumps(test_data_json),
    headers={"Content-Type": "application/json"}
)

print(json.dumps(response.json(), indent=2, ensure_ascii=False))
```

### Sample Output
```json
{
  "predictions": [
    {
      "input_text": "Galatasaray şampiyonluk maçına çıkıyor, tüm biletler satıldı",
      "predicted_category": "spor"
    },
    {
      "input_text": "Enflasyon rakamları açıklandı, merkez bankası faiz kararı bekleniyor",
      "predicted_category": "genel"
    },
    {
      "input_text": "Yapay zeka modelleri metin özetlemede çığır açtı",
      "predicted_category": "guncel"
    }
  ]
}
```


##  MLOps Best Practices Demonstrated

- **Experiment Tracking**: All runs logged with parameters and metrics
- **Model Versioning**: Automatic versioning in Model Registry
- **Staged Deployments**: Staging → Production workflow
- **Quality Gates**: Automated validation with F1 threshold
- **Model Serving**: Production-ready REST API
- **Reproducibility**: All artifacts and metadata tracked
