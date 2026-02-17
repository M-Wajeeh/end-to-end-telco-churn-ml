# End-to-End Telco Churn ML

An enterprise-grade MLOps project for predicting telecom customer churn using machine learning, with containerized deployment, model tracking, and real-time inference capabilities.

## 🎯 Project Overview

This project demonstrates a complete machine learning pipeline from data preprocessing to production deployment:

- **Data Processing**: Feature engineering and preprocessing pipelines
- **Model Training**: XGBoost-based churn prediction with hyperparameter tuning
- **Experiment Tracking**: MLflow integration for reproducible ML experiments
- **Containerization**: Docker support for consistent deployments
- **CI/CD Pipeline**: GitHub Actions for automated testing and building
- **API Service**: FastAPI application for real-time predictions
- **Data Validation**: Great Expectations for data quality checks

## 📁 Project Structure

```
.
├── app/                          # Web UI and API application
├── artifacts/                    # Model artifacts and outputs
├── configs/                      # Configuration files
├── data/
│   ├── raw/                      # Original dataset
│   ├── processed/                # Preprocessed data
│   └── external/                 # External reference data
├── mlruns/                       # MLflow experiment tracking
├── notebooks/                    # Jupyter notebooks for EDA
├── scripts/
│   ├── prepared_data.py          # Data preparation script
│   └── run_pipeline.py           # End-to-end pipeline runner
├── src/
│   ├── app/
│   │   └── main.py              # FastAPI application
│   ├── data/
│   │   ├── load_data.py         # Data loading utilities
│   │   └── preprocessing.py     # Data preprocessing
│   ├── features/
│   │   └── build_features.py    # Feature engineering
│   ├── models/
│   │   ├── train.py             # Model training
│   │   ├── tune.py              # Hyperparameter tuning
│   │   └── evaluate.py          # Model evaluation
│   ├── serving/
│   │   └── inference.py         # Inference logic
│   └── utils/
│       └── validate_data.py     # Data validation
├── tests/                        # Unit and integration tests
├── .github/workflows/            # CI/CD workflows
├── dockerfile                    # Docker configuration
├── .dockerignore                 # Docker ignore rules
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Docker & Docker Compose (optional)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/M-Wajeeh/end-to-end-telco-churn-ml.git
   cd end-to-end-telco-churn-ml
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Data

The project uses a telecom customer churn dataset with features like:
- Customer demographics (age, gender)
- Account information (tenure, contract type)
- Service usage (internet, phone, streaming)
- Billing information (charges, payment method)
- Target: Churn (Yes/No)

**Dataset locations:**
- Raw: `data/raw/Dataset.csv`
- Processed: `data/processed/Dataset_processed.csv`

## 🔄 Pipeline Execution

### Run the full pipeline
```bash
python scripts/run_pipeline.py
```

### Or run individual steps

**Data Preparation:**
```bash
python scripts/prepared_data.py
```

**Model Training:**
```bash
python src/models/train.py
```

**Hyperparameter Tuning:**
```bash
python src/models/tune.py
```

**Model Evaluation:**
```bash
python src/models/evaluate.py
```

## 🤖 Model Training

The project uses XGBoost for churn prediction with:
- Automated hyperparameter tuning
- Cross-validation for robust evaluation
- MLflow integration for experiment tracking
- Model persistence and versioning

**Key Metrics:**
- F1-Score
- Precision & Recall
- ROC-AUC
- Training & Prediction Time

## 🎛️ Experiment Tracking

MLflow tracks all experiments with:
- Hyperparameters
- Metrics
- Model artifacts
- Run metadata

View experiments:
```bash
mlflow ui
```

## 🐳 Docker Deployment

### Build Docker Image
```bash
docker build -t telco-churn-ml:latest .
```

### Run Container
```bash
docker run -p 8000:8000 telco-churn-ml:latest
```

### Using Docker Compose
```bash
docker-compose up
```

## 🌐 FastAPI Application

The application provides real-time inference endpoints:

**Start the server:**
```bash
python src/app/main.py
```

**API will be available at:** `http://localhost:8000`

**Interactive API docs:** `http://localhost:8000/docs`

## ✅ Testing

Run the test suite:
```bash
pytest tests/ -v --cov=src
```

## 🔄 CI/CD Pipeline

GitHub Actions workflows:
- **CI Pipeline** (`.github/workflows/ci.yaml`):
  - Python linting with flake8
  - Unit tests with pytest
  - Docker image building
  - Code coverage reporting
  
- **Code Quality**:
  - Black code formatting
  - isort import sorting
  - Pylint static analysis

Workflows trigger on:
- Push to `main` or `develop`
- Pull requests

## 📋 Data Validation

Great Expectations integration for:
- Data quality checks
- Schema validation
- Statistical profiling
- Automated test suites

## 🛠️ Configuration

Configuration files are located in `configs/` directory for:
- Data processing parameters
- Model hyperparameters
- Feature engineering settings
- Validation rules

## 📈 Performance Metrics

Latest model performance metrics are stored in `artifacts/` and tracked in MLflow.

## 🤝 Contributing

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Commit changes: `git commit -am 'Add feature'`
3. Push to branch: `git push origin feature/your-feature`
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the project maintainer.

## 🙏 Acknowledgments

- Dataset source: [Telecom Churn Dataset]
- MLflow: Experiment tracking and model registry
- XGBoost: Gradient boosting framework
- FastAPI: Modern Python web framework
