# 🕵️ Spot The Scam - Job Fraud Detector

A machine learning-powered tool to detect fraudulent job postings using advanced NLP techniques and model explainability.

google drive link of ppt presentation and video discussion link👇👇👇👇👇
https://drive.google.com/drive/folders/1cDdrq97c_L0FUr4ncmnUn9W3jgGEDw94

## 📋 Overview

Spot The Scam helps job seekers and recruiters identify potentially fraudulent job postings by analyzing various aspects of job listings, including:
- Job description and requirements
- Company information
- Salary details
- Location data
- Communication patterns

The system uses machine learning models trained on a dataset of known legitimate and fraudulent job postings to provide risk assessments and detailed explanations.

## ✨ Features

### 🔍 Analysis Capabilities
- Batch processing of job listings via CSV upload
- Real-time single job posting analysis
- Fraud probability scoring
- Risk categorization (High, Medium, Low)
- Detailed feature importance analysis

### 📊 Visualizations
- Interactive fraud probability distribution
- Feature importance plots
- SHAP (SHapley Additive exPlanations) analysis
- Correlation matrix of fraud indicators
- Risk distribution charts

### 🤖 ML Pipeline
- Text preprocessing and cleaning
- Advanced feature engineering
- XGBoost classification model
- Model explainability using SHAP
- Cross-validation and performance metrics

### 🌐 Deployment Options
- Streamlit web dashboard
- FastAPI REST API (optional)
- Batch processing capabilities
- Model retraining pipeline

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/spot-the-scam.git
cd spot-the-scam
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required language models:
```bash
python -m spacy download en_core_web_sm
python -m nltk.downloader stopwords
```

### Running the Application

1. Start the Streamlit dashboard:
```bash
streamlit run dashboard/app.py
```

2. (Optional) Start the FastAPI server:
```bash
uvicorn api.app:app --reload
```

## 📊 Usage

### Dashboard Interface

1. **Upload Data**:
   - Prepare your job listings in CSV format
   - Required columns: title, description
   - Optional columns: location, company_profile, salary

2. **Analysis**:
   - View fraud probability scores
   - Examine risk categories
   - Explore feature importance
   - Generate SHAP explanations

3. **Export**:
   - Download analysis results
   - Save visualization plots
   - Export detailed reports

### API Endpoints (Optional)

```python
# Predict single job posting
POST /predict
{
    "title": "Data Entry Position",
    "description": "Work from home opportunity...",
    "company_profile": "ABC Corp",
    "location": "Remote"
}

# Batch prediction
POST /predict_batch
{
    "jobs": [
        {
            "title": "...",
            "description": "..."
        }
    ]
}
```

## 📁 Project Structure

```
spot-the-scam/
├── data/               # Dataset storage
├── pipeline/           # ML pipeline components
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model.py
│   └── explainability.py
├── dashboard/          # Streamlit web interface
│   ├── app.py
│   └── assets/
├── api/               # FastAPI service (optional)
│   └── app.py
├── models/            # Trained model storage
├── tests/            # Unit tests
├── requirements.txt   # Dependencies
└── README.md
```

## 🛠️ Model Training

To train a new model:

1. Prepare your training data in CSV format
2. Run the training pipeline:
```bash
python pipeline/model.py --data path/to/training_data.csv --output models/
```

3. The trained model will be saved in the `models/` directory

## 📈 Performance Metrics

The model is evaluated using:
- F1 Score
- Precision
- Recall
- ROC-AUC
- Confusion Matrix

Current model performance:
- F1 Score: 0.92
- Precision: 0.89
- Recall: 0.95
- ROC-AUC: 0.96

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- SHAP library for model explanations
- Streamlit for the web interface
- XGBoost team for the gradient boosting framework
- spaCy and NLTK for NLP capabilities

## 📧 Contact

Your Name - Chaitnaya and Harsh Kumar - email-mulechaitu3@gmail.com and sharmaharsh9708@gmail.com.

Project Link: https://github.com/Chaitureddy1606/spot-the-scam
