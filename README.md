# 🧠 Disease Prediction Toolkit

A beginner-friendly machine learning project that predicts diseases from healthcare datasets using **Logistic Regression**, **Decision Tree**, and **Random Forest** models.  
This project follows the AI/ML Bootcamp guidelines to build, evaluate, and document a professional ML pipeline for healthcare.

---

## 📌 Project Overview
- Learn machine learning basics with healthcare data.
- Preprocess datasets (handle missing values, encode categoricals, scale numerics).
- Train and compare ML models for disease prediction.
- Evaluate with metrics: Accuracy, Precision, Recall, F1, ROC-AUC.
- Visualize results with Confusion Matrices and ROC Curves.
- Document results with a professional GitHub repo, demo video, and 5-slide presentation.

---

## 🗂️ Datasets
You can use:
- ✅ **Breast Cancer Dataset** (built into scikit-learn — runs instantly in Colab).

⚙️ Installation

Run this project on Google Colab (recommended).
For local setup, install dependencies:

pip install -r requirements.txt

🚀 How to Run

Clone this repo:

git clone https://github.com/<your-username>/Disease-Predictor.git
cd Disease-Predictor

Open notebooks/Disease_Predictor.ipynb in Google Colab or Jupyter.
Run all cells step by step.
Check model results (metrics + plots).
The best model will be saved as best_model.joblib.
Try predictions with new patient data.

🛠️ Technologies Used

Python 3.x NumPy, Pandas → Data handling 
Scikit-learn → Machine learning algorithms
Matplotlib, Seaborn → Visualization 
Joblib → Model persistence

📊 Model Evaluation

We evaluate models using:

Accuracy
Precision
Recall
F1-score
ROC-AUC
Confusion Matrix
ROC Curve

Sample Results (Breast Cancer dataset):

Model	                 Accuracy Precision	Recall	  F1	   ROC-AUC
Logistic Regression	      95%      96%	   94%	   95%   	  0.99
Decision Tree             91%	     92%	   90%	   91%	    0.92
Random Forest	            97%	     98%	   96%	   97%	    0.99

📂 Repository Structure
Disease-Predictor/
│
├── notebooks/
│   └── Disease_Predictor.ipynb    # Colab-ready notebook
│
├── models/
│   └── best_model.joblib          # Saved Random Forest model
│
├── presentation/
│   └── Disease_Predictor_Presentation_With_Flowchart.pptx
│
├── test_predictions.csv           # Sample predictions
├── README.md                      # Documentation
├── requirements.txt               # Dependencies
└── demo_script.txt                # 30s demo narration


🎥 Demo
🎬 30-second demo video
👉 Show dataset load, preprocessing, model training, evaluation plots, and a single prediction.

🔮 Future Work
Train on multiple healthcare datasets
Handle class imbalance (e.g., SMOTE)
Add model explainability (SHAP, LIME)
Deploy as an API (FastAPI/Flask)


