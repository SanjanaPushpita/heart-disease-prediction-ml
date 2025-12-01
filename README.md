# Heart Disease Prediction Notebook

An end-to-end Google Colab-friendly workflow for exploring the UCI Heart Disease dataset, cleaning and preprocessing the data, training multiple neural networks and tree-based baselines, and comparing their performance through clear visualizations and summary tables.

## Key Features
- **Data acquisition helpers** for KaggleHub downloads or manual CSV uploads.
- **Comprehensive preprocessing pipeline**: duplicate removal, median/mode imputation, one-hot encoding, and feature scaling.
- **Multiple model families**: three regularized multilayer perceptrons plus logistic regression, random forest, gradient boosting, and XGBoost baselines.
- **Rich evaluation tooling**: confusion matrices, ROC curves, learning curves, and consolidated metrics tables including precision/recall/F1.
- **Best-model snapshot** that highlights the top-performing configuration at a glance.

## Dataset
- Source: [Kaggle – Heart Disease Data](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
- Primary file used: `heart_disease_uci.csv` (renamed to `data/heart.csv` in the notebook for consistency).

## Running the Notebook
1. Open `heart_disease_prediction.ipynb` in Google Colab (recommended) or VS Code with the Jupyter extension.
2. Execute the runtime setup cell to install dependencies (`pandas`, `numpy`, `tensorflow`, `scikit-learn`, `xgboost`, etc.).
3. Choose one dataset option in Section 2:
   - **Option A**: Download directly with KaggleHub (no API token required for public datasets).
   - **Option B**: Upload the `heart.csv` file manually when prompted.
4. Run Sections 3–6 to configure imports, inspect the data, and build the feature matrix.
5. Execute Section 8 to train the neural networks, followed by Section 9 to view learning curves.
6. Run Section 10 to train the classical baselines and continue through Sections 11–13 for evaluation plots and consolidated metrics.

## Repository Structure
```
heart_disease_prediction/
├── heart_disease_prediction.ipynb  # Main analysis and modeling notebook
├── README.md                       # Project overview and usage guide
└── LICENSE                         # MIT License
```

## Author
Created by **Maliha Sanjana**.

## License
This project is released under the MIT License. See [LICENSE](LICENSE) for details.
