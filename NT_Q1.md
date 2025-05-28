AI Community JE Assignment

NAME: P Manthan    DPT : MEMS    ROLL NO : 24B2511

---

**Non-Technical Questions**

**2.1 NT Q1: Hackathorn Preparation Timeline**

**1-Month AI Hackathon Preparation Plan**  
**Week 1: Problem Selection & Quantitative Formulation**

**Day 1-2: Problem Identification**

* Brainstorming Session: Discuss trending AI domains (NLP, CV, Time Series, etc.).  
* Evaluate Feasibility:  
  * Does the problem have measurable success metrics?  
  * Is quality data available?  
  * Can it be solved in the hackathon timeframe?  
* Finalize Problem Statement:  
  * Example: "Predict customer churn using transaction history."  
  * Define success metrics (Accuracy, F1-score, AUC-ROC).

**Day 3-4: Quantitative Formulation**

* Define Inputs & Outputs:  
  * Input: Structured (CSV, SQL) vs. Unstructured (Images, Text).  
  * Output: Classification, Regression, or Generation.  
* Baseline Metrics:  
  * If existing solutions exist, note benchmark scores.  
  * Example: "Current churn models achieve 85% accuracy."

**Day 5-7: Data Sourcing & Initial Exploration**

* Find Datasets:  
  * Kaggle, UCI, Google Dataset Search, APIs (Twitter, Reddit).  
  * Synthetic data generation if real data is scarce.  
* Quick EDA:  
  * Check missing values, class imbalance, outliers.  
  * Use pandas\_profiling or Sweetviz for automated reports.

**Week 2: Data Preprocessing & EDA**

**Day 8-10: Data Cleaning**

* Handle Missing Data: Imputation (mean/median) or removal.  
* Feature Engineering:  
  * Normalization (MinMax, StandardScaler).  
  * Text: TF-IDF, BERT embeddings.  
  * Images: Augmentation (flips, rotations).

**Day 11-12: Exploratory Data Analysis (EDA)**

* Visualizations:  
  * Correlation heatmaps, distribution plots.  
  * NLP: Word clouds, topic modeling (LDA).  
  * CV: Sample images with annotations.  
* Insights:  
  * Identify key features influencing the target variable.

**Day 13-14: Baseline Model**

* Quick Prototyping:  
  * Use AutoML (PyCaret, H2O) or simple scikit-learn models (Logistic Regression, Random Forest).  
  * Compare against dummy baselines (majority class, mean prediction).

**Week 3: Model Development & Optimization**

**Day 15-17: Model Selection**

* Algorithm Shortlist:  
  * Tabular Data: XGBoost, LightGBM, CatBoost.  
  * NLP: BERT, GPT-3 (if API access).  
  * CV: ResNet, EfficientNet.  
* Compute Requirements:  
  * Use free tiers (Google Colab, Kaggle GPUs).  
  * Optimize for speed (quantization, mixed precision).

**Day 18-20: Training & Validation**

* Cross-Validation: Stratified K-Fold to prevent leakage.  
* Hyperparameter Tuning:  
  * Bayesian Optimization (Optuna) over Grid Search.  
* Track Experiments: Weights & Biases (W\&B) or MLflow.

**Day 21: Ensemble & Explainability**

* Blending Models: Stacking or weighted averaging.  
* Explainability: SHAP, LIME for stakeholder trust.

**Week 4: Deployment & Risk Mitigation**

**Day 22-23: Deployment Prep**

* Minimal Viable Product (MVP):  
  * FastAPI backend \+ Streamlit frontend.  
  * Dockerize for reproducibility.  
* Edge Cases:  
  * Test model on noisy/out-of-distribution data.

**Day 24-25: Presentation & Storytelling**

* Slide Deck:  
  * Problem → Solution → Impact.  
  * Visuals: Confusion matrices, ROC curves.  
* Demo Video: Record a 2-min screencast.

**Day 26-28: Contingency Planning**

* Backup Models: Keep a simpler model if primary fails.  
* Data Leakage Check: Ensure no train-test contamination.  
* Team Roles:  
  * Lead: Coordinates deadlines.  
  * Data Engineer: Handles preprocessing.  
  * ML Engineer: Optimizes models.  
  * Frontend Dev: Builds demo.

| Day | Milestone |
| :---- | :---- |
| 7 | Finalized problem statement \+ dataset |
| 14 | EDA report \+ baseline model |
| 21 | Optimized model \+ validation results |
| 28 | Deployed demo \+ presentation |

**Key Deliverables**Risks & Mitigation:

* Data Quality Issues: Have backup datasets.  
* Model Underperformance: Use AutoML fallbacks.  
* Last-Minute Bugs: Test deployment early.

This structured approach balances speed and rigor, ensuring a competition-ready solution in 4 weeks. 

