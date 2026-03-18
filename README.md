# 🌍 Earthquake Pattern Analysis: Clustering, Forecasting & Machine Learning (1960–2023)

[![Research Paper](https://img.shields.io/badge/Research%20Paper-ResearchGate-00CCBB?style=for-the-badge&logo=researchgate&logoColor=white)](https://www.researchgate.net/publication/393362033_Earthquake_Pattern_Analysis_Using_Clustering_Forecasting_and_Machine_Learning_A_Global_Study_1960-2023)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

> **Published Research:** *Earthquake Pattern Analysis Using Clustering, Forecasting, and Machine Learning: A Global Study (1960–2023)*  
> 📄 [Read the full paper on ResearchGate →](https://www.researchgate.net/publication/393362033_Earthquake_Pattern_Analysis_Using_Clustering_Forecasting_and_Machine_Learning_A_Global_Study_1960-2023)

---

## 📌 Project Overview

This project performs a comprehensive global analysis of earthquake activity over six decades (1960–2023) using a combination of geospatial analysis, unsupervised clustering, time-series forecasting, and supervised machine learning. The goal is to identify seismic patterns, classify earthquake risk levels, and forecast future activity based on historical data.

The analysis is built on **28,113 cleaned earthquake records** sourced from the USGS global earthquake catalog.

---

## 🔬 Key Findings

| Finding | Result |
|---|---|
| Dataset size (after cleaning) | **28,113 records** |
| Depth range | **−4.0 km to 700.0 km** |
| Random Forest Accuracy (tuned) | **72.22%** |
| Decision Tree Accuracy (tuned) | **72.22%** |
| Cross-validation avg score | **0.54** |
| K-Means clusters identified | **5 seismic zones** |
| DBSCAN parameters | `eps=0.5`, `min_samples=10` |

---

## 🗂️ Project Structure

```
earthquake-analysis/
│
├── CODE.ipynb                        # Main Jupyter Notebook (full analysis)
├── CODE.html                         # Exported HTML view of the notebook
│
├── data/
│   └── earthquake_data_1960_2023.csv # Cleaned dataset
│
├── outputs/
│   ├── Earthquake_Heatmap.html       # Interactive Folium heatmap
│   └── figures/                      # Plots and visualizations
│
└── README.md
```

---

## 🧪 Methodology

### 1. Data Cleaning & Preprocessing
- Converted timestamps to datetime format; filtered records from **1960 to 2023**
- Removed rows with negative depth values and missing critical fields
- Final cleaned dataset: **28,113 records** (from original raw data)

### 2. Exploratory Data Analysis (EDA)
- Depth distribution analysis (min: −4 km, max: 700 km)
- Feature cardinality: 4 earthquake types, 54 location sources, 34 magnitude sources
- Seasonal analysis of average earthquake magnitudes by month

### 3. Geospatial Analysis
- Plotted global earthquake locations using **GeoPandas**
- Built an interactive magnitude heatmap using **Folium** → saved as `Earthquake_Heatmap.html`

### 4. Clustering Analysis
- **K-Means Clustering** — 5 clusters representing distinct seismic zones
- **DBSCAN** — Density-based clustering with `eps=0.5`, `min_samples=10`

### 5. Risk Classification
- Earthquakes classified into `Low`, `Moderate`, and `High` risk based on magnitude thresholds
- Applied **SMOTE** to address class imbalance before model training

### 6. Machine Learning Models

#### Random Forest Classifier
- Hyperparameter tuning via **Grid Search**
- Best params: `max_depth=3`, `min_samples_split=2`, `criterion=gini`
- **Final Accuracy: 72.22%**

#### Decision Tree Classifier
- Fine-tuned params: `max_depth=10`, `min_samples_split=10`, `criterion=entropy`
- **Final Accuracy: 72.22%**
- Extracted interpretable decision rules based on depth categories

### 7. Top Earthquake-Prone Locations
- Identified top 10 highest-probability seismic coordinates
- Example: coordinate `(37.1, −116.0)` — probability **0.000705**

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Folium |
| Geospatial | GeoPandas |
| Machine Learning | Scikit-learn (RandomForest, DecisionTree, GridSearchCV) |
| Clustering | K-Means, DBSCAN |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Environment | Jupyter Notebook |

---

## ⚙️ Installation & Usage

```bash
# Clone the repository
git clone https://github.com/harsh-stack/earthquake-pattern-analysis.git
cd earthquake-pattern-analysis

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook CODE.ipynb
```

### Requirements (`requirements.txt`)
```
pandas
numpy
matplotlib
seaborn
scikit-learn
geopandas
folium
imbalanced-learn
jupyter
```

---

## 📊 Sample Outputs

### Decision Tree Rules (Depth Category Splitting)
```
|--- Depth_Category <= 0.50
|   |--- class: 1
|--- Depth_Category >  0.50
|   |--- Depth_Category <= 1.50
|   |   |--- class: 1
|   |--- Depth_Category >  1.50
|       |--- class: 1
```

### Random Forest — Best Grid Search Parameters
```python
{'max_depth': 3, 'min_samples_split': 2, 'criterion': 'gini'}
# Best CV Accuracy: 0.7222
```

---

## 📄 Citation

If you use this code or findings in your research, please cite the associated paper:

```bibtex
@article{malviya2025earthquake,
  title     = {Earthquake Pattern Analysis Using Clustering, Forecasting, and Machine Learning: A Global Study (1960–2023)},
  author    = {Malviya, Harsh},
  journal   = {Journal of Research in Environmental and Earth Sciences},
  publisher = {QUEST Journals},
  year      = {2025},
  url       = {https://www.researchgate.net/publication/393362033}
}
```

---

## 👤 Author

**Harsh Malviya**  
MS Data Science, UMass Dartmouth  
🌐 [harshmalviya.com](https://harshmalviya.com) · 📄 [ResearchGate Paper](https://www.researchgate.net/publication/393362033_Earthquake_Pattern_Analysis_Using_Clustering_Forecasting_and_Machine_Learning_A_Global_Study_1960-2023)

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
