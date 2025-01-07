# Statistical Application: Beginner-Friendly Guide

Welcome to the **Statistical Application**! This guide will help you understand how to use this app step by step, even if you are new to statistics or programming.

---

## 1. Introduction
This app allows you to upload your dataset and perform various statistical and machine learning analyses interactively. The application supports:

- Data Upload & Processing
- Descriptive Statistics
- Inferential Statistics
- Regression Analysis
- Clustering
- Advanced Features (e.g., PCA and Time-Series Analysis)
- Machine Learning Models
- Interactive Dashboard

Each section is designed to be user-friendly and includes helpful visualizations and options to download the results.

---

## 2. Installation & Requirements
To run this application locally, you need:

- Python 3.8 or above
- Libraries: Streamlit, pandas, numpy, seaborn, matplotlib, plotly, scikit-learn, scipy

Install required libraries by running:
```bash
pip install streamlit pandas numpy seaborn matplotlib plotly scikit-learn scipy
```

Launch the app using:
```bash
streamlit run statistical_app.py
```

---

## 3. Application Workflow

### 3.1. **Data Upload**
1. Go to the **Data Upload** section.
2. Upload your dataset in CSV or Excel format.
3. Once uploaded, you will see a preview of your data.
4. The app stores the uploaded data for further analysis.

---

### 3.2. **Data Processing**
This section allows you to clean and preprocess your dataset. Steps include:

- **Remove Missing Values**: Deletes rows with missing data.
- **Fill Missing Values**: Replaces missing data with mean, median, or a custom value.
- **Remove Duplicate Rows**: Eliminates duplicate rows to avoid redundancy.
- **Rename Columns**: Lets you rename columns for clarity.
- **Change Column Data Types**: Converts columns into numeric, string, or date format.
- **Identify Outliers**: Detects outliers using IQR or Z-Score methods.
- **Scale Data**: Scales numeric columns using Standard Scaling or Min-Max Scaling.

Every change updates the dataset, and you can preview the processed data.

---

### 3.3. **Descriptive Statistics**

1. **Summary Statistics**: Provides metrics like mean, median, min, max, and standard deviation for numeric columns.
2. **Correlation Heatmap**: Displays relationships between numeric variables.
3. **Data Distribution**: Allows you to explore the distribution of each column using histograms.

---

### 3.4. **Inferential Statistics**
This section helps you test hypotheses using statistical tests.

- **T-Test**: Compares means between two groups.
- **Paired T-Test**: Compares means between two related groups (e.g., before vs. after).
- **Chi-Square Test**: Tests relationships between categorical variables.
- **ANOVA**: Compares means across multiple groups.
- **Mann-Whitney U Test**: Non-parametric alternative to T-Test.
- **Kruskal-Wallis Test**: Non-parametric alternative to ANOVA.

You can view results in a table and visualize them with boxplots or heatmaps.

---

### 3.5. **Regression Analysis**
Build predictive models using different regression techniques:

- **Linear Regression**
- **Ridge Regression**
- **Lasso Regression**
- **Decision Tree Regression**
- **Random Forest Regression**
- **Support Vector Regressor (SVR)**

Visualizations include:
- Scatter plots of actual vs. predicted values.
- Residual plots to evaluate model fit.

---

### 3.6. **Clustering**
Group similar data points into clusters:

- **K-Means Clustering**: Creates user-defined clusters.
- **DBSCAN**: Identifies clusters based on density.
- **Agglomerative Clustering**: Hierarchical clustering.

Visualizations include scatter plots with clusters highlighted.

---

### 3.7. **Advanced Features**

#### Principal Component Analysis (PCA)
- Select numeric columns for dimensionality reduction.
- Visualize the reduced dimensions (PC1, PC2).
- Download PCA results.

#### Time-Series Analysis
- Select a date column and a numeric value column.
- Analyze trends and create rolling averages.
- Generate dynamic visualizations for forecasting.
- Download forecast data.

---

### 3.8. **Machine Learning Models**
Build advanced predictive models using:

- **Linear Regression**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **Support Vector Regressor (SVR)**
- **K-Nearest Neighbors (KNN)**

Steps include:
1. Select a target variable and predictor variables.
2. Configure model parameters (e.g., number of trees, learning rate).
3. Train the model and view performance metrics (RMSE, R²).
4. Visualize predictions and residuals.
5. Download prediction results.

---

### 3.9. **Dashboard**
The dashboard provides an interactive way to explore and visualize data:

#### Overview
- View dataset summary and quick stats (total rows, missing values).
- Download the dataset.

#### Insights
- Generate correlation heatmaps.
- Explore data distributions interactively.
- Download correlation matrices.

#### Custom Visualizations
- Create scatter plots, line charts, and bar charts.
- Filter data interactively to refine visualizations.

#### Filtered Data
- Preview and filter data using sliders and dropdowns.
- Download filtered datasets.

---

## 4. How to Use the Application

1. **Upload Your Data**: Start by uploading a CSV or Excel file.
2. **Choose Your Analysis**: Navigate through the sidebar to select a feature.
3. **Customize Parameters**: Modify parameters for visualizations or models.
4. **Interpret Results**: View tables, charts, and performance metrics.
5. **Download Outputs**: Export results like filtered data, forecasts, or prediction results.

---

## 5. Support & Contact
For queries or issues, contact the development team at:

- Email: thanuka.ellepola@gmail.com
- GitHub: (https://github.com/Thanuka9/Statapp)

---

Enjoy exploring your data with ease and precision!

---

Happy Analyzing!

