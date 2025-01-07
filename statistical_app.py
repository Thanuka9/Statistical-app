# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import io
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.tree import DecisionTreeRegressor
import plotly.graph_objects as go
from pmdarima import auto_arima
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, roc_curve, auc
from scipy.stats import ttest_ind, chi2_contingency, f_oneway, mannwhitneyu, kruskal, shapiro
from scipy.stats import ttest_rel, ttest_ind, chi2_contingency, f_oneway, mannwhitneyu, kruskal
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score, classification_report, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
import os
import sys
from PIL import Image


def resource_path(relative_path):
    """
    Get the absolute path to a resource, works for both
    development and when packaged with PyInstaller.

    :param relative_path: Path to the resource file relative to the script.
    :return: Absolute path to the resource file.
    """
    try:
        # For PyInstaller (when the app is bundled into an .exe)
        base_path = sys._MEIPASS
    except Exception:
        # For running locally (in development mode)
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Path to App.pdf
app_pdf_path = resource_path("App.pdf")
    
# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Matplotlib backend fix for Windows
import matplotlib
matplotlib.use("Agg")

def main():
    st.title("Statistical Application")

    # Sidebar for navigation
    menu = [
        "Data Upload", "Data Processing","Dashboard", "Compare Multiple Datasets", "Descriptive Statistics",
        "Inferential Statistics", "Regression Analysis", "Clustering", "Advanced Features", "Machine Learning Models", 
        "Predictive Analysis", "Probabilistic Insights"
    ]
    choice = st.sidebar.selectbox("Select an Option", menu)

    if choice == "Data Upload":
        data_upload()
    elif choice == "Data Processing":
        data_processing()
    elif choice == "Dashboard":
        dashboard()
    elif choice == "Descriptive Statistics":
        descriptive_statistics()
    elif choice == "Inferential Statistics":
        inferential_statistics()
    elif choice == "Regression Analysis":
        regression_analysis()
    elif choice == "Clustering":
        clustering_visualizations()
    elif choice == "Advanced Features":
        advanced_features()
    elif choice == "Machine Learning Models":
        machine_learning_models()
    elif choice == "Predictive Analysis":
        predictive_analysis()
    elif choice == "Probabilistic Insights":
        probabilistic_insights()
    elif choice == "Compare Multiple Datasets":
        compare_datasets()

def data_upload():
    st.header("Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            # Load data based on file type
            if uploaded_file.name.endswith(".csv"):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)

            # Store the dataset in session_state
            st.session_state['data'] = data
            st.session_state['original_data'] = data.copy()  # Store original dataset for reference

            # Display dataset information
            st.success(f"Dataset uploaded successfully! Total rows: {data.shape[0]}, Total columns: {data.shape[1]}")
            
            # Display a preview of the dataset
            st.write("### Dataset Preview")
            st.dataframe(data.head(), use_container_width=True)

            # Add download button for the full dataset
            csv_data = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Full Dataset as CSV",
                data=csv_data,
                file_name='uploaded_dataset.csv',
                mime='text/csv',
            )

        except Exception as e:
            st.error(f"Error loading the file: {e}")

#Data Processing Function
def data_processing():
    st.header("Data Processing and Wrangling")
    if 'data' in st.session_state:
        data = st.session_state['data']
        st.write("### Initial Dataset")
        st.dataframe(data.head(), use_container_width=True)

        # Handle missing values
        if st.checkbox("Remove Missing Values"):
            data = data.dropna()
            st.success("Missing values removed.")

        if st.checkbox("Fill Missing Values"):
            fill_method = st.selectbox("Fill Method", ["Mean", "Median", "Custom Value"])
            columns_to_fill = st.multiselect("Select Columns to Fill", data.columns)
            if columns_to_fill:
                if fill_method == "Mean":
                    for col in columns_to_fill:
                        if pd.api.types.is_numeric_dtype(data[col]):
                            data[col].fillna(data[col].mean(), inplace=True)
                    st.success("Missing values filled with mean.")
                elif fill_method == "Median":
                    for col in columns_to_fill:
                        if pd.api.types.is_numeric_dtype(data[col]):
                            data[col].fillna(data[col].median(), inplace=True)
                    st.success("Missing values filled with median.")
                elif fill_method == "Custom Value":
                    custom_value = st.text_input("Enter Custom Value")
                    for col in columns_to_fill:
                        data[col].fillna(custom_value, inplace=True)
                    st.success("Missing values filled with custom value.")

        # Remove duplicate rows
        if st.checkbox("Remove Duplicate Rows"):
            initial_count = data.shape[0]
            data = data.drop_duplicates()
            final_count = data.shape[0]
            st.success(f"Removed {initial_count - final_count} duplicate rows.")

        # Rename columns
        if st.checkbox("Rename Columns"):
            col_mapping = {}
            for col in data.columns:
                new_name = st.text_input(f"Rename column '{col}'", value=col)
                if new_name != col:
                    col_mapping[col] = new_name
            if col_mapping:
                data.rename(columns=col_mapping, inplace=True)
                st.success("Columns renamed.")

        # Convert column data types
        if st.checkbox("Change Column Data Type"):
            col_to_convert = st.selectbox("Select Column to Convert", data.columns)
            new_type = st.selectbox("Select Data Type", ["Numeric", "String", "Date"])
            try:
                if new_type == "Numeric":
                    data[col_to_convert] = pd.to_numeric(data[col_to_convert], errors="coerce")
                elif new_type == "String":
                    data[col_to_convert] = data[col_to_convert].astype(str)
                elif new_type == "Date":
                    data[col_to_convert] = pd.to_datetime(data[col_to_convert], errors="coerce")
                st.success(f"Column '{col_to_convert}' converted to {new_type}.")
            except Exception as e:
                st.error(f"Error converting column: {e}")

        # Filter rows by value
        if st.checkbox("Filter Rows by Value"):
            filter_col = st.selectbox("Select Column to Filter", data.columns)
            filter_val = st.text_input("Value to Filter by (supports partial matching)")
            if filter_val:
                data = data[data[filter_col].astype(str).str.contains(filter_val, na=False)]
                st.success("Rows filtered by the specified value.")

        # Identify outliers
        if st.checkbox("Identify Outliers"):
            numeric_columns = data.select_dtypes(include=np.number).columns
            outlier_method = st.selectbox("Select Outlier Detection Method", ["IQR", "Z-Score"])
            if numeric_columns.any():
                for col in numeric_columns:
                    if outlier_method == "IQR":
                        Q1 = data[col].quantile(0.25)
                        Q3 = data[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
                    elif outlier_method == "Z-Score":
                        z_scores = (data[col] - data[col].mean()) / data[col].std()
                        outliers = data[(z_scores.abs() > 3)]
                    
                    st.write(f"Column: {col}, Outliers: {len(outliers)}")
                    if not outliers.empty:
                        st.dataframe(outliers)
                        fig = px.box(data, y=col, points="outliers", title=f"Boxplot of {col} (Outliers Highlighted)")
                        st.plotly_chart(fig)

        # Scaling options
        if st.checkbox("Scale Data"):
            scaler_type = st.selectbox("Select Scaling Method", ["Standard Scaling", "Min-Max Scaling"])
            numeric_columns = data.select_dtypes(include=np.number).columns
            if scaler_type == "Standard Scaling":
                scaler = StandardScaler()
            elif scaler_type == "Min-Max Scaling":
                scaler = MinMaxScaler()
            data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
            st.success(f"Data scaled using {scaler_type}.")

        # Show real-time changes
        st.session_state['data'] = data
        st.write("### Processed Dataset")
        st.dataframe(data.head(), use_container_width=True)
    else:
        st.error("Please upload a dataset first.")

#Dashboard functions
def dashboard():
    st.header("Interactive Dashboard")
    if 'data' not in st.session_state or st.session_state['data'] is None:
        st.error("Please upload and process a dataset first.")
        return

    data = st.session_state['data'].copy()  # Work on a copy of the dataset to avoid mutating the original

    # Data Sanitization
    try:
        # Identify numeric and non-numeric columns
        numeric_columns = data.select_dtypes(include=np.number).columns
        non_numeric_columns = data.select_dtypes(exclude=np.number).columns

        # Convert non-numeric columns to strings
        for col in non_numeric_columns:
            data[col] = data[col].astype(str)

        # Coerce numeric columns with invalid values to NaN
        for col in numeric_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    except Exception as e:
        st.error(f"Error sanitizing dataset: {e}")
        return

    # Sidebar Controls
    st.sidebar.header("Dashboard Controls")
    analysis_type = st.sidebar.radio(
        "Select Dashboard Section:",
        ["Overview", "Insights", "Custom Visualization", "Filtered Data"]
    )

    # Overview Section
    if analysis_type == "Overview":
        st.subheader("Dataset Overview")
        try:
            st.write("### Dataset Summary")
            sanitized_data = data.copy()
            st.dataframe(sanitized_data.describe(include="all"), use_container_width=True)

            st.write("### Data Preview")
            st.dataframe(data.head(), use_container_width=True)

            # Quick Stats
            st.write("### Quick Statistics")
            st.metric("Total Rows", len(data))
            st.metric("Total Columns", len(data.columns))
            st.metric("Missing Values", data.isnull().sum().sum())

            # Download the dataset
            csv_data = data.to_csv(index=False)
            st.download_button(
                label="Download Dataset as CSV",
                data=csv_data,
                file_name="dataset.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"An error occurred in the Overview section: {e}")

    # Insights Section
    elif analysis_type == "Insights":
        st.subheader("Data Insights")
        try:
            # Correlation Heatmap
            st.write("### Correlation Heatmap")
            if len(numeric_columns) > 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(
                    data[numeric_columns].corr(),
                    annot=True,
                    fmt=".2f",
                    cmap="coolwarm",
                    linewidths=0.5,
                    linecolor='gray',
                    ax=ax
                )
                ax.set_title("Correlation Heatmap", fontsize=14)
                plt.xticks(rotation=45)
                plt.yticks(rotation=0)
                st.pyplot(fig)

                # Download correlation matrix
                corr_matrix = data[numeric_columns].corr()
                st.download_button(
                    label="Download Correlation Matrix as CSV",
                    data=corr_matrix.to_csv(),
                    file_name="correlation_matrix.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Not enough numeric columns available for correlation heatmap.")

            # Distribution Plots
            st.write("### Data Distribution")
            dist_col = st.selectbox("Select Column for Distribution", data.columns, key="dist_col")
            if dist_col:
                bins = st.slider("Number of Bins", min_value=5, max_value=50, value=20, key="dist_bins")

                # Option to choose between Seaborn and Plotly
                plot_option = st.radio("Select Plot Type:", ["Seaborn", "Plotly"], horizontal=True, key="dist_plot_option")

                if data[dist_col].dtype in [np.float64, np.int64]:
                    if plot_option == "Seaborn":
                        fig, ax = plt.subplots()
                        sns.histplot(data[dist_col], bins=bins, kde=True, color="blue", edgecolor="black", ax=ax)
                        ax.set_title(f"Distribution of {dist_col}", fontsize=14)
                        st.pyplot(fig)
                    else:
                        fig = px.histogram(data, x=dist_col, nbins=bins, marginal="box", title=f"Distribution of {dist_col}")
                        fig.update_layout(bargap=0.2)
                        st.plotly_chart(fig)
                else:
                    st.warning("Selected column is non-numeric. Use categorical plots instead.")
        except Exception as e:
            st.error(f"An error occurred in the Insights section: {e}")

    # Custom Visualization Section
    elif analysis_type == "Custom Visualization":
        st.subheader("Custom Visualization")
        try:
            # 2D or 3D Selection
            visualization_mode = st.radio(
                "Select Visualization Mode:",
                ["2D", "3D"],
                horizontal=True,
                key="viz_mode"
            )

            if visualization_mode == "2D":
                # X, Y, and Color Controls for 2D
                x_column = st.selectbox("Select X-axis Variable", data.columns, key="2d_x_col")
                y_column = st.selectbox("Select Y-axis Variable", data.columns, key="2d_y_col")
                color_column = st.selectbox("Select Column for Color (Optional)", [None] + list(data.columns), index=0, key="2d_color_col")
                plot_type = st.selectbox(
                    "Select Visualization Type:",
                    ["Scatter Plot", "Line Chart", "Bar Chart"],
                    key="2d_plot_type"
                )

                if st.button("Generate 2D Plot", key="2d_generate_plot"):
                    if plot_type == "Scatter Plot":
                        fig = px.scatter(
                            data,
                            x=x_column,
                            y=y_column,
                            color=color_column,
                            title="2D Scatter Plot",
                            labels={"color": color_column} if color_column else None
                        )
                    elif plot_type == "Line Chart":
                        fig = px.line(
                            data,
                            x=x_column,
                            y=y_column,
                            color=color_column,
                            title="2D Line Chart",
                            labels={"color": color_column} if color_column else None
                        )
                    elif plot_type == "Bar Chart":
                        fig = px.bar(
                            data,
                            x=x_column,
                            y=y_column,
                            color=color_column,
                            title="2D Bar Chart",
                            labels={"color": color_column} if color_column else None
                        )
                    st.plotly_chart(fig)

            elif visualization_mode == "3D":
                # X, Y, Z, and Color Controls for 3D
                x_column = st.selectbox("Select X-axis Variable", data.columns, key="3d_x_col")
                y_column = st.selectbox("Select Y-axis Variable", data.columns, key="3d_y_col")
                z_column = st.selectbox("Select Z-axis Variable", [None] + list(data.columns), key="3d_z_col")
                color_column = st.selectbox("Select Column for Color (Optional)", [None] + list(data.columns), index=0, key="3d_color_col")
                chart_type = st.selectbox(
                    "Select 3D Chart Type:",
                    ["3D Scatter Plot", "3D Surface Plot", "3D Line Plot"],
                    key="3d_chart_type"
                )

                # Color Mapping Options
                color_scale = st.selectbox(
                    "Select Color Scale:",
                    ["Viridis", "Plasma", "Cividis", "Inferno", "Blues", "Greens"],
                    key="3d_color_scale"
                )

                if st.button("Generate 3D Plot", key="3d_generate_plot"):
                    if z_column:
                        if chart_type == "3D Scatter Plot":
                            fig = px.scatter_3d(
                                data,
                                x=x_column,
                                y=y_column,
                                z=z_column,
                                color=color_column,
                                color_continuous_scale=color_scale,
                                title="3D Scatter Plot"
                            )
                        elif chart_type == "3D Surface Plot":
                            try:
                                # Create grid data for the surface plot
                                grid_data = data.pivot_table(
                                    index=y_column,
                                    columns=x_column,
                                    values=z_column,
                                    aggfunc='mean'  # Use mean aggregation for simplicity
                                )
                                if grid_data.isnull().values.any():
                                    st.warning("Some values in the grid are missing; filling with zeros for visualization.")
                                    grid_data = grid_data.fillna(0)  # Fill NaN values with 0

                                # Generate mesh grid for x and y
                                x_vals = grid_data.columns.values
                                y_vals = grid_data.index.values
                                z_vals = grid_data.values

                                fig = go.Figure(data=[go.Surface(
                                    z=z_vals,
                                    x=x_vals,
                                    y=y_vals,
                                    colorscale=color_scale
                                )])
                                fig.update_layout(
                                    title="3D Surface Plot",
                                    scene=dict(
                                        xaxis_title=x_column,
                                        yaxis_title=y_column,
                                        zaxis_title=z_column
                                    )
                                )
                            except Exception as e:
                                st.error(f"Failed to generate 3D Surface Plot: {e}")
                                return
                        elif chart_type == "3D Line Plot":
                            fig = px.line_3d(
                                data,
                                x=x_column,
                                y=y_column,
                                z=z_column,
                                color=color_column,
                                title="3D Line Plot"
                            )
                        st.plotly_chart(fig)
                    else:
                        st.warning("Please select a Z-axis variable for 3D visualization.")
        except Exception as e:
            st.error(f"An error occurred in Custom Visualization: {e}")

    # Filtered Data Section
    elif analysis_type == "Filtered Data":
        st.subheader("Filtered Data Preview")
        try:
            filters = {}
            for col in data.columns:
                if data[col].dtype in ['object', 'category']:
                    filters[col] = st.multiselect(f"Filter {col}", options=data[col].unique(), key=f"filter_{col}")
                elif col in numeric_columns:
                    filters[col] = st.slider(
                        f"Filter {col}",
                        min_value=float(data[col].min()),
                        max_value=float(data[col].max()),
                        value=(float(data[col].min()), float(data[col].max())),
                        key=f"filter_{col}_slider"
                    )

            filtered_data = data
            for col, condition in filters.items():
                if isinstance(condition, list) and condition:
                    filtered_data = filtered_data[filtered_data[col].isin(condition)]
                elif isinstance(condition, tuple):
                    filtered_data = filtered_data[(filtered_data[col] >= condition[0]) & (filtered_data[col] <= condition[1])]

            st.dataframe(filtered_data.head(), use_container_width=True)

            # Optional Pairplot for Filtered Data
            if not filtered_data.empty and len(filtered_data.columns) > 1:
                st.write("### Pairplot of Filtered Data")
                pairplot_fig = sns.pairplot(filtered_data, diag_kind="kde", corner=True)
                st.pyplot(pairplot_fig)

            # Download filtered data
            filtered_csv = filtered_data.to_csv(index=False)
            st.download_button(
                label="Download Filtered Data as CSV",
                data=filtered_csv,
                file_name="filtered_data.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"An error occurred in Filtered Data section: {e}")

    # Filtered Data Section
    elif analysis_type == "Filtered Data":
        st.subheader("Filtered Data Preview")
        try:
            filters = {}
            for col in data.columns:
                if data[col].dtype in ['object', 'category']:
                    filters[col] = st.multiselect(f"Filter {col}", options=data[col].unique(), key=f"filter_{col}")
                elif col in numeric_columns:
                    filters[col] = st.slider(
                        f"Filter {col}",
                        min_value=float(data[col].min()),
                        max_value=float(data[col].max()),
                        value=(float(data[col].min()), float(data[col].max())),
                        key=f"filter_{col}_slider"
                    )

            filtered_data = data
            for col, condition in filters.items():
                if isinstance(condition, list) and condition:
                    filtered_data = filtered_data[filtered_data[col].isin(condition)]
                elif isinstance(condition, tuple):
                    filtered_data = filtered_data[(filtered_data[col] >= condition[0]) & (filtered_data[col] <= condition[1])]

            st.dataframe(filtered_data.head(), use_container_width=True)

            # Optional Pairplot for Filtered Data
            if not filtered_data.empty and len(filtered_data.columns) > 1:
                st.write("### Pairplot of Filtered Data")
                pairplot_fig = sns.pairplot(filtered_data, diag_kind="kde", corner=True)
                st.pyplot(pairplot_fig)

            # Download filtered data
            filtered_csv = filtered_data.to_csv(index=False)
            st.download_button(
                label="Download Filtered Data as CSV",
                data=filtered_csv,
                file_name="filtered_data.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"An error occurred in Filtered Data section: {e}")
            
# Descriptive Statistics Function
def descriptive_statistics():
    st.header("Descriptive Statistics")

    if 'data' in st.session_state:
        data = st.session_state['data']

        # Separate numeric and categorical columns
        numeric_columns = data.select_dtypes(include=np.number).columns
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns

        # Sanitize the dataset
        try:
            # Convert numeric columns to proper numeric values (coercing errors to NaN)
            for col in numeric_columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

            # Convert categorical/object columns to strings (if they aren't already)
            for col in categorical_columns:
                data[col] = data[col].astype(str)

            st.success("Dataset sanitized successfully.")
        except Exception as e:
            st.error(f"Error sanitizing dataset: {e}")
            return

        # Summary Statistics
        st.write("### Summary Statistics")
        try:
            # Generate summary statistics (separately for numeric and non-numeric)
            numeric_summary = data.describe(include=[np.number]).transpose()
            categorical_summary = data.describe(include=['object', 'category']).transpose()

            # Display numeric summary
            if not numeric_summary.empty:
                st.write("#### Numeric Columns Summary")
                st.dataframe(numeric_summary.style.highlight_max(axis=0), use_container_width=True)

                # Download Numeric Summary Statistics
                csv_numeric_summary = numeric_summary.to_csv()
                st.download_button(
                    "Download Numeric Summary Statistics as CSV",
                    csv_numeric_summary,
                    "numeric_summary_statistics.csv",
                    "text/csv"
                )
            else:
                st.warning("No numeric columns found for summary statistics.")

            # Display categorical summary
            if not categorical_summary.empty:
                st.write("#### Categorical Columns Summary")
                st.dataframe(categorical_summary, use_container_width=True)

                # Download Categorical Summary Statistics
                csv_categorical_summary = categorical_summary.to_csv()
                st.download_button(
                    "Download Categorical Summary Statistics as CSV",
                    csv_categorical_summary,
                    "categorical_summary_statistics.csv",
                    "text/csv"
                )
            else:
                st.warning("No categorical columns found for summary statistics.")
        except Exception as e:
            st.error(f"Error generating summary statistics: {e}")

        # Missing Values Heatmap
        st.write("### Missing Values Heatmap")
        try:
            if data.isnull().sum().sum() > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(data.isnull(), cbar=False, cmap="YlGnBu", ax=ax)
                ax.set_title("Missing Values Heatmap")
                st.pyplot(fig)
            else:
                st.success("No missing values found!")
        except Exception as e:
            st.error(f"Error generating missing values heatmap: {e}")

        # Correlation Heatmap
        st.write("### Correlation Heatmap")
        try:
            if len(numeric_columns) > 1:
                correlation_threshold = st.slider("Filter Correlation (Absolute Value)", 0.0, 1.0, 0.5, 0.05)
                corr_matrix = data[numeric_columns].corr()
                filtered_corr = corr_matrix.abs() >= correlation_threshold
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(
                    corr_matrix,
                    mask=~filtered_corr,
                    annot=True,
                    cmap="coolwarm",
                    fmt=".2f",
                    linewidths=0.5,
                    ax=ax
                )
                ax.set_title(f"Correlation Heatmap (Threshold: {correlation_threshold})")
                st.pyplot(fig)

                # Download Correlation Matrix
                csv_corr = corr_matrix.to_csv()
                st.download_button("Download Correlation Matrix as CSV", csv_corr, "correlation_matrix.csv", "text/csv")
            else:
                st.warning("Not enough numeric columns for correlation analysis.")
        except Exception as e:
            st.error(f"Error generating correlation heatmap: {e}")

        # Data Distribution
        st.write("### Data Distribution")
        try:
            if len(numeric_columns) > 0:
                selected_column = st.selectbox("Select a Numeric Column", numeric_columns)
                plot_type = st.radio("Select Plot Type", ["Histogram", "Boxplot", "Violin Plot"], index=0)

                if plot_type == "Histogram":
                    fig, ax = plt.subplots()
                    sns.histplot(data[selected_column], kde=True, color='skyblue', ax=ax)
                    ax.set_title(f"Distribution of {selected_column}")
                    ax.set_xlabel(selected_column)
                    ax.set_ylabel("Frequency")
                    st.pyplot(fig)

                elif plot_type == "Boxplot":
                    fig, ax = plt.subplots()
                    sns.boxplot(x=data[selected_column], color='lightgreen', ax=ax)
                    ax.set_title(f"Boxplot of {selected_column}")
                    st.pyplot(fig)

                elif plot_type == "Violin Plot":
                    fig, ax = plt.subplots()
                    sns.violinplot(x=data[selected_column], color='orange', ax=ax)
                    ax.set_title(f"Violin Plot of {selected_column}")
                    st.pyplot(fig)
            else:
                st.warning("No numeric columns found for distribution analysis.")
        except Exception as e:
            st.error(f"Error generating data distribution: {e}")
    else:
        st.error("Please upload a dataset first.")

# Inferential Statistics Function
def inferential_statistics():
    st.header("Inferential Statistics")
    
    # Check if data is available
    if 'data' in st.session_state and st.session_state['data'] is not None:
        data = st.session_state['data']
        numeric_columns = data.select_dtypes(include=np.number).columns
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns

        # Visualization Library Selection
        st.sidebar.header("Visualization Settings")
        viz_library = st.sidebar.radio(
            "Select Visualization Library",
            options=["Seaborn", "Plotly", "Matplotlib"],
            index=0
        )

        # Ensure there are numeric columns for analysis
        if len(numeric_columns) < 1 and len(categorical_columns) < 2:
            st.error("Dataset needs at least two numeric columns or two categorical columns for analysis.")
            return

        # Test Type Selection
        test_type = st.selectbox(
            "Select Statistical Test",
            [
                "T-test", "Paired T-test", "Chi-Square Test",
                "ANOVA", "Mann-Whitney U Test", "Kruskal-Wallis Test"
            ],
            help="Choose the appropriate statistical test for your data."
        )

        # Display Hypotheses
        st.write("### Hypotheses")
        if test_type == "T-test":
            st.info("**Null Hypothesis:** The means of the two groups are equal.\n\n"
                    "**Alternative Hypothesis:** The means of the two groups are not equal.")
        elif test_type == "Paired T-test":
            st.info("**Null Hypothesis:** The means of the paired groups are equal.\n\n"
                    "**Alternative Hypothesis:** The means of the paired groups are not equal.")
        elif test_type == "Chi-Square Test":
            st.info("**Null Hypothesis:** There is no association between the two categorical variables.\n\n"
                    "**Alternative Hypothesis:** There is an association between the two categorical variables.")
        elif test_type == "ANOVA":
            st.info("**Null Hypothesis:** All group means are equal.\n\n"
                    "**Alternative Hypothesis:** At least one group mean is different.")
        elif test_type == "Mann-Whitney U Test":
            st.info("**Null Hypothesis:** The distributions of the two groups are equal.\n\n"
                    "**Alternative Hypothesis:** The distributions of the two groups are not equal.")
        elif test_type == "Kruskal-Wallis Test":
            st.info("**Null Hypothesis:** All group distributions are equal.\n\n"
                    "**Alternative Hypothesis:** At least one group distribution is different.")

        # Option to Apply Log Transformation
        log_transform = st.checkbox("Apply Log Transformation to Data (for non-normal data)")

        # Helper Function for Visualizations
        def visualize_boxplot(data, columns, title):
            if viz_library == "Seaborn":
                fig, ax = plt.subplots()
                sns.boxplot(data=[data[col].dropna() for col in columns], ax=ax, palette="coolwarm")
                ax.set_xticks(range(len(columns)))
                ax.set_xticklabels(columns)
                ax.set_title(title)
                st.pyplot(fig)
            elif viz_library == "Matplotlib":
                fig, ax = plt.subplots()
                ax.boxplot([data[col].dropna() for col in columns], patch_artist=True)
                ax.set_xticks(range(1, len(columns) + 1))
                ax.set_xticklabels(columns)
                ax.set_title(title)
                st.pyplot(fig)
            else:  # Plotly
                fig = px.box(
                    data_frame=data,
                    y=columns,
                    title=title,
                    points="all"
                )
                st.plotly_chart(fig)

        # Statistical Tests
        if test_type == "T-test":
            col1 = st.selectbox("Select First Numeric Column", numeric_columns, key="t1")
            col2 = st.selectbox("Select Second Numeric Column", numeric_columns, key="t2")
            if st.button("Run T-test"):
                try:
                    if log_transform:
                        data[col1] = np.log1p(data[col1])
                        data[col2] = np.log1p(data[col2])

                    t_stat, p_value = ttest_ind(data[col1].dropna(), data[col2].dropna())
                    st.write(f"T-statistic: {t_stat:.2f}, P-value: {p_value:.2e}")
                    visualize_boxplot(data, [col1, col2], "T-test Boxplot")
                except Exception as e:
                    st.error(f"Error: {e}")

        elif test_type == "Paired T-test":
            col1 = st.selectbox("Select First Numeric Column", numeric_columns, key="paired1")
            col2 = st.selectbox("Select Second Numeric Column", numeric_columns, key="paired2")
            if st.button("Run Paired T-test"):
                try:
                    if log_transform:
                        data[col1] = np.log1p(data[col1])
                        data[col2] = np.log1p(data[col2])

                    t_stat, p_value = ttest_rel(data[col1].dropna(), data[col2].dropna())
                    st.write(f"T-statistic: {t_stat:.2f}, P-value: {p_value:.2e}")
                    visualize_boxplot(data, [col1, col2], "Paired T-test Boxplot")
                except Exception as e:
                    st.error(f"Error: {e}")

        elif test_type == "Chi-Square Test":
            cat_col1 = st.selectbox("Select First Categorical Column", categorical_columns, key="chi1")
            cat_col2 = st.selectbox("Select Second Categorical Column", categorical_columns, key="chi2")
            if st.button("Run Chi-Square Test"):
                try:
                    contingency_table = pd.crosstab(data[cat_col1], data[cat_col2])
                    st.write("Contingency Table:")
                    st.dataframe(contingency_table)
                    chi2, p, dof, expected = chi2_contingency(contingency_table)
                    st.write(f"Chi-Square Statistic: {chi2:.2f}, P-value: {p:.2e}")
                    fig, ax = plt.subplots()
                    sns.heatmap(contingency_table, annot=True, fmt="d", cmap="coolwarm", ax=ax)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error: {e}")

        elif test_type == "ANOVA":
            selected_columns = st.multiselect("Select Numeric Columns", numeric_columns, default=numeric_columns[:3], key="anova")
            if len(selected_columns) >= 2 and st.button("Run ANOVA"):
                try:
                    f_stat, p_value = f_oneway(*(data[col].dropna() for col in selected_columns))
                    st.write(f"F-statistic: {f_stat:.2f}, P-value: {p_value:.2e}")
                    visualize_boxplot(data, selected_columns, "ANOVA Boxplot")
                except Exception as e:
                    st.error(f"Error: {e}")

        elif test_type == "Mann-Whitney U Test":
            col1 = st.selectbox("Select First Numeric Column", numeric_columns, key="mw1")
            col2 = st.selectbox("Select Second Numeric Column", numeric_columns, key="mw2")
            if st.button("Run Mann-Whitney U Test"):
                try:
                    u_stat, p_value = mannwhitneyu(data[col1].dropna(), data[col2].dropna())
                    st.write(f"U-statistic: {u_stat:.2f}, P-value: {p_value:.2e}")
                    visualize_boxplot(data, [col1, col2], "Mann-Whitney U Test Boxplot")
                except Exception as e:
                    st.error(f"Error: {e}")

        elif test_type == "Kruskal-Wallis Test":
            selected_columns = st.multiselect("Select Numeric Columns", numeric_columns, key="kruskal")
            if len(selected_columns) >= 2 and st.button("Run Kruskal-Wallis Test"):
                try:
                    h_stat, p_value = kruskal(*(data[col].dropna() for col in selected_columns))
                    st.write(f"H-statistic: {h_stat:.2f}, P-value: {p_value:.2e}")
                    visualize_boxplot(data, selected_columns, "Kruskal-Wallis Test Boxplot")
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.error("Please upload a dataset first.")

# Regression Analysis Function
def regression_analysis():
    st.header("Regression Analysis")

    if 'data' in st.session_state:
        data = st.session_state['data']
        numeric_columns = data.select_dtypes(include=np.number).columns.tolist()

        if len(numeric_columns) < 2:
            st.error("Dataset needs at least two numeric columns for regression.")
        else:
            # User Inputs
            x_column = st.selectbox("Select Independent Variable (X)", numeric_columns)
            y_column = st.selectbox("Select Dependent Variable (Y)", numeric_columns)
            regression_type = st.selectbox(
                "Select Regression Type",
                [
                    "Linear Regression",
                    "Ridge Regression",
                    "Lasso Regression",
                    "Decision Tree Regression",
                    "Random Forest Regression",
                    "Support Vector Regression (SVR)"
                ]
            )

            # Advanced Options
            scaling_option = st.radio("Scaling Options", ["None", "Standard Scaling", "Min-Max Scaling"], index=0)
            polynomial_degree = st.slider("Polynomial Degree (For Non-Linear)", 1, 5, 1)
            train_test_split_percentage = st.slider("Train/Test Split (%)", 50, 90, 80, help="Percentage of data for training.")
            cross_validation = st.checkbox("Enable Cross-Validation (5-Fold)", value=False)
            show_feature_importance = st.checkbox("Show Feature Importance (Tree Models Only)", value=False)

            # Additional Model Parameters
            if regression_type in ["Ridge Regression", "Lasso Regression"]:
                alpha = st.slider("Alpha (Regularization Strength)", 0.1, 10.0, 1.0)
            if regression_type in ["Decision Tree Regression", "Random Forest Regression"]:
                max_depth = st.slider("Max Depth", 2, 20, 5)
                if regression_type == "Random Forest Regression":
                    n_estimators = st.slider("Number of Trees", 10, 200, 100)
            if regression_type == "Support Vector Regression (SVR)":
                C = st.slider("C (Regularization Parameter)", 0.1, 10.0, 1.0)
                epsilon = st.slider("Epsilon (Margin of Tolerance)", 0.01, 1.0, 0.1)

            if st.button("Run Regression"):
                try:
                    # Data Preparation
                    valid_data = data[[x_column, y_column]].dropna()
                    X = valid_data[[x_column]].values
                    y = valid_data[y_column].values

                    # Train/Test Split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=(100 - train_test_split_percentage) / 100, random_state=42
                    )

                    # Scaling
                    if scaling_option == "Standard Scaling":
                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)
                    elif scaling_option == "Min-Max Scaling":
                        scaler = MinMaxScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)

                    # Polynomial Features
                    if polynomial_degree > 1:
                        poly = PolynomialFeatures(degree=polynomial_degree)
                        X_train = poly.fit_transform(X_train)
                        X_test = poly.transform(X_test)

                    # Initialize Model
                    if regression_type == "Linear Regression":
                        model = LinearRegression()
                    elif regression_type == "Ridge Regression":
                        model = Ridge(alpha=alpha)
                    elif regression_type == "Lasso Regression":
                        model = Lasso(alpha=alpha)
                    elif regression_type == "Decision Tree Regression":
                        model = DecisionTreeRegressor(max_depth=max_depth)
                    elif regression_type == "Random Forest Regression":
                        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
                    elif regression_type == "Support Vector Regression (SVR)":
                        model = SVR(C=C, epsilon=epsilon)

                    # Cross-validation or Training
                    if cross_validation:
                        try:
                            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_squared_error")
                            st.write(f"Cross-Validation RMSE: {(-cv_scores.mean()) ** 0.5:.2f}")
                        except Exception as e:
                            st.error(f"Cross-validation failed: {e}")
                    else:
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)

                        # Metrics
                        mse = mean_squared_error(y_test, predictions)
                        rmse = mse ** 0.5
                        mae = mean_absolute_error(y_test, predictions)
                        r2 = r2_score(y_test, predictions)

                        st.write("### Model Performance Metrics")
                        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
                        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
                        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
                        st.write(f"R² Score: {r2:.2f}")

                        # Feature Importance
                        if show_feature_importance and hasattr(model, "feature_importances_"):
                            st.write("### Feature Importance")
                            importance = model.feature_importances_
                            fig, ax = plt.subplots()
                            sns.barplot(x=importance, y=[f"Feature {i+1}" for i in range(len(importance))], ax=ax)
                            ax.set_title("Feature Importance")
                            st.pyplot(fig)

                        # Visualizations
                        st.write("### Regression Visualization")
                        fig, ax = plt.subplots()
                        sns.scatterplot(x=X_test.flatten(), y=y_test, color="blue", label="Actual Data", ax=ax)
                        sns.lineplot(x=X_test.flatten(), y=predictions, color="red", label="Regression Line", ax=ax)
                        ax.set_title(f"{regression_type} Regression Analysis")
                        ax.set_xlabel(f"{x_column} (Independent Variable)")
                        ax.set_ylabel(f"{y_column} (Dependent Variable)")
                        st.pyplot(fig)

                        # Residual Plot
                        residuals = y_test - predictions
                        st.write("### Residual Plot")
                        fig, ax = plt.subplots()
                        sns.residplot(x=predictions, y=residuals, ax=ax, color="purple")
                        ax.set_title("Residual Analysis")
                        ax.set_xlabel("Predicted Values")
                        ax.set_ylabel("Residuals")
                        st.pyplot(fig)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
    else:
        st.error("Please upload a dataset first.")

# Clustering Analysis Function
def clustering_visualizations():
    st.header("Clustering Visualizations")
    
    if 'data' in st.session_state:
        data = st.session_state['data']
        numeric_columns = data.select_dtypes(include=np.number).columns.tolist()

        if len(numeric_columns) < 2:
            st.error("Dataset needs at least two numeric columns for clustering.")
        else:
            # Column Selection
            x_column = st.selectbox("Select X-Axis Variable", numeric_columns, key="x_axis")
            y_column = st.selectbox("Select Y-Axis Variable", numeric_columns, key="y_axis")
            clustering_method = st.selectbox(
                "Select Clustering Method",
                ["K-Means", "DBSCAN", "Agglomerative Clustering", "Gaussian Mixture Model (GMM)", "Spectral Clustering", "OPTICS"]
            )

            # Additional parameters for specific methods
            if clustering_method == "K-Means":
                n_clusters = st.slider("Number of Clusters (k)", min_value=2, max_value=10, value=3)
                if st.checkbox("Use Elbow Method to Find Optimal k"):
                    try:
                        wcss = []
                        for k in range(2, 11):
                            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
                            kmeans.fit(data[[x_column, y_column]].dropna())
                            wcss.append(kmeans.inertia_)

                        fig, ax = plt.subplots()
                        ax.plot(range(2, 11), wcss, marker="o")
                        ax.set_title("Elbow Method for Optimal k")
                        ax.set_xlabel("Number of Clusters (k)")
                        ax.set_ylabel("WCSS")
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"An error occurred during the Elbow Method: {e}")

            elif clustering_method == "DBSCAN":
                eps = st.slider("Epsilon (eps)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
                min_samples = st.slider("Minimum Samples", min_value=1, max_value=20, value=5)

                if st.checkbox("Show Nearest Neighbor Distance Plot"):
                    try:
                        from sklearn.neighbors import NearestNeighbors
                        neighbors = NearestNeighbors(n_neighbors=min_samples)
                        neighbors_fit = neighbors.fit(data[[x_column, y_column]].dropna())
                        distances, indices = neighbors_fit.kneighbors()
                        distances = np.sort(distances[:, -1])

                        fig, ax = plt.subplots()
                        ax.plot(distances)
                        ax.set_title("Nearest Neighbor Distance Plot")
                        ax.set_xlabel("Points Sorted by Distance")
                        ax.set_ylabel("Distance to Nearest Neighbor")
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"An error occurred during Nearest Neighbor Plot: {e}")

            elif clustering_method == "Agglomerative Clustering":
                n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=3)
                linkage_method = st.selectbox("Linkage Method", ["ward", "complete", "average", "single"])
                if st.checkbox("Show Dendrogram"):
                    try:
                        from scipy.cluster.hierarchy import dendrogram, linkage
                        linkage_matrix = linkage(data[[x_column, y_column]].dropna(), method=linkage_method)
                        
                        # Create Dendrogram
                        fig, ax = plt.subplots(figsize=(12, 6))  # Adjust figure size for better readability
                        dendrogram(
                            linkage_matrix,
                            ax=ax,
                            leaf_rotation=90,  # Rotate x-axis labels for better readability
                            leaf_font_size=8,  # Adjust font size
                            color_threshold=0  # Customize colors if needed
                        )
                        ax.set_title("Dendrogram")
                        ax.set_xlabel("Data Points")
                        ax.set_ylabel("Distance")
                        st.pyplot(fig)
                        
                        # Download Dendrogram as image
                        buffer = io.BytesIO()
                        fig.savefig(buffer, format="png")
                        buffer.seek(0)
                        st.download_button(
                            label="Download Dendrogram Image",
                            data=buffer,
                            file_name="dendrogram.png",
                            mime="image/png"
                        )
                    except Exception as e:
                        st.error(f"An error occurred during Dendrogram generation: {e}")

            elif clustering_method == "Gaussian Mixture Model (GMM)":
                n_components = st.slider("Number of Components", min_value=2, max_value=10, value=3)
                covariance_type = st.selectbox("Covariance Type", ["full", "tied", "diag", "spherical"])

            elif clustering_method == "Spectral Clustering":
                n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=3)

            elif clustering_method == "OPTICS":
                eps = st.slider("Epsilon (eps)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
                min_samples = st.slider("Minimum Samples", min_value=1, max_value=20, value=5)

            # Visualization Library Selection
            visualization_library = st.selectbox("Select Visualization Library", ["Seaborn", "Plotly"])

            if st.button("Run Clustering"):
                try:
                    X = data[[x_column, y_column]].dropna()

                    # Initialize the clustering model
                    if clustering_method == "K-Means":
                        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
                    elif clustering_method == "DBSCAN":
                        model = DBSCAN(eps=eps, min_samples=min_samples)
                    elif clustering_method == "Agglomerative Clustering":
                        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
                    elif clustering_method == "Gaussian Mixture Model (GMM)":
                        from sklearn.mixture import GaussianMixture
                        model = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=42)
                    elif clustering_method == "Spectral Clustering":
                        from sklearn.cluster import SpectralClustering
                        model = SpectralClustering(n_clusters=n_clusters, affinity="nearest_neighbors", random_state=42)
                    elif clustering_method == "OPTICS":
                        from sklearn.cluster import OPTICS
                        model = OPTICS(min_samples=min_samples, eps=eps)

                    clusters = model.fit_predict(X)
                    data['Cluster'] = clusters

                    st.success("Clustering completed successfully!")
                    st.write("### Cluster Assignments Preview")
                    st.dataframe(data[['Cluster']].value_counts(), use_container_width=True)

                    # Enhanced visualization
                    st.write("### Cluster Visualization")
                    if visualization_library == "Seaborn":
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.scatterplot(
                            x=x_column,
                            y=y_column,
                            hue=data['Cluster'].astype(str),
                            palette="tab10",
                            data=data,
                            ax=ax,
                            s=100,
                            alpha=0.8
                        )
                        ax.set_title(f"{clustering_method} Clustering")
                        ax.set_xlabel(x_column)
                        ax.set_ylabel(y_column)
                        ax.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc="upper left")
                        st.pyplot(fig)

                        # Download visualization
                        buffer = io.BytesIO()
                        fig.savefig(buffer, format="png")
                        buffer.seek(0)
                        st.download_button(
                            label="Download Cluster Visualization as PNG",
                            data=buffer,
                            file_name="cluster_visualization.png",
                            mime="image/png"
                        )

                    elif visualization_library == "Plotly":
                        import plotly.express as px
                        fig = px.scatter(
                            data,
                            x=x_column,
                            y=y_column,
                            color=data['Cluster'].astype(str),
                            color_discrete_sequence=px.colors.qualitative.T10,
                            title=f"{clustering_method} Clustering",
                            labels={"color": "Cluster"}
                        )
                        st.plotly_chart(fig)

                        # Download Plotly visualization
                        st.download_button(
                            label="Download Clustered Data as CSV",
                            data=data.to_csv(index=False),
                            file_name="clustered_data.csv",
                            mime="text/csv"
                        )

                except Exception as e:
                    st.error(f"An error occurred during clustering: {e}")
    else:
        st.error("Please upload a dataset first.")

# Advanced Features Function
def advanced_features():
    st.header("Advanced Features")
    if 'data' in st.session_state:
        data = st.session_state['data']
        numeric_columns = data.select_dtypes(include=np.number).columns
        date_columns = data.select_dtypes(include=['datetime64', 'object']).columns

        # Visualization Library Selection
        st.sidebar.header("Visualization Settings")
        viz_library = st.sidebar.radio(
            "Select Visualization Library",
            options=["Seaborn", "Plotly", "Matplotlib"],
            index=0
        )

        # Debugging: Display available columns
        st.write("### Available Columns in Dataset")
        st.dataframe(data.head())

        # Principal Component Analysis (PCA)
        st.write("### Principal Component Analysis (PCA)")
        selected_columns = st.multiselect(
            "Select Numeric Columns for PCA",
            numeric_columns,
            help="Select at least 2 numeric columns."
        )
        min_variance = st.slider(
            "Minimum Cumulative Explained Variance (%)",
            min_value=80,
            max_value=100,
            value=90,
            step=1
        )
        color_column = st.selectbox(
            "Select a Column to Color by (Optional)",
            [None] + list(data.columns),
            index=0
        )
        if st.button("Run PCA"):
            try:
                if len(selected_columns) < 2:
                    st.error("Please select at least two numeric columns for PCA.")
                else:
                    valid_data = data[selected_columns].dropna()
                    if valid_data.shape[0] < 2:
                        st.error("Insufficient data: PCA requires at least two samples.")
                    else:
                        # Apply PCA
                        pca = PCA()
                        components = pca.fit_transform(valid_data)
                        explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_ * 100)
                        num_components = np.argmax(explained_variance_ratio >= min_variance) + 1

                        st.write(f"PCA selected {num_components} components to explain at least {min_variance}% variance.")

                        pca = PCA(n_components=num_components)
                        components = pca.fit_transform(valid_data)
                        pca_df = pd.DataFrame(components, columns=[f'PC{i+1}' for i in range(num_components)])
                        pca_df["Original_Index"] = valid_data.index

                        # Scree Plot
                        st.write("### Scree Plot")
                        if viz_library == "Seaborn":
                            fig, ax = plt.subplots()
                            sns.barplot(
                                x=[f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
                                y=pca.explained_variance_ratio_,
                                ax=ax
                            )
                            ax.set_title("PCA Scree Plot")
                            ax.set_ylabel("Explained Variance Ratio")
                            ax.set_xlabel("Principal Components")
                            st.pyplot(fig)
                        elif viz_library == "Matplotlib":
                            fig, ax = plt.subplots()
                            ax.bar(
                                [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
                                pca.explained_variance_ratio_,
                                color="blue"
                            )
                            ax.set_title("PCA Scree Plot")
                            ax.set_ylabel("Explained Variance Ratio")
                            ax.set_xlabel("Principal Components")
                            st.pyplot(fig)
                        else:  # Plotly
                            fig = px.bar(
                                x=[f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
                                y=pca.explained_variance_ratio_,
                                labels={"x": "Principal Components", "y": "Explained Variance Ratio"},
                                title="PCA Scree Plot"
                            )
                            st.plotly_chart(fig)

                        # Scatter Plot
                        st.write("### PCA Results")
                        if color_column and color_column != "None":
                            pca_df[color_column] = data.loc[valid_data.index, color_column]
                        if viz_library == "Seaborn":
                            fig, ax = plt.subplots()
                            sns.scatterplot(
                                data=pca_df,
                                x='PC1',
                                y='PC2',
                                hue=color_column,
                                palette="viridis",
                                ax=ax
                            )
                            ax.set_title("PCA Scatter Plot")
                            st.pyplot(fig)
                        elif viz_library == "Matplotlib":
                            fig, ax = plt.subplots()
                            scatter = ax.scatter(
                                pca_df['PC1'],
                                pca_df['PC2'],
                                c=pca_df[color_column] if color_column else "blue",
                                cmap="viridis",
                                edgecolor="k",
                                alpha=0.7
                            )
                            ax.set_title("PCA Scatter Plot")
                            ax.set_xlabel("PC1")
                            ax.set_ylabel("PC2")
                            st.pyplot(fig)
                        else:  # Plotly
                            fig = px.scatter(
                                pca_df,
                                x='PC1',
                                y='PC2',
                                color=color_column,
                                title="PCA Scatter Plot",
                                labels={'color': color_column},
                                hover_data=['Original_Index']
                            )
                            st.plotly_chart(fig)

                        # Provide Download Option
                        csv_data = pca_df.to_csv(index=False)
                        st.download_button(
                            "Download PCA Results as CSV",
                            csv_data,
                            f"PCA_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv"
                        )
            except Exception as e:
                st.error(f"An error occurred during PCA: {e}")

        # Time-Series Analysis
        st.write("### Time-Series Analysis and Advanced Forecasting")
        if date_columns.empty:
            st.warning("No date column found in the dataset.")
        elif numeric_columns.empty:
            st.warning("No numeric column found in the dataset.")
        else:
            date_column = st.selectbox("Select Date Column", date_columns)
            value_column = st.selectbox("Select Numeric Value Column", numeric_columns)
            forecast_window = st.slider(
                "Select Forecast Window (Days)",
                min_value=1,
                max_value=30,
                value=7,
                key="forecast_slider"
            )
            decomposition_model = st.radio("Decomposition Model", options=["Additive", "Multiplicative"], index=0)

            if st.button("Run Time-Series Analysis"):
                try:
                    if date_column not in data.columns:
                        st.error(f"Selected column '{date_column}' does not exist in the dataset.")
                        return

                    # Convert to datetime
                    data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
                    if data[date_column].isnull().all():
                        st.error(f"Unable to parse the column '{date_column}' as datetime.")
                        return

                    valid_data = data[[date_column, value_column]].dropna()

                    if valid_data.empty:
                        st.error("No valid data available for time-series analysis.")
                    else:
                        # Decompose Time Series
                        st.write("### Trend and Seasonality Decomposition")
                        from statsmodels.tsa.seasonal import seasonal_decompose
                        decomposition = seasonal_decompose(
                            valid_data[value_column],
                            model=decomposition_model.lower(),
                            period=forecast_window,
                            extrapolate_trend='freq'
                        )
                        fig, ax = plt.subplots(3, 1, figsize=(10, 8))
                        decomposition.trend.plot(ax=ax[0], title="Trend")
                        decomposition.seasonal.plot(ax=ax[1], title="Seasonal")
                        decomposition.resid.plot(ax=ax[2], title="Residual")
                        st.pyplot(fig)

                        # Forecasting (Simple Moving Average)
                        valid_data['Forecast'] = valid_data[value_column].rolling(window=forecast_window).mean()

                        # Time-Series Plot
                        st.write("### Time-Series with Forecasting")
                        if viz_library == "Seaborn":
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.lineplot(data=valid_data, x=date_column, y=value_column, label="Actual", ax=ax)
                            sns.lineplot(data=valid_data, x=date_column, y='Forecast', label="Forecast", ax=ax, color="orange")
                            ax.set_title("Time-Series Forecasting")
                            ax.legend()
                            st.pyplot(fig)
                        elif viz_library == "Matplotlib":
                            fig, ax = plt.subplots()
                            ax.plot(valid_data[date_column], valid_data[value_column], label="Actual", color="blue")
                            ax.plot(valid_data[date_column], valid_data['Forecast'], label="Forecast", color="orange")
                            ax.fill_between(
                                valid_data.index,
                                valid_data['Forecast'] - 1.96 * valid_data[value_column].std(),
                                valid_data['Forecast'] + 1.96 * valid_data[value_column].std(),
                                color='orange',
                                alpha=0.2,
                                label="Confidence Interval"
                            )
                            ax.set_title("Time-Series Forecasting")
                            ax.legend()
                            st.pyplot(fig)
                        else:  # Plotly
                            fig = px.line(
                                valid_data,
                                x=date_column,
                                y=[value_column, 'Forecast'],
                                labels={"value": "Values"},
                                title="Time-Series Forecasting"
                            )
                            st.plotly_chart(fig)

                        # Provide download option for forecast data
                        csv_forecast = valid_data.reset_index().to_csv(index=False)
                        st.download_button(
                            "Download Forecast Data as CSV",
                            csv_forecast,
                            f"Forecast_Data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv"
                        )
                except Exception as e:
                    st.error(f"An error occurred: {e}")
    else:
        st.error("Please upload a dataset first.")

# Machine Learning Models Function
def machine_learning_models():
    st.header("Machine Learning Models")

    # Check if data exists in session state
    if 'data' not in st.session_state or st.session_state['data'] is None:
        st.error("Please upload a dataset first.")
        return

    data = st.session_state['data']
    numeric_columns = data.select_dtypes(include=np.number).columns
    non_numeric_columns = data.select_dtypes(exclude=np.number).columns

    # Ensure there are numeric columns for predictors
    if len(numeric_columns) < 1:
        st.error("Dataset needs at least one numeric column for modeling.")
        return

    # Task Selection
    task_type = st.radio(
        "Select Task Type",
        ["Regression", "Classification", "Model Comparison"],
        index=0
    )

    # Persistent train/test splits
    if 'X_train' not in st.session_state:
        st.session_state.X_train, st.session_state.X_test = None, None
        st.session_state.y_train, st.session_state.y_test = None, None

    # Visualization options
    visualization_library = st.radio("Select Visualization Library", ["Seaborn", "Plotly"], index=0)

    if task_type in ["Regression", "Classification"]:
        target_column = st.selectbox(
            "Select Target Variable",
            numeric_columns if task_type == "Regression" else data.columns
        )
        predictor_columns = st.multiselect(
            "Select Predictor Variables",
            [col for col in numeric_columns if col != target_column]
        )
        test_size = st.slider("Select Test Set Size (%)", 10, 50, 20)

        if not predictor_columns:
            st.warning("Please select at least one predictor variable.")
            return

        scale_data = st.checkbox("Apply Feature Scaling", value=False)
        scaling_option = st.radio("Select Scaling Type", ["Standard Scaling", "Min-Max Scaling"], index=0) if scale_data else None

        model_type = st.selectbox(
            "Select Model",
            [
                "Linear Regression", "Random Forest Regressor", "Gradient Boosting Regressor",
                "Support Vector Regressor (SVR)", "K-Nearest Neighbors (KNN)"
            ] if task_type == "Regression" else [
                "Logistic Regression", "Random Forest Classifier", "Gradient Boosting Classifier",
                "Support Vector Machine (SVM)", "K-Nearest Neighbors (KNN)"
            ]
        )

        if st.button("Run Model"):
            # Encode non-numeric target for classification
            if task_type == "Classification" and target_column in non_numeric_columns:
                label_encoder = LabelEncoder()
                data[target_column] = label_encoder.fit_transform(data[target_column])

            # Check for missing data
            if data[predictor_columns].isnull().any().any() or data[target_column].isnull().any():
                st.error("Data contains missing values. Please clean the data before proceeding.")
                return

            # Prepare feature matrix X and target variable y
            X = data[predictor_columns]
            y = data[target_column]

            # Apply scaling if required
            if scale_data:
                scaler = StandardScaler() if scaling_option == "Standard Scaling" else MinMaxScaler()
                X = scaler.fit_transform(X)

            # Train/test split
            st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = train_test_split(
                X, y, test_size=test_size / 100, random_state=42
            )

            # Model Selection
            model = None
            if model_type == "Linear Regression":
                model = LinearRegression()
            elif model_type == "Random Forest Regressor":
                model = RandomForestRegressor(random_state=42)
            elif model_type == "Gradient Boosting Regressor":
                model = GradientBoostingRegressor(random_state=42)
            elif model_type == "Support Vector Regressor (SVR)":
                model = SVR(kernel='rbf')
            elif model_type == "K-Nearest Neighbors (KNN)":
                model = KNeighborsClassifier() if task_type == "Classification" else KNeighborsRegressor()
            elif model_type == "Logistic Regression":
                model = LogisticRegression()
            elif model_type == "Random Forest Classifier":
                model = RandomForestClassifier(random_state=42)
            elif model_type == "Gradient Boosting Classifier":
                model = GradientBoostingClassifier(random_state=42)
            elif model_type == "Support Vector Machine (SVM)":
                model = SVC(kernel='rbf', probability=True)

            # Train Model
            model.fit(st.session_state.X_train, st.session_state.y_train)
            predictions = model.predict(st.session_state.X_test)

            # Outputs and Visualizations
            if task_type == "Regression":
                mse = mean_squared_error(st.session_state.y_test, predictions)
                r2 = r2_score(st.session_state.y_test, predictions)
                st.write(f"Mean Squared Error (MSE): {mse:.2f}")
                st.write(f"R² Score: {r2:.2f}")

                # Visualization
                if visualization_library == "Seaborn":
                    fig, ax = plt.subplots()
                    sns.scatterplot(x=st.session_state.y_test, y=predictions, ax=ax, color='blue')
                    sns.lineplot(x=st.session_state.y_test, y=st.session_state.y_test, ax=ax, color='red')
                    ax.set_title("Regression Results")
                    st.pyplot(fig)
                elif visualization_library == "Plotly":
                    fig = px.scatter(
                        x=st.session_state.y_test, y=predictions,
                        labels={"x": "Actual", "y": "Predicted"},
                        title="Regression Results"
                    )
                    st.plotly_chart(fig)

            elif task_type == "Classification":
                accuracy = accuracy_score(st.session_state.y_test, predictions)
                precision = precision_score(st.session_state.y_test, predictions, average='weighted')
                recall = recall_score(st.session_state.y_test, predictions, average='weighted')
                f1 = f1_score(st.session_state.y_test, predictions, average='weighted')

                st.write(f"Accuracy: {accuracy:.2f}")
                st.write(f"Precision: {precision:.2f}")
                st.write(f"Recall: {recall:.2f}")
                st.write(f"F1 Score: {f1:.2f}")

                # Confusion Matrix
                if visualization_library == "Seaborn":
                    fig, ax = plt.subplots()
                    sns.heatmap(pd.crosstab(st.session_state.y_test, predictions), annot=True, cmap="Blues", ax=ax)
                    ax.set_title("Confusion Matrix")
                    st.pyplot(fig)
                elif visualization_library == "Plotly":
                    fig = px.imshow(
                        pd.crosstab(st.session_state.y_test, predictions),
                        text_auto=True,
                        color_continuous_scale="Blues",
                        labels={"x": "Predicted", "y": "Actual"},
                        title="Confusion Matrix"
                    )
                    st.plotly_chart(fig)

            # Download Predictions
            st.download_button(
                label="Download Predictions",
                data=pd.DataFrame({"Actual": st.session_state.y_test, "Predicted": predictions}).to_csv(index=False),
                file_name="predictions.csv",
                mime="text/csv"
            )

    elif task_type == "Model Comparison":
        selected_models = st.multiselect("Select Models to Compare", [
            "Logistic Regression", "Random Forest", "Gradient Boosting", "SVM", "K-Nearest Neighbors"
        ])
        metric_to_evaluate = st.selectbox("Select Metric", ["Accuracy", "Precision", "Recall", "F1-Score"])

        if st.button("Compare Models"):
            if st.session_state.X_train is None:
                st.error("Please run a Regression or Classification model first.")
                return

            results = {}
            for model_name in selected_models:
                if model_name == "Logistic Regression":
                    model = LogisticRegression()
                elif model_name == "Random Forest":
                    model = RandomForestClassifier(random_state=42)
                elif model_name == "Gradient Boosting":
                    model = GradientBoostingClassifier(random_state=42)
                elif model_name == "SVM":
                    model = SVC(kernel='rbf', probability=True)
                elif model_name == "K-Nearest Neighbors":
                    model = KNeighborsClassifier()

                model.fit(st.session_state.X_train, st.session_state.y_train)
                predictions = model.predict(st.session_state.X_test)

                if metric_to_evaluate == "Accuracy":
                    score = accuracy_score(st.session_state.y_test, predictions)
                elif metric_to_evaluate == "Precision":
                    score = precision_score(st.session_state.y_test, predictions, average='weighted')
                elif metric_to_evaluate == "Recall":
                    score = recall_score(st.session_state.y_test, predictions, average='weighted')
                elif metric_to_evaluate == "F1-Score":
                    score = f1_score(st.session_state.y_test, predictions, average='weighted')

                results[model_name] = score

            # Display Results
            st.table(pd.DataFrame.from_dict(results, orient='index', columns=[metric_to_evaluate]))

            # Visualization
            if visualization_library == "Seaborn":
                fig, ax = plt.subplots()
                sns.barplot(x=list(results.keys()), y=list(results.values()), ax=ax)
                ax.set_title(f"Model Comparison ({metric_to_evaluate})")
                ax.set_ylabel(metric_to_evaluate)
                st.pyplot(fig)
            elif visualization_library == "Plotly":
                fig = px.bar(
                    x=list(results.keys()), y=list(results.values()),
                    labels={"x": "Model", "y": metric_to_evaluate},
                    title=f"Model Comparison ({metric_to_evaluate})"
                )
                st.plotly_chart(fig)


# Predictive Analysis Function
def predictive_analysis():
    st.header("Predictive Analysis")

    # Check if data exists in session state
    if 'data' not in st.session_state or st.session_state['data'] is None:
        st.error("Please upload a dataset first.")
        return

    data = st.session_state['data']

    # Detect datetime columns
    time_columns = [
        col for col in data.columns
        if pd.api.types.is_datetime64_any_dtype(data[col]) or pd.to_datetime(data[col], errors='coerce').notna().all()
    ]
    numeric_columns = data.select_dtypes(include=np.number).columns

    # Ensure there are datetime columns for time series
    if len(time_columns) == 0:
        st.error("No valid datetime column found in the dataset.")
        return

    # Time series column selection
    time_column = st.selectbox("Select Time Column", time_columns)
    forecast_column = st.selectbox("Select Column to Forecast", numeric_columns)

    # Forecast horizon input
    prediction_period_type = st.radio("Select Prediction Period Type", ["Fixed Period", "Custom Date Range"], index=0)
    if prediction_period_type == "Fixed Period":
        forecast_horizon = st.slider("Select Forecast Horizon (Periods)", 1, 365, 7)
    else:
        start_date = st.date_input("Select Start Date for Prediction", value=data[time_column].max())
        end_date = st.date_input("Select End Date for Prediction", value=start_date + pd.Timedelta(days=30))
        if end_date <= start_date:
            st.error("End date must be after the start date.")
            return
        forecast_horizon = (end_date - start_date).days

    visualization_library = st.radio("Select Visualization Library", ["Matplotlib", "Seaborn"], index=0)

    if st.button("Run Predictive Analysis"):
        try:
            # Prepare the data
            df = data[[time_column, forecast_column]].dropna()
            df[time_column] = pd.to_datetime(df[time_column])
            df = df.set_index(time_column)
            st.write("### Data Preview")
            st.dataframe(df.head())

            # Use auto_arima for automatic order selection
            st.write("### ARIMA Model Training")
            arima_model = auto_arima(
                df[forecast_column],
                seasonal=False,
                trace=True,
                error_action='ignore',
                suppress_warnings=True
            )

            # Display ARIMA Model Summary in a collapsible section
            with st.expander("View ARIMA Model Summary"):
                st.text(arima_model.summary())

            # Forecasting
            forecast = arima_model.predict(n_periods=forecast_horizon)
            confidence_intervals = arima_model.predict(n_periods=forecast_horizon, return_conf_int=True)[1]
            forecast_dates = pd.date_range(start=df.index[-1], periods=forecast_horizon + 1, freq='D')[1:]

            forecast_df = pd.DataFrame({
                "Date": forecast_dates,
                "Forecast": forecast,
                "Lower Bound": confidence_intervals[:, 0],
                "Upper Bound": confidence_intervals[:, 1]
            })

            st.write("### Forecasted Results")
            st.dataframe(forecast_df)

            # Visualization with Confidence Interval
            st.write("### Forecast Visualization")
            if visualization_library == "Matplotlib":
                fig, ax = plt.subplots(figsize=(10, 6))
                df[forecast_column].plot(ax=ax, label="Actual", color="blue")
                ax.plot(forecast_df["Date"], forecast_df["Forecast"], label="Forecast", color="red")
                ax.fill_between(forecast_df["Date"], forecast_df["Lower Bound"], forecast_df["Upper Bound"], color='k', alpha=0.1, label="Confidence Interval")
                ax.set_title("Time Series Forecast with Confidence Interval")
                ax.set_xlabel("Date")
                ax.set_ylabel("Value")
                ax.legend()
                st.pyplot(fig)
            elif visualization_library == "Seaborn":
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.lineplot(data=df, x=df.index, y=forecast_column, label="Actual", ax=ax, color="blue")
                sns.lineplot(data=forecast_df, x="Date", y="Forecast", label="Forecast", ax=ax, color="red")
                ax.fill_between(forecast_df["Date"], forecast_df["Lower Bound"], forecast_df["Upper Bound"], color='k', alpha=0.1, label="Confidence Interval")
                ax.set_title("Time Series Forecast with Confidence Interval")
                ax.set_xlabel("Date")
                ax.set_ylabel("Value")
                ax.legend()
                st.pyplot(fig)

            # Residual Diagnostics
            st.write("### Residual Analysis")
            residuals = arima_model.arima_res_.resid

            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            sns.histplot(residuals, kde=True, ax=axs[0], color="green")
            axs[0].set_title("Residual Histogram")
            axs[0].set_xlabel("Residuals")
            axs[0].set_ylabel("Frequency")

            plot_acf(residuals, ax=axs[1], lags=20)
            axs[1].set_title("Residual Autocorrelation")
            st.pyplot(fig)

            # Metrics
            st.write("### Model Evaluation Metrics")
            mse = mean_squared_error(df[forecast_column][-forecast_horizon:], forecast[:len(df[forecast_column][-forecast_horizon:])])
            rmse = np.sqrt(mse)
            st.write(f"Mean Squared Error (MSE): {mse:.2f}")
            st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

            # Download forecast data
            st.download_button(
                label="Download Forecast Data",
                data=forecast_df.to_csv(index=False),
                file_name="forecast_data.csv",
                mime="text/csv"
            )

            # Download ARIMA model summary
            summary_text = arima_model.summary().as_text()
            st.download_button(
                label="Download ARIMA Model Summary",
                data=summary_text,
                file_name="arima_model_summary.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"An error occurred during predictive analysis: {e}")

#Probabilistic Insights function
def probabilistic_insights():
    st.header("Probabilistic Insights")
    
    # User inputs for prior and likelihood
    prior_mean = st.number_input("Prior Mean", value=0.5, min_value=0.0, max_value=1.0, step=0.1)
    prior_std = st.number_input("Prior Standard Deviation", value=0.1, min_value=0.01, max_value=1.0, step=0.01)
    likelihood_mean = st.number_input("Likelihood Mean", value=0.7, min_value=0.0, max_value=1.0, step=0.1)
    likelihood_std = st.number_input("Likelihood Standard Deviation", value=0.1, min_value=0.01, max_value=1.0, step=0.01)
    
    if st.button("Calculate Posterior"):
        posterior_mean = (prior_mean / prior_std**2 + likelihood_mean / likelihood_std**2) / \
                         (1 / prior_std**2 + 1 / likelihood_std**2)
        posterior_std = (1 / (1 / prior_std**2 + 1 / likelihood_std**2)) ** 0.5
        st.success(f"Posterior Mean: {posterior_mean:.3f}, Posterior Std Dev: {posterior_std:.3f}")
        
        # Visualization
        fig = plt.figure(figsize=(10, 5))
        x = np.linspace(0, 1, 500)
        plt.plot(x, stats.norm.pdf(x, prior_mean, prior_std), label="Prior", color="blue")
        plt.plot(x, stats.norm.pdf(x, likelihood_mean, likelihood_std), label="Likelihood", color="orange")
        plt.plot(x, stats.norm.pdf(x, posterior_mean, posterior_std), label="Posterior", color="green")
        plt.legend()
        plt.title("Bayesian Probability Distributions")
        st.pyplot(fig)

#Compare Multiple Datasets function
def compare_datasets():
    st.header("Compare Multiple Datasets")

    # Upload multiple datasets
    uploaded_files = st.file_uploader("Upload Multiple Datasets", type=["csv", "xlsx"], accept_multiple_files=True)
    if uploaded_files:
        datasets = {f.name: pd.read_csv(f) for f in uploaded_files}  # Load datasets into a dictionary

        # Display uploaded datasets
        st.write("Uploaded Datasets:")
        for name, df in datasets.items():
            st.write(f"**{name}**")
            st.dataframe(df.head())

        # Compare datasets
        selected_datasets = st.multiselect("Select Datasets to Compare", list(datasets.keys()))
        if len(selected_datasets) < 2:
            st.warning("Please select at least two datasets to compare.")
            return

        # Column comparison
        common_columns = set.intersection(*[set(datasets[ds].columns) for ds in selected_datasets])
        selected_column = st.selectbox("Select a Column for Comparison", list(common_columns))

        # Visualization
        if selected_column:
            fig = plt.figure(figsize=(10, 5))
            for ds in selected_datasets:
                sns.histplot(datasets[ds][selected_column], kde=True, label=ds, alpha=0.5)
            plt.legend()
            plt.title(f"Comparison of {selected_column}")
            st.pyplot(fig)

if __name__ == "__main__":
    main()
