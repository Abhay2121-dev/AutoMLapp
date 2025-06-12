import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, r2_score, mean_squared_error, 
                           mean_absolute_error, confusion_matrix, classification_report,
                           precision_recall_fscore_support, roc_curve, auc)
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set page config
st.set_page_config(page_title="ML Model Training App", layout="wide")

# Display current date/time and user information
current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
st.sidebar.write(f"Current Date and Time (UTC): {current_time}")
st.sidebar.write("Current User: Abhay2121-dev")

# Title
st.title("Machine Learning Model Training App")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'data_types' not in st.session_state:
    st.session_state.data_types = None

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the data
    st.session_state.df = pd.read_csv(uploaded_file)
    if st.session_state.data_types is None:
        st.session_state.data_types = {col: str(st.session_state.df[col].dtype) for col in st.session_state.df.columns}
    
    # Display initial missing values summary
    st.subheader("Missing Values Summary")
    missing_info = pd.DataFrame({
        'Column': st.session_state.df.columns,
        'Data Type': st.session_state.df.dtypes,
        'Missing Values': st.session_state.df.isnull().sum(),
        'Missing Percentage': (st.session_state.df.isnull().sum() / len(st.session_state.df) * 100).round(2)
    })
    missing_info_filtered = missing_info[missing_info['Missing Values'] > 0]
    
    if not missing_info_filtered.empty:
        st.write("Columns containing missing values:")
        missing_info_filtered['Missing Percentage'] = missing_info_filtered['Missing Percentage'].apply(lambda x: f"{x}%")
        st.dataframe(missing_info_filtered, use_container_width=True)
        
        # Visualization of missing values
        if st.checkbox("Show Missing Values Visualization"):
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=missing_info_filtered['Column'], 
                       y=missing_info_filtered['Missing Values'])
            plt.xticks(rotation=45, ha='right')
            plt.title('Missing Values by Column')
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.success("No missing values found in the dataset!")
    
    st.write(f"Dataset Shape: {st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} columns")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Data Cleaning", "Data Preview", "Model Training"])

# Data Cleaning Tab
with tab1:
    if st.session_state.df is not None:
        st.header("Data Cleaning")

        # Data type modification section
        st.subheader("Modify Data Types")
        st.write("Current data types for each column:")
        
        cols = st.columns(3)
        for idx, col in enumerate(st.session_state.df.columns):
            with cols[idx % 3]:
                new_type = st.selectbox(
                    f"{col}", 
                    ["int64", "float64", "object", "category", "datetime64[ns]"],
                    index=["int64", "float64", "object", "category", "datetime64[ns]"].index(
                        "object" if "datetime" in str(st.session_state.df[col].dtype) 
                        else str(st.session_state.df[col].dtype) if str(st.session_state.df[col].dtype) in ["int64", "float64", "object", "category"] 
                        else "object"
                    ),
                    key=f"dtype_{col}"
                )
                st.session_state.data_types[col] = new_type

        # Missing values handling section
        st.subheader("Handle Missing Values")
        missing_data = pd.DataFrame({
            'Missing Values': st.session_state.df.isnull().sum(),
            'Percentage': (st.session_state.df.isnull().sum() / len(st.session_state.df)) * 100
        })
        missing_data = missing_data[missing_data['Missing Values'] > 0]
        
        if not missing_data.empty:
            st.write("Columns with missing values:")
            st.dataframe(missing_data)
            
            for col in missing_data.index:
                st.write(f"\n{col}")
                handling_method = st.selectbox(
                    f"How to handle missing values in {col}?",
                    ["Select an option", "Drop rows", "Mean", "Median", "Mode", "Fill with value", "Drop column"],
                    key=f"missing_{col}"
                )
                
                if handling_method != "Select an option":
                    if handling_method == "Drop rows":
                        st.session_state.df = st.session_state.df.dropna(subset=[col])
                        st.success(f"Rows with missing values in {col} have been dropped.")
                    elif handling_method == "Mean":
                        if pd.api.types.is_numeric_dtype(st.session_state.df[col]):
                            st.session_state.df[col].fillna(st.session_state.df[col].mean(), inplace=True)
                            st.success(f"Missing values in {col} have been filled with mean.")
                        else:
                            st.error(f"Cannot calculate mean for non-numeric column {col}")
                    elif handling_method == "Median":
                        if pd.api.types.is_numeric_dtype(st.session_state.df[col]):
                            st.session_state.df[col].fillna(st.session_state.df[col].median(), inplace=True)
                            st.success(f"Missing values in {col} have been filled with median.")
                        else:
                            st.error(f"Cannot calculate median for non-numeric column {col}")
                    elif handling_method == "Mode":
                        mode_value = st.session_state.df[col].mode()[0]
                        st.session_state.df[col].fillna(mode_value, inplace=True)
                        st.success(f"Missing values in {col} have been filled with mode.")
                    elif handling_method == "Fill with value":
                        fill_value = st.text_input(f"Enter value to fill in {col}:", key=f"fill_{col}")
                        if st.button(f"Fill {col}", key=f"fill_button_{col}"):
                            try:
                                if pd.api.types.is_numeric_dtype(st.session_state.df[col]):
                                    st.session_state.df[col].fillna(float(fill_value), inplace=True)
                                else:
                                    st.session_state.df[col].fillna(fill_value, inplace=True)
                                st.success(f"Missing values in {col} have been filled with {fill_value}")
                            except ValueError:
                                st.error("Please enter a valid value")
                    elif handling_method == "Drop column":
                        st.session_state.df = st.session_state.df.drop(columns=[col])
                        st.success(f"Column {col} has been dropped.")
        else:
            st.write("No missing values found in the dataset!")

        # Apply data type changes button
        if st.button("Apply Data Type Changes"):
            try:
                for col, dtype in st.session_state.data_types.items():
                    if col in st.session_state.df.columns:
                        if dtype == "datetime64[ns]":
                            st.session_state.df[col] = pd.to_datetime(st.session_state.df[col])
                        else:
                            st.session_state.df[col] = st.session_state.df[col].astype(dtype)
                st.success("Data types updated successfully!")
            except Exception as e:
                st.error(f"Error updating data types: {str(e)}")

# Data Preview Tab
with tab2:
    if st.session_state.df is not None:
        st.header("Data Preview")
        st.write(st.session_state.df.head())
        
        st.subheader("Dataset Information")
        buffer = StringIO()
        st.session_state.df.info(buf=buffer)
        st.text(buffer.getvalue())
        
        st.subheader("Basic Statistics")
        st.write(st.session_state.df.describe())

# Model Training Tab
with tab3:
    if st.session_state.df is not None:
        st.header("Model Training")
        
        # Select target variable
        target_variable = st.selectbox(
            "Select Target Variable",
            st.session_state.df.columns.tolist()
        )
        
        # Model selection
        model_types = {
            "Random Forest": (RandomForestClassifier, RandomForestRegressor),
            "Linear Model": (LogisticRegression, LinearRegression),
            "Gradient Boosting": (GradientBoostingClassifier, GradientBoostingRegressor),
            "XGBoost": (XGBClassifier, XGBRegressor)
        }
        
        selected_model = st.selectbox(
            "Select Model Type",
            list(model_types.keys())
        )
        
        # Validation strategy
        validation_strategy = st.radio(
            "Select Validation Strategy",
            ["Train-Test Split", "Cross Validation"]
        )
        
        if validation_strategy == "Train-Test Split":
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
        else:
            n_folds = st.slider("Number of Folds", 2, 10, 5)
        
        # Force classification/regression selection
        problem_type = st.radio(
            "Force Problem Type",
            ["Auto Detect", "Classification", "Regression"]
        )
        
        if st.button("Train Model"):
            try:
                # Prepare the data
                X = st.session_state.df.drop(columns=[target_variable])
                y = st.session_state.df[target_variable]
                
                # Handle categorical variables
                categorical_columns = X.select_dtypes(include=['object', 'category']).columns
                for col in categorical_columns:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                
                # Determine if classification or regression
                if problem_type == "Auto Detect":
                    is_classification = y.dtype == 'object' or y.dtype == 'category' or len(np.unique(y)) < 10
                else:
                    is_classification = problem_type == "Classification"
                
                if is_classification and y.dtype != 'category':
                    le = LabelEncoder()
                    y = le.fit_transform(y.astype(str))
                
                if validation_strategy == "Train-Test Split":
                    # Split the data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )
                    
                    # Select and train the model
                    if is_classification:
                        model = model_types[selected_model][0]()
                    else:
                        model = model_types[selected_model][1]()
                    
                    st.info("Training in progress... Please wait.")
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Display results
                    st.subheader("Model Performance")
                    
                    if is_classification:
                        # Calculate metrics
                        accuracy = accuracy_score(y_test, y_pred)
                        conf_matrix = confusion_matrix(y_test, y_pred)
                        class_report = classification_report(y_test, y_pred)
                        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("Classification Metrics:")
                            st.write(f"Accuracy: {accuracy:.4f}")
                            st.write(f"Precision: {precision:.4f}")
                            st.write(f"Recall: {recall:.4f}")
                            st.write(f"F1 Score: {f1:.4f}")
                        
                        with col2:
                            st.write("Confusion Matrix:")
                            fig, ax = plt.subplots(figsize=(8, 6))
                            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
                            plt.title('Confusion Matrix')
                            plt.ylabel('True Label')
                            plt.xlabel('Predicted Label')
                            st.pyplot(fig)
                        
                        st.subheader("Detailed Classification Report")
                        st.text(class_report)
                        
                        # ROC Curve for binary classification
                        if len(np.unique(y)) == 2:
                            fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
                            roc_auc = auc(fpr, tpr)
                            
                            fig, ax = plt.subplots(figsize=(8, 6))
                            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                                    label=f'ROC curve (AUC = {roc_auc:.2f})')
                            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                            plt.xlim([0.0, 1.0])
                            plt.ylim([0.0, 1.05])
                            plt.xlabel('False Positive Rate')
                            plt.ylabel('True Positive Rate')
                            plt.title('Receiver Operating Characteristic (ROC) Curve')
                            plt.legend(loc="lower right")
                            st.pyplot(fig)
                    else:
                        r2 = r2_score(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y_test, y_pred)
                        
                        st.write(f"R²: {r2:.4f}")
                        st.write(f"MSE: {mse:.4f}")
                        st.write(f"RMSE: {rmse:.4f}")
                        st.write(f"MAE: {mae:.4f}")
                        
                        # Scatter plot of predicted vs actual values
                        fig, ax = plt.subplots(figsize=(8, 6))
                        plt.scatter(y_test, y_pred, alpha=0.5)
                        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                        plt.xlabel('Actual Values')
                        plt.ylabel('Predicted Values')
                        plt.title('Actual vs Predicted Values')
                        st.pyplot(fig)
                
                else:  # Cross Validation
                    # Select and initialize the model
                    if is_classification:
                        model = model_types[selected_model][0]()
                        scoring = 'accuracy'
                    else:
                        model = model_types[selected_model][1]()
                        scoring = 'r2'
                    
                    # Perform cross-validation
                    cv_scores = cross_val_score(model, X, y, cv=n_folds, scoring=scoring)
                    
                    # Display cross-validation results
                    st.subheader("Cross Validation Results")
                    st.write(f"Mean {scoring}: {cv_scores.mean():.4f}")
                    st.write(f"Standard Deviation: {cv_scores.std():.4f}")
                    
                    # Box plot of cross-validation scores
                    fig, ax = plt.subplots(figsize=(8, 6))
                    plt.boxplot(cv_scores)
                    plt.title(f'Cross-validation Scores ({n_folds} folds)')
                    plt.ylabel(scoring.capitalize())
                    st.pyplot(fig)
                
                # Feature importance plot
                if hasattr(model, 'feature_importances_'):
                    st.subheader("Feature Importance")
                    importances = pd.DataFrame({
                        'feature': X.columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    plt.bar(importances['feature'][:10], importances['importance'][:10])
                    plt.xticks(rotation=45, ha='right')
                    plt.title('Top 10 Feature Importance')
                    st.pyplot(fig)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.write("Error details:", e.__class__.__name__)
                st.write("Please check your data and try again.")