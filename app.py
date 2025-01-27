import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("Medicaldataset.csv")  # Specify the correct path to your dataset

# Encode 'Gender' column (if exists)
if 'Gender' in data.columns:
    data['Gender'] = data['Gender'].map({'male': 1, 'female': 0})

# Encode 'Result' column to numeric (negative=0, positive=1)
if 'Result' in data.columns:
    data['Result'] = data['Result'].map({'negative': 0, 'positive': 1})

# Sidebar for navigation
st.sidebar.title("Heart Disease Prediction❤️")
option = st.sidebar.selectbox("Select a section", ("Check your health", "Model Training", "Data Analysis"))

# Main app logic based on selected option
if option == "Check your health":
    st.title("Heart Disease Prediction🩷")
    st.write("Please enter the details below to predict heart disease:")

    # Create input fields for user input
    gender = st.selectbox("Select Gender", ["Female", "Male"])
    sample_input = {col: st.number_input(f"Enter {col}", value=0.0) for col in data.columns if col != 'Result' and col != 'Gender'}

    # Map gender to 0 for Female, 1 for Male
    sample_input['Gender'] = 0 if gender == "Female" else 1

    # Store the user input data in a DataFrame
    user_data = pd.DataFrame([sample_input])

    # Model training and prediction
    if 'Result' in data.columns:
        X = data.drop('Result', axis=1)
        y = data['Result']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scaling the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train the model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Make prediction on user input after scaling
        if st.button("Predict"):
            user_data_scaled = scaler.transform(user_data[X.columns])
            prediction = model.predict(user_data_scaled)
            result = "Positive for Heart Disease" if prediction[0] == 1 else "Negative for Heart Disease"
            st.write(f"### Prediction Result: {result}")

            # Thoughtful message based on result
            if prediction[0] == 1:
                st.write("It's important to consult with a healthcare professional for further diagnosis and potential preventive measures. Stay healthy and take care!😊")
            else:
                st.write("Great! Your results indicate no heart disease risk. Keep maintaining a healthy lifestyle!😊")

            # Show user input
            st.write("### User Input Submitted:")
            st.dataframe(user_data)

            # Model Evaluation and Accuracy
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"### Model Accuracy: {accuracy * 100:.2f}%")

elif option == "Model Training":
    st.title("Heart Disease Prediction - Model Training")
    st.write("This section trains a Random Forest model and evaluates its performance.")

    # Split the data for training
    if 'Result' in data.columns:
        X = data.drop('Result', axis=1)
        y = data['Result']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scaling the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train the model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Model Evaluation
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"### Model Accuracy: {accuracy * 100:.2f}%")

        # Display the feature importance
        st.write("### Feature Importance")
        importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)

        st.write(feature_importance_df)

elif option == "Data Analysis":
    st.title("Heart Disease Prediction - EDA")
    st.write("This section provides an overview of the dataset through EDA.")

    # Show dataset preview
    st.write("### Dataset Preview:")
    st.dataframe(data.head())

    # Show dataset statistics
    st.write("### Dataset Statistics:")
    st.write(data.describe())

    # Visualize correlation matrix
    st.write("### Correlation Heatmap:")
    # Calculate correlation only on numeric columns
    corr = data.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='hot', fmt='.2f', linewidths=0.5, ax=ax)
    st.pyplot(fig)

    # Visualize feature distributions
    st.write("### Feature Distributions:")
    # Remove specific columns from distribution plots
    exclude_columns = ['CK-MB', 'Troponin', 'Systolic blood pressure', 'Diastolic blood pressure']
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    numeric_columns = [col for col in numeric_columns if col not in exclude_columns]

    for col in numeric_columns:
        st.write(f"#### {col} Distribution")
        fig, ax = plt.subplots()
        sns.histplot(data[col], kde=True, ax=ax)
        st.pyplot(fig)

    # Count plot for the target variable (Result)
    st.write("### Target Variable Distribution (Result):")
    fig, ax = plt.subplots()
    sns.countplot(x='Result', data=data, ax=ax)
    st.pyplot(fig)
