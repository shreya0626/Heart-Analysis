
# Heart Disease Prediction App ❤️

## Overview
The **Heart Disease Prediction App** is a user-friendly tool built with **Streamlit** that uses machine learning to predict the likelihood of heart disease. It also provides an intuitive interface for exploratory data analysis (EDA) and model training.

You can access the live app here: Heart Disease Prediction App.

## Features
1. **Heart Disease Prediction**:  
   - Users can input personal and medical details to predict their likelihood of having heart disease.
   - Provides actionable recommendations and thoughtful advice based on predictions.
   - Displays user input data and model accuracy after predictions.

2. **Model Training**:  
   - Trains a **Random Forest Classifier** on the provided medical dataset.
   - Displays the model's accuracy and feature importance.

3. **Exploratory Data Analysis (EDA)**:  
   - Displays dataset statistics and previews.
   - Visualizes:
     - Correlation matrix (heatmap).
     - Feature distributions.
     - Target variable distribution.

## Installation and Setup
1. Clone the repository or download the `app.py` file.
2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place your dataset (e.g., `Medicaldataset.csv`) in the same directory as the `app.py` file.
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Dataset Requirements
- The dataset should include the following:
  - `Gender`: Male/Female.
  - `Result`: Target column with `positive` or `negative` values.
  - Additional medical features relevant to heart disease prediction.
  
Ensure the file is named `Medicaldataset.csv` or modify the file path in the `app.py` file accordingly.

## Dependencies
This project requires the following Python libraries:
- `streamlit`
- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `scikit-learn`

Install these using:
```bash
pip install streamlit pandas numpy seaborn matplotlib scikit-learn
```

## Usage
1. **Check Your Health**:  
   - Input your details (e.g., gender, age, medical stats) in the app.
   - Click on the "Predict" button to see results.
   - View prediction outcomes, advice, and model accuracy.

2. **Model Training**:  
   - Navigate to the "Model Training" section in the sidebar.
   - Train a Random Forest model on the dataset and see feature importance.

3. **Data Analysis**:  
   - Explore the dataset, visualize correlations, and analyze feature distributions.

## Screenshots
(Include relevant screenshots of the app's interface here.)


## Future Improvements
- Add support for additional machine learning models.
- Enable data upload functionality for user-specific datasets.
- Enhance the UI for better interactivity and accessibility.

## License
This project is licensed under the MIT License.

---

Let me know if you'd like additional customization or clarification! 😊
