import streamlit as st
import pandas as pd
import joblib

# Load the trained model
# The model expects all 13 features it was trained on.
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# --- Preprocessing Mappings ---
# These dictionaries replicate the LabelEncoder from the notebook.
# The keys are the user-friendly options, and the values are the encoded numbers the model expects.

education_map = {
    "Bachelors": 13, "HS-grad": 9, "11th": 7, "Masters": 14, "9th": 5, "Some-college": 10,
    "Assoc-acdm": 12, "Assoc-voc": 11, "7th-8th": 4, "Doctorate": 16, "Prof-school": 15,
    "5th-6th": 3, "10th": 6, "1st-4th": 2, "Preschool": 1, "12th": 8
}

workclass_map = {
    "Private": 3, "Self-emp-not-inc": 5, "Local-gov": 1, "State-gov": 6, "Self-emp-inc": 4,
    "Federal-gov": 0, "Others": 2
}

occupation_map = {
    "Prof-specialty": 10, "Craft-repair": 2, "Exec-managerial": 3, "Adm-clerical": 0, "Sales": 12,
    "Other-service": 7, "Machine-op-inspct": 6, "Transport-moving": 14, "Handlers-cleaners": 5,
    "Farming-fishing": 4, "Tech-support": 13, "Protective-serv": 11, "Priv-house-serv": 9,
    "Armed-Forces": 1, "Others": 8
}

marital_status_map = {
    "Married-civ-spouse": 2, "Never-married": 4, "Divorced": 0, "Separated": 5,
    "Widowed": 6, "Married-spouse-absent": 3, "Married-AF-spouse": 1
}

relationship_map = {
    "Husband": 0, "Not-in-family": 1, "Own-child": 3, "Unmarried": 4, "Wife": 5, "Other-relative": 2
}

race_map = {
    "White": 4, "Black": 2, "Asian-Pac-Islander": 1, "Amer-Indian-Eskimo": 0, "Other": 3
}

gender_map = {
    "Male": 1, "Female": 0
}

# The notebook encoded 42 countries. We'll use a smaller list for the app.
# The most common value is 'United-States', which encodes to 39.
native_country_map = {
    "United-States": 39, "Mexico": 26, "Philippines": 30, "Germany": 11, "Canada": 2, "Puerto-Rico": 32, "Other": 41
}


# --- Sidebar Inputs ---
# These must match your training feature columns from the notebook
st.sidebar.header("Input Employee Details")

# Numerical Inputs
age = st.sidebar.slider("Age", 17, 75, 30)

fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", 12285, 1490400, 189778)
capital_gain = st.sidebar.number_input("Capital Gain", 0, 99999, 0)
capital_loss = st.sidebar.number_input("Capital Loss", 0, 4356, 0)
hours_per_week = st.sidebar.slider("Hours per week", 1, 99, 40)

# Categorical Inputs (using the maps for encoding)
education_label = st.sidebar.selectbox("Education Level", list(education_map.keys()))
educational_num = education_map[education_label]

workclass_label = st.sidebar.selectbox("Work Class", list(workclass_map.keys()))
workclass = workclass_map[workclass_label]

occupation_label = st.sidebar.selectbox("Job Role", list(occupation_map.keys()))
occupation = occupation_map[occupation_label]

marital_status_label = st.sidebar.selectbox("Marital Status", list(marital_status_map.keys()))
marital_status = marital_status_map[marital_status_label]

relationship_label = st.sidebar.selectbox("Relationship", list(relationship_map.keys()))
relationship = relationship_map[relationship_label]

race_label = st.sidebar.selectbox("Race", list(race_map.keys()))
race = race_map[race_label]

gender_label = st.sidebar.selectbox("Gender", list(gender_map.keys()))
gender = gender_map[gender_label]

native_country_label = st.sidebar.selectbox("Native Country", list(native_country_map.keys()))
native_country = native_country_map[native_country_label]


# Build input DataFrame (⚠️ must match the columns and order from training)
input_data = {
    'age': [age],
    'workclass': [workclass],
    'fnlwgt': [fnlwgt],
    'educational-num': [educational_num],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country]
}
input_df = pd.DataFrame(input_data)

st.write("### 🔎 Input Data (after encoding)")
st.write(input_df)

# Predict button
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f"✅ Prediction: {prediction[0]}")

# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
st.markdown("Upload a CSV file with the same 13 columns as the training data (e.g., `age`, `workclass`, `fnlwgt`...). The data must already be preprocessed (label encoded).")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)

        # Ensure the uploaded data has the correct columns
        expected_cols = list(input_data.keys())
        if list(batch_data.columns) != expected_cols:
            st.error(f"The uploaded CSV does not have the correct columns. Expected: {expected_cols}")
        else:
            st.write("Uploaded data preview:", batch_data.head())
            batch_preds = model.predict(batch_data)
            batch_data['PredictedClass'] = batch_preds
            st.write("✅ Predictions:")
            st.write(batch_data.head())
            csv = batch_data.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
    except Exception as e:
        st.error(f"An error occurred during batch prediction: {e}")
