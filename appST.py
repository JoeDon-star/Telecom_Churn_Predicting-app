import streamlit as st
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Load the model
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as file:
        return joblib.load(file)

model = load_model()

# Define expected feature names (must match trained model)
feature_names = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
    'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'InternetService', 'Contract', 'PaymentMethod',
    'tenure', 'MonthlyCharges', 'TotalCharges'
]

# Streamlit app title
st.title("Telecom Churn Prediction")
st.divider()
st.markdown("""This application predicts the likelihood of a customer churning based on various features. 
Please upload a CSV file with customer data or fill out the form at the side bar and click the predict button below to get the churn prediction.""")
st.divider()

#Ploting the list of the importance features
feature_importance = model.feature_importances_

feature_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values(by='Importance', ascending=False)

st.subheader("Feature Importance")
st.write("The chart below indicates how variables contribute to the model prediction")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=feature_df, x='Importance',y= 'Feature', hue='Feature',legend = False, ax=ax, palette="cividis")
ax.set_title("Feature Importance for Churn Prediction")
st.pyplot(fig)


# Sidebar setup
st.sidebar.image("dataset-cover.png", use_container_width=True)
st.sidebar.title("Telecom Churn Prediction")
st.sidebar.markdown("""[Example CSV input file](https://drive.google.com/file/d/1R4udxBpVc4l9zuJ5oPiSEKhWRwdyoYwS/view?usp=sharing)""")
uploaded_file = st.sidebar.file_uploader("Upload your csv file", type=["csv"])

# Encoding mappings
encoding_maps = {
    'gender': {"Female": 0, "Male": 1},
    'radio': {"Yes": 1, "No": 0},
    'options': {"No": 0, "No internet/phone service": 1, "Yes": 2},
    'contract': {"Month-to-month": 0, "One year": 1, "Two year": 2},
    'payment_method': {
        "Bank transfer (automatic)": 0, "Credit card (automatic)": 1, 
        "Electronic check": 2, "Mailed check": 3
    },
    'internet_service': {"DSL": 0, "Fiber optic": 1, "No": 2}
}

# Function to encode categorical data
def encode_data(input_df):
    input_df['gender'] = input_df['gender'].map(encoding_maps['gender'])
    input_df['SeniorCitizen'] = input_df['SeniorCitizen'].map(encoding_maps['radio'])
    input_df['Partner'] = input_df['Partner'].map(encoding_maps['radio'])
    input_df['Dependents'] = input_df['Dependents'].map(encoding_maps['radio'])
    input_df['PhoneService'] = input_df['PhoneService'].map(encoding_maps['radio'])
    input_df['PaperlessBilling'] = input_df['PaperlessBilling'].map(encoding_maps['radio'])

    # Map multiple options for categorical columns
    options_columns = [
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    for col in options_columns:
        input_df[col] = input_df[col].map(encoding_maps['options'])

    input_df['InternetService'] = input_df['InternetService'].map(encoding_maps['internet_service'])
    input_df['Contract'] = input_df['Contract'].map(encoding_maps['contract'])
    input_df['PaymentMethod'] = input_df['PaymentMethod'].map(encoding_maps['payment_method'])
    
    # Convert numeric columns safely
    input_df['tenure'] = pd.to_numeric(input_df['tenure'], errors='coerce')
    input_df['MonthlyCharges'] = pd.to_numeric(input_df['MonthlyCharges'], errors='coerce')
    input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce')
    
    return input_df #If NumPy array is required

# Function to process uploaded CSV file
def handle_uploaded_file(uploaded_file):
        input_df = pd.read_csv(uploaded_file)

        st.write("### Preview of Uploaded Data")
        st.write(input_df.head())

        # Ensure required columns are present
        missing_columns = [col for col in feature_names if col not in input_df.columns]
        if missing_columns:
            st.error(f"Missing columns in uploaded file: {missing_columns}")
            return None

        # Encode data
        encoded_data = encode_data(input_df)
        if encoded_data is None:
            return None

        # Ensure input has only the required 19 features
        encoded_data = encoded_data[feature_names]

        # Process multiple rows for prediction
        predictions = []
        for index, row in encoded_data.iterrows():
            try:
                # Convert row to 2D array for prediction
                features = row.values.reshape(1, -1)

                # Make predictions
                prediction = model.predict(features)
                probability = model.predict_proba(features)[0][1]  # Get churn probability

                prediction_result = 'Churn' if prediction[0] == 1 else 'Not Churn'
                predictions.append({
                    'CustomerID': input_df.iloc[index]['CustomerID'] if 'CustomerID' in input_df.columns else index,
                    'Prediction': prediction_result,
                    'Churn Probability': f'{probability * 100:.2f}%'
                })
            except Exception as e:
                st.error(f"Error processing row {index}: {e}")
                continue

        # Convert predictions to DataFrame and display
        prediction_df = pd.DataFrame(predictions)
        st.write("### Prediction Results")
        st.write(prediction_df)

# Function for user input form
def user_input_form():
    st.sidebar.divider()
    st.sidebar.header("Enter Customer Information")
    st.sidebar.divider()

    # Radio options
    gender = st.sidebar.radio("Select gender", ['Female', 'Male'])
    SeniorCitizen = st.sidebar.radio("Is the customer a senior citizen", ['Yes', 'No'])
    Partner = st.sidebar.radio("Do they have a partner", ['Yes', 'No'])
    Dependents = st.sidebar.radio("Do they have a dependent", ['Yes', 'No'])
    PhoneService = st.sidebar.radio("Do they have a phone service", ['Yes', 'No'])
    PaperlessBilling = st.sidebar.radio("Do they have paperless billing", ['Yes', 'No'])

    # Numerical inputs
    tenure = st.sidebar.number_input("Enter tenure(month)", min_value=0, max_value=72, value=30)
    MonthlyCharges = st.sidebar.number_input("Enter monthly charge", min_value=18.25, max_value=118.75, value=30.00)
    TotalCharges = st.sidebar.number_input("Enter total charge", min_value=18.80, max_value=8684.80, value=3300.00)

    # Option selection (Yes/No/No internet service)
    options = ["No", "No internet/phone service", "Yes"]
    MultipleLines = st.sidebar.selectbox("Do they have multiple lines", options)
    OnlineSecurity = st.sidebar.selectbox("Do they have online security", options)
    OnlineBackup = st.sidebar.selectbox("Do they have online backup", options)
    DeviceProtection = st.sidebar.selectbox("Do they have device protection", options)
    TechSupport = st.sidebar.selectbox("Do they have tech support", options)
    StreamingTV = st.sidebar.selectbox("Do they have streaming TV", options)
    StreamingMovies = st.sidebar.selectbox("Do they have streaming Movies", options)

    # Internet service and contract
    InternetService = st.sidebar.selectbox("Which internet service do they use", ["DSL", "Fiber optic", "No"])
    Contract = st.sidebar.selectbox("State their contract", ["Month-to-month", "One year", "Two year"])
    PaymentMethod = st.sidebar.selectbox("What is their payment method", ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"])

    # Encoding the form data
    encoded_data = {
        'gender': encoding_maps['gender'][gender],
        'SeniorCitizen': encoding_maps['radio'][SeniorCitizen],
        'Partner': encoding_maps['radio'][Partner],
        'Dependents': encoding_maps['radio'][Dependents],
        'PhoneService': encoding_maps['radio'][PhoneService],
        'PaperlessBilling': encoding_maps['radio'][PaperlessBilling],
        'MultipleLines': encoding_maps['options'][MultipleLines],
        'OnlineSecurity': encoding_maps['options'][OnlineSecurity],
        'OnlineBackup': encoding_maps['options'][OnlineBackup],
        'DeviceProtection': encoding_maps['options'][DeviceProtection],
        'TechSupport': encoding_maps['options'][TechSupport],
        'StreamingTV': encoding_maps['options'][StreamingTV],
        'StreamingMovies': encoding_maps['options'][StreamingMovies],
        'InternetService': encoding_maps['internet_service'][InternetService],
        'Contract': encoding_maps['contract'][Contract],
        'PaymentMethod': encoding_maps['payment_method'][PaymentMethod],
        'tenure': tenure,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }

    return pd.DataFrame([encoded_data])

# Handle uploaded file or user form input
if uploaded_file is not None:
    encoded_data = handle_uploaded_file(uploaded_file)  # Process uploaded file
else:
    encoded_data = user_input_form()  # Get user input from form

# Disable Predict button if there's no valid input
predict_disabled = encoded_data is None

# Prediction on input data
st.divider()
st.write("Get prediction on the input data")

if st.button("Predict", disabled=predict_disabled):
    try:
        prediction = model.predict(encoded_data.values)
        prediction_prob = model.predict_proba(encoded_data.values)

        churn_prob = prediction_prob[0][1]
        not_churn_prob = prediction_prob[0][0]

        if prediction[0] == 1:
            st.success("Prediction: Churn")
        else:
            st.success("Prediction: Not Churn")

        st.write(f"Probability of Churn: {churn_prob:.2f}")
        st.write(f"Probability of Not Churn: {not_churn_prob:.2f}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
