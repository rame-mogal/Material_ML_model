import streamlit as st
import pandas as pd
import joblib

# Load trained model and encoder
model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")
X_train_columns = joblib.load("X_train_columns.pkl")  # Store column order

# Load reference material dataset
df = pd.read_excel("Cleaned_Filtered_Materials.xlsx")

# Unique sorted options
material_options = sorted(df['Material'].dropna().unique().tolist())
category_options = sorted(df['Category'].dropna().unique().tolist())
make_options = sorted(df['Make'].dropna().unique().tolist())
unit_options = sorted(df['Unit'].dropna().unique().tolist())

# App Title
st.title("Material Price Prediction")

# Sidebar input form
st.sidebar.header("Input Material Details")

material = st.sidebar.selectbox("Material", options=material_options)
category = st.sidebar.selectbox("Category", options=category_options)
make = st.sidebar.selectbox("Make", options=make_options)
unit = st.sidebar.selectbox("Unit", options=unit_options)

# Prediction
if st.sidebar.button("Predict Price"):
    # Create dataframe from input
    input_data = pd.DataFrame([{
        'Material': material,
        'Category': category,
        'Make': make,
        'Unit': unit
    }])

    # Encode categorical features
    encoded_features = encoder.transform(input_data[['Material', 'Category', 'Make', 'Unit']])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['Material', 'Category', 'Make', 'Unit']))

    # Add missing columns and align with training columns
    for col in set(X_train_columns) - set(encoded_df.columns):
        encoded_df[col] = 0
    encoded_df = encoded_df[X_train_columns]

    # Predict
    predicted_price = model.predict(encoded_df)[0]
    st.success(f"Predicted Price: ₹{predicted_price:.2f}")
