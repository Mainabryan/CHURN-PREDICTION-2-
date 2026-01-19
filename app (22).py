import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

# Set streamlit layout to wide
st.set_page_config(layout="wide")

# Load the trained model
with open('best_model.pkl','rb') as file:
    model = pickle.load(file)

# Load the MinMaxScaler
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# ================================================
# YOUR GITHUB IMAGE URLs (CORRECT FORMAT)
# ================================================
SIDEBAR_IMAGE_URL = "https://raw.githubusercontent.com/Mainabryan/CHURN-PREDICTION-2-/main/Screenshot%202026-01-19%20041922.png"
HEADER_IMAGE_URL = "https://raw.githubusercontent.com/Mainabryan/CHURN-PREDICTION-2-/main/Screenshot%202026-01-19%20041922.png"

# Define the input features for the model
Feature_names = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary',
                 'Geography_France', 'Geography_Germany', 'Geography_Spain', 'Gender_Female',
                 'Gender_Male', 'HasCrCard_0', 'HasCrCard_1', 'IsActiveMember_0', 'IsActiveMember_1']

# Columns requiring scaling
scales_vars = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

# Updated default values
default_values = {
    'CreditScore': 600,   
    'Age': 30,    
    'Tenure': 2,     
    'Balance': 60000.0, 
    'NumOfProducts': 1,     
    'EstimatedSalary': 50000.0, 
    'Geography_France': True,  
    'Geography_Germany': False, 
    'Geography_Spain': False, 
    'Gender_Female': True,  
    'Gender_Male': False, 
    'HasCrCard_0': False, 
    'HasCrCard_1': True,  
    'IsActiveMember_0': False, 
    'IsActiveMember_1': True   
}

# ================================================
# Sidebar with GitHub image
# ================================================
try:
    st.sidebar.image(SIDEBAR_IMAGE_URL, width=150, 
                    caption="Churn Analysis Dashboard")
except Exception as e:
    st.sidebar.warning(f"Sidebar image not loading: {e}")

st.sidebar.header('Customer Input Parameters')

# Collect user inputs
user_inputs = {}
for feature in Feature_names:
    if feature == 'CreditScore':
        user_inputs[feature] = st.sidebar.slider(feature, min_value=350, max_value=850, 
                                                value=default_values[feature], step=1)
    elif feature == 'Age':
        user_inputs[feature] = st.sidebar.slider(feature, min_value=18, max_value=92, 
                                                value=default_values[feature], step=1)
    elif feature == 'Tenure':
        user_inputs[feature] = st.sidebar.slider(feature, min_value=0, max_value=10, 
                                                value=default_values[feature], step=1)
    elif feature == 'Balance':
        user_inputs[feature] = st.sidebar.slider(feature, min_value=0.0, max_value=250000.0, 
                                                value=default_values[feature], step=100.0)
    elif feature == 'NumOfProducts':
        user_inputs[feature] = st.sidebar.slider(feature, min_value=1, max_value=4, 
                                                value=default_values[feature], step=1)
    elif feature == 'EstimatedSalary':
        user_inputs[feature] = st.sidebar.slider(feature, min_value=0.0, max_value=200000.0, 
                                                value=default_values[feature], step=100.0)
    elif isinstance(default_values[feature], bool):
        user_inputs[feature] = st.sidebar.selectbox(feature, [0, 1], 
                                                   index=1 if default_values[feature] else 0)
    else:
        user_inputs[feature] = st.sidebar.number_input(feature, value=default_values[feature])

# ================================================
# Main header with GitHub image
# ================================================
try:
    st.image(HEADER_IMAGE_URL, width=700, 
             caption="Customer Churn Prediction System")
except Exception as e:
    st.warning(f"Header image not loading: {e}")

st.title('Customer Churn Prediction Dashboard')

# Page layout
left_col, right_col = st.columns(2)

# Left page (Feature Importance)
with left_col:
    st.subheader('Feature Importance Analysis')
    try:
        # Load feature importance
        feature_importance = pd.read_excel('feature_importance.xlsx')
        # Plot the feature
        fig = px.bar(
            feature_importance.sort_values(by='Importance', ascending=False),
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top Features Influencing Churn',
            color='Importance',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load feature importance: {e}")

# Right page (Predictions)
with right_col:
    st.subheader('Churn Prediction')
    st.markdown("Click the button below to predict customer churn risk")
    
    if st.button('🔮 Predict Churn Risk', type='primary', use_container_width=True):
        # Convert user inputs to DataFrame
        input_df = pd.DataFrame([user_inputs])

        # Apply MinMaxScaler to the required columns
        input_df[scales_vars] = scaler.transform(input_df[scales_vars])

        # Make predictions
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]  # Probability of churn

        prediction_label = '🚨 HIGH RISK - Likely to CHURN' if prediction == 1 else '✅ LOW RISK - Likely to STAY'

        # Display results with better formatting
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        if prediction == 1:
            st.error(prediction_label)
        else:
            st.success(prediction_label)
            
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Churn Probability", f"{probability:.1%}")
        with col2:
            st.metric("Retention Probability", f"{(1-probability):.1%}")
            
        # Progress bar for visualization
        st.progress(probability)
        st.caption(f"Churn Risk Level: {probability:.1%}")

# Footer
st.markdown("---")
st.caption("Customer Churn Prediction App • Powered by Machine Learning • Built with Streamlit")
