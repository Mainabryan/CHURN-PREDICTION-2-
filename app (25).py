import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# ================================================
# PAGE CONFIGURATION
# ================================================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================
# YOUR GITHUB IMAGE URLs
# ================================================
HEADER_IMAGE_URL = "https://raw.githubusercontent.com/Mainabryan/CHURN-PREDICTION-2-/main/Screenshot%202026-01-19%20041922.png"
SIDEBAR_IMAGE_URL = "https://raw.githubusercontent.com/Mainabryan/CHURN-PREDICTION-2-/main/Screenshot%202026-01-19%20041922.png"

# ================================================
# FIX: GET FEATURE ORDER FROM MODEL ITSELF
# ================================================
@st.cache_resource
def load_resources():
    try:
        with open('best_model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        
        # Try to get feature names from model
        try:
            if hasattr(model, 'feature_names_in_'):
                feature_order = list(model.feature_names_in_)
            elif hasattr(model, 'get_booster'):
                feature_order = model.get_booster().feature_names
            else:
                # Default order from error message
                feature_order = [
                    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary',
                    'Geography_France', 'Geography_Germany', 'Geography_Spain',
                    'Gender_Female', 'Gender_Male',
                    'HasCrCard_1', 'HasCrCard_0',  # CRITICAL: This order!
                    'IsActiveMember_1', 'IsActiveMember_0'  # CRITICAL: This order!
                ]
        except:
            # Fallback order
            feature_order = [
                'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary',
                'Geography_France', 'Geography_Germany', 'Geography_Spain',
                'Gender_Female', 'Gender_Male',
                'HasCrCard_1', 'HasCrCard_0',
                'IsActiveMember_1', 'IsActiveMember_0'
            ]
        
        return model, scaler, feature_order
        
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        return None, None, None

model, scaler, FEATURE_ORDER = load_resources()
SCALING_FEATURES = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

# ================================================
# ENLARGED HEADER SECTION
# ================================================
st.markdown(
    """
    <style>
    .big-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 30px;
        color: white;
    }
    .sidebar-image {
        text-align: center;
        padding: 20px 0;
    }
    .stButton>button {
        width: 100%;
        font-size: 1.2rem;
        height: 50px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header image - FULL WIDTH
try:
    st.image(HEADER_IMAGE_URL, caption="Customer Churn Prediction Dashboard")
except:
    st.markdown('<div class="big-header"><h1>🏦 Customer Churn Prediction</h1></div>', unsafe_allow_html=True)

st.markdown("---")

# ================================================
# SIDEBAR WITH LARGER IMAGE
# ================================================
with st.sidebar:
    try:
        st.image(SIDEBAR_IMAGE_URL, width=300, caption="Analysis Dashboard")
    except:
        st.markdown("### 🔮 Churn Predictor")
    
    st.markdown("---")
    st.header("📋 Customer Details")
    
    # User inputs
    st.subheader("📊 Financial Details")
    credit_score = st.slider("Credit Score", 300, 850, 650)
    balance = st.slider("Account Balance (€)", 0, 250000, 50000, step=1000)
    salary = st.slider("Estimated Salary (€)", 0, 250000, 80000, step=1000)
    
    st.subheader("👤 Personal Details")
    age = st.slider("Age", 18, 100, 38)
    tenure = st.slider("Tenure (years)", 0, 10, 3)
    num_products = st.slider("Number of Products", 1, 4, 2)
    
    st.subheader("📍 Geography")
    geography = st.selectbox("Country", ["France", "Germany", "Spain"])
    
    st.subheader("⚡ Account Status")
    has_credit_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active = st.selectbox("Active Member?", ["Yes", "No"])
    
    st.subheader("👫 Gender")
    gender = st.selectbox("Gender", ["Male", "Female"])

# ================================================
# PREPARE DATA IN CORRECT ORDER
# ================================================
if FEATURE_ORDER:
    user_data = {}
    for feature in FEATURE_ORDER:
        if feature == 'CreditScore':
            user_data[feature] = credit_score
        elif feature == 'Age':
            user_data[feature] = age
        elif feature == 'Tenure':
            user_data[feature] = tenure
        elif feature == 'Balance':
            user_data[feature] = balance
        elif feature == 'NumOfProducts':
            user_data[feature] = num_products
        elif feature == 'EstimatedSalary':
            user_data[feature] = salary
        elif feature == 'Geography_France':
            user_data[feature] = (geography == "France")
        elif feature == 'Geography_Germany':
            user_data[feature] = (geography == "Germany")
        elif feature == 'Geography_Spain':
            user_data[feature] = (geography == "Spain")
        elif feature == 'Gender_Female':
            user_data[feature] = (gender == "Female")
        elif feature == 'Gender_Male':
            user_data[feature] = (gender == "Male")
        elif feature == 'HasCrCard_1':
            user_data[feature] = (has_credit_card == "Yes")
        elif feature == 'HasCrCard_0':
            user_data[feature] = (has_credit_card == "No")
        elif feature == 'IsActiveMember_1':
            user_data[feature] = (is_active == "Yes")
        elif feature == 'IsActiveMember_0':
            user_data[feature] = (is_active == "No")

# ================================================
# MAIN CONTENT AREA
# ================================================
st.title("📈 Customer Churn Analysis Dashboard")

# Create two columns
col1, col2 = st.columns([1.2, 1])

# COLUMN 1: FEATURE IMPORTANCE
with col1:
    st.markdown("### 🔍 Feature Importance Analysis")
    
    try:
        feature_importance = pd.read_excel('feature_importance.xlsx')
        fig = px.bar(
            feature_importance.sort_values('Importance', ascending=True),
            x='Importance',
            y='Feature',
            orientation='h',
            title='<b>Top Factors Influencing Customer Churn</b>',
            color='Importance',
            color_continuous_scale='Viridis',
            height=500
        )
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Could not load feature importance: {e}")
        sample_data = pd.DataFrame({
            'Feature': ['Age', 'Balance', 'CreditScore', 'NumOfProducts', 'Tenure'],
            'Importance': [0.25, 0.20, 0.18, 0.15, 0.12]
        })
        fig = px.bar(sample_data, x='Importance', y='Feature', orientation='h')
        st.plotly_chart(fig)

# COLUMN 2: PREDICTION
with col2:
    st.markdown("### 🎯 Make Prediction")
    
    if st.button("🚀 **Predict Churn Risk**", type="primary"):
        if model is None or scaler is None or FEATURE_ORDER is None:
            st.error("Model not loaded. Please check if model files are available.")
        else:
            with st.spinner("Analyzing customer data..."):
                # Create DataFrame IN CORRECT ORDER
                input_df = pd.DataFrame([user_data])
                input_df = input_df[FEATURE_ORDER]
                
                # Scale features
                input_df[SCALING_FEATURES] = scaler.transform(input_df[SCALING_FEATURES])
                
                # Make prediction
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0][1]
                
                # Display results
                st.markdown("---")
                st.markdown("### 📊 Prediction Results")
                
                if prediction == 1:
                    st.error("## 🚨 HIGH RISK - Likely to CHURN")
                else:
                    st.success("## ✅ LOW RISK - Likely to STAY")
                
                # Probability display
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Churn Probability", f"{probability:.1%}")
                with col_b:
                    st.metric("Retention Probability", f"{(1-probability):.1%}")
                
                # Progress bar
                st.progress(float(probability))
                
                # Customer summary
                st.markdown("---")
                st.markdown("#### 📋 Customer Summary")
                st.write(f"**Age:** {age} years")
                st.write(f"**Balance:** €{balance:,}")
                st.write(f"**Credit Score:** {credit_score}")
                st.write(f"**Tenure:** {tenure} years")
                st.write(f"**Active Member:** {is_active}")
                st.write(f"**Geography:** {geography}")

# ================================================
# FOOTER
# ================================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>Customer Churn Prediction System • Powered by Machine Learning</p>
        <p>Built with ❤️ using Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
