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
# CRITICAL FIX: CORRECT FEATURE ORDER
# ================================================
# Based on the error, your model expects this EXACT order:
CORRECT_FEATURE_ORDER = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary',
    'Geography_France', 'Geography_Germany', 'Geography_Spain',
    'Gender_Female', 'Gender_Male',
    'HasCrCard_1', 'HasCrCard_0',  # NOTE: This order is different!
    'IsActiveMember_1', 'IsActiveMember_0'   # NOTE: This order is different!
]

SCALING_FEATURES = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

# ================================================
# LOAD MODEL AND SCALER
# ================================================
@st.cache_resource
def load_resources():
    try:
        with open('best_model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        return None, None

model, scaler = load_resources()

# ================================================
# ENLARGED HEADER SECTION
# ================================================
# Full-width header with larger image
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
    .big-header h1 {
        font-size: 3rem;
        margin-bottom: 10px;
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
    .metric-card {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #667eea;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header image - FULL WIDTH
try:
    st.image(
        HEADER_IMAGE_URL, 
        width=None,  # Let it use full container width
        caption="Customer Churn Prediction Dashboard"
    )
except:
    st.markdown(
        """
        <div class="big-header">
            <h1>🏦 Customer Churn Prediction</h1>
            <p>AI-Powered Customer Retention Analysis</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# ================================================
# SIDEBAR WITH LARGER IMAGE
# ================================================
with st.sidebar:
    # Enlarged sidebar image
    try:
        st.image(
            SIDEBAR_IMAGE_URL,
            width=300,  # Increased size
            caption="Analysis Dashboard"
        )
    except:
        st.markdown("### 🔮 Churn Predictor")
    
    st.markdown("---")
    
    # Customer Inputs Section
    st.header("📋 Customer Details")
    
    # User inputs in correct order for the model
    st.subheader("📊 Financial Details")
    credit_score = st.slider("Credit Score", 300, 850, 650)
    balance = st.slider("Account Balance (€)", 0, 250000, 50000, step=1000)
    salary = st.slider("Estimated Salary (€)", 0, 250000, 80000, step=1000)
    
    st.subheader("👤 Personal Details")
    age = st.slider("Age", 18, 100, 38)
    tenure = st.slider("Tenure (years)", 0, 10, 3)
    num_products = st.slider("Number of Products", 1, 4, 2)
    
    st.subheader("📍 Geography")
    geography = st.radio("Country", ["France", "Germany", "Spain"], horizontal=True)
    
    st.subheader("⚡ Account Status")
    has_credit_card = st.radio("Has Credit Card?", ["Yes", "No"], horizontal=True)
    is_active = st.radio("Active Member?", ["Yes", "No"], horizontal=True)
    
    st.subheader("👫 Gender")
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)

# ================================================
# PREPARE USER DATA IN CORRECT ORDER
# ================================================
user_data = {}
for feature in CORRECT_FEATURE_ORDER:
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
st.markdown("Predict customer retention risk with AI-powered insights")

# Create two columns
col1, col2 = st.columns([1.2, 1])

# ================================================
# COLUMN 1: FEATURE IMPORTANCE
# ================================================
with col1:
    st.markdown("### 🔍 Feature Importance Analysis")
    
    try:
        # Load feature importance
        feature_importance = pd.read_excel('feature_importance.xlsx')
        
        # Create enhanced plot
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
        
        # Enhance layout
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12),
            xaxis_title="<b>Importance Score</b>",
            yaxis_title="<b>Features</b>",
            title_font_size=20,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Could not load feature importance: {e}")
        # Show sample data
        sample_data = pd.DataFrame({
            'Feature': ['Age', 'Balance', 'CreditScore', 'NumOfProducts', 'Tenure'],
            'Importance': [0.25, 0.20, 0.18, 0.15, 0.12]
        })
        fig = px.bar(sample_data, x='Importance', y='Feature', orientation='h')
        st.plotly_chart(fig, use_container_width=True)

# ================================================
# COLUMN 2: PREDICTION
# ================================================
with col2:
    st.markdown("### 🎯 Make Prediction")
    
    if st.button("🚀 **Predict Churn Risk**", type="primary"):
        if model is None or scaler is None:
            st.error("Model not loaded. Please check if model files are available.")
        else:
            with st.spinner("Analyzing customer data..."):
                # Create DataFrame IN CORRECT ORDER
                input_df = pd.DataFrame([user_data])
                
                # Ensure columns are in correct order
                input_df = input_df[CORRECT_FEATURE_ORDER]
                
                # Scale features
                input_df[SCALING_FEATURES] = scaler.transform(input_df[SCALING_FEATURES])
                
                # Make prediction
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0][1]
                
                # Display results
                st.markdown("---")
                st.markdown("### 📊 Prediction Results")
                
                # Risk indicator
                if prediction == 1:
                    st.error("## 🚨 HIGH RISK - Likely to CHURN")
                else:
                    st.success("## ✅ LOW RISK - Likely to STAY")
                
                # Probability cards
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h3>Churn Probability</h3>
                            <h1>{probability:.1%}</h1>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                with col_b:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h3>Retention Probability</h3>
                            <h1>{(1-probability):.1%}</h1>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                # Progress bar
                st.progress(float(probability))
                st.caption(f"Churn Risk Level: {probability:.1%}")
                
                # Customer summary
                st.markdown("---")
                st.markdown("#### 📋 Customer Summary")
                
                summary_data = {
                    "Parameter": ["Age", "Balance", "Credit Score", "Tenure", 
                                 "Active Member", "Geography"],
                    "Value": [f"{age} years", f"€{balance:,}", f"{credit_score}", 
                             f"{tenure} years", is_active, geography]
                }
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

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
