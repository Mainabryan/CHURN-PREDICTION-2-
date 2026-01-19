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
# Replace with your actual GitHub raw URLs
HEADER_IMAGE_URL = "https://raw.githubusercontent.com/Mainabryan/CHURN-PREDICTION-2-/main/Screenshot%202026-01-19%20041922.png"
SIDEBAR_IMAGE_URL = "https://raw.githubusercontent.com/Mainabryan/CHURN-PREDICTION-2-/main/Screenshot%202026-01-19%20041922.png"

# ================================================
# LOAD MODEL AND SCALER (with error handling)
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
    .big-header p {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    .sidebar-image {
        text-align: center;
        padding: 20px 0;
    }
    .stButton>button {
        width: 100%;
        font-size: 1.2rem;
        height: 50px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Full-width header container
header_col1, header_col2, header_col3 = st.columns([1, 3, 1])
with header_col2:
    try:
        # Enlarged header image
        st.image(
            HEADER_IMAGE_URL, 
            width=800,  # Increased width
            caption="Customer Churn Prediction Dashboard",
            use_column_width='auto'
        )
    except:
        # Fallback if image fails
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
    # Enlarged sidebar image with styling
    st.markdown('<div class="sidebar-image">', unsafe_allow_html=True)
    try:
        st.image(
            SIDEBAR_IMAGE_URL,
            width=250,  # Increased size
            caption="Analysis Dashboard",
            use_column_width='auto'
        )
    except:
        st.markdown("### 🔮 Churn Predictor")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Customer Inputs Section (now more prominent)
    st.header("📋 Customer Details")
    st.markdown("*Adjust the parameters below*")
    
    # Define feature names
    FEATURE_NAMES = [
        'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary',
        'Geography_France', 'Geography_Germany', 'Geography_Spain',
        'Gender_Female', 'Gender_Male',
        'HasCrCard_0', 'HasCrCard_1',
        'IsActiveMember_0', 'IsActiveMember_1'
    ]
    
    # User inputs with better spacing
    st.subheader("📊 Financial Details")
    credit_score = st.slider("Credit Score", 300, 850, 650, help="Customer's credit score")
    balance = st.slider("Account Balance (€)", 0, 250000, 50000, step=1000, 
                       help="Current account balance")
    salary = st.slider("Estimated Salary (€)", 0, 250000, 80000, step=1000,
                      help="Annual estimated salary")
    
    st.subheader("👤 Personal Details")
    age = st.slider("Age", 18, 100, 38, help="Customer's age")
    tenure = st.slider("Tenure (years)", 0, 10, 3, help="Years with the bank")
    num_products = st.slider("Number of Products", 1, 4, 2, help="Bank products owned")
    
    st.subheader("📍 Geography")
    geography = st.radio("Country", ["France", "Germany", "Spain"], 
                        horizontal=True, label_visibility="collapsed")
    
    st.subheader("⚡ Account Status")
    has_credit_card = st.radio("Has Credit Card?", ["Yes", "No"], 
                              horizontal=True, label_visibility="collapsed")
    is_active = st.radio("Active Member?", ["Yes", "No"], 
                        horizontal=True, label_visibility="collapsed")
    
    st.subheader("👫 Gender")
    gender = st.radio("Gender", ["Male", "Female"], 
                     horizontal=True, label_visibility="collapsed")
    
    # Prepare data dictionary
    user_data = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'EstimatedSalary': salary,
        'Geography_France': geography == "France",
        'Geography_Germany': geography == "Germany",
        'Geography_Spain': geography == "Spain",
        'Gender_Female': gender == "Female",
        'Gender_Male': gender == "Male",
        'HasCrCard_1': has_credit_card == "Yes",
        'HasCrCard_0': has_credit_card == "No",
        'IsActiveMember_1': is_active == "Yes",
        'IsActiveMember_0': is_active == "No"
    }

# ================================================
# MAIN CONTENT AREA
# ================================================
# Title
st.title("📈 Customer Churn Analysis Dashboard")
st.markdown("Predict customer retention risk with AI-powered insights")

# Create two columns for main content
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
            height=500,
            text='Importance'
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
        
        fig.update_traces(
            texttemplate='%{text:.3f}',
            textposition='outside',
            marker_line_color='rgb(8,48,107)',
            marker_line_width=1.5
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Could not load feature importance: {e}")
        # Show sample data for testing
        sample_data = pd.DataFrame({
            'Feature': ['Age', 'Balance', 'CreditScore', 'NumOfProducts', 'Tenure', 
                       'IsActiveMember', 'Geography', 'Gender', 'HasCrCard'],
            'Importance': [0.25, 0.20, 0.18, 0.15, 0.12, 0.08, 0.06, 0.04, 0.02]
        })
        
        fig = px.bar(sample_data, x='Importance', y='Feature', orientation='h',
                    title='Sample Feature Importance')
        st.plotly_chart(fig, use_container_width=True)

# ================================================
# COLUMN 2: PREDICTION
# ================================================
with col2:
    st.markdown("### 🎯 Make Prediction")
    
    # Prediction button
    if st.button("🚀 **Predict Churn Risk**", type="primary", use_container_width=True):
        if model is None or scaler is None:
            st.error("Model not loaded. Please check if model files are available.")
        else:
            with st.spinner("Analyzing customer data..."):
                # Create DataFrame
                input_df = pd.DataFrame([user_data])
                
                # Scale features
                scaling_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 
                                   'NumOfProducts', 'EstimatedSalary']
                input_df[scaling_features] = scaler.transform(input_df[scaling_features])
                
                # Make prediction
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0][1]
                
                # Display results
                st.markdown("---")
                st.markdown("### 📊 Prediction Results")
                
                # Risk indicator
                risk_col1, risk_col2 = st.columns(2)
                with risk_col1:
                    if prediction == 1:
                        st.error(f"## 🚨 HIGH RISK")
                        st.markdown("**Likely to Churn**")
                    else:
                        st.success(f"## ✅ LOW RISK")
                        st.markdown("**Likely to Stay**")
                
                with risk_col2:
                    # Gauge visualization
                    st.markdown(f"### {probability:.1%}")
                    st.progress(float(probability))
                    st.caption(f"Churn Probability: {probability:.1%}")
                
                # Probability breakdown
                st.markdown("---")
                st.markdown("#### Probability Breakdown")
                
                prob_col1, prob_col2 = st.columns(2)
                with prob_col1:
                    st.metric(
                        label="Churn Probability",
                        value=f"{probability:.1%}",
                        delta=f"{(probability-0.5)*100:+.1f}% vs neutral"
                    )
                
                with prob_col2:
                    st.metric(
                        label="Retention Probability",
                        value=f"{1-probability:.1%}",
                        delta=f"{((1-probability)-0.5)*100:+.1f}% vs neutral"
                    )
                
                # Key factors
                st.markdown("---")
                st.markdown("#### 📋 Customer Summary")
                
                summary_data = {
                    "Parameter": ["Age", "Balance", "Credit Score", "Tenure", 
                                 "Active Member", "Geography"],
                    "Value": [f"{age} years", f"€{balance:,}", f"{credit_score}", 
                             f"{tenure} years", is_active, geography]
                }
                st.table(pd.DataFrame(summary_data))

# ================================================
# FOOTER
# ================================================
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])
with footer_col2:
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 20px;">
            <p>Customer Churn Prediction System • Powered by Machine Learning</p>
            <p>Built with ❤️ using Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )
