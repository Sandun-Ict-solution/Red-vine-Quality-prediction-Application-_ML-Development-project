import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Wine Quality Predictor", layout="wide", page_icon="🍷")

# ---------------------------
# Helpers
# ---------------------------
@st.cache_data
def load_data(path="winequality-red.csv"):
    if not os.path.exists(path):
        st.error(f"Dataset not found at {path}. Please run model_training.py first.")
        return None
    df = pd.read_csv(path)
    # Add binary quality column
    df['quality_binary'] = (df['quality'] >= 6).astype(int)
    df['quality_label'] = df['quality_binary'].map({1: 'Good Wine', 0: 'Bad Wine'})
    return df

@st.cache_resource
def load_model(path="model.pkl"):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

@st.cache_data
def load_model_comparison():
    if os.path.exists("model_comparison.csv"):
        return pd.read_csv("model_comparison.csv")
    return None

@st.cache_data
def load_feature_importance():
    if os.path.exists("feature_importance.csv"):
        return pd.read_csv("feature_importance.csv")
    return None

# Load data and model
df = load_data()
model = load_model()
model_comparison = load_model_comparison()
feature_importance = load_feature_importance()

# ---------------------------
# Sidebar navigation
# ---------------------------
st.sidebar.title("🍷 Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Data Explorer", "📈 Visualizations", "🤖 Predict Quality", "📋 Model Performance"])

# ---------------------------
# Home
# ---------------------------
if page == "🏠 Home":
    st.title("🍷 Wine Quality Prediction App")
    st.markdown("""
    ### Welcome to the Wine Quality Predictor!
    
    This application uses machine learning to predict the quality of red wine based on its chemical properties.
    The model classifies wines as **Good** (quality ≥ 6) or **Bad** (quality < 6) based on features like:
    
    - **Fixed Acidity** - Most acids involved with wine
    - **Volatile Acidity** - Amount of acetic acid in wine
    - **Citric Acid** - Found in small quantities, adds 'freshness'
    - **Residual Sugar** - Amount of sugar remaining after fermentation
    - **Chlorides** - Amount of salt in the wine
    - **Free Sulfur Dioxide** - Prevents microbial growth and oxidation
    - **Total Sulfur Dioxide** - Amount of free and bound forms of SO2
    - **Density** - Depends on percent alcohol and sugar content
    - **pH** - Describes how acidic or basic a wine is
    - **Sulphates** - Wine additive that contributes to SO2 levels
    - **Alcohol** - Percent alcohol content of the wine
    """)
    
    col1, col2, col3 = st.columns(3)
    
    if df is not None:
        with col1:
            st.metric("Total Wines", len(df))
        with col2:
            good_wines = (df['quality'] >= 6).sum()
            st.metric("Good Wines", good_wines)
        with col3:
            st.metric("Average Quality", f"{df['quality'].mean():.2f}")
    
    if model is None:
        st.warning("⚠️ Model not found. Please run model_training.py first to train the model.")
    else:
        st.success("✅ Model loaded successfully! Navigate to different sections using the sidebar.")
    
    if df is not None:
        st.subheader("📋 Dataset Sample")
        st.dataframe(df.sample(10).round(3), use_container_width=True)

# ---------------------------
# Data Explorer
# ---------------------------
elif page == "📊 Data Explorer":
    st.title("📊 Dataset Overview")
    
    if df is None:
        st.error("Dataset missing. Please ensure winequality-red.csv is in the directory.")
    else:
        # Dataset info
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 Dataset Information")
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.write(f"**Features:** {df.shape[1] - 2} (excluding quality columns)")
            
        with col2:
            st.subheader("🎯 Quality Distribution")
            quality_counts = df['quality'].value_counts().sort_index()
            fig_quality = px.bar(x=quality_counts.index, y=quality_counts.values,
                               title="Wine Quality Distribution",
                               labels={'x': 'Quality Rating', 'y': 'Count'})
            st.plotly_chart(fig_quality, use_container_width=True)
        
        # Data filtering
        st.subheader("🔍 Interactive Data Filtering")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            alcohol_range = st.slider("Alcohol Content (%)", 
                                    float(df['alcohol'].min()), 
                                    float(df['alcohol'].max()), 
                                    (float(df['alcohol'].min()), float(df['alcohol'].max())))
        with col2:
            quality_filter = st.multiselect("Quality Ratings", 
                                           sorted(df['quality'].unique()), 
                                           default=sorted(df['quality'].unique()))
        with col3:
            ph_range = st.slider("pH Level", 
                               float(df['pH'].min()), 
                               float(df['pH'].max()), 
                               (float(df['pH'].min()), float(df['pH'].max())))
        
        # Apply filters
        filtered_df = df[
            (df['alcohol'] >= alcohol_range[0]) & 
            (df['alcohol'] <= alcohol_range[1]) &
            (df['quality'].isin(quality_filter)) &
            (df['pH'] >= ph_range[0]) & 
            (df['pH'] <= ph_range[1])
        ]
        
        st.write(f"**Filtered Dataset:** {len(filtered_df)} wines")
        st.dataframe(filtered_df.round(3), use_container_width=True)
        
        # Statistics
        st.subheader("📊 Statistical Summary")
        st.dataframe(filtered_df.describe().round(3), use_container_width=True)

# ---------------------------
# Visualizations
# ---------------------------
elif page == "📈 Visualizations":
    st.title("📈 Wine Quality Visualizations")
    
    if df is None:
        st.error("Dataset missing.")
    else:
        # Quality vs Alcohol
        st.subheader("🍷 Quality vs Alcohol Content")
        fig1 = px.box(df, x='quality', y='alcohol', 
                     title="Alcohol Content by Wine Quality",
                     labels={'quality': 'Quality Rating', 'alcohol': 'Alcohol %'})
        st.plotly_chart(fig1, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("🔥 Feature Correlation Heatmap")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig2 = px.imshow(corr_matrix, 
                        title="Correlation Matrix of Wine Features",
                        aspect="auto", 
                        color_continuous_scale="RdBu_r")
        st.plotly_chart(fig2, use_container_width=True)
        
        # Distribution plots
        st.subheader("📊 Feature Distributions by Wine Quality")
        
        col1, col2 = st.columns(2)
        
        with col1:
            feature1 = st.selectbox("Select Feature 1", numeric_cols, index=0)
            fig3 = px.histogram(df, x=feature1, color='quality_label', 
                              title=f"{feature1.title()} Distribution by Quality",
                              barmode='overlay', opacity=0.7)
            st.plotly_chart(fig3, use_container_width=True)
            
        with col2:
            feature2 = st.selectbox("Select Feature 2", numeric_cols, index=1)
            fig4 = px.violin(df, y=feature2, x='quality_label', 
                           title=f"{feature2.title()} by Wine Quality",
                           box=True)
            st.plotly_chart(fig4, use_container_width=True)
        
        # Scatter plot
        st.subheader("🎯 Feature Relationships")
        col1, col2 = st.columns(2)
        with col1:
            x_feature = st.selectbox("X-axis Feature", numeric_cols, index=10)
        with col2:
            y_feature = st.selectbox("Y-axis Feature", numeric_cols, index=0)
            
        fig5 = px.scatter(df, x=x_feature, y=y_feature, color='quality_label',
                         title=f"{x_feature.title()} vs {y_feature.title()}",
                         hover_data=['quality'])
        st.plotly_chart(fig5, use_container_width=True)

# ---------------------------
# Predict Quality
# ---------------------------
elif page == "🤖 Predict Quality":
    st.title("🤖 Wine Quality Prediction")
    
    if model is None:
        st.error("Model not found. Run model_training.py to create model.pkl")
    else:
        st.write("Enter wine characteristics to predict its quality:")
        
        # Input widgets
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, max_value=20.0, value=8.0, step=0.1)
            volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, max_value=2.0, value=0.5, step=0.01)
            citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
            residual_sugar = st.number_input("Residual Sugar", min_value=0.0, max_value=20.0, value=2.0, step=0.1)
            
        with col2:
            chlorides = st.number_input("Chlorides", min_value=0.0, max_value=1.0, value=0.08, step=0.001, format="%.3f")
            free_so2 = st.number_input("Free Sulfur Dioxide", min_value=0.0, max_value=100.0, value=15.0, step=1.0)
            total_so2 = st.number_input("Total Sulfur Dioxide", min_value=0.0, max_value=300.0, value=45.0, step=1.0)
            density = st.number_input("Density", min_value=0.9, max_value=1.1, value=0.997, step=0.001, format="%.4f")
            
        with col3:
            ph = st.number_input("pH", min_value=2.0, max_value=5.0, value=3.3, step=0.01)
            sulphates = st.number_input("Sulphates", min_value=0.0, max_value=3.0, value=0.65, step=0.01)
            alcohol = st.number_input("Alcohol (%)", min_value=8.0, max_value=16.0, value=10.5, step=0.1)
        
        # Create input dataframe
        input_data = pd.DataFrame([{
            "fixed acidity": fixed_acidity,
            "volatile acidity": volatile_acidity,
            "citric acid": citric_acid,
            "residual sugar": residual_sugar,
            "chlorides": chlorides,
            "free sulfur dioxide": free_so2,
            "total sulfur dioxide": total_so2,
            "density": density,
            "pH": ph,
            "sulphates": sulphates,
            "alcohol": alcohol
        }])
        
        # Prediction
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if st.button("🔮 Predict Wine Quality", type="primary"):
                try:
                    prediction = model.predict(input_data)[0]
                    probability = model.predict_proba(input_data)[0]
                    
                    st.subheader("🎯 Prediction Result")
                    
                    if prediction == 1:
                        st.success("🍷 **Good Wine** (Quality ≥ 6)")
                        confidence = probability[1]
                    else:
                        st.error("🚫 **Bad Wine** (Quality < 6)")
                        confidence = probability[0]
                    
                    st.write(f"**Confidence:** {confidence*100:.2f}%")
                    
                    # Probability gauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = probability[1]*100,
                        title = {'text': "Good Wine Probability (%)"},
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkred"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 100], 'color': "gray"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=300)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
        
        with col2:
            # Show input summary
            st.subheader("📝 Input Summary")
            st.dataframe(input_data.T.rename(columns={0: 'Value'}), use_container_width=True)

# ---------------------------
# Model Performance
# ---------------------------
elif page == "📋 Model Performance":
    st.title("📋 Model Performance Analysis")
    
    if model_comparison is not None:
        st.subheader("🏆 Model Comparison")
        
        # Display comparison table
        st.dataframe(model_comparison, use_container_width=True)
        
        # Comparison chart
        fig_comp = px.bar(model_comparison, x='Model', y=['Train Accuracy', 'Test Accuracy'],
                         title="Model Performance Comparison", barmode='group')
        st.plotly_chart(fig_comp, use_container_width=True)
    
    if feature_importance is not None:
        st.subheader("🎯 Feature Importance")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.dataframe(feature_importance, use_container_width=True)
            
        with col2:
            fig_importance = px.bar(feature_importance.head(10), 
                                  x='Importance', y='Feature',
                                  title="Top 10 Most Important Features",
                                  orientation='h')
            fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_importance, use_container_width=True)
    
    if df is not None:
        st.subheader("📊 Dataset Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Samples", len(df))
            st.metric("Good Wines", (df['quality'] >= 6).sum())
            st.metric("Bad Wines", (df['quality'] < 6).sum())
            
        with col2:
            st.metric("Average Quality", f"{df['quality'].mean():.2f}")
            st.metric("Quality Std Dev", f"{df['quality'].std():.2f}")
            st.metric("Features", len([col for col in df.columns if col not in ['quality', 'quality_binary', 'quality_label']]))
    
    # Additional insights
    if df is not None:
        st.subheader("💡 Key Insights")
        
        insights = []
        good_wines = df[df['quality'] >= 6]
        bad_wines = df[df['quality'] < 6]
        
        # Compare means
        if len(good_wines) > 0 and len(bad_wines) > 0:
            alcohol_diff = good_wines['alcohol'].mean() - bad_wines['alcohol'].mean()
            if alcohol_diff > 0:
                insights.append(f"🍷 Good wines have {alcohol_diff:.2f}% higher alcohol content on average")
            
            ph_diff = good_wines['pH'].mean() - bad_wines['pH'].mean()
            if abs(ph_diff) > 0.05:
                direction = "higher" if ph_diff > 0 else "lower"
                insights.append(f"⚗️ Good wines have {direction} pH levels (difference: {abs(ph_diff):.3f})")
            
            volatile_diff = bad_wines['volatile acidity'].mean() - good_wines['volatile acidity'].mean()
            if volatile_diff > 0:
                insights.append(f"🔴 Bad wines have {volatile_diff:.3f} higher volatile acidity on average")
        
        for insight in insights:
            st.write(insight)