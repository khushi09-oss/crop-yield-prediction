"""
================================================================================
CROP YIELD PREDICTION & WATER OPTIMIZATION SYSTEM
================================================================================
A Machine Learning-powered web application for:
- Predicting crop yield based on environmental and soil parameters
- Recommending optimal irrigation water usage
- Supporting sustainable agriculture practices
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras import layers, callbacks, regularizers
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')
tf.random.set_seed(42)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="🌾 Crop Yield & Water Optimization",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@st.cache_data
def generate_sample_dataset(n_samples=1000):
    """
    Generate synthetic agricultural dataset with REALISTIC feature interactions
    This creates balanced feature importance for better demonstration
    """
    np.random.seed(42)

    crops = ['Wheat', 'Rice', 'Maize', 'Cotton', 'Sugarcane']
    seasons = ['Kharif', 'Rabi', 'Zaid']

    # Generate base features with realistic distributions
    data = {
        'Crop_Type': np.random.choice(crops, n_samples),
        'Temperature': np.random.normal(28, 5, n_samples).clip(15, 40),
        'Rainfall': np.random.gamma(2, 50, n_samples).clip(0, 500),
        'Humidity': np.random.normal(65, 15, n_samples).clip(30, 95),
        'Soil_Moisture': np.random.normal(45, 15, n_samples).clip(10, 90),
        'Soil_pH': np.random.normal(6.5, 0.8, n_samples).clip(4.5, 8.5),
        'Nitrogen': np.random.normal(50, 20, n_samples).clip(10, 120),
        'Phosphorus': np.random.normal(40, 15, n_samples).clip(10, 100),
        'Potassium': np.random.normal(45, 18, n_samples).clip(10, 110),
        'Season': np.random.choice(seasons, n_samples),
        'Historical_Irrigation_Water': np.random.normal(200, 60, n_samples).clip(50, 500)
    }

    df = pd.DataFrame(data)

    # Generate REALISTIC crop yield with BALANCED feature contributions
    # Base yield by crop type
    yield_base = {
        'Wheat': 3.5, 'Rice': 4.2, 'Maize': 5.1,
        'Cotton': 2.8, 'Sugarcane': 70
    }

    # Create yield with multiple contributing factors (not dominated by one feature)
    df['Crop_Yield'] = df.apply(lambda row:
        yield_base[row['Crop_Type']] *
        # Nitrogen contribution (15-20% impact)
        (1 + 0.15 * (row['Nitrogen'] - 50) / 50) *
        # Phosphorus contribution (10-15% impact)
        (1 + 0.12 * (row['Phosphorus'] - 40) / 40) *
        # Potassium contribution (10-15% impact)
        (1 + 0.12 * (row['Potassium'] - 45) / 45) *
        # Rainfall contribution (15-20% impact)
        (1 + 0.15 * np.tanh((row['Rainfall'] - 100) / 100)) *
        # Soil moisture contribution (10-15% impact)
        (1 + 0.12 * (row['Soil_Moisture'] - 45) / 45) *
        # Temperature contribution (10-15% impact) - optimal around 25-30°C
        (1 - 0.10 * ((row['Temperature'] - 27.5) / 10) ** 2) *
        # Soil pH contribution (8-12% impact) - optimal around 6.5
        (1 - 0.10 * abs(row['Soil_pH'] - 6.5) / 2) *
        # Humidity contribution (5-10% impact)
        (1 + 0.08 * (row['Humidity'] - 65) / 65) *
        # Season effect (5-8% impact)
        (1.05 if row['Season'] == 'Kharif' else 0.98 if row['Season'] == 'Rabi' else 1.0) *
        # Realistic random variation
        np.random.normal(1, 0.08),
        axis=1
    ).clip(lower=0.5)  # Ensure minimum realistic yield

    return df


def preprocess_data(df):
    """
    Preprocess the dataset:
    - Handle missing values
    - Encode categorical features
    - Scale numerical features
    Returns: processed dataframe, encoders, and scaler
    """
    df_processed = df.copy()

    # Handle missing values
    df_processed.fillna(df_processed.median(numeric_only=True), inplace=True)
    df_processed.fillna(df_processed.mode().iloc[0], inplace=True)

    # Encode categorical features
    encoders = {}
    categorical_cols = df_processed.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        if col != 'Crop_Yield':  # Don't encode target
            le = LabelEncoder()
            df_processed[col + '_Encoded'] = le.fit_transform(df_processed[col])
            encoders[col] = le

    return df_processed, encoders


def train_yield_model(df):
    """
    Train Random Forest model for crop yield prediction
    Returns: trained model, test metrics, feature importance, scaler
    """
    # Select features for modeling
    feature_cols = [
        'Temperature', 'Rainfall', 'Humidity', 'Soil_Moisture',
        'Soil_pH', 'Nitrogen', 'Phosphorus', 'Potassium',
        'Crop_Type_Encoded', 'Season_Encoded'
    ]

    X = df[feature_cols]
    y = df['Crop_Yield']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest model with optimized parameters
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',  # Use sqrt for better feature distribution
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_scaled, y_train)

    # Evaluate model
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)

    # Calculate training metrics to check for overfitting
    r2_train = r2_score(y_train, y_pred_train)

    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    metrics = {
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'R2': r2,
        'MAE': mae,
        'R2_train': r2_train
    }

    return model, metrics, feature_importance, scaler


def build_deep_learning_model(input_dim):
    """
    Build a deep learning regression model for crop yield prediction
    """
    model = tf.keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )

    return model


def train_deep_learning_model(df):
    """
    Train deep learning model for crop yield prediction
    Returns: trained model, test metrics, scaler
    """
    feature_cols = [
        'Temperature', 'Rainfall', 'Humidity', 'Soil_Moisture',
        'Soil_pH', 'Nitrogen', 'Phosphorus', 'Potassium',
        'Crop_Type_Encoded', 'Season_Encoded'
    ]

    X = df[feature_cols]
    y = df['Crop_Yield']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42
    )

    model = build_deep_learning_model(input_dim=X_train_scaled.shape[1])

    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )

    model.fit(
        X_train_sub,
        y_train_sub,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=64,
        verbose=0,
        callbacks=[early_stop]
    )

    y_pred_test = model.predict(X_test_scaled, verbose=0).flatten()
    y_pred_train = model.predict(X_train_scaled, verbose=0).flatten()
    
    mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)

    metrics = {
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'R2': r2,
        'MAE': mae,
        'R2_train': r2_train
    }

    return model, metrics, scaler


def calculate_irrigation_recommendation(
    crop_type, temperature, rainfall, humidity,
    soil_moisture, soil_ph, season
):
    """
    Water Optimization Logic:
    Calculates optimal irrigation water based on:
    - Current soil moisture
    - Recent rainfall
    - Crop water requirements
    - Evapotranspiration factors

    Returns: recommended water (mm), water saved (mm), sustainability message
    """

    # Base water requirements by crop (mm/week)
    crop_water_needs = {
        'Wheat': 25,
        'Rice': 50,
        'Maize': 30,
        'Cotton': 35,
        'Sugarcane': 60
    }

    base_water = crop_water_needs.get(crop_type, 30)

    # Adjust for soil moisture
    # If soil moisture > 60%, reduce irrigation significantly
    if soil_moisture > 70:
        moisture_factor = 0.2
    elif soil_moisture > 60:
        moisture_factor = 0.4
    elif soil_moisture > 50:
        moisture_factor = 0.7
    elif soil_moisture > 40:
        moisture_factor = 1.0
    else:
        moisture_factor = 1.3

    # Adjust for recent rainfall
    # If rainfall > 30mm, reduce irrigation
    if rainfall > 50:
        rainfall_factor = 0.1
    elif rainfall > 30:
        rainfall_factor = 0.3
    elif rainfall > 15:
        rainfall_factor = 0.6
    else:
        rainfall_factor = 1.0

    # Adjust for temperature (higher temp = more evaporation)
    if temperature > 35:
        temp_factor = 1.3
    elif temperature > 30:
        temp_factor = 1.15
    elif temperature > 25:
        temp_factor = 1.0
    else:
        temp_factor = 0.9

    # Adjust for humidity (higher humidity = less evaporation)
    if humidity > 80:
        humidity_factor = 0.8
    elif humidity > 60:
        humidity_factor = 1.0
    else:
        humidity_factor = 1.2

    # Calculate recommended irrigation
    recommended_water = base_water * moisture_factor * rainfall_factor * temp_factor * humidity_factor
    recommended_water = max(0, recommended_water)  # Ensure non-negative

    # Calculate water saved compared to traditional irrigation (assume 30mm fixed)
    traditional_irrigation = 30
    water_saved = max(0, traditional_irrigation - recommended_water)

    # Generate sustainability message
    if recommended_water < 5:
        message = "🌧️ Excellent! No irrigation needed. Soil moisture and rainfall are sufficient."
        water_status = "Optimal"
    elif recommended_water < 15:
        message = "💧 Minimal irrigation required. Good soil moisture levels."
        water_status = "Good"
    elif recommended_water < 25:
        message = "💦 Moderate irrigation recommended for optimal growth."
        water_status = "Moderate"
    else:
        message = "🚰 Higher irrigation needed. Consider checking soil drainage."
        water_status = "High"

    return recommended_water, water_saved, message, water_status


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():

    # Header Section
    st.title("🌾 Crop Yield Prediction & Water Optimization System")
    st.markdown("""
    ### 🎯 Sustainable Agriculture through Machine Learning
    This system helps farmers and agricultural planners:
    - **Predict crop yield** based on environmental and soil conditions
    - **Optimize irrigation water usage** to reduce wastage
    - **Support sustainable farming** practices for better resource management
    """)

    st.markdown("---")

    # ========================================================================
    # SIDEBAR - USER INPUTS
    # ========================================================================

    st.sidebar.header("🌱 Input Parameters")
    st.sidebar.markdown("### Enter Crop and Environmental Details")

    # Crop Selection
    crop_type = st.sidebar.selectbox(
        "🌾 Crop Type",
        options=['Wheat', 'Rice', 'Maize', 'Cotton', 'Sugarcane'],
        index=0
    )

    # Season Selection
    season = st.sidebar.selectbox(
        "📅 Growing Season",
        options=['Kharif', 'Rabi', 'Zaid'],
        index=0,
        help="Kharif: Monsoon crops | Rabi: Winter crops | Zaid: Summer crops"
    )

    st.sidebar.markdown("### 🌡️ Environmental Conditions")

    # Temperature
    temperature = st.sidebar.slider(
        "Temperature (°C)",
        min_value=15.0,
        max_value=40.0,
        value=28.0,
        step=0.5,
        help="Average temperature during growing period"
    )

    # Rainfall
    rainfall = st.sidebar.slider(
        "Rainfall (mm)",
        min_value=0.0,
        max_value=500.0,
        help="Total rainfall in recent period"
    )

    # Humidity
    humidity = st.sidebar.slider(
        "Humidity (%)",
        min_value=30.0,
        max_value=95.0,
        value=65.0,
        step=1.0,
        help="Relative humidity level"
    )

    st.sidebar.markdown("### 🌱 Soil Conditions")

    # Soil Moisture
    soil_moisture = st.sidebar.slider(
        "Soil Moisture (%)",
        min_value=10.0,
        max_value=90.0,
        value=45.0,
        step=1.0,
        help="Current soil moisture content"
    )

    # Soil pH
    soil_ph = st.sidebar.slider(
        "Soil pH",
        min_value=4.5,
        max_value=8.5,
        value=6.5,
        step=0.1,
        help="Soil acidity/alkalinity level"
    )

    st.sidebar.markdown("### 🧪 Soil Nutrients (kg/ha)")

    # Nitrogen
    nitrogen = st.sidebar.slider(
        "Nitrogen (N)",
        min_value=10.0,
        max_value=120.0,
        value=50.0,
        step=1.0
    )

    # Phosphorus
    phosphorus = st.sidebar.slider(
        "Phosphorus (P)",
        min_value=10.0,
        max_value=100.0,
        value=40.0,
        step=1.0
    )

    # Potassium
    potassium = st.sidebar.slider(
        "Potassium (K)",
        min_value=10.0,
        max_value=110.0,
        value=45.0,
        step=1.0
    )

    st.sidebar.markdown("### 🧠 Model Selection")
    model_choice = st.sidebar.selectbox(
        "Prediction Model",
        options=["Random Forest", "Deep Learning (Neural Net)"],
        index=0
    )
    
    # Prediction Button
    predict_button = st.sidebar.button("🚀 Predict Yield & Optimize Water", type="primary")

    # ========================================================================
    # LOAD AND TRAIN MODEL
    # ========================================================================

    with st.spinner("🔄 Loading and training model..."):
        # Generate or load dataset
        df = generate_sample_dataset(n_samples=1200)

        # Preprocess data
        df_processed, encoders = preprocess_data(df)

        # Train model
        if model_choice == "Random Forest":
            model, metrics, feature_importance, scaler = train_yield_model(df_processed)
        else:
            model, metrics, scaler = train_deep_learning_model(df_processed)
            feature_importance = None
    
    # ========================================================================
    # MAIN CONTENT AREA
    # ========================================================================

    # Model Performance Section
    st.header("📊 Model Performance")
    st.caption(f"Current model: {model_choice}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="R² Score (Test)",
            value=f"{metrics['R2']:.4f}",
            help="Proportion of variance explained by the model"
        )

    with col2:
        st.metric(
            label="RMSE",
            value=f"{metrics['RMSE']:.4f}",
            help="Root Mean Squared Error"
        )

    with col3:
        st.metric(
            label="MAE",
            value=f"{metrics['MAE']:.4f}",
            help="Mean Absolute Error"
        )

    with col4:
        st.metric(
            label="R² Score (Train)",
            value=f"{metrics['R2_train']:.4f}",
            help="Training score - check for overfitting"
        )

    # Overfitting check
    if metrics['R2_train'] - metrics['R2'] > 0.1:
        st.warning("⚠️ Model shows some overfitting (train R² >> test R²). Consider regularization.")
    else:
        st.success("✅ Model shows good generalization (minimal overfitting)")

    st.markdown("---")

    # ========================================================================
    # PREDICTION RESULTS
    # ========================================================================

    if predict_button:

        st.header("🎯 Prediction Results")

        # Prepare input data for prediction
        input_data = {
            'Crop_Type': crop_type,
            'Temperature': temperature,
            'Rainfall': rainfall,
            'Humidity': humidity,
            'Soil_Moisture': soil_moisture,
            'Soil_pH': soil_ph,
            'Nitrogen': nitrogen,
            'Phosphorus': phosphorus,
            'Potassium': potassium,
            'Season': season
        }

        # Encode categorical variables
        input_df = pd.DataFrame([input_data])
        input_df['Crop_Type_Encoded'] = encoders['Crop_Type'].transform([crop_type])[0]
        input_df['Season_Encoded'] = encoders['Season'].transform([season])[0]

        # Prepare features for model
        feature_cols = [
            'Temperature', 'Rainfall', 'Humidity', 'Soil_Moisture',
            'Soil_pH', 'Nitrogen', 'Phosphorus', 'Potassium',
            'Crop_Type_Encoded', 'Season_Encoded'
        ]

        X_input = input_df[feature_cols]
        X_input_scaled = scaler.transform(X_input)

        # Make prediction
        predicted_yield = model.predict(X_input_scaled)
        predicted_yield = float(np.asarray(predicted_yield).squeeze())
        
        # Calculate irrigation recommendation
        recommended_water, water_saved, message, water_status = calculate_irrigation_recommendation(
            crop_type, temperature, rainfall, humidity, soil_moisture, soil_ph, season
        )

        # Display Results
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🌾 Crop Yield Prediction")
            
            # Determine yield unit based on crop
            if crop_type == 'Sugarcane':
                yield_unit = "tons/hectare"
            else:
                yield_unit = "tons/hectare"
            
            # Yield card with gradient background
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; text-align: center; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h2 style="margin: 0; font-size: 2.5em;">📊 {predicted_yield:.2f}</h2>
                <p style="margin: 5px 0 0 0; font-size: 1.1em; opacity: 0.9;">{yield_unit}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Yield interpretation
            crop_avg_yields = {
                'Wheat': 3.5, 'Rice': 4.2, 'Maize': 5.1,
                'Cotton': 2.8, 'Sugarcane': 70
            }

            avg_yield = crop_avg_yields.get(crop_type, 3.5)
            
            st.markdown("")
            if predicted_yield > avg_yield * 1.1:
                st.markdown("""
                <div style="background: #d4edda; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745;">
                    <p style="margin: 0; color: #155724;"><b>✅ Excellent!</b> Yield is <b>above average</b> for this crop.</p>
                </div>
                """, unsafe_allow_html=True)
            elif predicted_yield > avg_yield * 0.9:
                st.markdown("""
                <div style="background: #d1ecf1; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8;">
                    <p style="margin: 0; color: #0c5460;"><b>✔️ Good!</b> Yield is <b>around average</b> for this crop.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107;">
                    <p style="margin: 0; color: #856404;"><b>⚠️ Below Average.</b> Consider improving <b>soil conditions or nutrients</b>.</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 💧 Water Optimization")
            
            # Water card with gradient background
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #00d4ff 0%, #0099ff 100%); padding: 20px; border-radius: 10px; text-align: center; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h2 style="margin: 0; font-size: 2.5em;">💧 {recommended_water:.2f}</h2>
                <p style="margin: 5px 0 0 0; font-size: 1.1em; opacity: 0.9;">mm per week</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background: rgba(76, 175, 80, 0.1); padding: 12px; border-radius: 8px; text-align: center; border: 1px solid rgba(76, 175, 80, 0.3);">
                <p style="margin: 0; color: #2e7d32;"><b>💰 Water Saved:</b> {:.2f} mm vs traditional</p>
            </div>
            """.format(water_saved), unsafe_allow_html=True)
            
            st.markdown("")
            # Water status indicator with enhanced styling
            if water_status == "Optimal":
                st.markdown(f"""
                <div style="background: #d4edda; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745;">
                    <p style="margin: 0; color: #155724;"><b>🌟 {message.split('.')[0]}</b></p>
                </div>
                """, unsafe_allow_html=True)
            elif water_status == "Good":
                st.markdown(f"""
                <div style="background: #d1ecf1; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8;">
                    <p style="margin: 0; color: #0c5460;"><b>✅ {message}</b></p>
                </div>
                """, unsafe_allow_html=True)
            elif water_status == "Moderate":
                st.markdown(f"""
                <div style="background: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107;">
                    <p style="margin: 0; color: #856404;"><b>⚠️ {message}</b></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: #f8d7da; padding: 15px; border-radius: 8px; border-left: 4px solid #dc3545;">
                    <p style="margin: 0; color: #721c24;"><b>🚰 {message}</b></p>
                </div>
                """, unsafe_allow_html=True)
        
        # Sustainability Impact
        st.markdown("---")
        st.markdown("### 🌍 Sustainability Impact")
        
        col1, col2, col3 = st.columns(3)
        
        water_saved_liters = water_saved * 10  # 1mm = 10 liters per sq meter
        annual_savings = water_saved_liters * 20  # 20 irrigation cycles per season
        efficiency_improvement = (water_saved / 30) * 100 if water_saved > 0 else 0
        
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #40E0D0 0%, #00CED1 100%); padding: 20px; border-radius: 10px; text-align: center; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h3 style="margin: 0 0 10px 0; font-size: 0.9em; opacity: 0.9;">💧 Water Saved/Hectare</h3>
                <p style="margin: 0; font-size: 2em;"><b>{water_saved_liters:.0f}</b></p>
                <p style="margin: 5px 0 0 0; font-size: 0.95em; opacity: 0.9;">Liters</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #76B900 0%, #90EE90 100%); padding: 20px; border-radius: 10px; text-align: center; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h3 style="margin: 0 0 10px 0; font-size: 0.9em; opacity: 0.9;">📈 Annual Savings</h3>
                <p style="margin: 0; font-size: 2em;"><b>{annual_savings:.0f}</b></p>
                <p style="margin: 5px 0 0 0; font-size: 0.95em; opacity: 0.9;">Liters/Season</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%); padding: 20px; border-radius: 10px; text-align: center; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h3 style="margin: 0 0 10px 0; font-size: 0.9em; opacity: 0.9;">⚡ Efficiency Gain</h3>
                <p style="margin: 0; font-size: 2em;"><b>{efficiency_improvement:.1f}%</b></p>
                <p style="margin: 5px 0 0 0; font-size: 0.95em; opacity: 0.9;">Improvement</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(144, 238, 144, 0.2), rgba(152, 251, 152, 0.2)); padding: 20px; border-radius: 10px; border: 2px solid #90EE90; margin-top: 20px;">
            <h4 style="margin: 0 0 15px 0; color: #2d5016;">🌱 Environmental Benefits</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.5em;">💧</span>
                    <p style="margin: 0; color: #333;"><b>Groundwater</b> protection & conservation</p>
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.5em;">⚡</span>
                    <p style="margin: 0; color: #333;"><b>Energy savings</b> from reduced pumping</p>
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.5em;">🌿</span>
                    <p style="margin: 0; color: #333;"><b>Fertilizer runoff</b> reduction</p>
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.5em;">🌍</span>
                    <p style="margin: 0; color: #333;"><b>Sustainable</b> farming practices</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================

    st.markdown("---")
    st.header("📈 Model Insights & Visualizations")

    tab1, tab2 = st.tabs(["Feature Importance", "Data Insights"])

    with tab1:
        st.subheader("🔍 Feature Importance Analysis")
        st.markdown("Understanding which factors most influence crop yield:")

        if feature_importance is None:
            st.info("Feature importance is available for the Random Forest model. Switch the model to view this chart.")
        else:
            # Feature Importance Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            feature_names = [
                'Temperature', 'Rainfall', 'Humidity', 'Soil Moisture',
                'Soil pH', 'Nitrogen', 'Phosphorus', 'Potassium',
                'Crop Type', 'Season'
            ]
            
            feature_importance_display = feature_importance.copy()
            feature_importance_display['Feature'] = feature_names
            
            colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(feature_importance_display)))
            
            bars = ax.barh(
                feature_importance_display['Feature'],
                feature_importance_display['Importance'],
                color=colors
            )
            
            ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
            ax.set_ylabel('Features', fontsize=12, fontweight='bold')
            ax.set_title('Feature Importance in Crop Yield Prediction', 
                         fontsize=14, fontweight='bold', pad=20)
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2,
                       f'{width:.3f}',
                       ha='left', va='center', fontsize=9, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.info("""
            **Interpretation:**
            - Higher importance = greater impact on crop yield
            - Focus on top features for yield improvement
            - Soil nutrients (N, P, K) and moisture are typically critical
            """)
    
    with tab2:
        st.subheader("📊 Dataset Correlation Heatmap")

        # Select numeric columns for correlation
        numeric_cols = [
            'Temperature', 'Rainfall', 'Humidity', 'Soil_Moisture',
            'Soil_pH', 'Nitrogen', 'Phosphorus', 'Potassium', 'Crop_Yield'
        ]

        correlation_matrix = df[numeric_cols].corr()

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )

        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        st.pyplot(fig)

        st.info("""
        **How to read:**
        - Green: Positive correlation (both increase together)
        - Red: Negative correlation (one increases, other decreases)
        - Values close to 1 or -1 indicate strong relationships
        """)

    # ========================================================================
    # ADDITIONAL INFORMATION
    # ========================================================================

    st.markdown("---")

    with st.expander("ℹ️ About This System"):
        st.markdown("""
        ### System Overview
        
        **Purpose:**
        This ML-powered system addresses critical challenges in modern agriculture:
        - Water scarcity and irrigation optimization
        - Crop yield prediction for better planning
        - Sustainable farming practices
        
        **Technology Stack:**
        - **Machine Learning:** Random Forest Regressor
        - **Framework:** Streamlit for web interface
        - **Data Processing:** Pandas, NumPy, Scikit-learn
        - **Visualization:** Matplotlib, Seaborn
        
        **Model Details:**
        - **Algorithm:** Random Forest (150 trees)
        - **Features:** 10 environmental and soil parameters
        - **Performance:** Evaluated using R², MSE, RMSE, MAE
        
        **Water Optimization Logic:**
        The system considers:
        1. Current soil moisture levels
        2. Recent rainfall data
        3. Crop-specific water requirements
        4. Temperature and humidity (evapotranspiration)
        5. Seasonal variations
        
        **Use Cases:**
        - Farm planning and resource allocation
        - Irrigation scheduling
        - Crop selection based on conditions
        - Environmental impact assessment
        
        **Developed for:**
        Academic research, sustainable agriculture initiatives, and precision farming applications.
        """)

    with st.expander("📖 How to Use This Application"):
        st.markdown("""
        ### Step-by-Step Guide
        
        1. **Select Crop Type:** Choose the crop you're growing
        2. **Choose Season:** Select the appropriate growing season
        3. **Enter Environmental Data:**
           - Temperature (from weather data)
           - Rainfall (recent precipitation)
           - Humidity levels
        4. **Input Soil Conditions:**
           - Soil moisture (can be measured with sensors)
           - Soil pH (from soil testing)
        5. **Add Nutrient Levels:**
           - Nitrogen, Phosphorus, Potassium (from soil test reports)
        6. **Click Predict:** Get instant predictions and recommendations
        
        ### Tips for Best Results
        - Use recent and accurate measurements
        - Perform soil tests regularly for NPK values
        - Update environmental data based on current weather
        - Consider seasonal variations in your region
        
        ### Understanding Results
        - **Crop Yield:** Expected harvest in tons/hectare
        - **Irrigation Amount:** Recommended water in millimeters
        - **Water Saved:** Efficiency gain vs traditional methods
        - **Sustainability Impact:** Environmental benefits achieved
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>🌾 Crop Yield Prediction & Water Optimization System | Powered by Machine Learning</p>
        <p>Supporting Sustainable Agriculture for a Better Tomorrow 🌍</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()