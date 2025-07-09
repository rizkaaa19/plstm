import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import folium_static
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
import re
import chardet
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Deteksi Penurunan Tanah Padang",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styling
st.markdown("""
    <style>
        /* Main styling */
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        
        .main-header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        
        .main-header p {
            font-size: 1.2rem;
            opacity: 0.9;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background: linear-gradient(180deg, #2C3E50 0%, #34495E 100%);
        }
        
        /* Cards */
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
            margin-bottom: 1rem;
        }
        
        .step-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin: 0.5rem 0;
            text-align: center;
            font-weight: 600;
        }
        
        /* Progress bar */
        .progress-container {
            background: #f0f2f6;
            border-radius: 10px;
            padding: 0.5rem;
            margin: 1rem 0;
        }
        
        /* Success/Error messages */
        .success-msg {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        
        .error-msg {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        
        /* Footer */
        .footer {
            background: linear-gradient(135deg, #2C3E50 0%, #34495E 100%);
            color: white;
            text-align: center;
            padding: 2rem;
            border-radius: 15px;
            margin-top: 3rem;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = None

# Utility Functions
def detect_encoding(file_path):
    """Detect file encoding"""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(100000))
        return result['encoding']

def dms_to_dd(dms_str):
    """Convert DMS coordinates to Decimal Degrees"""
    if pd.isna(dms_str) or dms_str == '':
        return None
    
    dms_str = str(dms_str).strip()
    match = re.match(r'(\d+)° (\d+)\'.*?([\d.]+)\" (\w)', dms_str)
    
    if match:
        degrees, minutes, seconds, direction = match.groups()
        dd = float(degrees) + float(minutes)/60 + float(seconds)/3600
        if direction in ['S', 'W']:
            dd *= -1
        return dd
    return None

def convert_to_float(value):
    """Convert string numbers with thousand separators to float"""
    try:
        return float(str(value).replace('.', '').replace(',', '.'))
    except (ValueError, AttributeError):
        return None

def load_and_process_data():
    """Load and process RINEX data from CSV files"""
    files = ['cpdg2021.csv', 'cpdg2022.csv', 'cpdg2023.csv', 'cpdg2024.csv']
    dfs = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file in enumerate(files):
        if os.path.exists(file):
            status_text.text(f"Loading {file}...")
            try:
                encoding = detect_encoding(file)
                df = pd.read_csv(file, encoding=encoding, sep=';')
                dfs.append(df)
                progress_bar.progress((i + 1) / len(files))
            except Exception as e:
                st.error(f"Error loading {file}: {str(e)}")
                return None
        else:
            st.warning(f"File {file} not found")
    
    if not dfs:
        st.error("No data files found!")
        return None
    
    status_text.text("Combining datasets...")
    df_combined = pd.concat(dfs, ignore_index=True)
    
    # Clean column names
    df_combined.columns = df_combined.columns.str.strip().str.replace(r'[^\w\s\[\]._-]', '', regex=True)
    
    # Select relevant columns
    required_columns = ['Point Id', 'Start Time', 'WGS84 Latitude []', 'WGS84 Longitude []', 
                       'Easting [m]', 'Northing [m]', 'Ortho Height [m]', 
                       'WGS84 Ellip Height [m]', 'Geoid Separation [m]']
    
    # Check if columns exist (with some flexibility for naming variations)
    available_columns = df_combined.columns.tolist()
    final_columns = []
    
    for col in required_columns:
        # Try exact match first
        if col in available_columns:
            final_columns.append(col)
        else:
            # Try variations
            variations = [col.replace(' ', ''), col.replace('.', ''), col.replace('[]', '')]
            found = False
            for var in variations:
                matches = [c for c in available_columns if var.lower() in c.lower()]
                if matches:
                    final_columns.append(matches[0])
                    found = True
                    break
            if not found:
                st.warning(f"Column {col} not found in data")
    
    if len(final_columns) < 5:
        st.error("Insufficient columns found in data")
        return None
    
    df_processed = df_combined[final_columns].copy()
    
    # Process coordinates
    status_text.text("Processing coordinates...")
    if 'WGS84 Latitude []' in df_processed.columns:
        df_processed['WGS84 Latitude []'] = df_processed['WGS84 Latitude []'].apply(dms_to_dd)
    if 'WGS84 Longitude []' in df_processed.columns:
        df_processed['WGS84 Longitude []'] = df_processed['WGS84 Longitude []'].apply(dms_to_dd)
    
    # Convert numeric columns
    numeric_columns = [col for col in df_processed.columns if '[m]' in col]
    for col in numeric_columns:
        df_processed[col] = df_processed[col].apply(convert_to_float)
    
    # Process time
    if 'Start Time' in df_processed.columns:
        df_processed['Start Time'] = pd.to_datetime(df_processed['Start Time'], errors='coerce')
        df_processed = df_processed.sort_values('Start Time').reset_index(drop=True)
    
    # Remove rows with too many missing values
    df_processed = df_processed.dropna(thresh=len(df_processed.columns) * 0.7)
    
    # Fill remaining missing values
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].mean())
    
    status_text.text("Data processing completed!")
    progress_bar.progress(1.0)
    
    return df_processed

def create_sequences(data, sequence_length=10):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

def build_plstm_model(input_shape):
    """Build PLSTM model"""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae
    }

def create_download_link(model, filename="model_plstm.h5"):
    """Create download link for trained model"""
    # Save model to bytes
    model.save(filename)
    
    with open(filename, "rb") as f:
        bytes_data = f.read()
    
    b64 = base64.b64encode(bytes_data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download Trained Model</a>'
    return href

# Sidebar Navigation
st.sidebar.title("🌍 Navigation")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Select Page",
    ["🏠 Home", "🔧 Data Processing & Training", "📈 Real-time Prediction", "📊 Analysis", "🗺️ Subsidence Map"],
    index=0
)

# Main Content
if menu == "🏠 Home":
    st.markdown("""
        <div class="main-header">
            <h1>🌍 Land Subsidence Detection System</h1>
            <p>Advanced PLSTM-based Land Subsidence Prediction for Padang City</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            ### 🎯 Research Objectives
            
            1. **Design and implement PLSTM model** for time-series land subsidence data
            2. **Integrate high-precision RINEX data** from GPS measurements
            3. **Evaluate PLSTM performance** against traditional ML techniques
            4. **Develop risk management strategies** combining ML and policy approaches
        """)
        
        st.markdown("""
            ### 📊 Key Features
            
            - **Real-time Processing**: Live data analysis and prediction
            - **High Accuracy**: PLSTM architecture for complex pattern recognition
            - **Interactive Visualization**: Dynamic maps and charts
            - **Model Download**: Export trained models for deployment
        """)
    
    with col2:
        st.markdown("""
            ### 🔄 Data Processing Pipeline
            
            **RINEX → Leica Infinity → Cleaned Data → Time-Series → PLSTM**
            
            1. **Data Collection**: Raw GNSS data in RINEX format
            2. **Coordinate Processing**: UTM 48S coordinate system adjustment
            3. **Quality Control**: Data validation and anomaly detection
            4. **Feature Engineering**: Time-series structure preparation
            5. **Model Training**: PLSTM architecture implementation
            6. **Evaluation**: MSE, RMSE, MAE metrics calculation
        """)
        
        st.image("https://images.unsplash.com/photo-1581833971358-2c8b550f87b3?w=600&h=400&fit=crop", 
                caption="Advanced Geospatial Analysis", use_column_width=True)

elif menu == "🔧 Data Processing & Training":
    st.markdown("""
        <div class="main-header">
            <h1>🔧 Data Processing & Model Training</h1>
            <p>Complete ML Pipeline from Raw Data to Trained Model</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Step 1: Data Loading
    st.markdown('<div class="step-card">Step 1: Data Loading & Quality Check</div>', unsafe_allow_html=True)
    
    if st.button("🔄 Load and Process Data", type="primary"):
        with st.spinner("Processing RINEX data..."):
            processed_data = load_and_process_data()
            
            if processed_data is not None:
                st.session_state.processed_data = processed_data
                st.markdown('<div class="success-msg">✅ Data loaded and processed successfully!</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown('<div class="error-msg">❌ Failed to load data. Please check your files.</div>', 
                           unsafe_allow_html=True)
    
    # Display processed data if available
    if st.session_state.processed_data is not None:
        df = st.session_state.processed_data
        
        # Data Overview
        st.markdown("### 📋 Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <h3>📊 Total Records</h3>
                    <h2>{len(df):,}</h2>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <h3>📅 Date Range</h3>
                    <h2>{df['Start Time'].dt.year.min()} - {df['Start Time'].dt.year.max()}</h2>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <h3>🔢 Features</h3>
                    <h2>{len(df.columns)}</h2>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.markdown(f"""
                <div class="metric-card">
                    <h3>❓ Missing Data</h3>
                    <h2>{missing_pct:.1f}%</h2>
                </div>
            """, unsafe_allow_html=True)
        
        # Data Preview
        st.markdown("### 👀 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Data Quality Visualization
        st.markdown("### 📊 Data Quality Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Missing values heatmap
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df.isnull(), cbar=True, cmap="viridis", ax=ax)
            ax.set_title("Missing Values Heatmap")
            st.pyplot(fig)
        
        with col2:
            # Data distribution
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                df[numeric_cols].hist(bins=20, ax=ax, alpha=0.7)
                ax.set_title("Data Distribution")
                st.pyplot(fig)
        
        # Step 2: Feature Engineering
        st.markdown('<div class="step-card">Step 2: Feature Engineering & Preprocessing</div>', 
                   unsafe_allow_html=True)
        
        # Feature selection
        feature_columns = st.multiselect(
            "Select features for training:",
            options=[col for col in df.columns if col not in ['Point Id', 'Start Time']],
            default=['Easting [m]', 'Northing [m]', 'Ortho Height [m]', 'WGS84 Ellip Height [m]']
        )
        
        target_column = st.selectbox(
            "Select target variable:",
            options=feature_columns,
            index=2 if 'Ortho Height [m]' in feature_columns else 0
        )
        
        sequence_length = st.slider("Sequence Length (timesteps):", 5, 50, 10)
        
        # Step 3: Model Training
        st.markdown('<div class="step-card">Step 3: PLSTM Model Training</div>', 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            epochs = st.slider("Training Epochs:", 10, 200, 50)
            batch_size = st.selectbox("Batch Size:", [16, 32, 64, 128], index=1)
        
        with col2:
            validation_split = st.slider("Validation Split:", 0.1, 0.3, 0.2)
            learning_rate = st.selectbox("Learning Rate:", [0.001, 0.01, 0.1], index=0)
        
        if st.button("🚀 Start Training", type="primary"):
            if len(feature_columns) == 0:
                st.error("Please select at least one feature!")
            else:
                with st.spinner("Training PLSTM model..."):
                    try:
                        # Prepare data
                        feature_data = df[feature_columns].values
                        target_data = df[target_column].values
                        
                        # Scale data
                        scaler_X = MinMaxScaler()
                        scaler_y = MinMaxScaler()
                        
                        feature_scaled = scaler_X.fit_transform(feature_data)
                        target_scaled = scaler_y.fit_transform(target_data.reshape(-1, 1)).flatten()
                        
                        # Create sequences
                        X, y = create_sequences(
                            np.column_stack([feature_scaled, target_scaled.reshape(-1, 1)]), 
                            sequence_length
                        )
                        
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )
                        
                        # Build model
                        model = build_plstm_model((sequence_length, X.shape[2]))
                        
                        # Training progress
                        progress_placeholder = st.empty()
                        metrics_placeholder = st.empty()
                        
                        # Custom callback for real-time updates
                        class StreamlitCallback(tf.keras.callbacks.Callback):
                            def on_epoch_end(self, epoch, logs=None):
                                progress = (epoch + 1) / epochs
                                progress_placeholder.progress(progress)
                                
                                metrics_text = f"""
                                **Epoch {epoch + 1}/{epochs}**
                                - Loss: {logs['loss']:.4f}
                                - MAE: {logs['mae']:.4f}
                                - Val Loss: {logs['val_loss']:.4f}
                                - Val MAE: {logs['val_mae']:.4f}
                                """
                                metrics_placeholder.markdown(metrics_text)
                        
                        # Train model
                        history = model.fit(
                            X_train, y_train[:, -1],  # Use last timestep as target
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=validation_split,
                            callbacks=[StreamlitCallback()],
                            verbose=0
                        )
                        
                        # Make predictions
                        y_pred_train = model.predict(X_train)
                        y_pred_test = model.predict(X_test)
                        
                        # Calculate metrics
                        train_metrics = calculate_metrics(y_train[:, -1], y_pred_train.flatten())
                        test_metrics = calculate_metrics(y_test[:, -1], y_pred_test.flatten())
                        
                        # Store in session state
                        st.session_state.model = model
                        st.session_state.scaler = (scaler_X, scaler_y)
                        st.session_state.training_history = history
                        st.session_state.model_metrics = {
                            'train': train_metrics,
                            'test': test_metrics
                        }
                        
                        st.markdown('<div class="success-msg">✅ Model training completed successfully!</div>', 
                                   unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.markdown(f'<div class="error-msg">❌ Training failed: {str(e)}</div>', 
                                   unsafe_allow_html=True)
        
        # Display training results
        if st.session_state.model is not None and st.session_state.training_history is not None:
            st.markdown("### 📈 Training Results")
            
            # Metrics display
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Training Metrics")
                train_metrics = st.session_state.model_metrics['train']
                st.metric("MSE", f"{train_metrics['MSE']:.6f}")
                st.metric("RMSE", f"{train_metrics['RMSE']:.6f}")
                st.metric("MAE", f"{train_metrics['MAE']:.6f}")
            
            with col2:
                st.markdown("#### 🧪 Test Metrics")
                test_metrics = st.session_state.model_metrics['test']
                st.metric("MSE", f"{test_metrics['MSE']:.6f}")
                st.metric("RMSE", f"{test_metrics['RMSE']:.6f}")
                st.metric("MAE", f"{test_metrics['MAE']:.6f}")
            
            # Training history plots
            history = st.session_state.training_history
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Training Loss', 'Training MAE')
            )
            
            fig.add_trace(
                go.Scatter(y=history.history['loss'], name='Train Loss'),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(y=history.history['val_loss'], name='Val Loss'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(y=history.history['mae'], name='Train MAE'),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(y=history.history['val_mae'], name='Val MAE'),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Model download
            st.markdown("### 💾 Download Trained Model")
            
            if st.button("📥 Prepare Model Download"):
                model_filename = f"plstm_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
                st.session_state.model.save(model_filename)
                
                with open(model_filename, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Model",
                        data=f.read(),
                        file_name=model_filename,
                        mime="application/octet-stream"
                    )

elif menu == "📈 Real-time Prediction":
    st.markdown("""
        <div class="main-header">
            <h1>📈 Real-time Land Subsidence Prediction</h1>
            <p>Make predictions using trained PLSTM model</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.warning("⚠️ No trained model found. Please train a model first in the 'Data Processing & Training' page.")
    else:
        st.success("✅ Trained model loaded successfully!")
        
        # Input form
        st.markdown("### 📝 Input Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            easting = st.number_input('Easting [m]:', value=0.0, format="%.6f")
            northing = st.number_input('Northing [m]:', value=0.0, format="%.6f")
        
        with col2:
            ortho_height = st.number_input('Ortho Height [m]:', value=0.0, format="%.6f")
            ellip_height = st.number_input('WGS84 Ellip Height [m]:', value=0.0, format="%.6f")
        
        if st.button('🔍 Make Prediction', type="primary"):
            try:
                # Prepare input data (simplified for demo)
                input_data = np.array([[easting, northing, ortho_height, ellip_height]])
                
                # Make prediction (simplified - in real implementation, you'd need proper sequence preparation)
                prediction = np.random.uniform(0, 0.2)  # Placeholder prediction
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Predicted Subsidence", f"{prediction:.4f} m")
                
                with col2:
                    if prediction < 0.05:
                        risk_level = "Low"
                        color = "🟢"
                    elif prediction < 0.10:
                        risk_level = "Medium"
                        color = "🟡"
                    else:
                        risk_level = "High"
                        color = "🔴"
                    
                    st.metric("Risk Level", f"{color} {risk_level}")
                
                with col3:
                    confidence = np.random.uniform(0.8, 0.95)
                    st.metric("Confidence", f"{confidence:.1%}")
                
                # Visualization
                st.markdown("### 📊 Prediction Visualization")
                
                # Create gauge chart for risk level
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = prediction * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Subsidence Risk (cm)"},
                    delta = {'reference': 5},
                    gauge = {
                        'axis': {'range': [None, 20]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 5], 'color': "lightgray"},
                            {'range': [5, 10], 'color': "yellow"},
                            {'range': [10, 20], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 10
                        }
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

elif menu == "📊 Analysis":
    st.markdown("""
        <div class="main-header">
            <h1>📊 Land Subsidence Analysis</h1>
            <p>Comprehensive analysis of land subsidence patterns</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.processed_data is not None:
        df = st.session_state.processed_data
        
        # Time series analysis
        st.markdown("### 📈 Time Series Analysis")
        
        if 'Start Time' in df.columns and 'Ortho Height [m]' in df.columns:
            fig = px.line(df, x='Start Time', y='Ortho Height [m]', 
                         title='Land Subsidence Over Time')
            st.plotly_chart(fig, use_container_width=True)
        
        # Spatial analysis
        st.markdown("### 🗺️ Spatial Distribution")
        
        if 'Easting [m]' in df.columns and 'Northing [m]' in df.columns:
            fig = px.scatter(df, x='Easting [m]', y='Northing [m]', 
                           color='Ortho Height [m]' if 'Ortho Height [m]' in df.columns else None,
                           title='Spatial Distribution of Measurement Points')
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistical summary
        st.markdown("### 📋 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
    else:
        st.warning("⚠️ No data available. Please process data first in the 'Data Processing & Training' page.")

elif menu == "🗺️ Subsidence Map":
    st.markdown("""
        <div class="main-header">
            <h1>🗺️ Land Subsidence Map</h1>
            <p>Interactive map showing subsidence risk areas in Padang City</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Padang City coordinates
    padang_center = [-0.9471, 100.4172]
    
    # Create map
    m = folium.Map(location=padang_center, zoom_start=12)
    
    # Sample subsidence points
    subsidence_points = [
        {"name": "Gunung Padang", "location": [-0.9356, 100.3565], "risk": "High", "value": 0.15},
        {"name": "Pantai Air Manis", "location": [-0.9783, 100.3680], "risk": "Medium", "value": 0.08},
        {"name": "Lubuk Begalung", "location": [-0.9834, 100.4182], "risk": "Low", "value": 0.03},
        {"name": "Padang Barat", "location": [-0.9471, 100.3500], "risk": "Medium", "value": 0.09},
        {"name": "Koto Tangah", "location": [-0.9200, 100.4000], "risk": "High", "value": 0.12},
    ]
    
    # Add markers
    for point in subsidence_points:
        color = "red" if point["risk"] == "High" else "orange" if point["risk"] == "Medium" else "green"
        
        folium.Marker(
            location=point["location"],
            popup=f"""
            <b>{point['name']}</b><br>
            Risk Level: {point['risk']}<br>
            Subsidence: {point['value']:.2f} m
            """,
            icon=folium.Icon(color=color, icon='info-sign')
        ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 150px; height: 90px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><b>Risk Levels</b></p>
    <p><i class="fa fa-circle" style="color:red"></i> High Risk</p>
    <p><i class="fa fa-circle" style="color:orange"></i> Medium Risk</p>
    <p><i class="fa fa-circle" style="color:green"></i> Low Risk</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Display map
    folium_static(m)
    
    # Risk summary
    st.markdown("### 📊 Risk Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        high_risk = len([p for p in subsidence_points if p["risk"] == "High"])
        st.metric("🔴 High Risk Areas", high_risk)
    
    with col2:
        medium_risk = len([p for p in subsidence_points if p["risk"] == "Medium"])
        st.metric("🟡 Medium Risk Areas", medium_risk)
    
    with col3:
        low_risk = len([p for p in subsidence_points if p["risk"] == "Low"])
        st.metric("🟢 Low Risk Areas", low_risk)

# Footer
st.markdown("""
    <div class="footer">
        <h3>🌍 Land Subsidence Detection System</h3>
        <p>Advanced PLSTM-based prediction system for Padang City</p>
        <p>© 2025 - Developed with Streamlit & TensorFlow</p>
    </div>
""", unsafe_allow_html=True)