import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

# Load the saved model and scaler
try:
    model = pickle.load(open('breast_cancer.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except:
    st.error("Error: Model files not found. Please ensure 'breast_cancer.pkl' and 'scaler.pkl' exist.")

def predict_cancer(features_df):
    try:
        # Scale the features
        features_scaled = scaler.transform(features_df)
        
        # Make prediction
        prediction = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)
        
        return {
            'prediction': prediction[0],
            'benign_probability': probabilities[0][0],
            'malignant_probability': probabilities[0][1]
        }
    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
        return None

def main():
    st.title('TUMO TRACK')
    st.write('Enter patient measurements to predict breast cancer diagnosis')
    
    # Sample data buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button('Load Benign Sample'):
            return_dict = {
                'radius_mean': 12.45, 'texture_mean': 15.70, 'perimeter_mean': 82.57,
                'area_mean': 477.1, 'smoothness_mean': 0.1278, 'compactness_mean': 0.0917,
                'concavity_mean': 0.0633, 'concave points_mean': 0.0345, 'symmetry_mean': 0.1809,
                'fractal_dimension_mean': 0.0634, 'radius_se': 0.2364, 'texture_se': 0.7127,
                'perimeter_se': 1.657, 'area_se': 24.53, 'smoothness_se': 0.006150,
                'compactness_se': 0.01777, 'concavity_se': 0.02196, 'concave points_se': 0.00563,
                'symmetry_se': 0.01649, 'fractal_dimension_se': 0.002967, 'radius_worst': 13.76,
                'texture_worst': 19.68, 'perimeter_worst': 91.10, 'area_worst': 580.6,
                'smoothness_worst': 0.1675, 'compactness_worst': 0.1795, 'concavity_worst': 0.1889,
                'concave points_worst': 0.0821, 'symmetry_worst': 0.2711, 'fractal_dimension_worst': 0.0799
            }
            st.session_state.update(return_dict)
            st.success("Benign sample data loaded! Scroll down and click 'Predict'")

    with col2:
        if st.button('Load Malignant Sample'):
            return_dict = {
                'radius_mean': 17.99, 'texture_mean': 10.38, 'perimeter_mean': 122.8,
                'area_mean': 1001.0, 'smoothness_mean': 0.1184, 'compactness_mean': 0.2776,
                'concavity_mean': 0.3001, 'concave points_mean': 0.1471, 'symmetry_mean': 0.2419,
                'fractal_dimension_mean': 0.07871, 'radius_se': 1.095, 'texture_se': 0.9053,
                'perimeter_se': 8.589, 'area_se': 153.4, 'smoothness_se': 0.006399,
                'compactness_se': 0.04904, 'concavity_se': 0.05373, 'concave points_se': 0.01587,
                'symmetry_se': 0.03003, 'fractal_dimension_se': 0.006193, 'radius_worst': 25.38,
                'texture_worst': 17.33, 'perimeter_worst': 184.6, 'area_worst': 2019.0,
                'smoothness_worst': 0.1622, 'compactness_worst': 0.6656, 'concavity_worst': 0.7119,
                'concave points_worst': 0.2654, 'symmetry_worst': 0.4601, 'fractal_dimension_worst': 0.1189
            }
            st.session_state.update(return_dict)
            st.success("Malignant sample data loaded! Scroll down and click 'Predict'")

    # Create columns for better layout
    col1, col2, col3 = st.columns(3)
    
    input_features = {}
    
    with col1:
        st.subheader('Mean Values')
        input_features['radius_mean'] = st.number_input('Radius Mean', value=st.session_state.get('radius_mean', 0.0), format="%.4f")
        input_features['texture_mean'] = st.number_input('Texture Mean', value=st.session_state.get('texture_mean', 0.0), format="%.4f")
        input_features['perimeter_mean'] = st.number_input('Perimeter Mean', value=st.session_state.get('perimeter_mean', 0.0), format="%.4f")
        input_features['area_mean'] = st.number_input('Area Mean', value=st.session_state.get('area_mean', 0.0), format="%.4f")
        input_features['smoothness_mean'] = st.number_input('Smoothness Mean', value=st.session_state.get('smoothness_mean', 0.0), format="%.4f")
        input_features['compactness_mean'] = st.number_input('Compactness Mean', value=st.session_state.get('compactness_mean', 0.0), format="%.4f")
        input_features['concavity_mean'] = st.number_input('Concavity Mean', value=st.session_state.get('concavity_mean', 0.0), format="%.4f")
        input_features['concave points_mean'] = st.number_input('Concave Points Mean', value=st.session_state.get('concave points_mean', 0.0), format="%.4f")
        input_features['symmetry_mean'] = st.number_input('Symmetry Mean', value=st.session_state.get('symmetry_mean', 0.0), format="%.4f")
        input_features['fractal_dimension_mean'] = st.number_input('Fractal Dimension Mean', value=st.session_state.get('fractal_dimension_mean', 0.0), format="%.4f")
    
    with col2:
        st.subheader('Standard Error Values')
        input_features['radius_se'] = st.number_input('Radius SE', value=st.session_state.get('radius_se', 0.0), format="%.4f")
        input_features['texture_se'] = st.number_input('Texture SE', value=st.session_state.get('texture_se', 0.0), format="%.4f")
        input_features['perimeter_se'] = st.number_input('Perimeter SE', value=st.session_state.get('perimeter_se', 0.0), format="%.4f")
        input_features['area_se'] = st.number_input('Area SE', value=st.session_state.get('area_se', 0.0), format="%.4f")
        input_features['smoothness_se'] = st.number_input('Smoothness SE', value=st.session_state.get('smoothness_se', 0.0), format="%.4f")
        input_features['compactness_se'] = st.number_input('Compactness SE', value=st.session_state.get('compactness_se', 0.0), format="%.4f")
        input_features['concavity_se'] = st.number_input('Concavity SE', value=st.session_state.get('concavity_se', 0.0), format="%.4f")
        input_features['concave points_se'] = st.number_input('Concave Points SE', value=st.session_state.get('concave points_se', 0.0), format="%.4f")
        input_features['symmetry_se'] = st.number_input('Symmetry SE', value=st.session_state.get('symmetry_se', 0.0), format="%.4f")
        input_features['fractal_dimension_se'] = st.number_input('Fractal Dimension SE', value=st.session_state.get('fractal_dimension_se', 0.0), format="%.4f")
    
    with col3:
        st.subheader('Worst Values')
        input_features['radius_worst'] = st.number_input('Radius Worst', value=st.session_state.get('radius_worst', 0.0), format="%.4f")
        input_features['texture_worst'] = st.number_input('Texture Worst', value=st.session_state.get('texture_worst', 0.0), format="%.4f")
        input_features['perimeter_worst'] = st.number_input('Perimeter Worst', value=st.session_state.get('perimeter_worst', 0.0), format="%.4f")
        input_features['area_worst'] = st.number_input('Area Worst', value=st.session_state.get('area_worst', 0.0), format="%.4f")
        input_features['smoothness_worst'] = st.number_input('Smoothness Worst', value=st.session_state.get('smoothness_worst', 0.0), format="%.4f")
        input_features['compactness_worst'] = st.number_input('Compactness Worst', value=st.session_state.get('compactness_worst', 0.0), format="%.4f")
        input_features['concavity_worst'] = st.number_input('Concavity Worst', value=st.session_state.get('concavity_worst', 0.0), format="%.4f")
        input_features['concave points_worst'] = st.number_input('Concave Points Worst', value=st.session_state.get('concave points_worst', 0.0), format="%.4f")
        input_features['symmetry_worst'] = st.number_input('Symmetry Worst', value=st.session_state.get('symmetry_worst', 0.0), format="%.4f")
        input_features['fractal_dimension_worst'] = st.number_input('Fractal Dimension Worst', value=st.session_state.get('fractal_dimension_worst', 0.0), format="%.4f")

    # Create a button to make predictions
    if st.button('Predict'):
        # Convert input features to DataFrame
        features_df = pd.DataFrame([input_features])
        
        # Make prediction
        result = predict_cancer(features_df)
        
        if result:
            # Create a nice display for the results
            st.subheader('Prediction Results')
            
            # Display the main prediction with color
            if result['prediction'] == 1:
                st.error('🚨 Prediction: MALIGNANT')
            else:
                st.success('✅ Prediction: BENIGN')
            
            # Create columns for probabilities
            col1, col2 = st.columns(2)
            
            # Display probabilities with progress bars
            with col1:
                st.metric("Benign Probability", f"{result['benign_probability']:.1%}")
                st.progress(result['benign_probability'])
                
            with col2:
                st.metric("Malignant Probability", f"{result['malignant_probability']:.1%}")
                st.progress(result['malignant_probability'])
            
            # Add interpretation
            st.write("---")
            st.write("📊 Interpretation:")
            if result['prediction'] == 1:
                st.write("""
                - The model predicts this case as **MALIGNANT** (cancerous)
                - There is a high probability of malignancy
                """)
            else:
                st.write("""
                - The model predicts this case as **BENIGN** (non-cancerous)
                - There is a low probability of malignancy
                """)
            
            # Add disclaimer
            st.warning("""
            ⚠️ **Medical Disclaimer**: 
            This prediction is based on a machine learning model and should not be used as the sole basis for medical decisions. 
            Please consult with healthcare professionals for proper diagnosis and treatment.
            """)

    # Add information about the features
    with st.expander("See Feature Descriptions"):
        st.write("""
        - **Mean Values**: Average values of the cell nuclei measurements
        - **SE Values**: Standard error of the measurements
        - **Worst Values**: Mean of the three largest values in the sample
        
        All measurements are taken from digitized images of a fine needle aspirate (FNA) of a breast mass.
        """)

if __name__ == '__main__':
    main() 