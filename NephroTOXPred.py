import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import os
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
import matplotlib.pyplot as plt

def get_fingerprints(smiles):
    # 解析 SMILES
    mol = Chem.MolFromSmiles(smiles)
    
    # 计算 MACCS 指纹
    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
    maccs_bits = np.array(maccs_fp, dtype=int).tolist()

    # 计算 ECFP4 指纹
    ecfp4_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    ecfp4_bits = np.array(ecfp4_fp, dtype=int).tolist()

    return maccs_bits, ecfp4_bits

def generate_feature_vector(smiles, feature_order):
    maccs_bits, ecfp4_bits = get_fingerprints(smiles)
    feature_vector = []

    for feature in feature_order:
        if feature.startswith("MACCS_"):
            index = int(feature.split("_")[1]) 
            feature_vector.append(maccs_bits[index])
        elif feature.startswith("ECFP4_bitvector"):
            index = int(feature.split("bitvector")[1])
            feature_vector.append(ecfp4_bits[index])

    return feature_vector

# Define feature names
feature_df = pd.read_csv('./features_for_ML.csv')
feature_names = feature_df['Features'].values.tolist()

# Load the model
model = joblib.load('./Model_final.joblib')

# add logo
st.image("./logo.png")

# Streamlit user interface
st.title("Nephrotoxic Component Predictor")

st.write("**Please enter a SMILE string for predicting nephrotoxic components.**")

# Smiles: string input
smiles = st.text_input("SMILE:", value="")

if st.button("Predict"):
    # Generate feature vector
    feature_vector = generate_feature_vector(smiles, feature_names)
    
    if feature_vector is None:
        st.write("**Invalid SMILES string. Please provide a correct SMILES notation.**")
    else:
        features = np.array([feature_vector])
        
        # Predict class and probabilities
        predicted_class = model.predict(features)[0]
        predicted_proba = model.predict_proba(features)[0]

        # Display a separator line
        st.write("---") 
        # Display prediction results
        st.write(f"**Predicted Class:** {predicted_class}")
        st.write(f"**Prediction Probabilities:** {predicted_proba}")

        # Generate advice based on prediction results
        probability = predicted_proba[predicted_class] * 100

        if predicted_class == 1:
            advice = (
                f"According to our model, the compound you submitted poses a high risk of nephrotoxicity. "
                f"The model predicts that your likelihood of experiencing nephrotoxicity is {probability:.2f}%. "
                "While this is only an estimation, it indicates that the compound may be at a significant risk. "
            )
        else:
            advice = (
                f"According to our model, the compound you submitted has a low risk of nephrotoxicity. "
                f"The model predicts that your likelihood of not experiencing nephrotoxicity is {probability:.2f}%. "
            )

        st.write(advice)

       # 删除旧的 SHAP force plot 图片文件（如果存在）
        if os.path.exists("./shap_force_plot.png"):
            os.remove("./shap_force_plot.png")
        if os.path.exists("./shap_waterfall_plot.png"):
            os.remove("./shap_waterfall_plot.png")

        # Calculate SHAP values and display force plot    
        explainer = shap.TreeExplainer(model)  
        shap_values = explainer.shap_values(pd.DataFrame([feature_vector], columns=feature_names))

        # Choose the index of the output you want to visualize
        output_index = 0  # Adjust this index based on your specific needs

        # Force plot for the specified output
        expected_value = explainer.expected_value[output_index]

        positive_color = "#FF8C69"
        negative_color = "#B23AEE"

        # Generate the force plot for the specified output
        shap.force_plot(
            expected_value,
            shap_values[output_index],
            feature_names=pd.DataFrame([feature_vector], columns=feature_names).columns,
            matplotlib=True,
            show=True,
            plot_cmap=[positive_color, negative_color]
        )

        # Save and display the image
        plt.savefig("./shap_force_plot.png", bbox_inches='tight', dpi=1200)
        # Display a separator line
        st.write("---") 
        st.write("**The generated SHAP force plot of this compound:**")
        st.image("./shap_force_plot.png")

        # Generate and display SHAP waterfall plot
        st.write("---")
        st.write("**The SHAP Waterfall plot of this compound:**")

        # Create waterfall plot
        shap.waterfall_plot(
            explainer,
            shap_values[0],
            feature_names=pd.DataFrame([feature_vector], columns=feature_names).columns.tolist()
        )

        # Save the waterfall plot as an image
        plt.savefig("./shap_waterfall_plot.png", bbox_inches='tight', dpi=1200)
        # Display the waterfall plot image
        st.image("./shap_waterfall_plot.png")
