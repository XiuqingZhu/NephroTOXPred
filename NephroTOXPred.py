import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
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

# Smiles: string input
smiles = st.text_input("SMILE:", value="")

if st.button("Predict"):

    #调用函数生成特征向量
    feature_vector = generate_feature_vector(smiles, feature_names)
    features = np.array([feature_vector])

    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"According to our model, the compound that you submitted has a high risk of nephrotoxicity. "
            f"The model predicts that your probability of having nephrotoxicity is {probability:.2f}%. "
            "While this is just an estimate, it suggests that the compound may be at significant risk. "
            "I recommend that you consult a cardiologist as soon as possible for further evaluation and "
            "to ensure you receive an accurate diagnosis and necessary treatment."
        )
    else:
        advice = (
            f"According to our model, the compound that you submitted has a low risk of nephrotoxicity. "
            f"The model predicts that your probability of not having nephrotoxicity is {probability:.2f}%. "
            "However, maintaining a healthy lifestyle is still very important. "
            "I recommend regular check-ups to monitor your heart health, "
            "and to seek medical advice promptly if you experience any symptoms."
        )

    st.write(advice)

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
    st.image("./shap_force_plot.png")
