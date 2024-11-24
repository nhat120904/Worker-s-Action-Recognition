import streamlit as st
import pandas as pd
import torch
import numpy as np
from model import TransformerClassifier, LSTMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib

# Load the trained models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformer (Preprocessed)
transformer_preprocessed = TransformerClassifier(
    input_dim=152,
    hidden_dim=256,
    num_classes=9,
    num_heads=8,
    num_layers=3,
    dim_feedforward=512,
    dropout=0.1,
    max_seq_length=60
)
transformer_preprocessed.load_state_dict(torch.load('./weights/trans_256_512.pt', map_location=device))
transformer_preprocessed.to(device)
transformer_preprocessed.eval()

# Transformer (Raw)
transformer_raw = TransformerClassifier(
    input_dim=138,
    hidden_dim=256,
    num_classes=9,
    num_heads=8,
    num_layers=3,
    dim_feedforward=512,
    dropout=0.1,
    max_seq_length=60
)
transformer_raw.load_state_dict(torch.load('./weights/trans_256_512_raw.pt', map_location=device))
transformer_raw.to(device)
transformer_raw.eval()

# Transformer (Raw) - Knife Sharpness
knife_transformer_raw = TransformerClassifier(
    input_dim=138,
    hidden_dim=256,
    num_classes=3,
    num_heads=8,
    num_layers=3,
    dim_feedforward=512,
    dropout=0.1,
    max_seq_length=60
)

knife_transformer_raw.load_state_dict(torch.load('./weights/trans_knife_raw.pt', map_location=device))
knife_transformer_raw.to(device)
knife_transformer_raw.eval()

# Transformer (Preprocessed) - Knife Sharpness
knife_transformer_processed= TransformerClassifier(
    input_dim=152,
    hidden_dim=256,
    num_classes=3,
    num_heads=8,
    num_layers=3,
    dim_feedforward=512,
    dropout=0.1,
    max_seq_length=60
)

knife_transformer_processed.load_state_dict(torch.load('./weights/trans_knife.pt', map_location=device))
knife_transformer_processed.to(device)
knife_transformer_processed.eval()

# LSTM (Raw)
lstm_raw = LSTMClassifier(
    input_dim=138,
    hidden_dim=256,
    num_classes=9,
)
lstm_raw.load_state_dict(torch.load('./weights/lstm_256_raw.pt', map_location=device))
lstm_raw.to(device)
lstm_raw.eval()

# LSTM (Preprocessed)
lstm_preprocessed = LSTMClassifier(
    input_dim=152,
    hidden_dim=256,
    num_classes=9,
)
lstm_preprocessed.load_state_dict(torch.load('./weights/lstm_256.pt', map_location=device))
lstm_preprocessed.to(device)
lstm_preprocessed.eval()


# LSTM (Raw) - Knife Sharpness
knife_lstm_raw = LSTMClassifier(
    input_dim=138,
    hidden_dim=64,
    num_classes=3,
)
knife_lstm_raw.load_state_dict(torch.load('./weights/lstm_knife_raw.pt', map_location=device))
knife_lstm_raw.to(device)
knife_lstm_raw.eval()

# LSTM (Preprocesses) - Knife Sharpness
knife_lstm_preprocessed = LSTMClassifier(
    input_dim=152,
    hidden_dim=64,
    num_classes=3,
)
knife_lstm_preprocessed.load_state_dict(torch.load('./weights/lstm_knife.pt', map_location=device))
knife_lstm_preprocessed.to(device)
knife_lstm_preprocessed.eval()

# Random Forest
RF_raw = RandomForestClassifier(n_estimators=100, random_state=42)

# SVM
SVM_raw = SVC(kernel='rbf', C=1, random_state=42)


def calculate_motion_features_df(df):
    feature_list = []

    # Helper function to extract x, y, z data for a body part
    def get_data(body_part, data_type):
        return df[[f"{body_part} x_{data_type}", 
                   f"{body_part} y_{data_type}", 
                   f"{body_part} z_{data_type}"]].to_numpy()

    # Process each row of the DataFrame
    for _, row in df.iterrows():
        features = {}

        # 1. Energy of Motion: sum of squared acceleration components
        def calc_energy(body_part):
            accel = row[[f"{body_part} x_acce", f"{body_part} y_acce", f"{body_part} z_acce"]].values
            return np.sum(accel ** 2)

        features['right_hand_energy'] = calc_energy('Right Hand')
        features['left_hand_energy'] = calc_energy('Left Hand')

        # 2. Movement Intensity: RMS of acceleration components
        def calc_intensity(body_part):
            accel = row[[f"{body_part} x_acce", f"{body_part} y_acce", f"{body_part} z_acce"]].values
            return np.sqrt(np.mean(accel ** 2))

        features['right_hand_intensity'] = calc_intensity('Right Hand')
        features['left_hand_intensity'] = calc_intensity('Left Hand')

        # 3. Hand-Arm Angle: Angle between hand and forearm acceleration vectors
        def calc_segment_angle(part1, part2):
            vec1 = row[[f"{part1} x_acce", f"{part1} y_acce", f"{part1} z_acce"]].values
            vec2 = row[[f"{part2} x_acce", f"{part2} y_acce", f"{part2} z_acce"]].values
            dot_product = np.dot(vec1, vec2)
            magnitudes = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            return np.arccos(np.clip(dot_product / magnitudes, -1.0, 1.0)) * 180.0 / np.pi if magnitudes != 0 else 0

        features['right_hand_arm_angle'] = calc_segment_angle('Right Hand', 'Right Forearm')
        features['left_hand_arm_angle'] = calc_segment_angle('Left Hand', 'Left Forearm')

        # 4. Wrist Rotation: Norm of cross product of hand and forearm acceleration vectors
        def calc_wrist_rotation(hand, forearm):
            hand_accel = row[[f"{hand} x_acce", f"{hand} y_acce", f"{hand} z_acce"]].values
            forearm_accel = row[[f"{forearm} x_acce", f"{forearm} y_acce", f"{forearm} z_acce"]].values
            return np.linalg.norm(np.cross(hand_accel, forearm_accel))

        features['right_wrist_rotation'] = calc_wrist_rotation('Right Hand', 'Right Forearm')
        features['left_wrist_rotation'] = calc_wrist_rotation('Left Hand', 'Left Forearm')

        # 5. Hands Symmetry: Norm of difference between left and right hand accelerations
        def calc_symmetry(part1, part2):
            accel1 = row[[f"{part1} x_acce", f"{part1} y_acce", f"{part1} z_acce"]].values
            accel2 = row[[f"{part2} x_acce", f"{part2} y_acce", f"{part2} z_acce"]].values
            return np.linalg.norm(accel1 - accel2)

        features['hands_symmetry'] = calc_symmetry('Right Hand', 'Left Hand')

        # 6. Posture Stability: Variance of accelerations across trunk sensors
        def calc_stability():
            trunk_parts = ['Pelvis', 'L5', 'L3', 'T12', 'T8']
            accels = np.array([row[[f"{part} x_acce", f"{part} y_acce", f"{part} z_acce"]].values for part in trunk_parts])
            return np.var(accels)

        features['posture_stability'] = calc_stability()

        # 7. Movement Efficiency: Ratio of direct path to actual path
        def calc_efficiency(body_part):
            accel = row[[f"{body_part} x_acce", f"{body_part} y_acce", f"{body_part} z_acce"]].values
            direct_path = np.linalg.norm(accel)
            actual_path = np.sum(np.abs(accel))
            return direct_path / actual_path if actual_path != 0 else 1

        features['right_hand_efficiency'] = calc_efficiency('Right Hand')
        features['left_hand_efficiency'] = calc_efficiency('Left Hand')

        # 8. Kinetic Power: Dot product of acceleration and velocity (proxy for power)
        def calc_kinetic_power(body_part):
            accel = row[[f"{body_part} x_acce", f"{body_part} y_acce", f"{body_part} z_acce"]].values
            velocity = row[[f"{body_part} x", f"{body_part} y", f"{body_part} z"]].values
            return np.sum(accel * velocity)

        features['right_hand_kinetic_power'] = calc_kinetic_power('Right Hand')
        features['left_hand_kinetic_power'] = calc_kinetic_power('Left Hand')

        feature_list.append(features)

    return pd.DataFrame(feature_list)

# Function to preprocess uploaded data
def preprocess_file(df, feature_type):
    try:
        if feature_type == 'Engineered Features':
            df = df.drop(columns=['Unnamed: 0', 'Frame', 'Marker', 'Frame_acce', 'Label'], errors='ignore')
            motion_features_df = calculate_motion_features_df(df)
            # Combine the original DataFrame with the motion features DataFrame
            df = pd.concat([df.reset_index(drop=True), motion_features_df.reset_index(drop=True)], axis=1)
        else:
            df = df.drop(columns=['Unnamed: 0', 'Frame', 'Marker', 'Frame_acce', 'Label'], errors='ignore')
        if len(df) < 60:
            st.error("Insufficient data: At least 60 frames are required.")
            return None
        sample = df.iloc[:60].to_numpy()
        return torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

# Streamlit UI
st.title("60-Frame Activity Prediction")
st.write("Upload a CSV file containing 60-frame data to predict activity.")

# Model and feature selection
model_type = st.sidebar.radio("Select Model:", ["Transformer", "LSTM", "Random Forest", "SVM"])
feature_type = st.sidebar.radio("Select Feature Type:", ["Raw Features", "Engineered Features"])

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

activity_labels = {0: "Idle", 1: "Walking", 2: "Steeling", 3: "Reaching", 4: "Cutting", 5: "Slicing", 6: "Pulling", 7: "Placing", 8: "Dropping"}
knife_sharpness = {0: "Sharp", 1: "Medium", 2: "Blunt"}

if uploaded_file is not None:
    st.write("Preview of Uploaded Data:")
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.drop(columns=['Unnamed: 0'], errors='ignore'))
    st.write("File uploaded successfully!")

    input_data = preprocess_file(df, feature_type)
    if input_data is not None:
        with torch.no_grad():
            if model_type == "Transformer":
                model = transformer_preprocessed if feature_type == "Engineered Features" else transformer_raw
                knife_model = knife_transformer_processed if feature_type == "Engineered Features" else knife_transformer_raw
            elif model_type == "LSTM":
                if feature_type == "Engineered Features":
                    model = lstm_preprocessed
                    knife_model = knife_lstm_preprocessed
                else:
                    model = lstm_raw
                    knife_model = knife_lstm_raw
            elif model_type == "Random Forest":
                if feature_type == "Engineered Features":
                    model = joblib.load('./weights/RF_knife_processed_activity.pkl')
                    knife_model = joblib.load('./weights/RF_knife_processed.pkl')
                else:
                    model = joblib.load('./weights/RF_activity_raw.pkl')
                    knife_model = joblib.load('./weights/RF_knife_raw.pkl')
            elif model_type == "SVM":
                if feature_type == "Engineered Features":
                    model = joblib.load('./weights/SVM_knife_processed_activity.pkl')
                    knife_model = joblib.load('./weights/SVM_knife_processed.pkl')
                else:
                    model = joblib.load('./weights/SVM_activity_raw.pkl')
                    knife_model = joblib.load('./weights/SVM_knife_raw.pkl')
            
            if model:
                try:
                    predictions = model(input_data)
                    knife_predictions = knife_model(input_data)
                    predicted_label = torch.argmax(predictions, dim=1).item()
                    knife_predicted_label = torch.argmax(knife_predictions, dim=1).item()
                except:
                    try:
                        predictions = model.predict(input_data.cpu().numpy().reshape(1, -1))
                        predicted_label = predictions[0]
                        knife_predictions = knife_model.predict(input_data.cpu().numpy().reshape(1, -1))
                        knife_predicted_label = knife_predictions[0]
                        # predictions = torch.tensor(predictions)
                    # print(input_data.cpu().numpy().shape)
                    # predictions = model.predict(input_data.cpu().numpy())
                    # predicted_label = predictions[0]
                    except Exception as e:
                        print(e)
                        predicted_label = -1
                st.success(f"The worker was {activity_labels[predicted_label]} (activity {predicted_label}) with a {knife_sharpness[knife_predicted_label]} knife")
                # confidence = torch.softmax(predictions, dim=1).cpu().numpy()[0]
                # st.write("Confidence Scores:", confidence)
