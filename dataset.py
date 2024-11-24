import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import re
import numpy as np
import pandas as pd

torch.manual_seed(0)
# COUNT = [0, 0, 0, 0, 0, 0, 0, 0, 0]

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

def categorize_sharpness(sharpness):
    if sharpness >= 85:
        return 0
    elif 70 <= sharpness < 85:
        return 1
    else:
        return 2    

def extract_and_categorize_sharpness(filename):
    # Extract sharpness value using regex (assumes sharpness is the number before the last dash)
    sharpness_value = int(re.search(r'-([0-9]+)-', filename).group(1))
    return categorize_sharpness(sharpness_value)

def split_to_chunk(df, frame_size=60, step=5):
    # Check and remove any unnecessary columns
    df = df.drop(columns=['Unnamed: 0', 'Marker', 'Frame_acce'], errors='ignore')
    # df["test"] = 0
    # df = df.drop(columns=['Frame', 'Marker', 'Frame_acce'], errors='ignore')
    # motion_features_df = calculate_motion_features_df(df)

    # Combine the original DataFrame with the motion features DataFrame
    # df = pd.concat([df.reset_index(drop=True), motion_features_df.reset_index(drop=True)], axis=1)
    # df = df.drop(columns=['Unnamed: 0', 'Frame', 'Marker'], errors='ignore')
    # df = df.drop(columns=['Unnamed: 0', 'Marker'], errors='ignore')

    if 'Label' not in df.columns:
        raise ValueError("DataFrame must contain 'Label' column")

    # Split into chunks based on changes in label value
    chunks = []
    current_chunk = [df.iloc[0]]

    for i in range(1, len(df)):
        if df['Label'].iloc[i] == df['Label'].iloc[i - 1]:
            current_chunk.append(df.iloc[i])
        else:
            chunks.append(pd.DataFrame(current_chunk))
            current_chunk = [df.iloc[i]]
            # display(pd.DataFrame(current_chunk))
    chunks.append(pd.DataFrame(current_chunk))  # Append the last chunk

    # print("Total chunks:", len(chunks))

    samples, labels = [], []
    output_dir = "chunk_output"
    
    # Iterate through each chunk and create samples
    for chunk_idx, chunk in enumerate(chunks):
        # print label of chunks
        # print("chunk label", chunk['Label'].iloc[0])
        # add to COUNT
        # COUNT[int(chunk['Label'].iloc[0])] += 1
        if len(chunk) >= frame_size:
            for start in range(0, len(chunk) - frame_size + 1, step):
                sample = chunk[start:start + frame_size]
                label = sample['Label'].iloc[0]
                # display(sample)
                sample_filename = os.path.join(output_dir, f"label_{label}_chunk_{chunk_idx}_sample_{start}.csv")
                sample.to_csv(sample_filename, index=False)
                # break
                samples.append(sample.drop(columns=['Label']))
                labels.append(sample['Label'].iloc[0])  # Use the first label in the sample

    # print("Generated samples:", len(samples), "Generated labels:", len(labels))
    # print("labels", labels)
    return samples, labels


class ActivityDataset(Dataset):
    def __init__(self, root, scaler=None):
        self.root = root
        self.scaler = scaler
        self.data = []
        self.label = []
        
        for file in tqdm(os.listdir(self.root)):
            if file.endswith(".csv"):  # Ensure only CSV files are processed
                # print(f"Processing {file}")
                df = pd.read_csv(os.path.join(self.root, file))
                try:
                    samples, labels = split_to_chunk(df)
                    self.data.extend(samples)
                    # self.label.extend(labels)
                except ValueError as e:
                    print(file)
                labels = [extract_and_categorize_sharpness(file)] * len(samples)
                self.label.extend(labels)
        
        # Normalize data if a scaler is provided
        if self.scaler is not None:
            self.data = [self.scaler.transform(sample) for sample in self.data]
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        # Convert DataFrame to NumPy array and then to a PyTorch tensor
        features = torch.tensor(self.data[idx].to_numpy(), dtype=torch.float32)
        label = torch.tensor(self.label[idx], dtype=torch.long)
        return features, label
