import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

class TONIoTLoader:
    def __init__(self, data_path=None, synthetic=False):
        """
        Loader for TON_IoT dataset.
        
        Args:
            data_path (str): Path to the CSV file.
            synthetic (bool): If True, generates synthetic data for testing.
        """
        self.data_path = data_path
        self.synthetic = synthetic
        self.label_encoders = {}
        self.scaler = MinMaxScaler()
        self.X_train, self.X_val, self.X_test = None, None, None
        self.y_train, self.y_val, self.y_test = None, None, None
        
        # Core features typically found in TON_IoT
        self.cat_cols = ['proto', 'service', 'conn_state']
        self.num_cols = ['duration', 'src_bytes', 'dst_bytes', 'src_pkts', 'dst_pkts', 
                         'missed_bytes', 'src_ip_bytes', 'dst_ip_bytes']
        self.label_col = 'type' # Multi-class label
        
    def load_data(self):
        if self.synthetic:
            print("Generating SYNTHETIC TON_IoT data for development...")
            df = self._generate_synthetic_data()
        elif self.data_path and os.path.exists(self.data_path):
            print(f"Loading real data from {self.data_path}...")
            df = pd.read_csv(self.data_path)
        else:
            raise FileNotFoundError(f"Data path {self.data_path} not found and synthetic=False.")

        # Basic cleanup
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        # Separate features and target
        # Assuming last column or 'type' is target.
        if self.label_col not in df.columns:
            # Fallback if 'type' isn't there, maybe 'label' (binary) or 'attack_cat'
            self.label_col = df.columns[-1]
            
        return df

    def preprocess(self, df):
        """
        Encodes categorical features and scales numerical ones.
        """
        # Encode Categorical
        for col in self.cat_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = df[col].astype(str)
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
        
        # Encode Label
        le_target = LabelEncoder()
        df[self.label_col] = le_target.fit_transform(df[self.label_col])
        self.label_encoders['target'] = le_target
        
        # Scale Numerical
        valid_num_cols = [c for c in self.num_cols if c in df.columns]
        df[valid_num_cols] = self.scaler.fit_transform(df[valid_num_cols])
        
        return df, valid_num_cols

    def get_splits(self):
        df = self.load_data()
        df, valid_cols = self.preprocess(df)
        
        X = df[valid_cols + [c for c in self.cat_cols if c in df.columns]].values
        y = df[self.label_col].values
        
        # Stratified Split 70/15/15
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
        
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def _generate_synthetic_data(self, n_samples=5000):
        """Generates mock data resembling TON_IoT structure."""
        data = {
            'src_bytes': np.random.randint(0, 10000, n_samples),
            'dst_bytes': np.random.randint(0, 10000, n_samples),
            'duration': np.random.exponential(1.0, n_samples),
            'src_pkts': np.random.randint(0, 100, n_samples),
            'dst_pkts': np.random.randint(0, 100, n_samples),
            'missed_bytes': np.random.randint(0, 10, n_samples),
            'src_ip_bytes': np.random.randint(0, 20000, n_samples),
            'dst_ip_bytes': np.random.randint(0, 20000, n_samples),
            'proto': np.random.choice(['tcp', 'udp', 'icmp'], n_samples),
            'service': np.random.choice(['http', 'ssh', 'ftp', 'dns', '-'], n_samples),
            'conn_state': np.random.choice(['SF', 'S0', 'REJ'], n_samples),
            'type': np.random.choice(['normal', 'ddos', 'dos', 'injection', 'xss', 'scanning', 'password', 'ransomware', 'backdoor', 'mitm'], n_samples, p=[0.5, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.025, 0.025])
        }
        return pd.DataFrame(data)

if __name__ == "__main__":
    # Quick Test
    loader = TONIoTLoader(synthetic=True)
    (X_tr, y_tr), _, _ = loader.get_splits()
    print(f"Data Loaded: Train Shape {X_tr.shape}, Classes {np.unique(y_tr)}")
