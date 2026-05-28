import sys
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Connect paths to find data_processing module safely
sys.path.append(os.path.dirname(__file__))
from data_processing import generate_mock_data, preprocess_data

def train_pipeline():
    print("Generating training logs...")
    raw_df = generate_mock_data()
    
    X, y = preprocess_data(raw_df, is_training=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test) * 100
    print(f"Model validation accuracy: {accuracy:.2f}%")
    
    # Save the model out to the main parent folder
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(parent_dir, "model.pkl")
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved deployment artifact model safely to: {model_path}")

if __name__ == "__main__":
    train_pipeline()
