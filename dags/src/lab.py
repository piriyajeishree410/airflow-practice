import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import pickle
import os
import base64

BASE_DIR = "/opt/airflow"
DATA_DIR = os.path.join(BASE_DIR, "dags/data")
MODEL_DIR = os.path.join(BASE_DIR, "working_data/model")

def load_data():
    """
    Loads data from CSV, serializes it, and returns base64 string for XCom.
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "file.csv"))
    serialized = pickle.dumps(df)
    return base64.b64encode(serialized).decode("ascii")


def data_preprocessing(data_b64: str):
    """
    Deserializes data, applies Standard scaling, and returns serialized result.
    """
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    df = df.dropna()
    clustering_data = df[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]

    scaler = StandardScaler()
    clustering_data_scaled = scaler.fit_transform(clustering_data)


    serialized = pickle.dumps(clustering_data_scaled)
    return base64.b64encode(serialized).decode("ascii")


def build_save_model(data_b64: str, filename: str):
    """
    Trains KMeans model and saves it. Returns SSE list.
    """
    data_bytes = base64.b64decode(data_b64)
    data = pickle.loads(data_bytes)

    sse = []
    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42,
    }

    # MODIFICATION: reduced cluster range (1–15 instead of 1–50)
    for k in range(1, 16):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, filename), "wb") as f:
        pickle.dump(kmeans, f)

    return sse


def load_model_elbow(filename: str, sse: list):
    """
    Loads model, applies elbow method, and predicts test data.
    """
    model_path = os.path.join(MODEL_DIR, filename)
    with open(model_path, "rb") as f:
        model = pickle.load(f)


    kl = KneeLocator(range(1, 16), sse, curve="convex", direction="decreasing")
    print(f"Optimal clusters: {kl.elbow}")

    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    prediction = model.predict(test_df)[0]

    return int(prediction)
