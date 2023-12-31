from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

features = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


def initialize_models():
    # Load the heart dataset
    data = pd.read_csv("heart.csv")
    X = data[features]
    # Normalisasi pada subset fitur yang dipilih
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Inisialisasi pusat klaster menggunakan data ke-166 dan ke-302 untuk K-Means
    acuan_1 = X.iloc[165]
    acuan_2 = X.iloc[302]
    centroids_kmeans = [acuan_1.values, acuan_2.values]
    kmeans = KMeans(n_clusters=2, init=centroids_kmeans, n_init=1)
    kmeans.fit(X_scaled)

    # Inisialisasi pusat klaster menggunakan data ke-166 dan ke-302 untuk GMM
    acuan_1_gmm = X.iloc[165]
    acuan_2_gmm = X.iloc[302]
    centers_gmm = pd.concat([acuan_1_gmm, acuan_2_gmm], axis=1).T
    gmm = GaussianMixture(n_components=2, means_init=centers_gmm, n_init=1)
    gmm.fit(X_scaled)

    # Hierarchical Clustering dengan metode linkage 'ward'
    hierarchical = AgglomerativeClustering(n_clusters=2, linkage="ward")
    hierarchical.fit(X_scaled)

    return kmeans, gmm, hierarchical


def get_cluster_label(model, new_data, kmeans, gmm, hierarchical):
    if model == "kmeans":
        return kmeans.predict(new_data)
    elif model == "gmm":
        return gmm.predict(new_data)
    elif model == "hierarchical":
        if len(new_data) < 2:
            return np.array([0])  # Assign to a single cluster
        else:
            return hierarchical.fit_predict(new_data)


def map_to_category(label):
    if label == 0:
        return "Resiko Rendah"
    elif label == 1:
        return "Resiko Tinggi"
    else:
        return "Unknown"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        kmeans, gmm, hierarchical = initialize_models()

        input_data = [float(request.form.get(feature, 0)) for feature in features]
        new_data = pd.DataFrame(np.array([input_data]), columns=features)

        # Predict using K-Means
        kmeans_label = get_cluster_label("kmeans", new_data, kmeans, gmm, hierarchical)
        # Predict using GMM
        gmm_label = get_cluster_label("gmm", new_data, kmeans, gmm, hierarchical)

        hierarchical_label = get_cluster_label(
            "hierarchical", np.array(new_data).reshape(1, -1), kmeans, gmm, hierarchical
        )

        kmeans_category = map_to_category(kmeans_label[0])
        gmm_category = map_to_category(gmm_label[0])
        hierarchical_category = map_to_category(hierarchical_label[0])

        return render_template(
            "result.html",
            age=request.form.get("age", ""),
            sex=request.form.get("sex", ""),
            cp=request.form.get("cp", ""),
            trestbps=request.form.get("trestbps", ""),
            chol=request.form.get("chol", ""),
            fbs=request.form.get("fbs", ""),
            restecg=request.form.get("restecg", ""),
            thalach=request.form.get("thalach", ""),
            exang=request.form.get("exang", ""),
            oldpeak=request.form.get("oldpeak", ""),
            slope=request.form.get("slope", ""),
            ca=request.form.get("ca", ""),
            thal=request.form.get("thal", ""),
            risk_category=request.form.get("risk_category", ""),
            kmeans_label=kmeans_label[0],
            kmeans_category=kmeans_category,
            gmm_label=gmm_label[0],
            gmm_category=gmm_category,
            hierarchical_label=hierarchical_label[0],
            hierarchical_category=hierarchical_category,
        )


if __name__ == "__main__":
    app.run(debug=True)
