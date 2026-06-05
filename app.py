from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import os

app = Flask(__name__)
CORS(app)

model = pickle.load(open("kmeans_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    data = request.json

    income = float(data['income'])
    score = float(data['score'])

    cluster = model.predict([[income, score]])

    segments = {
        0: "Average Customers",
        1: "High Income - High Spending",
        2: "High Income - Low Spending",
        3: "Low Income - Low Spending",
        4: "Low Income - High Spending"
    }

    cluster_num = int(cluster[0])

    return jsonify({
        "cluster": cluster_num,
        "segment": segments[cluster_num]
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
