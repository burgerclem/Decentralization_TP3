from flask import Flask, request, jsonify
import requests

# URLs des APIs de chaque modèle (remplace avec tes vraies URLs ngrok)
API_ENDPOINTS = {
    "KNN": "https://knn123.ngrok.io/predict",
    "RandomForest": "https://rf456.ngrok.io/predict",
    "LogisticRegression": "https://lr789.ngrok.io/predict"
}

# Poids fixes attribués à chaque modèle
MODEL_WEIGHTS = {
    "KNN": 0.4,
    "RandomForest": 0.35,
    "LogisticRegression": 0.25
}

app = Flask(__name__)

@app.route('/consensus', methods=['GET'])
def consensus():
    try:
        # Récupérer les paramètres
        params = {
            "sepal_length": request.args.get("sepal_length"),
            "sepal_width": request.args.get("sepal_width"),
            "petal_length": request.args.get("petal_length"),
            "petal_width": request.args.get("petal_width")
        }
        
        # Obtenir les prédictions de chaque modèle
        predictions = {}
        for model, url in API_ENDPOINTS.items():
            try:
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    predictions[model] = response.json()["prediction"]
            except Exception as e:
                print(f"Erreur avec {model} ({url}): {e}")
                continue
        
        if not predictions:
            return jsonify({"status": "error", "message": "Aucune prédiction reçue"}), 500

        # Calcul du consensus pondéré
        weighted_votes = {}
        for model, pred in predictions.items():
            weighted_votes[pred] = weighted_votes.get(pred, 0) + MODEL_WEIGHTS[model]
        
        # Sélection de la prédiction ayant le poids total le plus élevé
        consensus_prediction = max(weighted_votes, key=weighted_votes.get)

        return jsonify({
            "status": "success",
            "individual_predictions": predictions,
            "model_weights": MODEL_WEIGHTS,
            "consensus_prediction": consensus_prediction
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Ce serveur tourne sur un port différent
