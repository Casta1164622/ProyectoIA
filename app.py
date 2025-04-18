from flask import Flask, request, render_template, jsonify
from model.naive_bayes import NaiveBayesClassifier
from utils.data_loader import load_bbc_news_dataset
import time
from sklearn.model_selection import train_test_split
from model.naive_bayes import evaluate_model


app = Flask(__name__)
classifier = NaiveBayesClassifier()

# ENTRENAMIENTO ANTES DE INICIAR EL SERVIDOR
# Cargar y dividir datos
print("Cargando dataset BBC News...")
data = load_bbc_news_dataset()
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

evaluation_report = {}

print(f"Total de entrenamiento: {len(train_data)}, prueba: {len(test_data)}")
classifier.train(train_data)
evaluation_report = evaluate_model(classifier, test_data)

@app.route('/report')
def report():
    return jsonify(evaluation_report)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    start = time.time()
    input_text = request.json.get("text", "")
    prediction = classifier.predict(input_text)
    elapsed = round((time.time() - start) * 1000, 2)
    return jsonify({
        "category": prediction,
        "elapsed_ms": elapsed
    })

if __name__ == '__main__':
    app.run(debug=True)
