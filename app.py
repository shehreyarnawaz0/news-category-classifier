from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load('trained_models/logistic_regression_model.joblib')
vectorizer = joblib.load('trained_models/vectorizer.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news']
    processed_text = news_text.lower()
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)[0]
    label = 'Real News' if prediction == 1 else 'Fake News'
    return render_template('index.html', prediction=label, text=news_text)

if __name__ == '__main__':
    app.run(debug=True)
