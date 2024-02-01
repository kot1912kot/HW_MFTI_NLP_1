import numpy as np
from flask import Flask, render_template, request, jsonify
import joblib
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Загрузка модели TF-IDF векторизатора и матрицы векторов
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
tfidf_matrix = joblib.load('tfidf_matrix.joblib')
all_dialogues = joblib.load('all_dialogues.joblib')

# Функция для нахождения наиболее подходящего ответа
def find_most_suitable_response(input_message):
    input_message_tfidf = tfidf_vectorizer.transform([input_message])
    cosine_similarities = cosine_similarity(input_message_tfidf, tfidf_matrix)
    most_similar_dialogue_idx = np.argmax(cosine_similarities)
    most_suitable_response = all_dialogues[most_similar_dialogue_idx]
    return most_suitable_response

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form.get('user_input', '')

    # Функция для нахождения наиболее подходящего ответа (уже определена ранее)
    response = find_most_suitable_response(user_input)

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)