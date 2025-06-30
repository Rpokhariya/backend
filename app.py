import os
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pickle
import numpy as np
import gzip

app = Flask(__name__, static_folder='../frontend/dist', static_url_path='/')
CORS(app)

# Loading pickled data
with open('pt.pkl', 'rb') as f:
    pt = pickle.load(f)

with open('similarity_score.pkl', 'rb') as f:
    similarity_score = pickle.load(f)

with open('top50_book_info.pkl', 'rb') as f:
    book_info = pickle.load(f)

with gzip.open("book_info.pkl.gz", "rb") as f:
    full_book_info = pickle.load(f)


# Normalizing keys for safe lookup
top_book_info = {k.strip(): v for k, v in book_info.items()}
full_book_info = {k.strip(): v for k, v in full_book_info.items()}

# Serving React frontend
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    file_path = os.path.join(app.static_folder, path)
    if path and os.path.exists(file_path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

# Return top 50 books directly from book_info
@app.route('/top-books')
def get_top_books():
    books = []
    for title, data in top_book_info.items():
        books.append({
            "title": title,
            "author": data.get("author", "Unknown Author"),
            "image": data.get("image", "")
        })
    return jsonify({"books": books})

# Recommend books from similarity matrix, enriched with book_info
@app.route('/recommend')
def recommend():
    q = request.args.get('book', '').strip().lower()

    # Match input to title in pt index
    matches = [t for t in pt.index if q in t.lower()]
    if not matches:
        return jsonify({"recommended": []})

    # Get index of the closest match
    idx = np.where(pt.index == matches[0])[0][0]
    sims = sorted(enumerate(similarity_score[idx]), key=lambda x: x[1], reverse=True)[0:5]

    recommendations = []
    for i, _ in sims:
        title = pt.index[i].strip()
        info = full_book_info.get(title, {})
        recommendations.append({
            "title": title,
            "author": info.get("author", "Unknown Author"),
            "image": info.get("image", "")
        })

    return jsonify({"recommended": recommendations})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
