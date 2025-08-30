import os
import pickle
import gzip
import uvicorn
import numpy as np
import pandas as pd # Ensure pandas is imported if pkl files use it
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Initialize the FastAPI app
app = FastAPI()

# --- CORS Middleware ---
# This is now CRITICAL because your frontend is on a different domain (Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://frontend-zeta-flax-60.vercel.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Loading ---
try:
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
    print("Data loaded successfully.")

except FileNotFoundError as e:
    print(f"Error loading data files: {e}")
    pt, similarity_score, top_book_info, full_book_info = None, None, {}, {}


# --- API Endpoints ---

@app.get('/top-books')
def get_top_books():
    if not top_book_info:
        return {"error": "Book data not loaded."}
        
    books = []
    for title, data in top_book_info.items():
        books.append({
            "title": title,
            "author": data.get("author", "Unknown Author"),
            "image": data.get("image", "")
        })
    return {"books": books}


@app.get('/recommend')
def recommend(book: str = ''):
    if not all([pt is not None, similarity_score is not None, full_book_info]):
         return {"error": "Recommendation engine not ready. Data not loaded."}

    q = book.strip().lower()
    matches = [t for t in pt.index if q in t.lower()]
    if not matches:
        return {"recommended": []}

    try:
        idx = np.where(pt.index == matches[0])[0][0]
        sims = sorted(list(enumerate(similarity_score[idx])), key=lambda x: x[1], reverse=True)[1:6]

        recommendations = []
        for i in sims:
            title = pt.index[i[0]].strip()
            info = full_book_info.get(title, {})
            recommendations.append({
                "title": title,
                "author": info.get("author", "Unknown Author"),
                "image": info.get("image", "")
            })
        
        return {"recommended": recommendations}
    except IndexError:
        return {"recommended": []}


# --- Running the App ---
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)

