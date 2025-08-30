import os
import pickle
import gzip
import uvicorn
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Initialize the FastAPI app
app = FastAPI()

# --- CORS Middleware ---
# This allows your frontend to communicate with this backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Data Loading ---
# This part remains the same as your Flask app. It loads the pickled data.
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
    print("Please make sure pt.pkl, similarity_score.pkl, top50_book_info.pkl, and book_info.pkl.gz are in the same directory.")
    pt, similarity_score, top_book_info, full_book_info = None, None, {}, {}


# --- API Endpoints ---

# FastAPI uses decorators to define routes. This is similar to Flask.
# The function returns a dictionary, and FastAPI automatically converts it to JSON.
@app.get('/top-books')
def get_top_books():
    """
    Returns a list of the top 50 books.
    """
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


# Query parameters are defined directly as function arguments with type hints.
# FastAPI handles validation and documentation for you.
@app.get('/recommend')
def recommend(book: str = ''):
    """
    Recommends books based on a given book title.
    Takes a 'book' query parameter. e.g., /recommend?book=1984
    """
    if not all([pt is not None, similarity_score is not None, full_book_info]):
         return {"error": "Recommendation engine not ready. Data not loaded."}

    q = book.strip().lower()

    # Match input to title in pt index
    matches = [t for t in pt.index if q in t.lower()]
    if not matches:
        return {"recommended": []}

    # Get index of the closest match
    try:
        idx = np.where(pt.index == matches[0])[0][0]
        sims = sorted(list(enumerate(similarity_score[idx])), key=lambda x: x[1], reverse=True)[1:6] # Get 5 similar books

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


# --- Static Files Serving ---
# This section serves your React frontend. It should be placed after your API routes.
# It tells FastAPI to serve static files from the '../frontend/dist' directory.
static_folder = "../frontend/dist"

# Mount the static files directory
app.mount("/assets", StaticFiles(directory=os.path.join(static_folder, "assets")), name="assets")

@app.get("/{full_path:path}")
async def serve_react_app(full_path: str):
    """
    Serves the static files for the React app.
    If a file is not found, it serves index.html to allow client-side routing.
    """
    path = os.path.join(static_folder, full_path)
    if os.path.isfile(path):
        return FileResponse(path)
    index_path = os.path.join(static_folder, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": "Frontend not found. Make sure the 'static_folder' path is correct."}


# --- Running the App ---
# This allows you to run the app directly using 'python app.py'
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
