import streamlit as st
import joblib
import pickle
import numpy as np
import pandas as pd

# --- Load model and movies ---
@st.cache_resource
def load_model():
    # Load hybrid recommender dictionary
    model = joblib.load("models/best_model.pkl")
    # Load movies dataframe
    with open("models/movies.pkl", "rb") as f:
        movies = pickle.load(f)
    return model, movies

model, movies = load_model()

# --- Streamlit layout ---
st.set_page_config(page_title="NextMovie 🎬", page_icon="🎥", layout="centered")
st.subheader("🎬 NextMoviePlease – Movie Recommender")
st.write("Get top movie recommendations based on your User ID!")

# --- Recommendation function ---
def recommend_for_user(user_id, N=5, alpha=0.3):
    user_id_map = model['user_id_map']
    idx_to_movie = model['idx_to_movie']
    user_item_matrix = model['user_item_matrix']
    item_similarity = model['item_similarity']
    user_factors = model['user_factors']
    item_factors = model['item_factors']

    if user_id not in user_id_map:
        return []

    user_idx = user_id_map[user_id]

    # Hybrid recommendation
    user_ratings = user_item_matrix[user_idx].toarray().flatten()
    rated_items = np.where(user_ratings > 0)[0]

    if len(rated_items) == 0:
        return []

    # Item-based CF
    cf_predictions = np.zeros(user_item_matrix.shape[1])
    for item_idx in range(user_item_matrix.shape[1]):
        if user_ratings[item_idx] > 0:
            continue
        similarities = item_similarity[item_idx, rated_items].toarray().flatten()
        ratings = user_ratings[rated_items]
        sim_sum = np.sum(np.abs(similarities))
        if sim_sum > 0:
            cf_predictions[item_idx] = np.dot(similarities, ratings) / sim_sum

    # SVD predictions
    svd_predictions = np.dot(user_factors[user_idx], item_factors.T)

    # Normalize predictions
    if cf_predictions.max() > 0:
        cf_predictions = cf_predictions / 5.0
    if svd_predictions.max() > svd_predictions.min():
        svd_predictions = (svd_predictions - svd_predictions.min()) / (svd_predictions.max() - svd_predictions.min())

    hybrid_predictions = alpha * cf_predictions + (1 - alpha) * svd_predictions
    hybrid_predictions[rated_items] = -np.inf

    # Get top N recommendations
    top_indices = np.argsort(hybrid_predictions)[::-1][:N]
    recommendations = [(idx_to_movie[idx], hybrid_predictions[idx]) for idx in top_indices if hybrid_predictions[idx] > -np.inf]

    return recommendations

# --- Streamlit input ---
user_id = st.number_input("Enter your User ID:", min_value=1, step=1)

if st.button("Get Recommendations"):
    recs = recommend_for_user(user_id, N=5, alpha=model.get('alpha', 0.3))
    
    if recs:
        result_data = []
        for movie_id, score in recs:
            title = movies[movies['movieId'] == movie_id]['title'].values[0]
            estimated_rating = min(5.0, max(0.0, score * 5.0))  # convert back to 0-5 scale
            result_data.append({
                'Movie ID': movie_id,
                'Title': title,
                'Predicted Rating': round(estimated_rating, 2)
            })
        st.subheader("Top 5 Recommendations:")
        st.table(pd.DataFrame(result_data))
    else:
        st.warning("No recommendations found for this user.")
