# NextMoviePlease – Movie Recommender System

NextMoviePlease is a hybrid movie recommendation system that combines **Item-Based Collaborative Filtering (CF)** and **Singular Value Decomposition (SVD)** to generate personalized movie recommendations for users. The system leverages user ratings from the MovieLens dataset to suggest top movies that a user may like.

The hybrid approach balances item similarity and latent factor modeling for more accurate recommendations.

---

## Features
- **Hybrid Recommendation**: Combines item-based CF and SVD predictions.
- **Interactive Web Interface**: Built with Streamlit for easy usage.
- **Top-N Recommendations**: Provides the top 5 or top N recommended movies for a given user.
- **Evaluation Metrics**: Supports Precision@k, Recall@k, and NDCG@k to evaluate model performance.

---

## Dataset
The project uses the **MovieLens Latest Small dataset**:
- `ratings.csv`: User ratings of movies.
- `movies.csv`: Movie metadata including movie titles.

Statistics:
- Total ratings: ~100,000
- Total users: ~700
- Total movies: ~9,000
- Rating scale: 0.5 – 5.0

---

## Usage

### Dependencies

```bash
pip install -r requirement.txt
```

### Training & Model Building

1. Run the Jupyter Notebook or Python script to:

   * Load and explore the dataset.
   * Split ratings into training and testing sets (80/20 split).
   * Build the user-item matrix.
   * Compute item-item similarity and perform SVD.
   * Generate hybrid recommendations.
   * Save the trained model and movies dataframe to `models/`.

### Running the Streamlit App

1. Start the app:

```bash
streamlit run app.py
```

2. Enter a **User ID** to get top 5 movie recommendations.

---

## Model Files

* `models/best_model.pkl`: Hybrid model including CF, SVD, user/item mappings, and trained factors.
* `models/movies.pkl`: Movies metadata.

---

## Recommendation Functions

### Item-Based Collaborative Filtering

* Computes weighted average of ratings from similar items.

### SVD-Based Recommendation

* Uses latent factor decomposition to predict missing ratings.

### Hybrid Recommendation

* Combines normalized CF and SVD predictions using a weighting factor `alpha`.

---

## Evaluation

The system is evaluated on **Precision@k**, **Recall@k**, and **NDCG@k** metrics using a sample of users from the test set.
Default `k` values used: 5 and 10.

---

## Notes

* The system assumes the user exists in the training data; new users with no ratings will not receive recommendations.
* Hybrid weighting `alpha` can be tuned to favor CF or SVD predictions.

---



