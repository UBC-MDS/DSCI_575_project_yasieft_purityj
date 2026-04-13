import streamlit as st
import pandas as pd
from io import StringIO
import random
import faiss


# Simulate some example data (Replace with your actual data or API for product retrieval)
def get_results(query, mode):
    # Simulated results, should be replaced with actual search/retrieval logic
    results = [
        {"title": f"Product {i}", "review": f"This is a sample review for product {i}. It is really good!" * random.randint(1, 3), "rating": random.uniform(1, 5), "score": random.uniform(0, 1)}
        for i in range(1, 11)
    ]
    return results[:3]  # Returning top 3 for now

# Create a CSV to store feedback (can be used locally for now)
def store_feedback(feedback_data):
    df = pd.DataFrame(feedback_data)
    df.to_csv("feedback.csv", mode="a", header=False, index=False)

# Title of the app
st.title('Product Search App')

# Search Mode Selector
search_mode = st.radio("Select Search Mode", ["BM25", "Semantic", "Hybrid"])

# Query Input
query = st.text_input("Enter your query:")

# Display results if the query is not empty
if query:
    # Retrieve the results based on the selected mode (replace with actual logic)
    results = get_results(query, search_mode)
    
    st.subheader(f"Top 3 results for '{query}' using {search_mode} search:")

    feedback_data = []  # Store feedback here
    
    # Loop through each result
    for idx, result in enumerate(results):
        with st.expander(f"Result {idx + 1}: {result['title']}"):
            # Display Product Title
            st.write(f"**Product Title:** {result['title']}")
            
            # Truncate the review text to 200 characters
            truncated_review = result["review"][:200] + "..." if len(result["review"]) > 200 else result["review"]
            st.write(f"**Review:** {truncated_review}")
            
            # Rating (show as stars or number)
            rating = result["rating"]
            st.write(f"**Rating:** {'⭐' * int(rating)} ({rating:.1f})")
            
            # Retrieval score
            st.write(f"**Retrieval Score:** {result['score']:.2f}")
            
            # Feedback (👍 / 👎 buttons)
            feedback = st.radio(f"Was this result helpful?", options=["👍", "👎"], key=f"feedback_{idx}")
            if feedback:
                feedback_data.append({
                    "product_title": result["title"],
                    "feedback": feedback,
                    "score": result["score"]
                })
    
    # Save feedback when the user submits
    if feedback_data:
        store_feedback(feedback_data)
        st.success("Feedback saved!")