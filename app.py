import streamlit as st
import pandas as pd
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Load the API key from Streamlit secrets (put this key in streamlit)
api_key = st.secrets["GROQ_API_KEY"]                                

# Initialize the Groq API client with the secret API key
client = Groq(api_key= api_key)

# Load the dataset
df = pd.read_csv('Hydra-Movie-Scrape.csv')

# Ensure there are no NaN values in the 'Summary' column
df['Summary'] = df['Summary'].fillna('')

# Use 'Summary' for embeddings
summaries = df['Summary'].tolist()

# Step 2: Generate embeddings for the data
model = SentenceTransformer('all-MiniLM-L6-v2')
summary_embeddings = model.encode(summaries)

# Step 3: Store embeddings in a vector database
vector_database = np.array(summary_embeddings)

# Streamlit UI
st.title("Movie Query and Recommendation System")

# User input
user_query = st.text_input("Enter a movie description or query:")

if st.button("Find Similar Movie"):
    if user_query:
        # Step 4: Retrieve the most similar movie and its details from the vector database
        def retrieve_similar_movie(query, vector_db, top_k=1):
            query_embedding = model.encode([query])
            similarities = cosine_similarity(query_embedding, vector_db)
            top_indices = np.argsort(similarities[0])[::-1][:top_k]
            return df.iloc[top_indices[0]], top_indices

        retrieved_movie, _ = retrieve_similar_movie(user_query, vector_database)

        # Prepare the message content using the retrieved data
        retrieved_content = f"Context: {retrieved_movie['Summary']}\n"
        answer_title = retrieved_movie['Title']

        # Use the retrieved content as context for the chat completion
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the user's question, but feel free to expand or provide additional information if necessary."},
                {"role": "user", "content": retrieved_content},
                {"role": "user", "content": user_query},
            ],
            model="llama3-8b-8192",
            temperature=0.5,
            max_tokens=1024,
            top_p=1,
            stop=None,
            stream=False,
        )

        # Display the generated response
        model_output = chat_completion.choices[0].message.content
        st.write(f"**Title:** {answer_title}\n")
        st.write(model_output)
    else:
        st.write("Please enter a query to proceed.")
