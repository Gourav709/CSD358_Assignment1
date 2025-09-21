import streamlit as st
from vsm import build_index, process_query, preprocess # Import your VSM functions

# --- Page Configuration ---
st.set_page_config(
    page_title="VSM Search Engine",
    page_icon="🔎",
    layout="centered",
)

# --- Caching the Index ---
# This decorator tells Streamlit to run this function only once and store the result.
# This prevents the slow index-building process from re-running every time the user
# types a new query.
@st.cache_resource
def load_index():
    """Builds and caches the search index."""
    print("Building the search index... (this will run only once)")
    postings, doc_term_norm, soundex_map, docs_tokens, N = build_index("corpus")
    print("Index built successfully.")
    return postings, doc_term_norm, soundex_map, N

# Load the data using the cached function
postings, doc_term_norm, soundex_map, N = load_index()

# --- User Interface Elements ---
st.title("🔎 Vector Space Model Search Engine")
st.write(
    "This app implements a ranked retrieval system using the lnc.ltc weighting scheme. "
    "Enter a query below to search the document corpus."
)

# Create a text input box for the user's query
query = st.text_input(
    "Enter your search query:",
    placeholder="e.g., ambitious student",
    help="Type your query and press Enter."
)

# --- Search and Display Results ---
if query:
    # Process the query using your existing function
    results = process_query(query, postings, doc_term_norm, soundex_map, N)

    st.subheader("Search Results")
    st.write(f"Found **{len(results)}** results for: *'{query}'*")

    if not results:
        st.warning("No matching documents found. Try a different query or check for typos.")
    else:
        # Display each result in a formatted way
        for i, (doc_id, score) in enumerate(results):
            with st.container():
                st.markdown(f"**{i+1}. {doc_id}**")
                st.markdown(f"> Score: `{score:.4f}`")
                st.divider() # Adds a horizontal line between results