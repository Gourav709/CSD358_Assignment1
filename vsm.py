#!/usr/bin/env python3
"""
vsm.py
Simple Vector Space Model implementing lnc.ltc weighting + Soundex fallback.
Usage:
  1) Put your corpus .txt files in a folder called 'corpus' (same folder as this script).
  2) Run: python vsm.py
  3) Enter queries when prompted, or run with a query via: python vsm.py "your query here"
"""

# Import necessary libraries
import os       # For interacting with the operating system (e.g., reading files)
import math     # For mathematical operations like log and sqrt
import re       # For regular expressions to clean text
import sys      # To access command-line arguments
from collections import defaultdict, Counter # Efficient data structures

# -------------------------
# Text Preprocessing
# -------------------------
def preprocess(text):
    """
    Lowercase, remove non-alphanumeric characters (keep digits/letters),
    and split on whitespace. Returns a list of tokens.
    """
    # Convert the entire text to lowercase to ensure case-insensitive matching
    text = text.lower()
    # Use a regular expression to replace any character that is not a letter, digit, or whitespace with a space
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Split the cleaned text into a list of words (tokens) based on whitespace
    tokens = text.split()
    return tokens

# -------------------------
# Soundex Algorithm
# -------------------------
def soundex(name):
    """
    Return a 4-character Soundex code for 'name' (simple implementation).
    Soundex is a phonetic algorithm for indexing names by sound, as pronounced in English.
    """
    # Convert the name to uppercase for consistent processing
    name = name.upper()
    if not name:
        return "0000" # Return a default code for empty strings

    # The first character of the Soundex code is the first letter of the name
    first = name[0]

    # Define the Soundex mapping for consonants
    mapping = {}
    for ch in "BFPV": mapping[ch] = "1"
    for ch in "CGJKQSXZ": mapping[ch] = "2"
    for ch in "DT": mapping[ch] = "3"
    mapping["L"] = "4"
    for ch in "MN": mapping[ch] = "5"
    mapping["R"] = "6"

    # Process the rest of the name to generate the numeric part of the code
    encoded = []
    # Get the code of the first letter to handle adjacent letters with the same code
    prev = mapping.get(first, "")
    for ch in name[1:]:
        code = mapping.get(ch, "0") # "0" for vowels, H, W, Y
        # Add the code if it's different from the previous one and not a vowel code
        if code != prev and code != "0":
            encoded.append(code)
        prev = code # Update the previous code

    # Combine the first letter with the encoded numbers
    code_str = first + "".join(encoded)
    # Pad with zeros and truncate to ensure a 4-character length
    code_str = (code_str + "000")[:4]
    return code_str

# -------------------------
# Index Construction
# -------------------------
def build_index(corpus_folder="corpus"):
    """
    Builds the core data structures for the search engine:
      - postings: An inverted index mapping each term to a list of (docID, term_frequency).
      - doc_term_norm: A dictionary mapping each docID to its normalized term weight vector.
      - soundex_map: Maps a Soundex code to the set of terms that share that code.
      - docs_tokens: A dictionary mapping each docID to its list of preprocessed tokens.
      - N: The total number of documents in the corpus.
    """
    # defaultdict(list) creates an inverted index where each term maps to a list of postings
    postings = defaultdict(list)
    # Stores the tokenized content of each document
    docs_tokens = {}

    # Find all .txt files in the specified corpus folder
    filenames = [f for f in os.listdir(corpus_folder) if f.endswith(".txt")]
    filenames.sort()  # Sort for a consistent, deterministic order
    for fname in filenames:
        path = os.path.join(corpus_folder, fname)
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            text = fh.read()
        tokens = preprocess(text)
        docs_tokens[fname] = tokens

    # N is the total number of documents
    N = len(docs_tokens)

    # This section calculates log-frequency weighted term weights for documents (lnc scheme)
    doc_raw_weights = {} # Temporarily stores un-normalized weights
    for docID, tokens in docs_tokens.items():
        freqs = Counter(tokens) # Count term frequencies in the current document
        doc_weights = {}
        for term, tf in freqs.items():
            # Calculate log-frequency weight: 1 + log10(tf)
            if tf > 0:
                wt = 1.0 + math.log10(tf)
            else:
                wt = 0.0 # Weight is 0 if term frequency is 0
            doc_weights[term] = wt
            # Add the term and its frequency to the postings list for that term
            postings[term].append((docID, tf))
        doc_raw_weights[docID] = doc_weights

    # Normalize the document term weights using cosine normalization
    doc_term_norm = {}
    for docID, weight_map in doc_raw_weights.items():
        # Calculate the length (Euclidean norm) of the document's weight vector
        length = math.sqrt(sum(w*w for w in weight_map.values())) if weight_map else 1.0
        if length == 0:
            length = 1.0 # Avoid division by zero for empty documents
        # Normalize each term's weight by dividing by the vector length
        norm_map = {t: (w / length) for t, w in weight_map.items()}
        doc_term_norm[docID] = norm_map

    # Build the Soundex map for all unique terms found in the corpus
    soundex_map = defaultdict(set)
    for term in postings.keys():
        code = soundex(term)
        soundex_map[code].add(term)

    return postings, doc_term_norm, soundex_map, docs_tokens, N


def process_query(query, postings, doc_term_norm, soundex_map, N, top_k=10):
    """
    Process a single free-text query and return up to top_k results.
    Implements ltc weighting for the query and uses precomputed lnc document weights.
    Includes a Soundex fallback for terms not found in the dictionary.
    """
    # Preprocess the query in the same way as the documents
    qtokens = preprocess(query)
    if not qtokens:
        return [] # Return empty list if query is empty after preprocessing

    q_freq = Counter(qtokens) # Count term frequencies in the query
    q_weights = {}  # Stores raw query term weights before normalization

    # Calculate log-frequency, inverse-document-frequency weights for the query (ltc scheme)
    for token, qtf in q_freq.items():
        if token in postings:
            # Term is in the dictionary
            df = len(postings[token]) # Document frequency
            if df > 0:
                idf = math.log10(N / df) if N > df else 0.0 # Inverse document frequency
            else:
                idf = 0.0
            # Calculate query term weight: (1 + log(tf)) * idf
            q_weights[token] = (1.0 + math.log10(qtf)) * idf
        else:
            # Soundex fallback: If term is not in the dictionary, find phonetically similar terms
            code = soundex(token)
            candidates = soundex_map.get(code, set()) # Get all terms with the same Soundex code
            for cand in candidates:
                # Calculate weight for each candidate term and add it to the query vector
                df = len(postings[cand])
                idf = math.log10(N / df) if df > 0 and N > df else 0.0
                # Aggregate weights in case multiple query terms map to the same candidate
                q_weights[cand] = q_weights.get(cand, 0.0) + (1.0 + math.log10(qtf)) * idf

    # Normalize the query vector using cosine normalization
    q_len = math.sqrt(sum(w*w for w in q_weights.values()))
    if q_len == 0:
        q_len = 1.0 # Avoid division by zero
    for t in list(q_weights.keys()):
        q_weights[t] = q_weights[t] / q_len

    # Calculate cosine similarity scores between the query vector and all document vectors
    scores = defaultdict(float)
    # The score is the dot product of the query and document vectors
    for term, qwt in q_weights.items():
        if term in postings:
            # Retrieve all documents containing the term
            for docID, tf in postings[term]:
                # Get the pre-computed normalized weight of the term in the document
                d_wt = doc_term_norm.get(docID, {}).get(term, 0.0)
                if d_wt:
                    # Accumulate the score for the document
                    scores[docID] += qwt * d_wt

    # Rank documents by their final scores in descending order
    # Tie-break by sorting docID lexicographically (ascending)
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    # Return the top K results
    return ranked[:top_k]

# -------------------------
# Main Application Driver
# -------------------------
def interactive_mode(corpus_folder="corpus"):
    """
    Handles the interactive command-line interface for the search engine.
    """
    print("Building index from corpus folder:", corpus_folder)
    if not os.path.isdir(corpus_folder):
        print("ERROR: corpus folder not found. Please create a folder named 'corpus' and add .txt files.")
        return

    # Build the index and other data structures once at the start
    postings, doc_term_norm, soundex_map, docs_tokens, N = build_index(corpus_folder)
    print(f"Indexed {N} documents. Dictionary size: {len(postings)} terms.")
    print("Enter a query (or type 'exit'):")

    # Start the interactive loop to accept user queries
    while True:
        try:
            query = input(">> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            break

        # Process the query and get ranked results
        results = process_query(query, postings, doc_term_norm, soundex_map, N, top_k=10)
        if not results:
            print("No matching documents found.")
        else:
            print("Top results (doc, score):")
            for doc, sc in results:
                print(f"{doc}\t{sc:.6f}")


def main():
    """
    Main function to run the script.
    It can either run a single query from command-line arguments or start interactive mode.
    """
    corpus_folder = "corpus"

    # Check if a query was passed as a command-line argument
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        postings, doc_term_norm, soundex_map, docs_tokens, N = build_index(corpus_folder)
        results = process_query(query, postings, doc_term_norm, soundex_map, N, top_k=10)
        if not results:
            print("No matching documents found.")
        else:
            for doc, sc in results:
                print(f"{doc}\t{sc:.6f}")
        return

    # If no arguments are provided, start the interactive mode
    interactive_mode(corpus_folder)

# Standard Python entry point
if __name__ == "__main__":
    main()
