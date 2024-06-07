## Retrieval System
The retrieval system in this project uses BERT text embeddings and the Faiss library for efficient similarity searches. This section explains the `SimilarityFinder` class, which is responsible for indexing and querying the company names in the database.

### SimilarityFinder Class

The `SimilarityFinder` class leverages the `SentenceTransformer` model to generate embeddings for the company names and Faiss for indexing these embeddings.

### How it Works

1. **Initialization**:
   - `database_file`: Path to the file containing company names.
   - `index_file`: Path to the Faiss index file.
   - `embeddings_file`: Path to the file containing the precomputed embeddings.

2. **Loading Data**:
   - The `_load_data` method reads the company names from the `database_file`.

3. **Loading or Creating Embeddings**:
   - The `_load_or_create_embeddings` method checks if the embeddings file exists. If it does, it loads the embeddings; otherwise, it creates them using the `SentenceTransformer` model.

4. **Initializing Faiss Index**:
   - The `_initialize_faiss_index` method initializes the Faiss index. If the index file exists, it loads the index; otherwise, it builds a new index.

5. **Building Faiss Index**:
   - The `_build_faiss_index` method adds the embeddings to the Faiss index and writes the index to a file.

6. **Finding Nearest Neighbors**:
   - The `find_nearest_neighbors` method encodes the query into an embedding and searches for the nearest neighbors in the Faiss index.

7. **Finding the Best Match**:
   - The `find_match` method compares the query with the nearest neighbors to find the most similar entry.

### Usage

To use the `SimilarityFinder` class, you need to instantiate it with the appropriate file paths and call its methods as needed:

```python
database_file = "140online/company_names.txt"  # Change this to your file path
similarity_finder = SimilarityFinder(database_file)

query = "Example Company Name"
nearest_neighbors = similarity_finder.find_nearest_neighbors(query, top_k=10)
best_match = similarity_finder.find_match(query, nearest_neighbors)
```

This setup allows you to efficiently search for the most similar company names based on text queries using the Faiss index and BERT embeddings.