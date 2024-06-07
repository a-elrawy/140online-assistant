import numpy as np
import csv
import faiss
from sentence_transformers import SentenceTransformer
from pyarabic import araby
from difflib import SequenceMatcher
import os
import time

class SimilarityFinder:
    def __init__(self, database_file, index_file='index/faiss_index.bin', embeddings_file='index/embeddings.npy'):
        self.database_file = database_file
        self.index_file = index_file
        self.embeddings_file = embeddings_file
        self.model = SentenceTransformer("CAMeL-Lab/bert-base-arabic-camelbert-msa")

        self._load_data()
        self._load_or_create_embeddings()
        self._initialize_faiss_index()
        print("Loaded Faiss Index")

    def _load_data(self):
        with open(self.database_file, 'r', encoding='utf-8') as file:
            self.db = [line.strip() for line in file.readlines()]

    def _load_or_create_embeddings(self):
        if os.path.exists(self.embeddings_file):
            self.embeddings = np.load(self.embeddings_file)
        else:
            self._create_embeddings()

    def _create_embeddings(self):
        self.embeddings = self.model.encode(self.db, convert_to_tensor=True).cpu().numpy()
        np.save(self.embeddings_file, self.embeddings)

    def _initialize_faiss_index(self):
        t = time.time()
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)  # Using L2 distance
        if os.path.exists(self.index_file):
            self.faiss_index = faiss.read_index(self.index_file)
        else:
            self._build_faiss_index()
        print("Time for index construction: ", time.time() - t)
    
    def _build_faiss_index(self):
        self.faiss_index.add(self.embeddings)
        faiss.write_index(self.faiss_index, self.index_file)

    def find_match(self, query, nearest_neighbors):
        max_ratio = 0
        most_similar_entry = None
        for entry in nearest_neighbors:
            ratio = self.similar(query, entry)
            if ratio > max_ratio:
                max_ratio = ratio
                most_similar_entry = entry
        return most_similar_entry

    def find_nearest_neighbors(self, query, top_k=10):
        query_embedding = self.model.encode([query], convert_to_tensor=True).cpu().numpy()
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        return [self.db[idx] for idx in indices[0]]

    def find_most_similar(self, query):
        nearest_neighbors = self.find_nearest_neighbors(query)
        return self.find_match(query, nearest_neighbors)

    @staticmethod
    def similar(a, b):
        return SequenceMatcher(None, a, b).ratio()


def append_to_csv(file_path, query, search_result, true_result, match):
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([query, search_result, true_result, match])


def main():

    database_file = '../140online/company_names.txt'
    query_results_file = 'retrieval_test_dataset.txt'
    csv_file_path = 'search_results.csv'

    finder = SimilarityFinder(database_file)
    queries_and_results = load_queries_and_results(query_results_file)
    total_matches = 0

    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["query", "search_result", "true_result", "match?"])

    total_time = 0
    for q, true_result in queries_and_results:
        t = time.time()
        search_result = finder.find_most_similar(q)
        match = araby.vocalizedlike(true_result.strip(), search_result.strip())
        append_to_csv(csv_file_path, q, search_result, true_result, match)
        total_matches += 1 if match else 0
        total_time += time.time() - t 

    accuracy = total_matches / len(queries_and_results)
    avg_time = total_time / len(queries_and_results)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Query Time: {avg_time:.2f}")



def load_queries_and_results(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.read().splitlines()
    return [(lines[i], lines[i + 1]) for i in range(0, 200, 2)]


if __name__ == "__main__":
    main()
