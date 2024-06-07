import numpy as np
from annoy import AnnoyIndex
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor


import csv
import faiss
from sentence_transformers import SentenceTransformer
from pyarabic import araby
from difflib import SequenceMatcher
import os
import time

class SimilarityFinder:
    def __init__(self, database_file, index_file='annoy_index.bin', embeddings_file='embeddings.npy', num_trees=100, batch_size=1000):
        self.database_file = database_file
        self.index_file = index_file
        self.embeddings_file = embeddings_file
        self.model = SentenceTransformer("CAMeL-Lab/bert-base-arabic-camelbert-msa")
        self.batch_size = batch_size
        self.num_trees = num_trees

        self._load_data()
        self._load_or_create_embeddings()
        self._initialize_annoy_index()
        print("Loaded Annoy Index")

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

    def _initialize_annoy_index(self):
        t = time.time()
        num_features = self.embeddings.shape[1]
        self.annoy_index = AnnoyIndex(num_features, 'angular')
        try:
            self.annoy_index.load(self.index_file)
        except Exception:
            self._build_annoy_index()
        print("Time for index construction: ", time.time() - t)


    def _build_annoy_index(self):
        with ThreadPoolExecutor() as executor:
            for i in range(0, len(self.embeddings), self.batch_size):
                batch = self.embeddings[i:i+self.batch_size]
                for j, vector in enumerate(batch):
                    self.annoy_index.add_item(i+j, vector)
        self.annoy_index.build(self.num_trees)
        self.annoy_index.save(self.index_file)

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
        query_embedding = self.model.encode([query], convert_to_tensor=True).cpu().numpy()[0]
        indices = self.annoy_index.get_nns_by_vector(query_embedding, 10)
        return [self.db[idx] for idx in indices]

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

    database_file = 'company_names.txt'
    query_results_file = '50k sentences.txt'
    csv_file_path = 'search_results.csv'

    finder = SimilarityFinder(database_file)
    queries_and_results = load_queries_and_results(query_results_file)
    total_matches = 0

    # with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["query", "search_result", "true_result", "match?"])
    total_time = 0
    for q, true_result in queries_and_results:
        t = time.time()
        search_result = finder.find_most_similar(q)
        match = araby.vocalizedlike(true_result.strip(), search_result.strip())
        # append_to_csv(csv_file_path, q, search_result, true_result, match)
        total_matches += 1 if match else 0
        total_time += time.time() - t 


    accuracy = total_matches / len(queries_and_results)
    avg_time = total_time / len(queries_and_results)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Query Time: {avg_time:.2f}")


def load_queries_and_results(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.read().splitlines()
    return [(lines[i], lines[i + 1]) for i in range(0, 5000, 2)]


if __name__ == "__main__":
    main()
