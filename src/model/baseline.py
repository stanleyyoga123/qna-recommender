import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Baseline():

    def __init__(self):
        self.n_taken = 5
        self.threshold = 0.2

        self.tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')

    def train(self, data):
        text = data['eng'].values

        print('[LOG] TFIDF Part')
        tfidf_mat = self.tfidf.fit_transform(text)
        
        print('[LOG] Cosine Similarity Part')
        sim_matrix = cosine_similarity(tfidf_mat, tfidf_mat)
        self.similarity = pd.concat([data, pd.DataFrame(sim_matrix)], axis=1)

    def recommend(self, q_id):
        recommendations = []

        sim = self.similarity[self.similarity['q_id'] == q_id].values[0][4:]
        sorted_val = np.argsort(sim)[-self.n_taken-1:]
        
        for el in sorted_val:
            if sim[el] > self.threshold:
                recommendations.append(self.similarity.loc[el, ['eng', 'class', 'chapter']])
        
        return recommendations