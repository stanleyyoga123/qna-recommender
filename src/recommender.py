import pickle
from tqdm import tqdm
import numpy as np

from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity

from src.model.convolution import ConvModel


class Recommender:
    def __init__(self, data, preprocessor_path, model_path):
        self.data = data
        self.preprocessor = pickle.load(open(preprocessor_path, 'rb'))
        self.model = ConvModel(len(self.preprocessor.tokenizer.word_index) + 1,
                               self.preprocessor.maxlen,
                               self.preprocessor.n_class)

        self.model.build((None, self.preprocessor.maxlen))
        self.model.load_weights(model_path)

    def __transform_data(self):
        self.x, self.label = self.preprocessor.transform(self.data, meta=True)

    def __featuring_data(self):
        self.features = []
        self.dict_id_feat = {}

        for instance, id in tqdm(zip(self.x, self.data['q_id'].values), total=len(self.x)):
            instance = instance.reshape(1, -1)
            feature = self.model(instance, feature_only=True)[0]
            self.features.append(feature)
            self.dict_id_feat[id] = feature

    def get_recommendations(self, id):
        recommendations = []
        similarity = cosine_similarity(
            [self.dict_id_feat[id]], self.features)[0]
        sorted_val = np.argsort(similarity)[-5:]
        for el in sorted_val:
            recommendations.append(
                self.data.loc[el, ['eng', 'class', 'chapter']])

        return recommendations

    def recommend(self):
        self.__transform_data()
        self.__featuring_data()
        while True:
            in_str = input("Put Id: ")
            try:
                recommendations = self.get_recommendations(in_str)
                for el in recommendations:
                    print(f'Class {el["class"]}')
                    print(f'Chapter {el["chapter"]}')
                    print(f'{el["eng"]}')
                    print()
            except Exception as e:
                print(e)
                print('No ID Available')
