from ewc_app.service.keras_service.models.keras_chatbot_model import KerasChatBotModel
from ewc_app import embedding_chatbot
import numpy as np
from ewc_app.utils.vocab import VocabLoader
from ewc_app.const.constants import *
from keras.preprocessing.sequence import pad_sequences


class QuestionService:

    def __init__(self, data):
        self.question = data["title"]
        self.intent = data["description"]

    def process_question(self, pretrained_model=False):

        km = KerasChatBotModel(embedding_chatbot, pretrained_model)
        prepared_question = self.prepare_question(self.question)
        intent = km.predict(prepared_question)[0]
        readable_intent = self.create_readable_labels(intent, INTENTS)
        return readable_intent

    def prepare_question(self, question):
        prepared_question = self.create_features([question])
        return prepared_question

    def create_features(self, x_data):
        vocab = VocabLoader.load()
        sequences = [vocab.encode_sentence(sentence) for sentence in x_data]
        data = pad_sequences(
            sequences, maxlen=111, padding='post', truncating='post')
        return data

    def create_readable_labels(self, y_data_ohe, y_data):
        unique_labels = np.unique(y_data)
        unique_index = np.argmax(y_data_ohe)
        return unique_labels[unique_index]



