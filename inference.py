import tensorflow_text
import tensorflow as tf
import tensorflow_hub as hub
from nltk.corpus import stopwords
import nltk
import re
import numpy as np


class Model:
    def __init__(self):
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
        self.model = None

    def _remove_stopwords(self, text):
        tokens = text.split()
        filtered_tokens = [word for word in tokens if word.lower() not in self.stop_words]
        return ' '.join(filtered_tokens)

    def _preprocess_pipeline(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        text = self._remove_stopwords(text)
        return text

    def predict(self, text):
        if self.model is None:
            raise ValueError("Model not instantiated.")

        preprocessed_text = self._preprocess_pipeline(text)
        prediction = int(np.argmax(self.model.predict([preprocessed_text]), axis=1))
        prediction_map = ["left", "center", "right"]

        return prediction_map[prediction]

    def load(self):
        self.model = tf.keras.models.load_model(
            "LSTM_MODEL.h5",
            custom_objects={'KerasLayer': hub.KerasLayer},
            compile=False
        )
        self.model.load_weights("model_weights.h5")
