from typing import List
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from src.dv.models.entities import Chunk, ValuationResult

class ProxyFilter:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = LinearRegression()

    def train(self, chunks: List[Chunk], results: List[ValuationResult], threshold: float = 0.0):
        """Trains a proxy filter to predict chunk value score."""
        # Map results to chunks
        chunk_map = {c.id: c.text for c in chunks}
        
        X_texts = []
        y = []
        
        for res in results:
            if res.chunk_id in chunk_map:
                X_texts.append(chunk_map[res.chunk_id])
                y.append(res.score)
        
        if not X_texts:
            return
            
        X = self.vectorizer.fit_transform(X_texts)
        self.model.fit(X, y)

    def predict_value(self, text: str) -> float:
        """Predicts the valuation score of a chunk."""
        X = self.vectorizer.transform([text])
        return float(self.model.predict(X)[0])

