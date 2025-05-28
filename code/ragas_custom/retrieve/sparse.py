from kiwipiepy import Kiwi
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

class BM25:
    def __init__(self, type: str = 'None', k: int = 15):
        self.k = k
        self.type = type
        self.kiwi = Kiwi()
        self.texts = []
        self.vectorizer = None

    
    def tokenize(self, text: str) -> List[str]:
        if self.type == 'None':
            return text.split()
        elif self.type == 'kiwi':
            return [token.form for token in self.kiwi.tokenize(text)]
        elif self.type == 'kiwi_pos':
            return [token.form for token in self.kiwi.tokenize(text) if token.tag.startswith(('NN', 'VV', 'VA'))]
        else:
            raise ValueError(f"지원하지 않는 토크나이저 타입입니다: {self.type}. 'None', 'kiwi', 'kiwi_pos' 중 하나를 선택해주세요.")
        

    def from_texts(self, texts: List[str]):
        self.texts = texts
        texts_processed = [self.tokenize(t) for t in texts]
        self.vectorizer = BM25Okapi(texts_processed)
    
    def search(self, query: str) -> List[str]:
        if not self.vectorizer:
            raise ValueError("BM25가 초기화되지 않았습니다. from_texts()를 먼저 호출하세요.")
        
        processed_query = self.tokenize(query)
        
        return self.vectorizer.get_top_n(processed_query, self.texts, n=self.k)