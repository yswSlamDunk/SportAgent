import os
from kiwipiepy import Kiwi
import cohere

from typing import List

from ragas.testset.graph import KnowledgeGraph
from rank_bm25 import BM25Okapi

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

class BM25:
    def __init__(self, type: str = 'None', k: int = 20):
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

class Rag:
    # 클래스 변수로 싱글톤 상태 관리
    _instance = None
    _initialized = False
    
    def __init__(self, alpha=0.4, threshold=0.1, k=20, kg_path=None):
        # 이미 초기화됐으면 재초기화 방지
        if Rag._initialized:
            return
            
        self.alpha = alpha
        self.threshold = threshold
        self.k = k
        self.rerank_k = 9
        
        # 프로젝트 루트 디렉토리 찾기
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_dir, '..', '..')
        
        # 기본 경로 설정
        if kg_path is None:
            kg_path = os.path.join(project_root, 'data', 'rag', 'kg.json')
        
        kg = KnowledgeGraph.load(kg_path)
        texts = [node.properties['page_content'] for node in kg.nodes]
        self.kiwi_pos = BM25(k=k, type='kiwi_pos')
        self.kiwi_pos.from_texts(texts)

        embeddings = OpenAIEmbeddings()
        db_path = os.path.join(project_root, 'data', 'rag', 'db')
        db = FAISS.load_local(folder_path=db_path, 
                              index_name="index",
                              embeddings=embeddings,
                              allow_dangerous_deserialization=True)
        self.db = db.as_retriever(search_type="similarity_score_threshold", 
                                  search_kwargs={"k": self.k, 
                                                 "score_threshold": self.threshold})

        self.cohere = cohere.ClientV2()
        
        # 싱글톤 인스턴스 저장
        Rag._instance = self
        Rag._initialized = True

    def rerank(self, query: str, contexts: List[str]):
        result = self.cohere.rerank(model='rerank-v3.5', query=query, documents=contexts)
        result = [re.index for re in result.results]
        return result

    def precompute(self, dense, sparse):
        results = []
        normalized_alpha = self.alpha / 100.0

        doc_scores = {}
        for i, doc in enumerate(dense):
            rank_score = 1.0 / (i + 1)
            doc_scores[doc] = normalized_alpha * rank_score
        for i, doc in enumerate(sparse):
            rank_score = 1.0 / (i + 1)
            if doc in doc_scores:
                doc_scores[doc] += (1 - normalized_alpha) * rank_score
            else:
                doc_scores[doc] = (1 - normalized_alpha) * rank_score

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:self.rerank_k]
        sorted_docs = [doc for doc, score in sorted_docs]
        results.extend(sorted_docs)

        return results

    def search(self, query: str) -> List[str]:
        db_results = self.db.invoke(query)
        if len(db_results) != 0:
            db_results = [result.page_content for result in db_results]
        else:
            db_results = []

        kiwi_pos_results = self.kiwi_pos.search(query)

        precomputed_results = self.precompute(db_results, kiwi_pos_results)

        print(precomputed_results)

        reranked_index = self.rerank(query, precomputed_results)
        reranked_results = [precomputed_results[i] for i in reranked_index]

        return reranked_results

    @staticmethod
    def search_static(query: str) -> List[str]:
        """정적 메서드: Rag.search_static('query') 형태로 사용"""
        if not Rag._initialized:
            Rag()  # 자동 초기화
        return Rag._instance.search(query)

# 편의 함수
def search(query: str) -> List[str]:
    """모듈 레벨 함수: search('query') 형태로 사용"""
    return Rag.search_static(query)