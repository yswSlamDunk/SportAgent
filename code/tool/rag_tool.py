from typing import List, Optional, Dict, Any
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent / 'ragas_custom'))

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools import tool

from ragas.testset.graph import KnowledgeGraph

from retrieve.sparse import BM25
from evaluation.retrieve import combine_hybrid_results

# Cohere Reranker
try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False
    raise ImportError("cohere 패키지가 설치되지 않았습니다. pip install cohere로 설치해주세요.")


class RAGTool:
    """RAG 검색 도구 - Hybrid Retrieval (Dense + Sparse) 및 Cohere Reranker 지원"""
    
    def __init__(
        self,
        kg_path: str = '../data/rag/kg.json'
    ):
        """
        RAG Tool 초기화
        
        Parameters:
        -----------
        kg_path : str
            KnowledgeGraph JSON 파일 경로
        """
        # 고정된 retriever 설정
        self.retriever_config = {
            'k': 20,
            'alpha': 40,
            'dense_type': 'threshold',
            'morphological_analyzer': 'bm25_kiwi_pos',
            'score_threshold': 0.1
        }
        
        # Reranker 설정
        self.reranker_type = 'cohere'
        self.rerank_top_k = 9
        
        # KnowledgeGraph 로드
        self.kg = KnowledgeGraph.load(kg_path)
        
        # Documents 생성
        self.documents = [
            Document(
                page_content=node.properties['page_content'],
                metadata=node.properties['document_metadata']
            )
            for node in self.kg.nodes
        ]
        
        # page_content와 Document 매핑 생성 (빠른 조회를 위해)
        self.content_to_doc = {doc.page_content: doc for doc in self.documents}
        
        # Dense Retriever (FAISS) 초기화
        self.embeddings = OpenAIEmbeddings()
        self.db = FAISS.from_documents(self.documents, self.embeddings)
        
        # Sparse Retriever (BM25) 초기화
        texts = [node.properties['page_content'] for node in self.kg.nodes]
        self.bm25 = BM25(k=self.retriever_config['k'], type='kiwi_pos')
        self.bm25.from_texts(texts)
        
        # Cohere Reranker 초기화
        if not COHERE_AVAILABLE:
            raise ImportError("cohere 패키지가 설치되지 않았습니다.")
        self.cohere_client = cohere.ClientV2()
    
    def retrieve(
        self,
        query: str,
        use_reranker: bool = True
    ) -> List[str]:
        """
        쿼리에 대한 관련 문서 검색 (문자열만 반환 - 호환성 유지)
        
        Parameters:
        -----------
        query : str
            사용자 쿼리
        use_reranker : bool
            Reranker 사용 여부 (기본값: True)
            
        Returns:
        --------
        List[str]
            관련 문서 리스트
        """
        documents = self.retrieve_documents(query, use_reranker)
        return [doc.page_content for doc in documents]
    
    def retrieve_documents(
        self,
        query: str,
        use_reranker: bool = True
    ) -> List[Document]:
        """
        쿼리에 대한 관련 문서 검색 (Document 객체 반환)
        
        Parameters:
        -----------
        query : str
            사용자 쿼리
        use_reranker : bool
            Reranker 사용 여부 (기본값: True)
            
        Returns:
        --------
        List[Document]
            관련 Document 객체 리스트 (page_content와 metadata 포함)
        """
        # Dense Retrieval (threshold 방식)
        dense_results = self.db.similarity_search_with_score(
            query,
            k=self.retriever_config['k']
        )
        dense_contexts = [
            doc.page_content
            for doc, score in dense_results
            if score >= self.retriever_config['score_threshold']
        ]
        
        # Sparse Retrieval (BM25 with kiwi_pos)
        sparse_contexts = self.bm25.search(query)
        
        # Hybrid Retrieval (alpha=40: Dense 40%, Sparse 60%)
        alpha = self.retriever_config['alpha']
        k = self.retriever_config['k']
        
        retrieved_contexts = combine_hybrid_results(
            [dense_contexts],
            [sparse_contexts],
            alpha,
            k
        )[0]
        
        # Cohere Reranker 적용
        if use_reranker and retrieved_contexts:
            retrieved_contexts = self._rerank_cohere(query, retrieved_contexts)
        
        # 최종 결과를 Document 객체로 변환
        final_contexts = retrieved_contexts[:self.rerank_top_k] if use_reranker else retrieved_contexts
        retrieved_docs = [
            self.content_to_doc[content] 
            for content in final_contexts 
            if content in self.content_to_doc
        ]
        
        return retrieved_docs
    
    def _rerank_cohere(self, query: str, contexts: List[str]) -> List[str]:
        """Cohere Reranker 적용"""
        response = self.cohere_client.rerank(
            model='rerank-v3.5',
            query=query,
            documents=contexts
        )
        # 결과를 점수 순으로 정렬된 인덱스로 변환
        ranked_indices = [result.index for result in response.results]
        return [contexts[i] for i in ranked_indices]


# LangChain tool로 사용하기 위한 래퍼 함수
def create_rag_tool(kg_path: str = '../data/rag/kg.json'):
    """
    LangChain tool로 사용할 수 있는 RAG 검색 함수 생성
    
    Parameters:
    -----------
    kg_path : str
        KnowledgeGraph JSON 파일 경로
        
    Returns:
    --------
    function
        RAG 검색 함수
    """
    rag_tool = RAGTool(kg_path=kg_path)
    
    @tool
    def rag_search(query: str) -> str:
        """
        Searches for documents related to the user's query based on the Weightlifting Coach Training Manual.

        This tool contains specialized weightlifting knowledge, including:
        1. Weightlifting training program composition and instruction plans
        2. Structure and training methods of weightlifting competition techniques
        3. Sports science principles of weightlifting
        4. Understanding weightlifting
        5. Structure and training methods for weightlifting physical fitness

        Args:
            query: The search query.

        Returns:
            A list of JSON-serializable dicts containing page content and metadata for each relevant document,
            where each entry is of the form: {"page_content": str, "metadata": dict}.
        """
        documents = rag_tool.retrieve_documents(query)
        
        # 각 Document를 dict 형태로 변환하여 리스트로 구성
        result_list = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in documents
        ]
        
        return result_list
    
    return rag_search