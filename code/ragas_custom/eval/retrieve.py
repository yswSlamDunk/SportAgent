import numpy as np
import pandas as pd
from tqdm import tqdm

def calculate_ndcg(predictions, ground_truth, k):
    """
    NDCG@k 직접 구현
    
    Parameters:
    -----------
    predictions : list
        retriever가 반환한 문서 리스트
    ground_truth : list
        실제 관련 문서 리스트
    k : int
        평가할 상위 k개 문서
    
    Returns:
    --------
    float
        NDCG@k 점수
    """
    def dcg_at_k(relevance_scores, k):
        """
        DCG@k 계산
        DCG@k = sum(rel_i / log_2(i + 1)) for i in range(k)
        """
        dcg = 0.0
        for i in range(min(len(relevance_scores), k)):
            dcg += relevance_scores[i] / np.log2(i + 2)  # i + 2 because log_2(1) = 0
        return dcg
    
    # 실제 관련 문서에 대한 relevance score 생성 (1: 관련, 0: 비관련)
    relevance = [1 if doc in ground_truth else 0 for doc in predictions[:k]]
    
    # Ideal ranking 생성 (모든 관련 문서가 상위에 있는 경우)
    ideal_ranking = [1] * min(len(ground_truth), k) + [0] * (k - len(ground_truth))
    
    # DCG 계산
    dcg = dcg_at_k(relevance, k)
    
    # IDCG 계산 (ideal ranking의 DCG)
    idcg = dcg_at_k(ideal_ranking, k)
    
    # NDCG 계산
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    return ndcg

def calculate_recall(predictions, ground_truth, k):
    """
    Recall@k 계산
    """
    # 상위 k개 예측 중 실제 관련 문서의 비율
    relevant_retrieved = len(set(predictions[:k]) & set(ground_truth))
    return relevant_retrieved / len(ground_truth) if ground_truth else 0.0

def calculate_map(predictions, ground_truth, k):
    """
    MAP@k 계산
    """
    if not ground_truth:
        return 0.0
        
    # 각 위치에서의 precision 계산
    precisions = []
    relevant_count = 0
    
    for i, doc in enumerate(predictions[:k]):
        if doc in ground_truth:
            relevant_count += 1
            precisions.append(relevant_count / (i + 1))
    
    return np.mean(precisions) if precisions else 0.0

def evaluate_metrics(predictions, ground_truth, k):
    """
    NDCG, Recall, MAP 평가 수행
    predictions와 ground_truth는 각각 리스트의 리스트 형태
    """
    ndcg_scores = []
    recall_scores = []
    map_scores = []
    
    # 각 쿼리별로 평가 수행
    for pred, truth in zip(predictions, ground_truth):
        ndcg_scores.append(calculate_ndcg(pred, truth, k))
        recall_scores.append(calculate_recall(pred, truth, k))
        map_scores.append(calculate_map(pred, truth, k))
    
    # 평균 점수 계산
    return {
        'ndcg': np.mean(ndcg_scores),
        'recall': np.mean(recall_scores),
        'map': np.mean(map_scores)
    }

def apply_mrr_ranking(docs_and_scores, lambda_mult):
    """
    Maximum Marginal Relevance (MRR) 랭킹 적용
    
    Parameters:
    -----------
    docs_and_scores : list of tuples
        (document, score) 튜플의 리스트
    lambda_mult : float
        MMR의 lambda 파라미터 (0~1 사이)
        - 1에 가까울수록 relevance에 더 가중치
        - 0에 가까울수록 diversity에 더 가중치
    
    Returns:
    --------
    list
        MRR로 재랭킹된 문서 리스트
    """
    if not docs_and_scores:
        return []
    
    # 초기 문서 선택 (가장 높은 점수의 문서)
    selected = [docs_and_scores[0][0]]
    remaining = docs_and_scores[1:]
    
    while remaining and len(selected) < len(docs_and_scores):
        # 각 남은 문서에 대해 MMR 점수 계산
        mmr_scores = []
        for doc, score in remaining:
            # relevance term
            relevance = score
            
            # diversity term (이미 선택된 문서들과의 최대 유사도)
            max_similarity = 0
            for selected_doc in selected:
                # 여기서는 간단히 문서 내용의 길이 차이를 유사도로 사용
                # 실제로는 더 정교한 유사도 계산 방법을 사용해야 함
                similarity = 1 - abs(len(doc) - len(selected_doc)) / max(len(doc), len(selected_doc))
                max_similarity = max(max_similarity, similarity)
            
            # MMR 점수 계산
            mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_similarity
            mmr_scores.append((doc, mmr_score))
        
        # 가장 높은 MMR 점수를 가진 문서 선택
        best_doc = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append(best_doc)
        
        # 선택된 문서를 remaining에서 제거
        remaining = [(doc, score) for doc, score in remaining if doc != best_doc]
    
    return selected

def combine_hybrid_results(dense_results, sparse_results, alpha, k):
    """
    Dense와 Sparse Retriever의 결과를 결합
    
    Parameters:
    -----------
    dense_results : list of lists
        Dense Retriever의 결과 리스트
    sparse_results : list of lists
        Sparse Retriever의 결과 리스트
    alpha : float
        Dense Retriever의 가중치 (0~100)
    k : int
        반환할 문서 수
    
    Returns:
    --------
    list of lists
        결합된 결과 리스트
    """
    combined_results = []
    
    # 각 쿼리별로 결과 결합
    for dense_pred, sparse_pred in zip(dense_results, sparse_results):
        # 문서와 점수 매핑 생성
        doc_scores = {}
        
        # Dense 결과 처리
        for i, doc in enumerate(dense_pred):
            # 순위 기반 점수 계산 (높은 순위일수록 높은 점수)
            rank_score = 1.0 / (i + 1)
            # alpha를 0~1 범위로 정규화
            normalized_alpha = alpha / 100.0
            doc_scores[doc] = normalized_alpha * rank_score
        
        # Sparse 결과 처리
        for i, doc in enumerate(sparse_pred):
            rank_score = 1.0 / (i + 1)
            # (1 - alpha)를 sparse의 가중치로 사용
            normalized_alpha = alpha / 100.0
            if doc in doc_scores:
                doc_scores[doc] += (1 - normalized_alpha) * rank_score
            else:
                doc_scores[doc] = (1 - normalized_alpha) * rank_score
        
        # 최종 점수로 정렬하여 상위 k개 선택
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        combined_results.append([doc for doc, _ in sorted_docs[:k]])
    
    return combined_results

def optimization(configs, precompute_df):
    results = []
    
    for config in tqdm(configs):
        if config['alpha'] == 0:  # Sparse Retriever
            if config['morphological_analyzer'] == 'bm25':
                predictions = precompute_df['precompute_sparse_bm25'].tolist()
            elif config['morphological_analyzer'] == 'bm25_kiwi':
                predictions = precompute_df['precompute_sparse_bm25_kiwi'].tolist()
            else:  # bm25_kiwi_pos
                predictions = precompute_df['precompute_sparse_bm25_kiwi_pos'].tolist()
                
            # 평가 수행
            scores = evaluate_metrics(predictions, precompute_df['reference_contexts'].tolist(), config['k'])
            
        elif config['alpha'] == 100:  # Dense Retriever
            if config['dense_type'] == 'mrr':
                # MRR 파라미터 적용
                fetch_k = int(config['dense_params']['fetch_k'] * config['k'])
                lambda_mult = config['dense_params']['lambda_mult']
                
                # dense_precompute에서 fetch_k만큼 가져와서 MRR 적용
                predictions = precompute_df['precompute_dense'].apply(
                    lambda x: apply_mrr_ranking(
                        [(doc.__dict__['page_content'], score) for doc, score in x[:fetch_k]], 
                        lambda_mult
                    )[:config['k']]
                ).tolist()
                
            else:  # threshold
                # Threshold 파라미터 적용
                score_threshold = config['dense_params']['score_threshold']
                
                # dense_precompute에서 threshold 적용
                predictions = precompute_df['precompute_dense'].apply(
                    lambda x: [doc.__dict__['page_content'] for doc, score in x if score >= score_threshold][:config['k']]
                ).tolist()
            
            # 평가 수행
            scores = evaluate_metrics(predictions, precompute_df['reference_contexts'].tolist(), config['k'])
            
        else:  # Hybrid Retriever
            # dense 결과 처리
            if config['dense_type'] == 'mrr':
                fetch_k = int(config['dense_params']['fetch_k'] * config['k'])
                lambda_mult = config['dense_params']['lambda_mult']
                dense_results = precompute_df['precompute_dense'].apply(
                    lambda x: apply_mrr_ranking(
                        [(doc.__dict__['page_content'], score) for doc, score in x[:fetch_k]], 
                        lambda_mult
                    )
                ).tolist()
            else:  # threshold
                score_threshold = config['dense_params']['score_threshold']
                dense_results = precompute_df['precompute_dense'].apply(
                    lambda x: [doc.__dict__['page_content'] for doc, score in x if score >= score_threshold]
                ).tolist()
            
            # sparse 결과 처리
            if config['morphological_analyzer'] == 'bm25':
                sparse_results = precompute_df['precompute_sparse_bm25'].tolist()
            elif config['morphological_analyzer'] == 'bm25_kiwi':
                sparse_results = precompute_df['precompute_sparse_bm25_kiwi'].tolist()
            else:  # bm25_kiwi_pos
                sparse_results = precompute_df['precompute_sparse_bm25_kiwi_pos'].tolist()
            
            # hybrid scoring 및 ranking
            predictions = combine_hybrid_results(
                dense_results, 
                sparse_results, 
                config['alpha'],
                config['k']
            )
            
            # 평가 수행
            scores = evaluate_metrics(predictions, precompute_df['reference_contexts'].tolist(), config['k'])
        
        # 결과 저장
        result = {
            'k': config['k'],
            'alpha': config['alpha'],
            'dense_type': config.get('dense_type', 'sparse'),
            'morphological_analyzer': config.get('morphological_analyzer', 'none'),
            'fetch_k': config.get('dense_params', {}).get('fetch_k', None),
            'lambda_mult': config.get('dense_params', {}).get('lambda_mult', None),
            'score_threshold': config.get('dense_params', {}).get('score_threshold', None),
            'ndcg': scores['ndcg'],
            'recall': scores['recall'],
            'map': scores['map']
        }
        results.append(result)
    
    # DataFrame으로 변환
    return pd.DataFrame(results)