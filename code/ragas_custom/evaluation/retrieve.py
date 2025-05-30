import numpy as np
import pandas as pd
from tqdm import tqdm

def calculate_ndcg(predictions, ground_truth, k):
    """
    NDCG@k 직접 구현0
    """
    def dcg_at_k(relevance_scores, k):
        dcg = 0.0
        for i in range(min(len(relevance_scores), k)):
            dcg += relevance_scores[i] / np.log2(i + 2)
        return dcg
    

    pred_contents = predictions[:k]
    truth_contents = ground_truth
    
    # 실제 관련 문서에 대한 relevance score 생성 (1: 관련, 0: 비관련)
    relevance = [1 if doc in truth_contents else 0 for doc in pred_contents]
    
    # Ideal ranking 생성 (모든 관련 문서가 상위에 있는 경우)
    ideal_ranking = [1] * min(len(truth_contents), k) + [0] * (k - len(truth_contents))
    
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

    pred_contents = predictions[:k]
    truth_contents = ground_truth
    
    # 상위 k개 예측 중 실제 관련 문서의 비율
    pred_set = set(pred_contents)
    truth_set = set(truth_contents)
    relevant_retrieved = len(pred_set & truth_set)
    return relevant_retrieved / len(truth_set) if truth_set else 0.0

def calculate_map(predictions, ground_truth, k):
    """
    MAP@k 계산
    """
    if not ground_truth:
        return 0.0

    pred_contents = predictions[:k]
    truth_contents = ground_truth
    
    # 각 위치에서의 precision 계산
    precisions = []
    relevant_count = 0
    
    for i, doc in enumerate(pred_contents):
        if doc in truth_contents:
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

def extract_text_from_docs(doc_score_list):
    return [pred.page_content for prediction in doc_score_list for pred, _ in prediction]

def optimization(configs, precompute_df):
    results = []
    
    for config in tqdm(configs):
        if config['alpha'] == 0:  # Sparse Retriever
            if config['morphological_analyzer'] == 'bm25':
                predictions = precompute_df['precompute_sparse_bm25']
            elif config['morphological_analyzer'] == 'bm25_kiwi':
                predictions = precompute_df['precompute_sparse_bm25_kiwi']
            else:  # bm25_kiwi_pos
                predictions = precompute_df['precompute_sparse_bm25_kiwi_pos']
                
        elif config['alpha'] == 100:  # Dense Retriever
            if config['dense_type'] == 'mmr':
                # 미리 계산된 MMR 결과 사용
                column_name = f'mmr_k{config["k"]}_fetch{config["dense_params"]["fetch_k"]:.1f}_lambda{config["dense_params"]["lambda_mult"]}'
                predictions = precompute_df[column_name].to_list()
                predictions = extract_text_from_docs(predictions)


            else:  # threshold
                # Threshold 파라미터 적용
                score_threshold = config['dense_params']['score_threshold']
                predictions = precompute_df['precompute_dense'].apply(
                    lambda x: [doc.__dict__['page_content'] for doc, score in x if score >= score_threshold][:config['k']]
                )
            
        else:  # Hybrid Retriever
            # dense 결과 처리
            if config['dense_type'] == 'mmr':
                # 미리 계산된 MMR 결과 사용
                column_name = f'mmr_k{config["k"]}_fetch{config["dense_params"]["fetch_k"]:.1f}_lambda{config["dense_params"]["lambda_mult"]}'
                dense_results = precompute_df[column_name].to_list()
                dense_results = extract_text_from_docs(dense_results)

            else:  # threshold
                score_threshold = config['dense_params']['score_threshold']
                dense_results = precompute_df['precompute_dense'].apply(
                    lambda x: [doc.__dict__['page_content'] for doc, score in x if score >= score_threshold]
                )
            
            # sparse 결과 처리
            if config['morphological_analyzer'] == 'bm25':
                sparse_results = precompute_df['precompute_sparse_bm25']
            elif config['morphological_analyzer'] == 'bm25_kiwi':
                sparse_results = precompute_df['precompute_sparse_bm25_kiwi']
            else:  # bm25_kiwi_pos
                sparse_results = precompute_df['precompute_sparse_bm25_kiwi_pos']
            
            # hybrid scoring 및 ranking
            predictions = combine_hybrid_results(
                dense_results, 
                sparse_results, 
                config['alpha'],
                config['k']
            )


        # 평가 수행
        scores = evaluate_metrics(predictions, precompute_df['reference_contexts'], config['k'])
        
        # 결과 저장
        result = {
            'k': config['k'],
            'alpha': config['alpha'],
            'dense_type': config.get('dense_type', None),
            'morphological_analyzer': config.get('morphological_analyzer', None),
            'fetch_k': config.get('dense_params', {}).get('fetch_k'),
            'lambda_mult': config.get('dense_params', {}).get('lambda_mult'),
            'score_threshold': config.get('dense_params', {}).get('score_threshold'),
            'ndcg': scores['ndcg'],
            'recall': scores['recall'],
            'map': scores['map']
        }
        results.append(result)
    
    return pd.DataFrame(results)