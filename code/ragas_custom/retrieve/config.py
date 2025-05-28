from itertools import product

def generate_sparse_configs(k_values, analyzers):
    """Sparse Retriever 설정 생성 (alpha = 0)"""
    return [
        {
            "k": k,
            "alpha": 0,
            "morphological_analyzer": analyzer
        }
        for k, analyzer in product(k_values, analyzers)
    ]

def generate_dense_configs(k_values, mrr_params, threshold_params):
    """Dense Retriever 설정 생성 (alpha = 100)"""
    configs = []
    
    # MRR 조합
    for k in k_values:
        for fetch_k, lambda_mult in product(mrr_params["fetch_k"], mrr_params["lambda_mult"]):
            configs.append({
                "k": k,
                "alpha": 100,
                "dense_type": "mrr",
                "dense_params": {
                    "fetch_k": fetch_k,
                    "lambda_mult": lambda_mult
                }
            })
    
    # Threshold 조합
    for k in k_values:
        for score_threshold in threshold_params["score_threshold"]:
            configs.append({
                "k": k,
                "alpha": 100,
                "dense_type": "threshold",
                "dense_params": {
                    "score_threshold": score_threshold
                }
            })
    
    return configs

def generate_hybrid_configs(k_values, hybrid_alphas, analyzers, mrr_params, threshold_params):
    """Hybrid Retriever 설정 생성 (0 < alpha < 100)"""
    configs = []
    
    for k, alpha, analyzer in product(k_values, hybrid_alphas, analyzers):
        # MRR 조합
        for fetch_k, lambda_mult in product(mrr_params["fetch_k"], mrr_params["lambda_mult"]):
            configs.append({
                "k": k,
                "alpha": alpha,
                "dense_type": "mrr",
                "dense_params": {
                    "fetch_k": fetch_k,
                    "lambda_mult": lambda_mult
                },
                "morphological_analyzer": analyzer
            })
        
        # Threshold 조합
        for score_threshold in threshold_params["score_threshold"]:
            configs.append({
                "k": k,
                "alpha": alpha,
                "dense_type": "threshold",
                "dense_params": {
                    "score_threshold": score_threshold
                },
                "morphological_analyzer": analyzer
            })
    
    return configs

def generate_retriever_configs():
    """모든 Retriever 설정 조합 생성"""
    # 기본 파라미터
    k_values = [5, 10, 15]
    analyzers = ["bm25", "bm25_kiwi", "bm25_kiwi_pos"]
    hybrid_alphas = [20, 40, 60, 80]
    
    # Dense Retriever 파라미터
    mrr_params = {
        "fetch_k": [2, 2.5, 3],
        "lambda_mult": [0.3, 0.5, 0.7]
    }
    threshold_params = {
        "score_threshold": [0.3, 0.5, 0.7, 0.9]
    }
    
    # 각 타입별 설정 생성
    sparse_configs = generate_sparse_configs(k_values, analyzers)
    dense_configs = generate_dense_configs(k_values, mrr_params, threshold_params)
    hybrid_configs = generate_hybrid_configs(k_values, hybrid_alphas, analyzers, mrr_params, threshold_params)
    
    # 모든 설정 합치기
    return sparse_configs + dense_configs + hybrid_configs