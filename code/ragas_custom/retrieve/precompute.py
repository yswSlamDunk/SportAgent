import itertools
from tqdm import tqdm

def precompute_retrievals(df, db, bm25, kiwi, kiwi_pos, 
                          k_list=[5, 10, 15], 
                          fetch_k_list=[1.5, 2.0, 2.5], 
                          lambda_mult_list=[0.3, 0.5, 0.7]):
    df['precompute_dense'] = df['user_input'].apply(lambda x : db.similarity_search_with_score(x, k=int(15)))
    df['precompute_sparse_bm25'] = df['user_input'].apply(lambda x : bm25.search(x))
    df['precompute_sparse_bm25_kiwi'] = df['user_input'].apply(lambda x : kiwi.search(x))
    df['precompute_sparse_bm25_kiwi_pos'] = df['user_input'].apply(lambda x : kiwi_pos.search(x))

    combinations = list(itertools.product(k_list, fetch_k_list, lambda_mult_list))
    for (k, fetch_k, lambda_mult) in tqdm(combinations):
        column_name = f'mmr_k{k}_fetch{fetch_k}_lambda{lambda_mult}'
        
        df[column_name] = df['user_input'].apply(
            lambda x: db.similarity_search_with_score(
                x, 
                k=k, 
                fetch_k=int(k*fetch_k),
                search_type='mmr', 
                lambda_mult=lambda_mult
            )
        )
    
    return df