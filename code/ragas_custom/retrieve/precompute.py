def precompute_retrievals(df, db, bm25, kiwi, kiwi_pos):
    df['precompute_dense'] = df['user_input'].apply(lambda x : db.similarity_search_with_score(x, k=int(15 * 2.5)))
    df['precompute_sparse_bm25'] = df['user_input'].apply(lambda x : bm25.search(x))
    df['precompute_sparse_bm25_kiwi'] = df['user_input'].apply(lambda x : kiwi.search(x))
    df['precompute_sparse_bm25_kiwi_pos'] = df['user_input'].apply(lambda x : kiwi_pos.search(x))
    
    return df