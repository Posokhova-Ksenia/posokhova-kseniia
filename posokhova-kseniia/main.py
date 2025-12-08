import os
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import lightgbm as lgb
import math
from sentence_transformers import SentenceTransformer
from sklearn.metrics import pairwise
import gc
from lightgbm import LGBMRanker
from lightgbm import early_stopping, log_evaluation



SEED = 993
random.seed(SEED)
np.random.seed(SEED)

TRAIN_PATH = "data/train.csv"
TEST_PATH  = "data/test.csv"
OUTPUT_SUBMISSION = "results/submission.csv"

SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2" 
EMBED_BATCH = 512
N_SPLITS = 5

EARLY_STOP = 100
NUM_BOOST_ROUND = 2000

def log(msg):
    print(f"[LOG] {msg}")

def read_df(path):
    return pd.read_csv(path)

def build_product_text(df):
    parts = []
    for col in ['product_title','product_description','product_bullet_point','product_brand','product_color']:
        if col in df.columns:
            parts.append(df[col].fillna("").astype(str))
    if not parts:
        return pd.Series([""]*len(df), index=df.index)
    txt = parts[0].copy()
    for p in parts[1:]:
        txt = txt + " " + p
    return txt

def embed_texts(model, texts, cache_path=None, batch_size=EMBED_BATCH):
    if cache_path and os.path.exists(cache_path):
        log(f"Loading cached embeddings from {cache_path}")
        return np.load(cache_path)
    
    emb = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding -> {cache_path or 'batch'}"):
        batch = texts[i:i+batch_size]
        emb_batch = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        emb.append(emb_batch)
    emb = np.vstack(emb)
    
    if cache_path:
        np.save(cache_path, emb)
        log(f"Saved embeddings to {cache_path}")
    
    return emb

def create_features_from_embeddings(df, query_emb_map, prod_emb_map):
    n = len(df)
    cos_sim = np.zeros(n, dtype=np.float32)
    dot_prod = np.zeros(n, dtype=np.float32)
    l2_dist = np.zeros(n, dtype=np.float32)
    absdiff_mean = np.zeros(n, dtype=np.float32)
    prod_len = np.zeros(n, dtype=np.int32)
    query_len = np.zeros(n, dtype=np.int32)
    common_tokens = np.zeros(n, dtype=np.int32)

    for i, row in enumerate(tqdm(df.itertuples(index=False), total=n, desc="Creating features")):
        qid = row.query_id
        pid = row.product_id
        qemb = query_emb_map[qid]
        pemb = prod_emb_map[pid]

        cos_sim[i] = float(cosine_similarity(qemb.reshape(1,-1), pemb.reshape(1,-1))[0,0])
        dot_prod[i] = float(np.dot(qemb, pemb))
        l2_dist[i] = float(np.linalg.norm(qemb - pemb))
        absdiff_mean[i] = float(np.mean(np.abs(qemb - pemb)))

        qtext = str(row.query) if hasattr(row, 'query') else ""
        ptext = str(row.product_title) + " " + (str(row.product_description) if hasattr(row, 'product_description') else "")
        query_len[i] = len(qtext.split())
        prod_len[i] = len(ptext.split())
        common_tokens[i] = len(set(qtext.lower().split()) & set(ptext.lower().split()))

    feat_df = pd.DataFrame({
        'id': df['id'].values,
        'cos_sim': cos_sim,
        'dot_prod': dot_prod,
        'l2_dist': l2_dist,
        'absdiff_mean': absdiff_mean,
        'query_len': query_len,
        'prod_len': prod_len,
        'common_tokens': common_tokens
    })
    return feat_df

def dcg_at_k(rels, k=10):
    rels = np.asarray(rels)[:k]
    if rels.size == 0:
        return 0.0
    gains = (2**rels - 1)
    discounts = np.log2(np.arange(2, rels.size + 2))
    return np.sum(gains / discounts)

def ndcg_at_10_for_group(true_rels, pred_scores, k=10):
    order = np.argsort(-np.asarray(pred_scores))
    preds_sorted = np.asarray(true_rels)[order]
    dcg = dcg_at_k(preds_sorted, k=k)
    ideal_order = np.argsort(-np.asarray(true_rels))
    ideal_sorted = np.asarray(true_rels)[ideal_order]
    idcg = dcg_at_k(ideal_sorted, k=k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg

def mean_ndcg10_from_merged(merged_df):
    grouped = merged_df.groupby("query_id")
    ndcgs = []
    for q, g in grouped:
        ndcg = ndcg_at_10_for_group(g['relevance'].values, g['prediction'].values, k=10)
        ndcgs.append(ndcg)
    return float(np.mean(ndcgs)) if len(ndcgs)>0 else 0.0

def get_group_sizes(qids):
    sizes = []
    prev = qids[0]
    count = 0
    for q in qids:
        if q == prev:
            count += 1
        else:
            sizes.append(count)
            prev = q
            count = 1
    sizes.append(count)
    return sizes

def main():
    log("Loading train/test data...")
    train = read_df(TRAIN_PATH)
    test  = read_df(TEST_PATH)

    log("Building concatenated product text...")
    train['product_text'] = build_product_text(train)
    test['product_text']  = build_product_text(test)

    log("Preparing unique queries and products...")
    unique_queries = train[['query_id','query']].drop_duplicates(subset=['query_id']).set_index('query_id')['query'].to_dict()
    unique_queries.update(test[['query_id','query']].drop_duplicates(subset=['query_id']).set_index('query_id')['query'].to_dict())
    prod_map = train[['product_id','product_text']].drop_duplicates(subset=['product_id']).set_index('product_id')['product_text'].to_dict()
    prod_map.update(test[['product_id','product_text']].drop_duplicates(subset=['product_id']).set_index('product_id')['product_text'].to_dict())

    log(f"Loading SBERT model: {SBERT_MODEL}")
    model = SentenceTransformer(SBERT_MODEL,trust_remote_code=True)
    log("SBERT loaded.")

    q_ids = list(unique_queries.keys())
    q_texts = [unique_queries[qid] for qid in q_ids]
    log(f"Embedding {len(q_ids)} items...")
    q_embs = embed_texts(model, q_texts, cache_path="query_emb.npy", batch_size=EMBED_BATCH)
    query_emb_map = {qid: emb for qid, emb in zip(q_ids, q_embs)}
    del q_ids, q_texts, q_embs
    gc.collect()

    p_ids = list(prod_map.keys())
    p_texts = [prod_map[pid] for pid in p_ids]
    log(f"Embedding {len(p_ids)} items...")
    p_embs = embed_texts(model, p_texts, cache_path="prod_emb.npy", batch_size=EMBED_BATCH)
    prod_emb_map = {pid: emb for pid, emb in zip(p_ids, p_embs)}
    del p_ids, p_texts, p_embs
    gc.collect()

    log("Creating features for train...")
    train_feats = create_features_from_embeddings(train, query_emb_map, prod_emb_map)
    log("Creating features for test...")
    test_feats  = create_features_from_embeddings(test, query_emb_map, prod_emb_map)

    train_feats = train_feats.merge(train[['id','query_id','relevance']], on='id', how='left')
    test_feats  = test_feats.merge(test[['id','query_id']], on='id', how='left')

    feature_cols = [c for c in train_feats.columns if c not in ['id','query_id','relevance']]
    log(f"Feature columns: {feature_cols}")

    X = train_feats[feature_cols].values
    y = train_feats['relevance'].astype(int).values
    qgroup = train_feats['query_id'].values
    test_X = test_feats[feature_cols].values

    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_preds = np.zeros(X.shape[0], dtype=np.float32)
    test_preds = np.zeros(test_X.shape[0], dtype=np.float32)

    log("Starting GroupKFold LightGBM training...")
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups=qgroup)):
        log(f"Fold {fold+1}/{N_SPLITS}")
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        q_tr = qgroup[tr_idx]
        q_val = qgroup[val_idx]

        train_group = get_group_sizes(q_tr)
        val_group   = get_group_sizes(q_val)

        model_lgb = LGBMRanker(
            objective='lambdarank',
            metric='ndcg',
            ndcg_at=[10],
            learning_rate=0.05,
            num_leaves=31,
            min_data_in_leaf=20,
            feature_fraction=0.8,
            bagging_freq=1,
            bagging_fraction=0.9,
            n_estimators=NUM_BOOST_ROUND,
            random_state=SEED,
            n_jobs=-1
        )

        model_lgb.fit(
            X_tr, y_tr,
            group=train_group,
            eval_set=[(X_val, y_val)],
            eval_group=[val_group],
            eval_metric='ndcg',
            callbacks=[early_stopping(stopping_rounds=EARLY_STOP), log_evaluation(period=100)]
        )

        oof_preds[val_idx] = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration_)
        test_preds += model_lgb.predict(test_X, num_iteration=model_lgb.best_iteration_) / N_SPLITS

        del model_lgb
        gc.collect()

    oof_df = pd.DataFrame({'id': train_feats['id'].values, 'prediction': oof_preds})
    merged_oof = train_feats[['id','query_id','relevance']].merge(oof_df, on='id', how='left')
    ndcg_oof = mean_ndcg10_from_merged(merged_oof)
    log(f"OOF mean nDCG@10: {ndcg_oof:.6f}")

    submission = pd.DataFrame({'id': test_feats['id'].values, 'prediction': test_preds})
    submission.to_csv(OUTPUT_SUBMISSION, index=False)
    log(f"Saved submission to {OUTPUT_SUBMISSION}")

if __name__ == "__main__":
    main()
