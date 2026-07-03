"""
Data loading, preprocessing, and text embedding utilities for Hy-LIN.
"""
import json
import os
import numpy as np
from collections import defaultdict


def load_raw_data(dataset_name):
    """Load raw JSON data for a given dataset."""
    data_path = os.path.join('data', f'{dataset_name}.json')
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def filter_and_build_vocab(data, min_bugs_per_dev=5):
    """
    Filter out unassigned bugs and low-activity developers.
    Returns filtered data and developer vocabulary.
    """
    # Count bugs per developer
    dev_counts = defaultdict(int)
    for item in data:
        dev_counts[item['owner']] += 1

    # Filter out unassigned / low-activity developers
    valid_devs = {dev for dev, cnt in dev_counts.items()
                  if cnt >= min_bugs_per_dev
                  and 'nobody' not in dev.lower()
                  and 'general' not in dev.lower()
                  and 'bugs@' not in dev.lower()}

    filtered = [item for item in data if item['owner'] in valid_devs]

    # Build vocabulary
    dev_list = sorted(valid_devs)
    dev2idx = {dev: idx for idx, dev in enumerate(dev_list)}
    idx2dev = {idx: dev for dev, idx in dev2idx.items()}

    return filtered, dev2idx, idx2dev


def build_bug_texts(data):
    """Combine title and description into bug text."""
    texts = []
    for item in data:
        text = item['issue_title']
        if item.get('description'):
            text += ' ' + item['description']
        texts.append(text)
    return texts


def compute_or_load_embeddings(texts, dataset_name, emb_dim=128,
                               cache_dir='data/embeddings', use_sbert=False):
    """
    Compute text embeddings. Uses TF-IDF + TruncatedSVD by default for
    efficiency and offline-compatibility. Optionally uses SBERT.

    Args:
        texts: list of strings
        dataset_name: name for caching
        emb_dim: embedding dimension (used for SVD)
        cache_dir: cache directory
        use_sbert: if True, try to use SBERT (requires network + GPU)

    Returns:
        numpy array [N, emb_dim]
    """
    os.makedirs(cache_dir, exist_ok=True)

    if use_sbert:
        cache_path = os.path.join(cache_dir, f'{dataset_name}_emb_sbert.npy')
    else:
        cache_path = os.path.join(cache_dir, f'{dataset_name}_emb_tfidf{emb_dim}.npy')

    if os.path.exists(cache_path):
        print(f'Loading cached embeddings from {cache_path}')
        return np.load(cache_path)

    if use_sbert:
        print(f'Computing SBERT embeddings for {len(texts)} texts...')
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts, batch_size=256,
                                  show_progress_bar=True, convert_to_numpy=True)
        emb_dim_actual = embeddings.shape[1]
        cache_path = os.path.join(cache_dir, f'{dataset_name}_emb_sbert.npy')
    else:
        print(f'Computing TF-IDF + SVD ({emb_dim}d) embeddings for {len(texts)} texts...')
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        # TF-IDF with sublinear scaling and reasonable vocabulary
        vectorizer = TfidfVectorizer(
            max_features=20000,
            sublinear_tf=True,
            stop_words='english',
            strip_accents='unicode',
            dtype=np.float32,
        )
        tfidf = vectorizer.fit_transform(texts)
        print(f'  TF-IDF shape: {tfidf.shape}')

        # SVD for dimensionality reduction
        svd_dim = min(emb_dim, tfidf.shape[1] - 1)
        svd = TruncatedSVD(n_components=svd_dim, random_state=42)
        embeddings = svd.fit_transform(tfidf)
        print(f'  SVD explained variance: {svd.explained_variance_ratio_.sum():.3f}')
        emb_dim_actual = svd_dim

        # If needed, pad with zeros to match emb_dim
        if emb_dim_actual < emb_dim:
            pad = np.zeros((embeddings.shape[0], emb_dim - emb_dim_actual),
                          dtype=np.float32)
            embeddings = np.concatenate([embeddings, pad], axis=1)

    np.save(cache_path, embeddings)
    print(f'Saved to {cache_path}')
    return embeddings


def split_data(num_samples, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Generate train/val/test indices with random split."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(num_samples)

    train_end = int(num_samples * train_ratio)
    val_end = int(num_samples * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return train_idx, val_idx, test_idx


def build_adjacency(num_bugs, num_devs, bug_ids, dev_ids, edge_weights=None):
    """Build sparse adjacency matrix (bug->dev) as scipy csr_matrix."""
    from scipy.sparse import csr_matrix
    if edge_weights is None:
        edge_weights = np.ones(len(bug_ids), dtype=np.float32)
    return csr_matrix((edge_weights, (bug_ids, dev_ids)),
                      shape=(num_bugs, num_devs), dtype=np.float32)


def prepare_dataset(dataset_name, min_bugs_per_dev=5, train_ratio=0.8,
                    val_ratio=0.1, seed=42, emb_dim=128):
    """
    Full preprocessing pipeline for a dataset.
    """
    print(f'\n{"="*60}')
    print(f'Preparing dataset: {dataset_name}')
    print(f'{"="*60}')

    # Load and filter
    raw_data = load_raw_data(dataset_name)
    print(f'Raw samples: {len(raw_data)}')
    filtered, dev2idx, idx2dev = filter_and_build_vocab(raw_data, min_bugs_per_dev)
    print(f'After filtering: {len(filtered)} bugs, {len(dev2idx)} developers')

    # Build texts and compute embeddings
    texts = build_bug_texts(filtered)
    X_b = compute_or_load_embeddings(texts, dataset_name, emb_dim=emb_dim)

    # Build assignments
    bug_ids = []
    dev_ids = []
    for i, item in enumerate(filtered):
        bug_ids.append(i)
        dev_ids.append(dev2idx[item['owner']])

    # Split
    num_bugs = len(filtered)
    num_devs = len(dev2idx)
    train_idx, val_idx, test_idx = split_data(num_bugs, train_ratio, val_ratio, seed)

    # Build adjacency matrices
    bug_ids = np.array(bug_ids)
    dev_ids = np.array(dev_ids)

    train_adj = build_adjacency(num_bugs, num_devs,
                                 bug_ids[train_idx], dev_ids[train_idx])
    val_adj = build_adjacency(num_bugs, num_devs,
                               bug_ids[val_idx], dev_ids[val_idx])
    test_adj = build_adjacency(num_bugs, num_devs,
                                bug_ids[test_idx], dev_ids[test_idx])

    emb_dim = X_b.shape[1]
    print(f'Embedding dim: {emb_dim}')
    print(f'Train edges: {train_adj.nnz}, Val edges: {val_adj.nnz}, Test edges: {test_adj.nnz}')

    # Build train_devs array: [N_b], dev index per bug in training, -1 if not in train
    train_devs = np.full(num_bugs, -1, dtype=np.int64)
    train_devs[bug_ids[train_idx]] = dev_ids[train_idx]

    return {
        'X_b': X_b,
        'train_adj': train_adj,
        'val_adj': val_adj,
        'test_adj': test_adj,
        'num_bugs': num_bugs,
        'num_devs': num_devs,
        'emb_dim': emb_dim,
        'dev2idx': dev2idx,
        'idx2dev': idx2dev,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'train_devs': train_devs,
    }
