"""TF-IDF + LinearSVM(ovr, calibrated-ish)：传统文本基线。按原文规矩：标题+描述前1500字符，word 1-2gram，min_df=3，线性SVM打分。"""
import sys, argparse, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_split, save_scores, model_path, collect_cost
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

def main(dataset: str, seed: int):
    import time as _t; t0 = _t.time()
    owners, texts, _, tr, va, te = load_split(dataset, seed)
    vec = TfidfVectorizer(max_features=60000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    Xtr = vec.fit_transform([texts[i] for i in tr])
    clf = LinearSVC(C=1.0, max_iter=3000)
    clf.fit(Xtr, owners[tr])
    def scores(idx):
        X = vec.transform([texts[i] for i in idx])
        return X @ clf.coef_.T + clf.intercept_
    ste = scores(te)
    sve = scores(va)
    import numpy as np
    import joblib
    joblib.dump({"vectorizer": vec, "clf": clf, "dev_names": None}, model_path("tfidf_svm", dataset, seed, "joblib"))
    save_scores("tfidf_svm", dataset, seed, np.asarray(ste), owners[te], np.asarray(sve), owners[va],
                {"C": 1.0, "features": 60000}, cost=collect_cost(t0))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("seed", type=int)
    a = ap.parse_args(); main(a.dataset, a.seed)
