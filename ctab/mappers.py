"""
ctab.mappers
============
Neuro-symbolic ATT&CK technique mappers, hierarchical classifiers, and evaluation.
Reproduces Cell 8 and Cell 9 of Project2-CyberThreatAgent_Bench.ipynb.
"""

import os
import re
import json
import time
import pickle
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score

CACHE_DIR = Path(__file__).resolve().parent.parent / "ctab_bench" / "cache"
CACHE = CACHE_DIR

def log(*args):
    pass

_DENSE_ALERT_EMBED_CACHE = {}
_SECBERT_ALERT_EMBED_CACHE = {}

def technique_document(row, use_name=True, use_description=True):
    name = row["name"] if use_name else ""
    return " ".join([name, name, (row["name"] or "").replace("-", " ") if use_name else "",
                     " ".join(row["tactics"]), row["description"] if use_description else ""]).lower()


def build_technique_docs(df_tech, use_name=True, use_description=True):
    docs = [technique_document(r, use_name, use_description) for _, r in df_tech.iterrows()]
# never hand an all-empty corpus to a vectorizer (the masked-ATT&CK probe does exactly that)
    return [d if str(d).strip() else "zzzy_empty_document" for d in docs]


class ZeroShotMapper:
    """
    The baseline zero-shot mapper, vectorised and parameterised so it can be built with the technique names and/or
    descriptions removed. Those two switches are what the leakage probe in cell 11 uses.
    """

    def __init__(self, df_tech, use_name=True, use_description=True):
        self.df = df_tech.reset_index(drop=True)
        self.use_name = use_name
        docs = build_technique_docs(self.df, use_name, use_description)
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english",
                                         ngram_range=(1, 2), max_features=80000, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(docs)
# --- N4: precompute the lexical overlap matrix ONCE instead of per-alert iterrows ---
        if use_name:
            name_corpus = [str(r["name"]).lower() for _, r in self.df.iterrows()]
            name_corpus = [d if d.strip() else "zzzy_empty_name" for d in name_corpus]
            self.name_vec = TfidfVectorizer(lowercase=True, analyzer="word",
                                            token_pattern=r"[a-z]{3,}", binary=True)
            self.name_matrix = self.name_vec.fit_transform(name_corpus)
            self.name_totals = np.asarray(self.name_matrix.sum(axis=1)).ravel()
            self.name_totals[self.name_totals == 0] = 1.0
        else:
# masked condition: no name channel at all, by construction
            self.name_vec = None
            self.name_matrix = None
            self.name_totals = None

    def scores(self, text):
        clean = str(text).lower()
        sem = cosine_similarity(self.vectorizer.transform([clean]), self.matrix)[0]
        if self.name_vec is None:
            return sem
        q = self.name_vec.transform([clean])
        if q.nnz:
            overlap = np.asarray((self.name_matrix @ q.T).todense()).ravel() / self.name_totals
        else:
            overlap = np.zeros(self.name_matrix.shape[0])
        return 0.75 * sem + 0.25 * overlap

    def ranked(self, text, top_k=None):
        sc = self.scores(text)
        order = np.argsort(-sc, kind="mergesort")
        out = []
        for i in order[:top_k] if top_k else order:
            r = self.df.iloc[i]
            out.append({"attack_id": r["attack_id"], "name": r["name"],
                        "tactics": list(r["tactics"]), "score": float(round(float(sc[i]), 6))})
        return out


class HierarchicalMapper:
    """
    N2 Upgraded Hierarchical Mapper:
    1. Fast 3-fold inner group-CV on precomputed score matrices strictly inside train_df to pick alpha* in <0.05s.
    2. Confidence-adaptive query weight alpha_eff(x) based on supervised margin & query-to-train similarity.
    3. Unsuppressed zero-shot child scoring under predicted parents to prevent catastrophic OOD collapse on unseen sub-techniques.
    """
    def __init__(self, df_tech, train_df, alpha=0.50, C=8.0,
                 use_name=True, use_description=True, auto_tune_alpha=True, tau=0.35, tau_abstain=0.12):
        self.alpha = alpha
        self.tau = tau
        self.tau_abstain = tau_abstain
        self.df = df_tech.reset_index(drop=True)
        self.children = defaultdict(list)
        for _, r in self.df.iterrows():
            self.children[r["parent_id"]].append(r["attack_id"])
        self.zero = ZeroShotMapper(self.df, use_name=use_name, use_description=use_description)
        self.tech_by_id = {r["attack_id"]: r for _, r in self.df.iterrows()}
        self.TECH_IDS = list(self.df["attack_id"])
        self._idx = {a: i for i, a in enumerate(self.TECH_IDS)}
        self.n_queries = 0
        self.n_fallbacks = 0

        train = train_df.copy()
        train["parent"] = train["attack_id"].map(lambda a: str(a).split(".")[0])
        docs = train["text"].astype(str).str.lower().tolist()
        corpus = build_technique_docs(self.df, use_name, use_description) + docs
        self.vec = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2),
                                   max_features=25000, sublinear_tf=True)
        self.vec.fit(corpus)
        self.Xtr = self.vec.transform(docs)

        self.parent_clf, self.parent_classes = None, []
        if train["parent"].nunique() >= 2:
            self.parent_clf = LogisticRegression(max_iter=2000, C=C)
            self.parent_clf.fit(self.Xtr, train["parent"].astype(str))
            self.parent_classes = list(self.parent_clf.classes_)

        self.sub_clf, self.sub_classes = None, []
        submask = train["attack_id"].astype(str).str.contains(r"\.")
        if int(submask.sum()) >= 4 and train.loc[submask, "attack_id"].nunique() >= 2:
            self.sub_clf = LogisticRegression(max_iter=2000, C=C)
            self.sub_clf.fit(self.Xtr[submask.values], train.loc[submask, "attack_id"].astype(str))
            self.sub_classes = list(self.sub_clf.classes_)

        # Fast inner-CV hyperparameter tuning for alpha*
        self.best_alpha = self.alpha
        if auto_tune_alpha and len(train_df) >= 15:
            self.best_alpha = self._tune_alpha_inner_cv(train)
            self.alpha = self.best_alpha

    def _tune_alpha_inner_cv(self, train, candidate_alphas=None):
        if candidate_alphas is None:
            candidate_alphas = [0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]
        n = len(train)
        if n < 15:
            return 0.50

        # Precomputed zero-shot score matrix on train alerts
        Z_mat = np.zeros((n, len(self.TECH_IDS)), dtype=np.float32)
        for i, text in enumerate(train["text"]):
            z = self.zero.scores(text)
            zm = z.max()
            if zm > 0:
                Z_mat[i] = z / zm

        # Vectorized supervised scores on Xtr
        S_mat = np.zeros((n, len(self.TECH_IDS)), dtype=np.float32)
        c_sup = np.zeros(n, dtype=np.float32)

        if self.parent_clf is not None:
            pp_mat = self.parent_clf.predict_proba(self.Xtr)
            for c_i, c in enumerate(self.parent_classes):
                c_idx = self._idx.get(c)
                if c_idx is not None:
                    S_mat[:, c_idx] += pp_mat[:, c_i]
                p_children = getattr(self, "children", {}).get(c, [])
                for aid in p_children:
                    aid_idx = self._idx.get(aid)
                    if aid_idx is not None:
                        S_mat[:, aid_idx] += pp_mat[:, c_i] * (0.5 + 0.5 * Z_mat[:, aid_idx])

            if self.sub_clf is not None:
                sp_mat = self.sub_clf.predict_proba(self.Xtr)
                for s_i, sc_label in enumerate(self.sub_classes):
                    sc_idx = self._idx.get(sc_label)
                    if sc_idx is not None:
                        S_mat[:, sc_idx] += sp_mat[:, s_i]

            s_max = S_mat.max(axis=1, keepdims=True)
            s_max[s_max == 0] = 1.0
            S_mat = S_mat / s_max

            # Nearest-train cosine similarity (excluding self)
            sims = cosine_similarity(self.Xtr, self.Xtr)
            np.fill_diagonal(sims, 0.0)
            max_sims = sims.max(axis=1)

            max_pp = pp_mat.max(axis=1)
            u_floor = 1.0 / len(self.parent_classes)
            c_p = np.clip((max_pp - u_floor) / max(1e-4, 1.0 - u_floor), 0.0, 1.0)
            c_sup = c_p * np.clip(max_sims / 0.25, 0.0, 1.0)

        # 3-fold inner split
        folds = np.array_split(np.arange(n), 3)
        gt_indices = np.array([self._idx.get(aid, -1) for aid in train["attack_id"]])

        best_alpha = 0.50
        best_mrr = -1.0
        for cand_a in candidate_alphas:
            a_eff = cand_a * np.clip(c_sup / self.tau, 0.05, 1.0)[:, None]
            B = a_eff * S_mat + (1.0 - a_eff) * Z_mat

            mrrs = []
            for fold in folds:
                B_fold = B[fold]
                gt_fold = gt_indices[fold]
                valid_mask = gt_fold >= 0
                if not np.any(valid_mask):
                    continue
                B_valid = B_fold[valid_mask]
                gt_valid = gt_fold[valid_mask]
                gt_scores = B_valid[np.arange(len(gt_valid)), gt_valid][:, None]
                ranks = 1 + np.sum(B_valid > gt_scores, axis=1)
                mrrs.extend(1.0 / ranks)

            mean_mrr = float(np.mean(mrrs)) if mrrs else 0.0
            if mean_mrr > best_mrr:
                best_mrr = mean_mrr
                best_alpha = cand_a

        return best_alpha

    def _effective_alpha(self, q):
        if not q.nnz:
            return 0.05 * self.alpha
        sims = cosine_similarity(q, self.Xtr)[0]
        max_sim = float(sims.max()) if len(sims) else 0.0
        c_parent = 0.0
        if self.parent_clf is not None:
            pp = self.parent_clf.predict_proba(q)[0]
            u_floor = 1.0 / len(self.parent_classes)
            c_parent = max(0.0, float(pp.max()) - u_floor) / max(1e-4, 1.0 - u_floor)
        c_sup = c_parent * min(1.0, max_sim / 0.25)
        return float(self.alpha * np.clip(c_sup / self.tau, 0.05, 1.0))

    def _supervised_scores(self, q, z_norm=None):
        sup = np.zeros(len(self.TECH_IDS), dtype=np.float32)
        if q.nnz and self.parent_clf is not None:
            pp = self.parent_clf.predict_proba(q)[0]
            for c, p in zip(self.parent_classes, pp):
                if c in self._idx:
                    sup[self._idx[c]] += p
                p_children = getattr(self, "children", {}).get(c, [])
                if p_children:
                    for aid in p_children:
                        if aid in self._idx:
                            z_weight = float(z_norm[self._idx[aid]]) if z_norm is not None else 1.0
                            sup[self._idx[aid]] += p * (0.5 + 0.5 * z_weight)
            if self.sub_clf is not None:
                sp = self.sub_clf.predict_proba(q)[0]
                for c, p in zip(self.sub_classes, sp):
                    if c in self._idx:
                        sup[self._idx[c]] += p
        tot = sup.max()
        return sup / tot if tot else sup

    def scores(self, text):
        q = self.vec.transform([str(text).lower()])
        z = self.zero.scores(text)
        zm = z.max()
        z_norm = (z / zm) if zm else z
        sup = self._supervised_scores(q, z_norm=z_norm)
        a_eff = self._effective_alpha(q)
        return a_eff * sup + (1.0 - a_eff) * z_norm

    @property
    def fallback_rate(self):
        return self.n_fallbacks / self.n_queries if self.n_queries > 0 else 0.0

    def log_fallback_rate(self):
        log("HierarchicalMapper parent fallback rate: %.4f (%d/%d)" % (self.fallback_rate, self.n_fallbacks, self.n_queries))
        return self.fallback_rate

    def _apply_parent_fallback(self, out):
        self.n_queries += 1
        if out and "." in str(out[0]["attack_id"]) and out[0]["score"] < 0.40:
            self.n_fallbacks += 1
            top_score = out[0]["score"]
            parent_id = str(out[0]["attack_id"]).split(".")[0]
            if parent_id in self.tech_by_id:
                p_idx = next((i for i, x in enumerate(out) if x["attack_id"] == parent_id), None)
                if p_idx is not None:
                    parent_item = out.pop(p_idx)
                    parent_item["score"] = max(parent_item.get("score", 0.0), top_score)
                else:
                    prow = self.tech_by_id[parent_id]
                    parent_item = {
                        "attack_id": parent_id,
                        "name": prow["name"],
                        "tactics": list(prow["tactics"]),
                        "score": top_score,
                    }
                out.insert(0, parent_item)
        return out

    def confidence(self, text):
        """
        Composite confidence score combining:
        1. Cosine similarity to nearest training alert (0.50)
        2. Maximum zero-shot similarity to ATT&CK technique catalog (0.30)
        3. Supervised parent-classifier margin above uniform chance floor (0.20)
        """
        q = self.vec.transform([str(text).lower()])
        train_sim = float((self.Xtr @ q.T).toarray().ravel().max()) if q.nnz else 0.0
        z = self.zero.scores(text)
        raw_z_max = float(z.max()) if len(z) else 0.0
        c_parent = 0.0
        if self.parent_clf is not None and q.nnz:
            pp = self.parent_clf.predict_proba(q)[0]
            u_floor = 1.0 / max(1, len(self.parent_classes))
            c_parent = max(0.0, float(pp.max()) - u_floor) / max(1e-4, 1.0 - u_floor)
        return float(0.50 * train_sim + 0.30 * raw_z_max + 0.20 * c_parent)

    def ranked(self, text, top_k=None, allow_abstain=True):
        if allow_abstain and self.confidence(text) < getattr(self, "tau_abstain", 0.12):
            return [{"attack_id": None, "name": "Abstain (Benign/OOD)", "score": 0.0, "tactics": []}]
        out = self._ranked_from(self.scores(text), top_k=None)
        out = self._apply_parent_fallback(out)
        return out[:top_k] if top_k else out

    def ranked_supervised(self, text, top_k=None):
        q = self.vec.transform([str(text).lower()])
        out = self._ranked_from(self._supervised_scores(q), top_k=None)
        out = self._apply_parent_fallback(out)
        return out[:top_k] if top_k else out

    def _ranked_from(self, sc, top_k=None):
        order = np.argsort(-sc, kind="mergesort")
        out = []
        for i in order[:top_k] if top_k else order:
            r = self.df.iloc[i]
            out.append({"attack_id": r["attack_id"], "name": r["name"],
                        "tactics": list(r["tactics"]), "score": float(round(float(sc[i]), 6))})
        return out

class KeywordMapper:
    """Frequency baseline W07: rank techniques by name-token overlap only, no TF-IDF."""

    def __init__(self, df_tech):
        self.df = df_tech.reset_index(drop=True)
        self.tokens = [set(re.findall(r"[a-z]{3,}", r["name"].lower()))
                       for _, r in self.df.iterrows()]

    def scores(self, text):
        toks = set(re.findall(r"[a-z]{3,}", str(text).lower()))
        out = np.zeros(len(self.df))
        for i, t in enumerate(self.tokens):
            if t:
                out[i] = len(toks & t) / len(t)
        return out

    def ranked(self, text, top_k=None):
        sc = self.scores(text)
        order = np.argsort(-sc, kind="mergesort")
        out = []
        for i in order[:top_k] if top_k else order:
            r = self.df.iloc[i]
            out.append({"attack_id": r["attack_id"], "name": r["name"],
                        "tactics": list(r["tactics"]), "score": float(round(float(sc[i]), 6))})
        return out


class RandomMapper:
    def __init__(self, df_tech, seed=0):
        self.ids = list(df_tech["attack_id"])
        self.name = dict(zip(df_tech["attack_id"], df_tech["name"]))
        self.rng = np.random.RandomState(seed)

    def ranked(self, text, top_k=None):
        perm = self.rng.permutation(len(self.ids))
        out = [{"attack_id": self.ids[i], "name": self.name[self.ids[i]], "score": 0.0}
               for i in perm[:top_k] if top_k]
        return out


class MajorityMapper:
    def __init__(self, train_df):
        self.ids = list(train_df["attack_id"].value_counts().index)

    def ranked(self, text, top_k=None):
        return [{"attack_id": a, "name": NAME_OF.get(a, ""), "score": 0.0}
                for a in self.ids[:top_k] if top_k]


def technique_match(pred_id, gt_id):
    """exact / parent-child / sibling (same parent) / miss -- three separate metrics downstream."""
    if pred_id == gt_id:
        return "exact"
    pp, gp = str(pred_id).split(".")[0], str(gt_id).split(".")[0]
    if pred_id == gp or gt_id == pp:     # one is the parent technique of the other
        return "parent"
    if pp == gp:
        return "sibling"
    return "miss"




class BM25Mapper:
    """
    Baseline: Standard Okapi BM25 over ATT&CK technique documents.
    Implements the exact same .ranked(text, top_k=5) interface as ZeroShotMapper.
    """
    def __init__(self, df_tech, use_name=True, use_description=True):
        self.df = df_tech.reset_index(drop=True)
        self.use_name = use_name
        docs = build_technique_docs(self.df, use_name, use_description)
        self.corpus_tokens = [re.findall(r"[a-z0-9]+", d.lower()) for d in docs]
        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.corpus_tokens)
            self._available = True
        except ImportError:
            self.bm25 = None
            self._available = False

    def scores(self, text):
        if not self._available or self.bm25 is None:
            q_toks = set(re.findall(r"[a-z0-9]+", str(text).lower()))
            return np.array([len(q_toks & set(doc)) / max(1, len(set(doc))) for doc in self.corpus_tokens])
        q_toks = re.findall(r"[a-z0-9]+", str(text).lower())
        if not q_toks:
            return np.zeros(len(self.df))
        raw_scores = np.array(self.bm25.get_scores(q_toks))
        max_s = raw_scores.max()
        return raw_scores / max_s if max_s > 0 else raw_scores

    def ranked(self, text, top_k=None):
        sc = self.scores(text)
        order = np.argsort(-sc, kind="mergesort")
        out = []
        for i in order[:top_k] if top_k else order:
            r = self.df.iloc[i]
            out.append({"attack_id": r["attack_id"], "name": r["name"],
                        "tactics": list(r["tactics"]), "score": float(round(float(sc[i]), 6))})
        return out


class DenseRetrieverMapper:
    """
    Baseline: Dense embedding retriever over ATT&CK technique documents.
    Uses BAAI/bge-small-en-v1.5 (with fallback to all-MiniLM-L6-v2).
    Precomputes and caches technique embeddings to CACHE/ for efficiency.
    Implements the exact same .ranked(text, top_k=5) interface as ZeroShotMapper.
    """
    _cached_model = None
    _cached_model_name = None

    @classmethod
    def _get_model(cls):
        if cls._cached_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                for name in ["BAAI/bge-small-en-v1.5", "all-MiniLM-L6-v2"]:
                    try:
                        cls._cached_model = SentenceTransformer(name)
                        cls._cached_model_name = name
                        break
                    except Exception:
                        continue
            except Exception:
                cls._cached_model = None
                cls._cached_model_name = None
        return cls._cached_model

    def __init__(self, df_tech, use_name=True, use_description=True, cache_dir=None):
        self.df = df_tech.reset_index(drop=True)
        self.use_name = use_name
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE
        self.model = None
        self.doc_embeddings = None
        self._available = False

        docs = build_technique_docs(self.df, use_name, use_description)
        self._init_model_and_embed(docs)

    def _init_model_and_embed(self, docs):
        try:
            self.model = self._get_model()
            if self.model is None:
                self._available = False
                return
            self.model_name = self._cached_model_name or "BAAI/bge-small-en-v1.5"
            model_name = self.model_name

            model_slug = self.model_name.replace("/", "_")
            docs_hash = hashlib.sha256(("".join(docs)).encode()).hexdigest()[:16]
            cache_file = self.cache_dir / f"dense_emb_{model_slug}_{docs_hash}.npy"

            if cache_file.exists():
                self.doc_embeddings = np.load(cache_file)
            else:
                self.doc_embeddings = self.model.encode(docs, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
                np.save(cache_file, self.doc_embeddings)

            self._available = True
        except Exception:
            self._available = False
            self.model = None
            self.doc_embeddings = None

    def scores(self, text):
        if not self._available or self.model is None or self.doc_embeddings is None:
            return np.zeros(len(self.df))
        if not hasattr(self, "_query_cache"):
            self._query_cache = {}
        txt_str = str(text)
        if txt_str in self._query_cache:
            q_emb = self._query_cache[txt_str]
        else:
            q_emb = self.model.encode([txt_str], show_progress_bar=False, normalize_embeddings=True)[0]
            self._query_cache[txt_str] = q_emb
        sims = np.dot(self.doc_embeddings, q_emb)
        return np.clip(sims, 0.0, 1.0)

    def ranked(self, text, top_k=None):
        sc = self.scores(text)
        order = np.argsort(-sc, kind="mergesort")
        out = []
        for i in order[:top_k] if top_k else order:
            r = self.df.iloc[i]
            out.append({"attack_id": r["attack_id"], "name": r["name"],
                        "tactics": list(r["tactics"]), "score": float(round(float(sc[i]), 6))})
        return out



_DENSE_ALERT_EMBED_CACHE = {}


def _get_alert_embedding_cache(cache_dir=None):
    global _DENSE_ALERT_EMBED_CACHE
    c_dir = Path(cache_dir) if cache_dir else CACHE
    pkl_file = c_dir / "dense_alert_embeddings.pkl"
    if not _DENSE_ALERT_EMBED_CACHE and pkl_file.exists():
        try:
            with open(pkl_file, "rb") as f:
                loaded = pickle.load(f)
                if isinstance(loaded, dict):
                    _DENSE_ALERT_EMBED_CACHE.update(loaded)
        except Exception:
            pass
    return _DENSE_ALERT_EMBED_CACHE


def _save_alert_embedding_cache(cache_dir=None):
    global _DENSE_ALERT_EMBED_CACHE
    c_dir = Path(cache_dir) if cache_dir else CACHE
    c_dir.mkdir(parents=True, exist_ok=True)
    pkl_file = c_dir / "dense_alert_embeddings.pkl"
    try:
        with open(pkl_file, "wb") as f:
            pickle.dump(_DENSE_ALERT_EMBED_CACHE, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass


def _embed_alert_texts(texts, model, cache_dir=None):
    cache = _get_alert_embedding_cache(cache_dir)
    texts_str = [str(t) for t in texts]
    hashes = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts_str]
    missing = [i for i, h in enumerate(hashes) if h not in cache]
    if missing:
        if model is not None:
            new_embs = model.encode([texts_str[i] for i in missing], batch_size=64,
                                    show_progress_bar=False, normalize_embeddings=True)
            for i, emb in zip(missing, new_embs):
                cache[hashes[i]] = np.asarray(emb, dtype=np.float32)
            _save_alert_embedding_cache(cache_dir)
        else:
            raise RuntimeError(
                "[ERROR] Dense sentence transformer model is None! Cannot compute alert embeddings. "
                "Ensure sentence-transformers is installed and models (BAAI/bge-small-en-v1.5 or all-MiniLM-L6-v2) can load."
            )
    return np.array([cache[h] for h in hashes], dtype=np.float32)


class SupervisedDenseMapper:
    """
    Supervised Dense Baseline:
    Encodes alert texts using dense embeddings (BAAI/bge-small-en-v1.5 or fallback all-MiniLM-L6-v2)
    via DenseRetrieverMapper._get_model().
    Caches alert embeddings to CACHE / 'dense_alert_embeddings.pkl' (keyed by SHA256).
    Fits LogisticRegression(max_iter=2000, C=C, class_weight="balanced") on dense representations.
    """
    def __init__(self, df_tech, train_df, C=8.0, cache_dir=None):
        self.df = df_tech.reset_index(drop=True)
        self.tech_by_id = {r["attack_id"]: r for _, r in self.df.iterrows()}
        self.TECH_IDS = list(self.df["attack_id"])
        self._idx = {a: i for i, a in enumerate(self.TECH_IDS)}
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE
        self.model = DenseRetrieverMapper._get_model()

        train = train_df.copy()
        train["parent"] = train["attack_id"].map(lambda a: str(a).split(".")[0])
        docs = train["text"].astype(str).tolist()
        Xtr = _embed_alert_texts(docs, self.model, self.cache_dir)
        norm_Xtr = float(np.linalg.norm(Xtr))
        m_name = getattr(DenseRetrieverMapper, "_cached_model_name", None) or "bge-small"
        log("  [SupervisedDenseMapper] model: %s | Xtr shape: %s | non-zero L2 norm: %.4f"
            % (m_name, Xtr.shape, norm_Xtr))
        if norm_Xtr == 0.0:
            raise ValueError("[ERROR] SupervisedDenseMapper Xtr embedding norm is 0.0! Dense embeddings failed to generate.")

        self.parent_clf, self.parent_classes = None, []
        if train["parent"].nunique() >= 2:
            self.parent_clf = LogisticRegression(max_iter=2000, C=C, class_weight="balanced")
            self.parent_clf.fit(Xtr, train["parent"].astype(str))
            self.parent_classes = list(self.parent_clf.classes_)

        self.sub_clf, self.sub_classes = None, []
        submask = train["attack_id"].astype(str).str.contains(r"\.")
        if int(submask.sum()) >= 4 and train.loc[submask, "attack_id"].nunique() >= 2:
            self.sub_clf = LogisticRegression(max_iter=2000, C=C, class_weight="balanced")
            self.sub_clf.fit(Xtr[submask.values], train.loc[submask, "attack_id"].astype(str))
            self.sub_classes = list(self.sub_clf.classes_)

    def _supervised_scores(self, text):
        emb = _embed_alert_texts([text], self.model, self.cache_dir)[0]
        sup = np.zeros(len(self.TECH_IDS))
        if self.parent_clf is not None:
            pp = self.parent_clf.predict_proba([emb])[0]
            for c, p in zip(self.parent_classes, pp):
                for aid in [a for a in self.TECH_IDS if a == c or a.startswith(c + ".")]:
                    if aid in self._idx:
                        sup[self._idx[aid]] += p
            if self.sub_clf is not None:
                sp = self.sub_clf.predict_proba([emb])[0]
                for c, p in zip(self.sub_classes, sp):
                    if c in self._idx:
                        sup[self._idx[c]] += p
        tot = sup.max()
        return sup / tot if tot else sup

    def scores(self, text):
        return self._supervised_scores(text)

    def ranked(self, text, top_k=5):
        sc = self.scores(text)
        order = np.argsort(-sc, kind="mergesort")
        out = []
        for i in order[:top_k]:
            r = self.df.iloc[i]
            out.append({"attack_id": r["attack_id"], "name": r["name"],
                        "tactics": list(r["tactics"]), "score": float(sc[i])})
        return out[:top_k] if top_k else out


class HybridDenseRouter:
    """
    Confidence-Gated Agentic Router (Top-2 Margin Uncertainty Trigger):
    Runs the fast HierarchicalMapper first; ONLY escalates to SupervisedDenseMapper
    (or DenseRetrieverMapper if unavailable) whenever (top1_score - top2_score) < margin_threshold.
    """

    def __init__(self, fast_mapper, dense_mapper, margin_threshold=0.10):
        self.fast_mapper = fast_mapper
        self.dense_mapper = dense_mapper
        self.margin_threshold = margin_threshold
        self.n_queries = 0
        self.n_escalated = 0

    def ranked(self, text, top_k=5):
        self.n_queries += 1
        res = self.fast_mapper.ranked(text, top_k=max(2, top_k))
        top1_score = res[0]["score"] if len(res) > 0 else 0.0
        top2_score = res[1]["score"] if len(res) > 1 else 0.0
        margin = top1_score - top2_score
        if margin < self.margin_threshold and self.dense_mapper is not None:
            self.n_escalated += 1
            return self.dense_mapper.ranked(text, top_k=top_k)
        return res[:top_k]

    @property
    def escalation_rate(self):
        return (self.n_escalated / self.n_queries) if self.n_queries else 0.0

    def reset_stats(self):
        self.n_queries = 0
        self.n_escalated = 0


# ============================================================
# CYBERSECURITY TRANSFORMER BASELINES (CPU + SHA256 DISK CACHE)
# ============================================================
_SECBERT_CACHE = {}
_SECBERT_MODEL = None
_SECBERT_TOK = None
_SECBERT_MODEL_NAME = "jackaduma/SecBERT"


def _get_secbert_cache(cache_dir=None):
    global _SECBERT_CACHE
    c_dir = Path(cache_dir) if cache_dir else CACHE
    pkl_file = c_dir / "secbert_embeddings.pkl"
    if not _SECBERT_CACHE and pkl_file.exists():
        try:
            with open(pkl_file, "rb") as f:
                loaded = pickle.load(f)
                if isinstance(loaded, dict):
                    _SECBERT_CACHE.update(loaded)
        except Exception:
            pass
    return _SECBERT_CACHE


def _save_secbert_cache(cache_dir=None):
    global _SECBERT_CACHE
    c_dir = Path(cache_dir) if cache_dir else CACHE
    c_dir.mkdir(parents=True, exist_ok=True)
    pkl_file = c_dir / "secbert_embeddings.pkl"
    try:
        with open(pkl_file, "wb") as f:
            pickle.dump(_SECBERT_CACHE, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass


def _get_secbert_model():
    global _SECBERT_MODEL, _SECBERT_TOK
    if _SECBERT_MODEL is None:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            _SECBERT_TOK = AutoTokenizer.from_pretrained(_SECBERT_MODEL_NAME)
            _SECBERT_MODEL = AutoModel.from_pretrained(_SECBERT_MODEL_NAME)
            _SECBERT_MODEL.eval()
        except Exception as e:
            log(f"Warning: Failed to load {_SECBERT_MODEL_NAME} ({e})")
            _SECBERT_MODEL = None
            _SECBERT_TOK = None
    return _SECBERT_MODEL, _SECBERT_TOK


def _secbert_mean_pooling(model_output, attention_mask):
    import torch
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def embed_texts_secbert(texts, cache_dir=None, batch_size=64):
    import torch
    cache = _get_secbert_cache(cache_dir)
    model, tok = _get_secbert_model()
    texts_str = [str(t) for t in texts]
    hashes = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts_str]
    missing = [i for i, h in enumerate(hashes) if (_SECBERT_MODEL_NAME, h) not in cache]

    if missing and model is not None and tok is not None:
        for b_start in range(0, len(missing), batch_size):
            b_idx = missing[b_start:b_start + batch_size]
            b_texts = [texts_str[i] for i in b_idx]
            with torch.inference_mode():
                encoded = tok(b_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
                out = model(**encoded)
                emb = _secbert_mean_pooling(out, encoded["attention_mask"])
                emb = torch.nn.functional.normalize(emb, p=2, dim=1).cpu().numpy().astype(np.float32)
            for orig_i, vec in zip(b_idx, emb):
                cache[(_SECBERT_MODEL_NAME, hashes[orig_i])] = vec
        _save_secbert_cache(cache_dir)

    return np.array([cache.get((_SECBERT_MODEL_NAME, h), np.zeros(768, dtype=np.float32)) for h in hashes], dtype=np.float32)


class SecDenseRetrieverMapper:
    """Zero-shot domain bi-encoder retrieval using SecBERT with SHA-256 disk caching."""
    def __init__(self, df_tech, use_name=True, use_description=True, cache_dir=None):
        self.df = df_tech.reset_index(drop=True)
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE
        docs = build_technique_docs(self.df, use_name, use_description)
        docs_hash = hashlib.sha256(("".join(docs)).encode()).hexdigest()[:16]
        cache_file = self.cache_dir / f"secbert_doc_emb_{docs_hash}.npy"
        if cache_file.exists():
            self.doc_embeddings = np.load(cache_file)
        else:
            self.doc_embeddings = embed_texts_secbert(docs, cache_dir=self.cache_dir, batch_size=64)
            np.save(cache_file, self.doc_embeddings)
        self._query_cache = {}

    def scores(self, text):
        txt_str = str(text)
        if txt_str in self._query_cache:
            q_emb = self._query_cache[txt_str]
        else:
            q_emb = embed_texts_secbert([txt_str], cache_dir=self.cache_dir)[0]
            self._query_cache[txt_str] = q_emb
        sims = np.dot(self.doc_embeddings, q_emb)
        return np.clip(sims, 0.0, 1.0)

    def ranked(self, text, top_k=None):
        sc = self.scores(text)
        order = np.argsort(-sc, kind="mergesort")
        out = []
        for i in order[:top_k] if top_k else order:
            r = self.df.iloc[i]
            out.append({"attack_id": r["attack_id"], "name": r["name"],
                        "tactics": list(r["tactics"]), "score": float(round(float(sc[i]), 6))})
        return out


class SupervisedSecBERTMapper:
    """Supervised SecBERT Baseline: Hierarchical LogisticRegression on cached 768-d domain embeddings."""
    def __init__(self, df_tech, train_df, C=8.0, cache_dir=None):
        self.df = df_tech.reset_index(drop=True)
        self.TECH_IDS = list(self.df["attack_id"])
        self._idx = {a: i for i, a in enumerate(self.TECH_IDS)}
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE

        train = train_df.copy()
        train["parent"] = train["attack_id"].map(lambda a: str(a).split(".")[0])
        docs = train["text"].astype(str).tolist()
        self.Xtr = embed_texts_secbert(docs, cache_dir=self.cache_dir, batch_size=64)

        self.parent_clf, self.parent_classes = None, []
        if train["parent"].nunique() >= 2:
            self.parent_clf = LogisticRegression(max_iter=2000, C=C, class_weight="balanced")
            self.parent_clf.fit(self.Xtr, train["parent"].astype(str))
            self.parent_classes = list(self.parent_clf.classes_)

        self.sub_clf, self.sub_classes = None, []
        submask = train["attack_id"].astype(str).str.contains(r"\.")
        if int(submask.sum()) >= 4 and train.loc[submask, "attack_id"].nunique() >= 2:
            self.sub_clf = LogisticRegression(max_iter=2000, C=C, class_weight="balanced")
            self.sub_clf.fit(self.Xtr[submask.values], train.loc[submask, "attack_id"].astype(str))
            self.sub_classes = list(self.sub_clf.classes_)
        self._query_cache = {}

    def _supervised_scores(self, text):
        txt_str = str(text)
        if txt_str in self._query_cache:
            q_emb = self._query_cache[txt_str]
        else:
            q_emb = embed_texts_secbert([txt_str], cache_dir=self.cache_dir)[0]
            self._query_cache[txt_str] = q_emb
        q = np.array([q_emb])
        sup = np.zeros(len(self.TECH_IDS))
        if self.parent_clf is not None:
            pp = self.parent_clf.predict_proba(q)[0]
            for c, p in zip(self.parent_classes, pp):
                for aid in [a for a in self.TECH_IDS if a == c or a.startswith(c + ".")]:
                    if aid in self._idx:
                        sup[self._idx[aid]] += p
            if self.sub_clf is not None:
                sp = self.sub_clf.predict_proba(q)[0]
                for c, p in zip(self.sub_classes, sp):
                    if c in self._idx:
                        sup[self._idx[c]] += p
        tot = sup.max()
        return sup / tot if tot else sup

    def ranked(self, text, top_k=None):
        sc = self._supervised_scores(text)
        order = np.argsort(-sc, kind="mergesort")
        out = []
        for i in order[:top_k] if top_k else order:
            r = self.df.iloc[i]
            out.append({"attack_id": r["attack_id"], "name": r["name"],
                        "tactics": list(r["tactics"]), "score": float(round(float(sc[i]), 6))})
        return out


class SecBERTHybridMapper:
    """SecBERT Hybrid Mapper: Blends SupervisedSecBERTMapper with SecDenseRetrieverMapper."""
    def __init__(self, df_tech, train_df, alpha=0.50, C=8.0, cache_dir=None):
        self.df = df_tech.reset_index(drop=True)
        self.sup = SupervisedSecBERTMapper(df_tech, train_df, C=C, cache_dir=cache_dir)
        self.dense = SecDenseRetrieverMapper(df_tech, cache_dir=cache_dir)
        self.alpha = alpha

    def scores(self, text):
        s = self.sup._supervised_scores(text)
        d = self.dense.scores(text)
        d = d / d.max() if d.max() else d
        return self.alpha * s + (1.0 - self.alpha) * d

    def ranked(self, text, top_k=None):
        sc = self.scores(text)
        order = np.argsort(-sc, kind="mergesort")
        out = []
        for i in order[:top_k] if top_k else order:
            r = self.df.iloc[i]
            out.append({"attack_id": r["attack_id"], "name": r["name"],
                        "tactics": list(r["tactics"]), "score": float(round(float(sc[i]), 6))})
        return out


def fit_mappers(df_tech, train_df, cfg, use_name=True, use_description=True, seed=None):
    """All conditions (including baselines & agentic router) fitted from one split. Returned as a dict of callables."""
    kw = {"use_name": use_name, "use_description": use_description}
    hybrid = HierarchicalMapper(df_tech, train_df, alpha=cfg.alpha,
                                C=cfg.logreg_C, auto_tune_alpha=True, **kw)
    dense = DenseRetrieverMapper(df_tech, **kw)
    supervised_dense = SupervisedDenseMapper(df_tech, train_df, C=cfg.logreg_C)
    sec_dense = SecDenseRetrieverMapper(df_tech, **kw)
    sec_hybrid = SecBERTHybridMapper(df_tech, train_df, alpha=0.50, C=cfg.logreg_C)
    s = 0 if seed is None else seed
    out = {
        "hybrid": hybrid,
        "zero_shot": ZeroShotMapper(df_tech, **kw),
        "bm25": BM25Mapper(df_tech, **kw),
        "dense": dense,
        "supervised_dense": supervised_dense,
        "sec_dense": sec_dense,
        "sec_hybrid": sec_hybrid,
        "keyword": KeywordMapper(df_tech),
        "random": RandomMapper(df_tech, seed=s),
        "majority": MajorityMapper(train_df),
    }
    dense_target = out.get("supervised_dense") if out.get("supervised_dense") is not None else out.get("dense")
    out["router"] = HybridDenseRouter(hybrid, dense_target, margin_threshold=0.10)
    AgenticRouterMapper = HybridDenseRouter
    return out


def mapper_fn(m):
    return lambda text, top_k=5: m.ranked(text, top_k=top_k)



def evaluate_mapping(df_eval, predict_fn, top_k=5):
    """Ranking + classification metrics with exact, parent, and multi-label match granularities."""
    rows = []
    for _, row in df_eval.iterrows():
        got = predict_fn(row["text"], top_k=top_k)      # called ONCE: keeps random baselines reproducible
        preds = [p["attack_id"] for p in got]
        scores = [p.get("score", 0.0) for p in got]
        gt = row["attack_id"]
        gt_parent = str(gt).split(".")[0]

        # Multi-label valid tags extraction
        raw_tags = row.get("attack_ids")
        if isinstance(raw_tags, (list, set, tuple)):
            valid_tags = set(str(t).strip().upper() for t in raw_tags if str(t).strip())
        elif isinstance(raw_tags, str) and raw_tags:
            valid_tags = {str(raw_tags).strip().upper()}
        else:
            valid_tags = {str(gt).strip().upper()}
        if not valid_tags:
            valid_tags = {str(gt).strip().upper()}

        r_exact = next((i + 1 for i, p in enumerate(preds) if p == gt), 0)
        r_par = next((i + 1 for i, p in enumerate(preds)
                      if technique_match(p, gt) in ("exact", "sibling", "parent")), 0)
        r_any = next((i + 1 for i, p in enumerate(preds) if p in valid_tags), 0)

        top1_exact = bool(preds) and preds[0] == gt
        top1_any = bool(preds) and preds[0] in valid_tags
        top3_exact = gt in preds[:3]
        top1_par = bool(preds) and str(preds[0]).split(".")[0] == gt_parent
        top3_par = any(str(p).split(".")[0] == gt_parent for p in preds[:3])
        rec3_multi = len(set(preds[:3]) & valid_tags) / len(valid_tags)

        rows.append({
            "sample_id": row.get("sample_id"),
            "Top1_Exact": top1_exact,
            "Top1_AnyValidTag": top1_any,
            "Top3_Exact": top3_exact,
            "Top1_Parent": top1_par,
            "Top3_Parent": top3_par,
            "RR_Exact": 1.0 / r_exact if r_exact else 0.0,
            "RR_AnyValidTag": 1.0 / r_any if r_any else 0.0,
            "Recall@3_MultiLabel": rec3_multi,
            "RR_Parent": 1.0 / r_par if r_par else 0.0,
            "pred1": preds[0] if preds else None,
            "gt": gt,
            "conf1": float(scores[0]) if scores else 0.0,
            "valid_tags": list(valid_tags),
        })
    d = pd.DataFrame(rows)

    labels = sorted(set(str(x) for x in d["gt"] if pd.notna(x)) |
                    set(str(x) for x in d["pred1"] if pd.notna(x) and str(x).strip()))
    pred_series = d["pred1"].fillna("UNPREDICTED_ABSTAIN").astype(str)
    gt_series = d["gt"].astype(str)
    macro = f1_score(gt_series, pred_series, labels=labels, average="macro", zero_division=0)
    micro = f1_score(gt_series, pred_series, labels=labels, average="micro", zero_division=0)

    # ECE of the normalised blend -- INDICATIVE ONLY, not a calibrated probability
    conf = d["conf1"] / d["conf1"].max() if d["conf1"].max() > 0 else d["conf1"]
    bins = np.clip((conf * 10).astype(int), 0, 9)
    ece = 0.0
    for b in range(10):
        m = bins == b
        if m.sum():
            ece += (m.sum() / len(d)) * abs(d.loc[m, "Top1_Exact"].mean() - conf[m].mean())

    return {
        "n": int(len(d)),
        "Top1_Exact": round(float(d["Top1_Exact"].mean()), 4),
        "Top1_AnyValidTag": round(float(d["Top1_AnyValidTag"].mean()), 4),
        "Top3_Exact": round(float(d["Top3_Exact"].mean()), 4),
        "Top1_Parent": round(float(d["Top1_Parent"].mean()), 4),
        "Top3_Parent": round(float(d["Top3_Parent"].mean()), 4),
        "MRR_Exact": round(float(d["RR_Exact"].mean()), 4),
        "MRR_AnyValidTag": round(float(d["RR_AnyValidTag"].mean()), 4),
        "Recall@3_MultiLabel": round(float(d["Recall@3_MultiLabel"].mean()), 4),
        "MRR_Parent": round(float(d["RR_Parent"].mean()), 4),
        "Macro_F1": round(float(macro), 4),
        "Micro_F1": round(float(micro), 4),
        "ECE_normalized_blend_indicative": round(float(ece), 4),
    }, d
