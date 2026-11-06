"""
OmniSense CSI - Biometric Gait Tracking & Live Hardware Enrollment Engine (v2.1)
Module: backend/ml_biometrics.py

Architecture:
1. Zero Pre-Trained State: Starts with a completely empty profile database.
2. First-Time Detection: Automatically labels any detected moving human as
   'UNKNOWN TARGET' (#ff0055) until live enrolled.
3. Single-Class Anomaly Safeguard:
   When only 1 profile is enrolled, uses normalized Euclidean / Mahalanobis distance
   from the profile centroid so an un-enrolled person is NOT falsely identified
   with 100% confidence.
4. Multi-Class Random Forest:
   Ensemble classification with 75% confidence threshold and anomaly gating.
5. Zero-Downtime Live Enrollment & Instant Profile Deletion.
"""

import os
import json
import time
import pickle
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

try:
    from scipy import stats
except ImportError:
    stats = None

try:
    from sklearn.ensemble import RandomForestClassifier
    import joblib
except ImportError:
    RandomForestClassifier = None
    joblib = None

logger = logging.getLogger("OmniSense.ML")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PROFILES_DB_PATH = os.path.join(BASE_DIR, "profiles_db.json")
MODEL_FILE_PATH = os.path.join(MODELS_DIR, "gait_classifier.joblib")
DATASET_FILE_PATH = os.path.join(MODELS_DIR, "gait_dataset.json")


class DecisionTreeNode:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, probs=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.probs = probs

    @property
    def is_leaf(self):
        return self.probs is not None


class NumpyRandomForestClassifier:
    """High-performance pure NumPy Random Forest implementation with identical API to sklearn."""

    def __init__(self, n_estimators: int = 50, max_depth: int = 12, min_samples_split: int = 3, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.rng = np.random.RandomState(random_state)
        self.trees: List[DecisionTreeNode] = []
        self.classes_: List[str] = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = sorted(list(set(y)))
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[item] for item in y], dtype=np.int32)

        n_samples, n_features = X.shape
        self.trees = []
        max_feat_sub = max(2, int(np.sqrt(n_features) * 1.5))

        for _ in range(self.n_estimators):
            boot_indices = self.rng.choice(n_samples, size=n_samples, replace=True)
            X_b = X[boot_indices]
            y_b = y_idx[boot_indices]
            tree = self._build_tree(X_b, y_b, depth=0, max_features=max_feat_sub)
            self.trees.append(tree)
        return self

    def _build_tree(self, X, y, depth, max_features):
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        if depth >= self.max_depth or n_samples < self.min_samples_split or len(set(y)) <= 1:
            counts = np.bincount(y, minlength=n_classes)
            probs = counts / max(1, n_samples)
            return DecisionTreeNode(probs=probs)

        feat_indices = self.rng.choice(n_features, size=min(n_features, max_features), replace=False)
        best_gini = 1e9
        best_feat = None
        best_thresh = None
        best_left_mask = None

        for feat in feat_indices:
            values = X[:, feat]
            thresholds = np.percentile(values, [20, 40, 60, 80])
            for thresh in thresholds:
                left_mask = values <= thresh
                right_mask = ~left_mask
                if not np.any(left_mask) or not np.any(right_mask):
                    continue
                gini = self._gini(y[left_mask], y[right_mask])
                if gini < best_gini:
                    best_gini = gini
                    best_feat = feat
                    best_thresh = thresh
                    best_left_mask = left_mask

        if best_feat is None:
            counts = np.bincount(y, minlength=n_classes)
            probs = counts / max(1, n_samples)
            return DecisionTreeNode(probs=probs)

        left_node = self._build_tree(X[best_left_mask], y[best_left_mask], depth + 1, max_features)
        right_node = self._build_tree(X[~best_left_mask], y[~best_left_mask], depth + 1, max_features)
        return DecisionTreeNode(feature_idx=best_feat, threshold=best_thresh, left=left_node, right=right_node)

    @staticmethod
    def _gini(y_left, y_right):
        n_l = len(y_left)
        n_r = len(y_right)
        n_total = n_l + n_r
        if n_total == 0:
            return 1.0
        p_l = np.bincount(y_left) / n_l
        p_r = np.bincount(y_right) / n_r
        gini_l = 1.0 - np.sum(p_l ** 2)
        gini_r = 1.0 - np.sum(p_r ** 2)
        return (n_l / n_total) * gini_l + (n_r / n_total) * gini_r

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        n_samples = len(X)
        n_classes = len(self.classes_)
        all_probs = np.zeros((n_samples, n_classes), dtype=np.float32)

        for tree in self.trees:
            for i in range(n_samples):
                all_probs[i] += self._predict_single(tree, X[i])

        all_probs /= max(1, len(self.trees))
        return all_probs

    def _predict_single(self, node: DecisionTreeNode, x):
        curr = node
        while not curr.is_leaf:
            if x[curr.feature_idx] <= curr.threshold:
                curr = curr.left
            else:
                curr = curr.right
        return curr.probs


def create_classifier(n_estimators=100, max_depth=12, random_state=42):
    """Instantiates RandomForestClassifier."""
    if RandomForestClassifier is not None:
        return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    return NumpyRandomForestClassifier(n_estimators=min(n_estimators, 50), max_depth=max_depth, random_state=random_state)


class GaitFeatureExtractor:
    """
    Extracts 24 statistical, spectral Doppler, and spatial correlation features
    from a 3-second CSI amplitude matrix of shape (window_length, num_subcarriers).
    """

    def __init__(self, fs: float = 25.0):
        self.fs = fs

    def extract(self, amp_matrix: np.ndarray) -> np.ndarray:
        if amp_matrix is None or len(amp_matrix) < 5:
            return np.zeros(24, dtype=np.float32)

        T, K = amp_matrix.shape
        amp_matrix = np.nan_to_num(amp_matrix, nan=0.0, posinf=50.0, neginf=0.0)

        # 1. Mean subcarrier amplitude statistics (4 features)
        sc_means = np.mean(amp_matrix, axis=0)
        f_mean_avg = float(np.mean(sc_means))
        f_mean_std = float(np.std(sc_means))
        f_mean_min = float(np.min(sc_means))
        f_mean_max = float(np.max(sc_means))

        # 2. Temporal variance statistics (4 features)
        sc_vars = np.var(amp_matrix, axis=0)
        f_var_avg = float(np.mean(sc_vars))
        f_var_std = float(np.std(sc_vars))
        f_var_min = float(np.min(sc_vars))
        f_var_max = float(np.max(sc_vars))

        # 3. Skewness statistics (3 features)
        if stats is not None:
            sc_skews = stats.skew(amp_matrix, axis=0)
            sc_skews = np.nan_to_num(sc_skews, nan=0.0)
            f_skew_avg = float(np.mean(sc_skews))
            f_skew_std = float(np.std(sc_skews))
            f_skew_max = float(np.max(sc_skews))
        else:
            diffs = amp_matrix - sc_means
            m3 = np.mean(diffs ** 3, axis=0)
            m2 = np.mean(diffs ** 2, axis=0) + 1e-8
            sc_skews = np.nan_to_num(m3 / (m2 ** 1.5), nan=0.0)
            f_skew_avg = float(np.mean(sc_skews))
            f_skew_std = float(np.std(sc_skews))
            f_skew_max = float(np.max(sc_skews))

        # 4. Kurtosis statistics (3 features)
        if stats is not None:
            sc_kurts = stats.kurtosis(amp_matrix, axis=0)
            sc_kurts = np.nan_to_num(sc_kurts, nan=0.0)
            f_kurt_avg = float(np.mean(sc_kurts))
            f_kurt_std = float(np.std(sc_kurts))
            f_kurt_max = float(np.max(sc_kurts))
        else:
            diffs = amp_matrix - sc_means
            m4 = np.mean(diffs ** 4, axis=0)
            m2 = np.mean(diffs ** 2, axis=0) + 1e-8
            sc_kurts = np.nan_to_num((m4 / (m2 ** 2)) - 3.0, nan=0.0)
            f_kurt_avg = float(np.mean(sc_kurts))
            f_kurt_std = float(np.std(sc_kurts))
            f_kurt_max = float(np.max(sc_kurts))

        # 5. Dominant Doppler Frequency & Harmonics (4 features)
        mean_stream = np.mean(amp_matrix, axis=1)
        mean_stream_detrend = mean_stream - np.mean(mean_stream)

        n_fft = max(128, int(2 ** np.ceil(np.log2(max(16, len(mean_stream_detrend))))))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / max(1.0, self.fs))
        fft_mag = np.abs(np.fft.rfft(mean_stream_detrend * np.hanning(len(mean_stream_detrend)), n=n_fft))

        gait_mask = (freqs >= 0.8) & (freqs <= 3.8)
        if np.any(gait_mask) and np.max(fft_mag[gait_mask]) > 1e-6:
            gait_freqs = freqs[gait_mask]
            gait_mags = fft_mag[gait_mask]
            dom_idx = np.argmax(gait_mags)
            dom_doppler_freq = float(gait_freqs[dom_idx])
            dom_doppler_power = float(gait_mags[dom_idx])
            mean_doppler_power = float(np.mean(gait_mags))
            doppler_snr = float(dom_doppler_power / (mean_doppler_power + 1e-6))
        else:
            dom_doppler_freq = 1.5
            dom_doppler_power = 0.0
            mean_doppler_power = 0.0
            doppler_snr = 0.0

        # 6. Subcarrier Correlation Matrix Eigenvalues (5 features)
        try:
            centered_matrix = amp_matrix - np.mean(amp_matrix, axis=0)
            cov_mat = np.dot(centered_matrix.T, centered_matrix) / max(1, (T - 1))
            eigenvalues = np.linalg.eigvalsh(cov_mat)
            eigenvalues = np.sort(np.maximum(0.0, eigenvalues))[::-1]
            total_eigen_sum = float(np.sum(eigenvalues)) + 1e-8
            norm_eigenvalues = (eigenvalues[:5] / total_eigen_sum).tolist()
            while len(norm_eigenvalues) < 5:
                norm_eigenvalues.append(0.0)
        except Exception:
            norm_eigenvalues = [0.2, 0.2, 0.2, 0.2, 0.2]

        # 7. Cadence Autocorrelation (1 feature)
        try:
            autocorr = np.correlate(mean_stream_detrend, mean_stream_detrend, mode='full')
            autocorr = autocorr[len(mean_stream_detrend) - 1 :]
            if autocorr[0] > 1e-6:
                autocorr = autocorr / autocorr[0]
                peak_window = autocorr[max(3, int(self.fs * 0.3)) : min(len(autocorr), int(self.fs * 1.5))]
                step_cadence_score = float(np.max(peak_window)) if len(peak_window) > 0 else 0.0
            else:
                step_cadence_score = 0.0
        except Exception:
            step_cadence_score = 0.0

        features = [
            f_mean_avg, f_mean_std, f_mean_min, f_mean_max,
            f_var_avg, f_var_std, f_var_min, f_var_max,
            f_skew_avg, f_skew_std, f_skew_max,
            f_kurt_avg, f_kurt_std, f_kurt_max,
            dom_doppler_freq, dom_doppler_power, mean_doppler_power, doppler_snr,
            norm_eigenvalues[0], norm_eigenvalues[1], norm_eigenvalues[2], norm_eigenvalues[3], norm_eigenvalues[4],
            step_cadence_score
        ]

        return np.nan_to_num(np.array(features, dtype=np.float32), nan=0.0)


class GaitBiometricClassifier:
    """
    Production Hardware Biometric Engine:
    - Starts with 0 profiles (Clean Slate).
    - Classifies unknown movers as 'UNKNOWN TARGET' (#ff0055).
    - Handles single-class enrollment via Mahalanobis/Euclidean centroid distance.
    - Handles multi-class enrollment via Random Forest.
    """

    CONFIDENCE_THRESHOLD = 0.75  # 75% requirement
    SINGLE_CLASS_MAX_DIST = 1.85  # Anomaly distance threshold for single-profile database

    def __init__(self, fs: float = 25.0):
        self.fs = fs
        self.feature_extractor = GaitFeatureExtractor(fs=fs)
        self.model: Optional[Any] = None
        self.classes_: List[str] = []
        self.profiles_db: Dict[str, Any] = {"profiles": [], "confidence_threshold": self.CONFIDENCE_THRESHOLD}
        self.profile_stats: Dict[str, Dict[str, np.ndarray]] = {}  # {name: {"mean": ..., "std": ...}}

        self.sliding_window: List[Tuple[float, np.ndarray]] = []  # [(timestamp, amplitudes), ...]
        self.window_duration = 3.0  # 3.0-second time-based window

        self._load_profiles_db()
        self._initialize_or_load_model()

    def _load_profiles_db(self):
        if os.path.exists(PROFILES_DB_PATH):
            try:
                with open(PROFILES_DB_PATH, 'r') as f:
                    self.profiles_db = json.load(f)
            except Exception as e:
                logger.error(f"Error loading profiles db: {e}")

    def _save_profiles_db(self):
        try:
            with open(PROFILES_DB_PATH, 'w') as f:
                json.dump(self.profiles_db, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving profiles db: {e}")

    def _initialize_or_load_model(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        enrolled = self.profiles_db.get("profiles", [])
        self.classes_ = [str(p["name"]) for p in enrolled]

        # Calculate centroids from dataset if dataset exists
        if os.path.exists(DATASET_FILE_PATH):
            try:
                with open(DATASET_FILE_PATH, 'r') as f:
                    ds = json.load(f)
                X_all = np.array(ds.get("X", []), dtype=np.float32)
                y_all = np.array(ds.get("y", []))
                for name in set(y_all):
                    mask = (y_all == name)
                    if np.any(mask):
                        sub_X = X_all[mask]
                        mu = np.mean(sub_X, axis=0)
                        raw_std = np.std(sub_X, axis=0)
                        sigma = np.maximum(0.20, np.maximum(raw_std, 0.08 * np.abs(mu)))
                        self.profile_stats[name] = {
                            "mean": mu,
                            "std": sigma
                        }
            except Exception as e:
                logger.warning(f"Error reading dataset stats: {e}")

        # If >= 2 classes, attempt loading model
        if len(self.classes_) >= 2 and os.path.exists(MODEL_FILE_PATH):
            try:
                if joblib is not None:
                    loaded = joblib.load(MODEL_FILE_PATH)
                else:
                    with open(MODEL_FILE_PATH, 'rb') as f:
                        loaded = pickle.load(f)
                self.model = loaded["model"]
                self.classes_ = [str(c) for c in loaded["classes"]]
                logger.info(f"Loaded existing gait classifier with classes: {self.classes_}")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")

    def add_frame(self, amplitudes: np.ndarray, timestamp: Optional[float] = None) -> None:
        """Adds amplitude frame with timestamp to adaptive sliding window."""
        if timestamp is not None:
            t = float(timestamp)
        else:
            if self.sliding_window:
                t = self.sliding_window[-1][0] + (1.0 / self.fs)
            else:
                t = time.time()
        self.sliding_window.append((t, amplitudes))

        # Prune older than 3.0 seconds
        cutoff = t - self.window_duration
        while self.sliding_window and self.sliding_window[0][0] < cutoff:
            self.sliding_window.pop(0)

    def _get_current_window_matrix(self) -> Optional[np.ndarray]:
        """Returns amplitude matrix for the current window if at least 1.0s of data exists."""
        if len(self.sliding_window) < 8:
            return None
        span = self.sliding_window[-1][0] - self.sliding_window[0][0]
        if span < 0.8:
            return None
        return np.array([item[1] for item in self.sliding_window], dtype=np.float32)

    def classify_current_window(self, disturbance_index: float) -> Dict[str, Any]:
        """
        Classifies gait motion based on calibrated disturbance index:
        - < 8%: AMBIENT_SILENCE (IDLE)
        - 8% - 22%: STATIONARY_TARGET (MONITORING_VITALS)
        - > 22%: MOTION_DETECTED (Evaluate walking gait)
        """
        if disturbance_index < 8.0:
            return {
                "identity": "AMBIENT_SILENCE",
                "confidence": 0.0,
                "is_known": False,
                "is_walking": False,
                "status_code": "IDLE",
                "probabilities": {},
                "color": "#00f3ff"
            }

        if disturbance_index <= 22.0:
            return {
                "identity": "STATIONARY_TARGET",
                "confidence": 0.0,
                "is_known": False,
                "is_walking": False,
                "status_code": "MONITORING_VITALS",
                "probabilities": {},
                "color": "#00f3ff"
            }

        # Subject is walking / in motion (> 22%)
        # Case A: Zero profiles enrolled
        if len(self.classes_) == 0:
            return {
                "identity": "UNKNOWN TARGET",
                "confidence": 0.0,
                "is_known": False,
                "is_walking": True,
                "status_code": "UNKNOWN_ANOMALY",
                "probabilities": {},
                "color": "#ff0055",
                "prompt_enrollment": True
            }

        amp_matrix = self._get_current_window_matrix()
        if amp_matrix is None:
            return {
                "identity": "CALIBRATING_GAIT_STREAM",
                "confidence": 0.0,
                "is_known": False,
                "is_walking": True,
                "status_code": "BUFFERING",
                "probabilities": {},
                "color": "#ff0055"
            }

        features = self.feature_extractor.extract(amp_matrix)

        # Case B: Exactly 1 profile enrolled (Single-Class Safeguard)
        if len(self.classes_) == 1:
            name = self.classes_[0]
            profile = next((p for p in self.profiles_db.get("profiles", []) if p["name"] == name), None)
            prof_color = profile.get("color", "#00ff88") if profile else "#00ff88"

            stats_dict = self.profile_stats.get(name)
            if stats_dict:
                mu = stats_dict["mean"]
                sigma = stats_dict["std"]
                # Normalized distance
                norm_diff = (features - mu) / sigma
                dist = float(np.sqrt(np.mean(norm_diff ** 2)))
            else:
                dist = 0.0

            if dist <= self.SINGLE_CLASS_MAX_DIST:
                # Target matches the enrolled profile
                conf = float(np.clip(100.0 - (dist / self.SINGLE_CLASS_MAX_DIST) * 20.0, 78.0, 99.0))
                return {
                    "identity": name,
                    "confidence": round(conf, 1),
                    "is_known": True,
                    "is_walking": True,
                    "status_code": "IDENTIFIED",
                    "probabilities": {name: round(conf, 1)},
                    "color": prof_color
                }
            else:
                # Anomaly detected: different gait cadence/energy
                conf = float(np.clip((1.0 - dist / 5.0) * 50.0, 5.0, 68.0))
                return {
                    "identity": "UNKNOWN TARGET",
                    "confidence": round(conf, 1),
                    "is_known": False,
                    "is_walking": True,
                    "status_code": "UNKNOWN_ANOMALY",
                    "probabilities": {name: round(conf, 1)},
                    "color": "#ff0055",
                    "prompt_enrollment": True
                }

        # Case C: 2 or more profiles enrolled (Random Forest + Centroid verification)
        if self.model is None:
            return {
                "identity": "UNKNOWN TARGET",
                "confidence": 0.0,
                "is_known": False,
                "is_walking": True,
                "status_code": "UNKNOWN_ANOMALY",
                "probabilities": {},
                "color": "#ff0055",
                "prompt_enrollment": True
            }

        probs = self.model.predict_proba(features.reshape(1, -1))[0]
        max_idx = int(np.argmax(probs))
        max_conf = float(probs[max_idx])
        pred_label = self.classes_[max_idx]

        prob_dict = {cls_name: round(float(p) * 100, 1) for cls_name, p in zip(self.classes_, probs)}

        # Check centroid distance to predicted class to reject out-of-distribution anomalies
        stats_dict = self.profile_stats.get(pred_label)
        if stats_dict:
            mu = stats_dict["mean"]
            sigma = stats_dict["std"]
            norm_diff = (features - mu) / sigma
            dist = float(np.sqrt(np.mean(norm_diff ** 2)))
        else:
            dist = 1.0

        profile = next((p for p in self.profiles_db.get("profiles", []) if p["name"] == pred_label), None)
        prof_color = profile.get("color", "#00ff88") if profile else "#00ff88"

        if max_conf >= self.CONFIDENCE_THRESHOLD and dist <= 2.2:
            return {
                "identity": pred_label,
                "confidence": round(max_conf * 100.0, 1),
                "is_known": True,
                "is_walking": True,
                "status_code": "IDENTIFIED",
                "probabilities": prob_dict,
                "color": prof_color
            }
        else:
            return {
                "identity": "UNKNOWN TARGET",
                "confidence": round(max_conf * 100.0, 1),
                "is_known": False,
                "is_walking": True,
                "status_code": "UNKNOWN_ANOMALY",
                "probabilities": prob_dict,
                "color": "#ff0055",
                "prompt_enrollment": True
            }

    def enroll_new_signature(self, name: str, recorded_frames: List[List[float]], description: str = "", color: str = "#e056fd") -> Dict[str, Any]:
        """
        Enrolls a new walking profile from live ESP32 hardware frames.
        Extracts sliding window features and retrains classifier in < 0.5s.
        """
        if not name or len(recorded_frames) < 20:
            return {"status": "error", "message": "Need at least 20 frames of walking data to enroll."}

        frames_arr = np.array(recorded_frames, dtype=np.float32)
        n_frames = len(frames_arr)

        # Slice into windows
        win_size = min(n_frames, max(15, int(self.fs * 2.5)))
        stride = max(2, (n_frames - win_size) // 15) if n_frames > win_size else 2

        new_features = []
        if n_frames <= win_size:
            feat = self.feature_extractor.extract(frames_arr)
            for _ in range(25):
                jitter = feat + np.random.normal(0, 0.04 * (np.abs(feat) + 0.1), len(feat))
                new_features.append(jitter)
        else:
            for start in range(0, n_frames - win_size + 1, stride):
                window = frames_arr[start : start + win_size]
                feat = self.feature_extractor.extract(window)
                new_features.append(feat)

        while len(new_features) < 20:
            base_f = new_features[len(new_features) % len(new_features)]
            jitter = base_f + np.random.normal(0, 0.03 * (np.abs(base_f) + 0.1), len(base_f))
            new_features.append(jitter)

        if len(new_features) == 0:
            return {"status": "error", "message": "Could not extract valid feature windows."}

        # Update dataset
        dataset = {"X": [], "y": []}
        if os.path.exists(DATASET_FILE_PATH):
            try:
                with open(DATASET_FILE_PATH, 'r') as f:
                    dataset = json.load(f)
            except Exception:
                pass

        for f in new_features:
            dataset["X"].append(f.tolist() if hasattr(f, "tolist") else list(f))
            dataset["y"].append(name)

        try:
            with open(DATASET_FILE_PATH, 'w') as f:
                json.dump(dataset, f)
        except Exception as e:
            logger.warning(f"Could not persist dataset: {e}")

        # Update centroid statistics
        new_feats_arr = np.array(new_features, dtype=np.float32)
        mu = np.mean(new_feats_arr, axis=0)
        raw_std = np.std(new_feats_arr, axis=0)
        sigma = np.maximum(0.20, np.maximum(raw_std, 0.08 * np.abs(mu)))

        self.profile_stats[name] = {
            "mean": mu,
            "std": sigma
        }

        # Retrain if >= 2 classes
        X_all = np.array(dataset["X"])
        y_all = np.array(dataset["y"])
        unique_classes = sorted(list(set(y_all)))
        self.classes_ = [str(c) for c in unique_classes]

        t0 = time.perf_counter()
        if len(unique_classes) >= 2:
            rf = create_classifier(n_estimators=80, max_depth=12, random_state=42)
            rf.fit(X_all, y_all)
            self.model = rf
            try:
                if joblib is not None:
                    joblib.dump({"model": self.model, "classes": self.classes_}, MODEL_FILE_PATH)
                else:
                    with open(MODEL_FILE_PATH, 'wb') as f:
                        pickle.dump({"model": self.model, "classes": self.classes_}, f)
            except Exception as e:
                logger.warning(f"Could not persist model: {e}")
        else:
            self.model = None

        train_duration_ms = round((time.perf_counter() - t0) * 1000, 1)

        # Update profiles DB
        profile_id = name.lower().replace(" ", "_")
        existing_profile = next((p for p in self.profiles_db.get("profiles", []) if p["id"] == profile_id), None)
        if not existing_profile:
            self.profiles_db["profiles"].append({
                "id": profile_id,
                "name": name,
                "enrolled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "sample_count": len(new_features),
                "color": color or "#e056fd",
                "description": description or f"Live enrolled target: {name}"
            })
        else:
            existing_profile["sample_count"] += len(new_features)
            if color:
                existing_profile["color"] = color

        self._save_profiles_db()
        logger.info(f"Enrolled profile '{name}' in {train_duration_ms}ms with {len(new_features)} windows. Classes: {self.classes_}")

        return {
            "status": "ok",
            "name": name,
            "samples_added": len(new_features),
            "train_duration_ms": train_duration_ms,
            "total_classes": self.classes_
        }

    def delete_profile(self, profile_id: str) -> Dict[str, Any]:
        """Deletes an enrolled profile and retrains or resets model."""
        pid_norm = profile_id.lower().replace(" ", "_")
        target_profile = next((p for p in self.profiles_db.get("profiles", []) if p["id"] == pid_norm or p["id"] == profile_id or p["name"].lower() == profile_id.lower()), None)

        if not target_profile:
            return {"status": "error", "message": f"Profile '{profile_id}' not found."}

        target_name = target_profile["name"]
        target_id = target_profile["id"]

        self.profiles_db["profiles"] = [p for p in self.profiles_db.get("profiles", []) if p["id"] != target_id]
        self._save_profiles_db()

        if target_name in self.profile_stats:
            del self.profile_stats[target_name]

        # Scrub from dataset
        if os.path.exists(DATASET_FILE_PATH):
            try:
                with open(DATASET_FILE_PATH, 'r') as f:
                    dataset = json.load(f)
                filtered_X = []
                filtered_y = []
                for x_val, y_val in zip(dataset.get("X", []), dataset.get("y", [])):
                    if y_val != target_name and y_val != target_id:
                        filtered_X.append(x_val)
                        filtered_y.append(y_val)

                dataset = {"X": filtered_X, "y": filtered_y}
                with open(DATASET_FILE_PATH, 'w') as f:
                    json.dump(dataset, f)

                unique_y = sorted(list(set(filtered_y)))
                self.classes_ = [str(c) for c in unique_y]

                if len(unique_y) >= 2:
                    X_all = np.array(dataset["X"])
                    y_all = np.array(dataset["y"])
                    rf = create_classifier(n_estimators=80, max_depth=12, random_state=42)
                    rf.fit(X_all, y_all)
                    self.model = rf
                    if joblib is not None:
                        joblib.dump({"model": self.model, "classes": self.classes_}, MODEL_FILE_PATH)
                    else:
                        with open(MODEL_FILE_PATH, 'wb') as f:
                            pickle.dump({"model": self.model, "classes": self.classes_}, f)
                else:
                    self.model = None
                    if os.path.exists(MODEL_FILE_PATH):
                        os.remove(MODEL_FILE_PATH)
            except Exception as e:
                logger.error(f"Error scrubbing dataset: {e}")
        else:
            self.classes_ = [str(p["name"]) for p in self.profiles_db.get("profiles", [])]
            self.model = None

        logger.info(f"Deleted profile '{target_name}'. Active classes: {self.classes_}")
        return {
            "status": "ok",
            "deleted_id": target_id,
            "deleted_name": target_name,
            "remaining_classes": self.classes_
        }
