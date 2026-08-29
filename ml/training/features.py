"""Shared feature engineering for CipherMind Sentinel.

The exported `feature_config.json` artifact is the single source of truth:
both this Python module and the TypeScript inference engine (mini-services/soc-engine)
implement the exact same transformations defined by the config.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9

# Raw numeric columns (order matters — must match EXPECTED_COLUMNS minus id/targets/categoricals)
RAW_NUMERIC = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl",
    "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit",
    "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat",
    "smean", "dmean", "trans_depth", "response_body_len", "ct_srv_src",
    "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
    "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd",
    "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
]
CATEGORICAL = ["proto", "service", "state"]

DERIVED = [
    {"name": "byte_ratio", "formula": "sbytes / (dbytes + eps)"},
    {"name": "packet_ratio", "formula": "spkts / (dpkts + eps)"},
    {"name": "total_bytes", "formula": "sbytes + dbytes"},
    {"name": "total_packets", "formula": "spkts + dpkts"},
    {"name": "total_loss", "formula": "sloss + dloss"},
    {"name": "payload_per_packet", "formula": "(sbytes + dbytes) / (spkts + dpkts + eps)"},
    {"name": "load_ratio", "formula": "sload / (dload + eps)"},
    {"name": "size_ratio", "formula": "smean / (dmean + eps)"},
    {"name": "rtt_ratio", "formula": "synack / (ackdat + eps)"},
    {"name": "jitter_ratio", "formula": "sjit / (djit + eps)"},
    {"name": "interpkt_ratio", "formula": "sinpkt / (dinpkt + eps)"},
    {"name": "flow_asymmetry", "formula": "(sbytes - dbytes) / (sbytes + dbytes + eps)"},
]


def compute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised derived features — mirrors the TS implementation exactly."""
    out = pd.DataFrame(index=df.index)
    sbytes = df["sbytes"].astype(float)
    dbytes = df["dbytes"].astype(float)
    spkts = df["spkts"].astype(float)
    dpkts = df["dpkts"].astype(float)
    sload = df["sload"].astype(float)
    dload = df["dload"].astype(float)
    out["byte_ratio"] = sbytes / (dbytes + EPS)
    out["packet_ratio"] = spkts / (dpkts + EPS)
    out["total_bytes"] = sbytes + dbytes
    out["total_packets"] = spkts + dpkts
    out["total_loss"] = df["sloss"].astype(float) + df["dloss"].astype(float)
    out["payload_per_packet"] = (sbytes + dbytes) / (spkts + dpkts + EPS)
    out["load_ratio"] = sload / (dload + EPS)
    out["size_ratio"] = df["smean"].astype(float) / (df["dmean"].astype(float) + EPS)
    out["rtt_ratio"] = df["synack"].astype(float) / (df["ackdat"].astype(float) + EPS)
    out["jitter_ratio"] = df["sjit"].astype(float) / (df["djit"].astype(float) + EPS)
    out["interpkt_ratio"] = df["sinpkt"].astype(float) / (df["dinpkt"].astype(float) + EPS)
    out["flow_asymmetry"] = (sbytes - dbytes) / (sbytes + dbytes + EPS)
    return out


def sanitize_numeric(num: pd.DataFrame) -> pd.DataFrame:
    """inf -> NaN (imputation happens with train-fitted medians later)."""
    return num.replace([np.inf, -np.inf], np.nan)


class FeatureBuilder:
    """Fits (train-only) and applies the full feature pipeline.

    Steps: sanitize -> median impute -> log1p on high-skew cols -> derived features
    -> categorical ordinal encoding with UNKNOWN fallback.
    """

    def __init__(self):
        self.medians: dict[str, float] = {}
        self.log_cols: list[str] = []
        self.cat_maps: dict[str, dict[str, int]] = {}
        self.unknown_code: dict[str, int] = {}
        self.feature_names: list[str] = []

    def fit(self, train: pd.DataFrame, skew_threshold: float = 3.0) -> "FeatureBuilder":
        num = sanitize_numeric(train[RAW_NUMERIC].apply(pd.to_numeric, errors="coerce"))
        self.medians = {c: float(num[c].median()) for c in RAW_NUMERIC}
        self.log_cols = [
            c for c in RAW_NUMERIC
            if abs(float(num[c].fillna(self.medians[c]).skew())) > skew_threshold
        ]
        for c in CATEGORICAL:
            vals = sorted(train[c].astype(str).unique().tolist())
            mapping = {v: i for i, v in enumerate(vals)}
            mapping["__unknown__"] = len(vals)
            self.cat_maps[c] = mapping
            self.unknown_code[c] = len(vals)
        self.feature_names = (
            [f"log1p_{c}" if c in self.log_cols else c for c in RAW_NUMERIC]
            + [d["name"] for d in DERIVED]
            + [f"cat_{c}" for c in CATEGORICAL]
        )
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        num = sanitize_numeric(df[RAW_NUMERIC].apply(pd.to_numeric, errors="coerce"))
        num = num.fillna(pd.Series(self.medians))
        if self.log_cols:
            num[self.log_cols] = np.log1p(num[self.log_cols].clip(lower=0))
        der = compute_derived(df)
        der = der.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        cats = np.column_stack([
            df[c].astype(str).map(self.cat_maps[c]).fillna(self.unknown_code[c]).astype(int).values
            for c in CATEGORICAL
        ])
        mat = np.column_stack([num.values, der.values, cats])
        return mat.astype(np.float64)

    def config(self) -> dict:
        return {
            "raw_numeric": RAW_NUMERIC,
            "categorical": CATEGORICAL,
            "medians": self.medians,
            "log_cols": self.log_cols,
            "cat_maps": self.cat_maps,
            "unknown_code": self.unknown_code,
            "derived": DERIVED,
            "eps": EPS,
            "feature_names": self.feature_names,
            "skew_threshold": 3.0,
        }
