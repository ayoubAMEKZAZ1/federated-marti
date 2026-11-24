"""Fast checks on a small generated population (about 20 s)."""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(ROOT, "generator", "generate_federated_marti_v2.py")


@pytest.fixture(scope="module")
def small():
    out = tempfile.mkdtemp()
    subprocess.run([sys.executable, GEN, "--out", out, "--customers", "2000", "--banks", "4", "--seed", "7"],
                   check=True, capture_output=True)
    return out


def test_outputs_and_meta(small):
    meta = json.load(open(os.path.join(small, "dataset_meta.json")))
    assert meta["n_banks"] == 4 and meta["n_transactions"] > 10000
    assert 0.002 < meta["fraud_rate"] < 0.05
    assert set(meta["split_counts"]) == {"train", "val", "test"}
    for b in range(1, 5):
        d = os.path.join(small, "clients", f"BANK_{b:02d}")
        for f in ("transactions.csv.gz", "edges.csv", "account_device_edges.csv", "sequences.csv.gz", "session_events.npz"):
            assert os.path.exists(os.path.join(d, f)), f


def test_shards_hide_global_identifiers(small):
    """A bank's shard must not contain global customer or device identifiers."""
    T = pd.read_csv(os.path.join(small, "clients", "BANK_01", "transactions.csv.gz"), nrows=2000)
    assert "global_customer_id" not in T.columns and "device_id" not in T.columns
    assert T.local_customer_id.str.startswith("LC_").all()
    G = pd.read_csv(os.path.join(small, "global", "transactions.csv.gz"), usecols=["global_customer_id"], nrows=2000)
    assert not set(T.local_customer_id) & set(G.global_customer_id)


def test_temporal_split_and_cross_bank(small):
    T = pd.read_csv(os.path.join(small, "global", "transactions.csv.gz"),
                    usecols=["timestamp", "split", "is_cross_bank", "dest_bank_id", "is_fraud", "fraud_type"])
    ts = pd.to_datetime(T.timestamp)
    assert ts[T.split == "train"].max() < ts[T.split == "val"].min() <= ts[T.split == "val"].max() < ts[T.split == "test"].min()
    internal = T[T.dest_bank_id != "EXTERNAL"]
    assert internal.is_cross_bank.mean() > 0.2
    assert set(T.loc[T.is_fraud == 1, "fraud_type"]) <= {
        "account_takeover", "social_engineering_app", "mule_account", "money_laundering_layering",
        "synthetic_identity", "first_party", "card_not_present"}


def test_event_tensors_match_rows(small):
    z = np.load(os.path.join(small, "clients", "BANK_02", "session_events.npz"))
    assert z["events"].shape[1:] == (64, 9)
    assert len(z["events"]) == len(z["txn_id"]) == len(z["is_fraud"])
    assert (z["lengths"] <= 64).all()


def test_deterministic():
    outs = []
    for _ in range(2):
        out = tempfile.mkdtemp()
        subprocess.run([sys.executable, GEN, "--out", out, "--customers", "800", "--banks", "3", "--seed", "3",
                        "--no-events"], check=True, capture_output=True)
        outs.append(json.load(open(os.path.join(out, "dataset_meta.json"))))
    assert outs[0] == outs[1]
