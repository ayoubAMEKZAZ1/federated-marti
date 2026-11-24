#!/usr/bin/env python3
"""Sanity, leakage and separability checks for the Federated-MARTI v2 synthetic dataset."""
import json, sys, os
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

out = sys.argv[1]
T = pd.read_csv(os.path.join(out, "global", "transactions.csv.gz"), low_memory=False)
A = pd.read_csv(os.path.join(out, "global", "account_identity_map.csv"))
R = {}
print("rows", len(T), "fraud rate", round(T.is_fraud.mean(), 4))
print(T.groupby("split").is_fraud.agg(["size", "mean"]).round(4))
print(T.fraud_type.value_counts())
print("cross-bank share of internal transfers:", round(T.loc[T.dest_bank_id != "EXTERNAL", "is_cross_bank"].mean(), 3))
print("channel mix:", T.channel.value_counts(normalize=True).round(3).to_dict())
print("has_behavioural share of digital:", round(T.loc[T.session_type != "none", "has_behavioural"].mean(), 3))
leg = T[(T.is_fraud == 0) & (T.has_behavioural == 1)]
print("\nLegit behavioural calibration (median by device type):")
print(leg.groupby("device_type")[["iki_mean_ms", "key_hold_mean_ms", "typing_speed_cps", "tap_hold_mean_ms",
                                   "touch_pressure_mean", "touch_size_mean", "accel_std_mean", "session_duration_s",
                                   "backspace_rate"]].median().round(3).T)
print("\nWPM-equivalent on phones by age band (legit, digits excluded not separable; indicative):")
c = pd.read_csv(os.path.join(out, "global", "global_customers.csv"))
x = leg[leg.device_type == "phone"].merge(c[["global_customer_id", "age_band"]], on="global_customer_id")
print((60 / (x.groupby("age_band").iki_mean_ms.median() / 1000) / 5).round(1).to_dict())
dev_per_acc = T[T.device_id.notna()].groupby("local_account_id").device_id.nunique()
print("\ndistinct devices per account (legit-only accounts):", dev_per_acc[A.set_index("local_account_id").account_role.reindex(dev_per_acc.index) == "normal"].describe().round(2).to_dict())
print("\nFraud signature table (medians, behavioural sessions):")
cols = ["is_new_device_for_account", "behaviour_deviation_score", "iki_mean_ms", "touch_pressure_mean", "accel_std_mean",
        "pause_count_2s", "session_duration_s", "active_call_flag", "remote_access_flag", "is_emulator",
        "device_n_accounts_at_bank", "ip_country_mismatch", "vpn_flag", "paste_count"]
B = T[T.has_behavioural == 1]
tab = B.groupby("fraud_type")[cols].agg(lambda s: s.mean() if s.dropna().isin([0, 1]).all() else s.median()).round(3)
print(tab.T.to_string())

# ---- separability: session/biometric features only, temporal split
feat_b = ["is_new_device_for_account", "device_age_days_at_bank", "device_n_accounts_at_bank", "is_emulator", "is_rooted",
          "remote_access_flag", "active_call_flag", "vpn_flag", "ip_country_mismatch", "geo_distance_km",
          "profile_change_in_session", "n_events", "n_screens", "session_duration_s", "time_to_first_action_s",
          "key_count", "key_hold_mean_ms", "key_hold_std_ms", "iki_mean_ms", "iki_std_ms", "typing_speed_cps",
          "backspace_rate", "paste_count", "tap_count", "tap_hold_mean_ms", "touch_pressure_mean", "touch_pressure_std",
          "touch_size_mean", "touch_size_std", "swipe_count", "swipe_velocity_mean", "swipe_straightness_mean",
          "mouse_velocity_mean", "mouse_straightness_mean", "accel_std_mean", "pause_count_2s", "longest_pause_s",
          "hesitation_ratio", "behaviour_deviation_score"]
feat_t = ["amount", "time_since_last_hrs", "velocity_1h", "velocity_24h", "amount_to_avg_ratio", "payee_is_new",
          "is_cross_bank", "hour_of_day", "day_of_week"]
D = T[T.session_type != "none"].copy()
for k in ["device_type", "device_os", "login_method", "session_type", "txn_type", "channel", "payee_entry_method"]:
    D[k + "_c"] = D[k].astype("category").cat.codes
cats = [k + "_c" for k in ["device_type", "device_os", "login_method", "session_type", "txn_type", "channel", "payee_entry_method"]]

def fit_eval(feats, name):
    tr, te = D[D.split == "train"], D[D.split == "test"]
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, class_weight="balanced", random_state=0)
    m.fit(tr[feats], tr.is_fraud)
    p = m.predict_proba(te[feats])[:, 1]
    res = {"all": (roc_auc_score(te.is_fraud, p), average_precision_score(te.is_fraud, p))}
    for ft in sorted(te.fraud_type.unique()):
        if ft == "none": continue
        mk = (te.fraud_type == ft) | (te.is_fraud == 0)
        if (te.fraud_type == ft).sum() >= 5:
            res[ft] = (roc_auc_score(te.is_fraud[mk], p[mk]), average_precision_score(te.is_fraud[mk], p[mk]))
    print(f"\n{name}: test ROC-AUC / PR-AUC (digital sessions, fraud rate {te.is_fraud.mean():.3%})")
    for k, (a, b) in res.items():
        print(f"  {k:28s} {a:.3f} / {b:.3f}")
    return {k: [round(a, 3), round(b, 3)] for k, (a, b) in res.items()}

R["txn_only"] = fit_eval(feat_t + cats[4:6], "Transaction features only")
R["biometric_only"] = fit_eval(feat_b + cats[:4] + [cats[6]], "Session + biometric features only")
R["combined"] = fit_eval(feat_t + feat_b + cats, "Combined")
# leakage probes: IDs / ordering must carry no signal
# identifier-format leakage: do account / customer / device ID numbers differ between fraud and legit entities?
from scipy.stats import ks_2samp
acc_lab = T.groupby("local_account_id").is_fraud.max()
an = acc_lab.index.str[-6:].astype(int)
cust_lab = T.groupby("global_customer_id").is_fraud.max()
cn = cust_lab.index.str[3:].astype(int)
dev_lab = T[T.device_id.notna()].groupby("device_id").is_fraud.max()
dn = dev_lab.index.str[4:10].map(lambda x: int(x, 16))
probe = {}
for name, num, lab in [("account_id", an, acc_lab.values), ("customer_id", cn, cust_lab.values), ("device_id", dn, dev_lab.values)]:
    num = np.asarray(num)
    probe[name] = dict(ks_p_value=float(ks_2samp(num[lab == 1], num[lab == 0]).pvalue),
                       auc=float(roc_auc_score(lab, num)))
print("\nIdentifier leakage probe (AUC of raw ID number vs fraud label; ~0.5 and p>0.01 = no leak):")
for k, v_ in probe.items(): print(f"  {k:12s} AUC {v_['auc']:.3f}  KS p={v_['ks_p_value']:.3f}")
R["id_leak_probe"] = probe
json.dump(R, open(os.path.join(out, "validation_report.json"), "w"), indent=1)
