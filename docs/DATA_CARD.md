# Federated-MARTI synthetic dataset v2: data card

A fully synthetic, seeded dataset of eight banks, 1,156,904 transactions and 346,053 behavioural-biometric sessions, built to train, validate and test Federated-MARTI. That model combines a heterogeneous GAT, a Transformer, a dynamic gate, cross-bank identity resolution and FedAvg clients. No real person, account, device or institution is represented.

Regenerate it exactly with:

```
python3 generate_federated_marti_v2.py --out ./fmarti_v2 --customers 40000 --banks 8 --seed 42
python3 validate_dataset.py ./fmarti_v2
```

Generation takes about 7 minutes on 2 CPU cores and needs about 4 GB of RAM.

## 1. What changed from v1

| Aspect | v1 (earlier version) | v2 |
| --- | --- | --- |
| Scale | 4 banks, 8,000 customers, 188,584 transactions | 8 banks, 40,630 customers (630 synthetic identities), 51,832 accounts, 1,156,904 transactions |
| Devices | OS drawn at random per transaction (≈4 OS per account, iOS at ATMs) | 67,790 stable devices; 1.58 devices per legitimate account; phone replacements, shared family phones, public PCs |
| Cross-bank | 0 cross-bank transfers or edges | 38.6% of account-to-account edges cross banks; mule and laundering flows cross banks by design |
| Fraud | Isolated rows; 40% of accounts touched | Episodes with mechanics (takeover → mule → cash-out chains, layering rings, coordinated bust-outs); 1.05% of transactions |
| Biometrics | None | 27 session features aggregated from raw events, 18 device/context features, and raw event tensors (64 × 9 per session) |
| Splits | None | Temporal: train Jan–Aug, val Sep–Oct, test Nov–Dec |
| Federated privacy | Global IDs in shards | Shards use bank-specific customer pseudonyms and salted device hashes |

## 2. Contents

```
fmarti_v2/
  dataset_meta.json
  validation_report.json
  global/                                    ground truth, all banks (evaluation / centralised baselines)
    transactions.csv.gz                      1,156,904 rows × 73 columns
    global_customers.csv                     incl. is_synthetic_identity
    account_identity_map.csv                 account → customer, account_role, account_fraud_label
    devices_ground_truth.csv                 device owner, type, OS, emulator, rooted
    customer_behaviour_profiles_ground_truth.csv   latent behavioural profile per customer
    graph_edges.csv                          account→account (n_txns, amount, is_cross_bank, is_fraud_edge)
    graph_account_device_edges.csv           account→device (n_sessions)
    identity_resolution_ground_truth.csv     13,082 positive pairs, 13,082 hard negatives, 5,677 random negatives
    account_sequences.csv.gz                 ordered per-account sequences (+ txn ids for joining features)
  clients/BANK_01 … BANK_08/                 what one bank is allowed to see
    transactions.csv.gz                      local_customer_id (bank pseudonym) and device_hash (bank-salted)
    edges.csv                                internal + boundary edges (counterparty in another bank)
    account_device_edges.csv
    sequences.csv.gz
    session_events.npz                       raw event tensors for this bank's behavioural sessions
```

Labels are `is_fraud` and `fraud_type` per transaction, plus `account_fraud_label` and `account_role` per account. Columns marked "ground truth" must never be used as model inputs.

## 3. Transaction table (73 columns)

**Transaction core.** These columns are compatible with v1. `txn_id`, `timestamp`, `step` (day 0–364), `split`, `bank_id`, `local_account_id`, `global_customer_id` (global file only), `dest_account_id`, `dest_bank_id`, `is_cross_bank`, `amount`, `txn_type`, `old_balance`, `new_balance` and `channel` (mobile, web, atm, branch, pos, system) carry the transaction itself. `hour_of_day`, `day_of_week`, `time_since_last_hrs`, `velocity_1h`, `velocity_24h`, `amount_to_avg_ratio` and `payee_is_new` add context. Every derived feature uses only the account's past.

**Session and device context.** These columns are empty for atm, branch, pos and system:

- Session and device: `session_id`, `session_type` (app, online_banking, 3ds), `app_version`, `device_id` (global) / `device_hash` (shard), `device_type`, `device_os` and `login_method` (biometric, password, autofill, otp).
- Device history: `has_behavioural`, `is_new_device_for_account`, `device_age_days_at_bank` and `device_n_accounts_at_bank`.
- Risk flags: `is_emulator` (as detected, about 60% of emulators), `is_rooted`, `remote_access_flag`, `active_call_flag` and `vpn_flag`.
- Location and actions: `ip_country`, `ip_country_mismatch`, `geo_distance_km`, `profile_change_in_session` and `payee_entry_method` (typed, pasted, saved).

**Behavioural biometrics.** These are aggregated from the session's own raw events.

- Session shape: `n_events`, `n_screens`, `session_duration_s` and `time_to_first_action_s`.
- Typing: `key_count`, `key_hold_mean/std_ms`, `iki_mean/std_ms` (inter-key interval, pauses over 2 s excluded), `typing_speed_cps`, `backspace_rate` and `paste_count`.
- Touch: `tap_count`, `tap_hold_mean_ms`, `touch_pressure_mean/std` (Android only), `touch_size_mean/std`, `swipe_count`, `swipe_velocity_mean` and `swipe_straightness_mean`.
- Mouse and motion: `mouse_velocity_mean`, `mouse_straightness_mean` (desktop only) and `accel_std_mean` (hand-motion proxy).
- Hesitation: `pause_count_2s`, `longest_pause_s` and `hesitation_ratio`.
- Deviation: `behaviour_deviation_score`, the mean absolute z-score against the account's own previous sessions on the same device type. It needs at least 3 prior sessions and uses no labels.

**Raw event tensors** (`session_events.npz`). Each file holds `events` [N, 64, 9] as float16, `lengths`, `session_id`, `txn_id`, `split`, `is_fraud` and `fields`.

- Fields: `event_type` (0 padding, 1 key, 2 backspace, 3 tap/click, 4 swipe/scroll, 5 mouse move, 6 paste, 7 screen change, 8 biometric auth), `dt_ms`, `hold_ms`, `pressure`, `touch_size`, `velocity`, `straightness`, `accel_std` and `screen_idx`.
- Encoding: `-1` means "not available on this device or event". `dt_ms` is clipped at 60,000.
- Truncation: 12% of sessions are longer than 64 events and keep their first and last 32. Aggregates are computed before truncation.

## 4. How the biometrics are made realistic

Each customer has a stable latent profile covering typing rhythm, key hold, touch pressure and size, swipe and mouse dynamics, hand tremor, think time, error rate and hesitation. The profile depends on age band and digital literacy, varies within a person from session to session, and is perturbed by context: phone held or on a table, the v5 app flow, step-up OTPs and new-payee entry. Session features are computed from the generated events, so the two levels always agree (verified: key counts and durations match in 100% of untruncated sessions).

Calibration against public typing studies:

| Measure | Reference | Generated (legitimate, median) |
| --- | --- | --- |
| Desktop inter-key interval | 238.7 ms mean ([Dhakal et al., CHI 2018](https://www.pokristensson.com/pubs/DhakalEtAlCHI2018.pdf)) | 262 ms (includes slower digit entry) |
| Desktop key hold | 116 ms mean (same study) | 125 ms |
| Mobile typing speed | 36.2 WPM mean, about 39.6 WPM (ages 10–19) down to 26.3 (ages 50–59) ([Palin et al., MobileHCI 2019](https://www.pokristensson.com/pubs/PalinEtAlMobileHCI2019.pdf)) | 34.8 WPM (18–25) → 19.3 WPM (65+) |
| Backspace / correction rate | low single-digit % in both studies | 5–6% |

Touch pressure, touch size, swipe velocity and tremor have no published population norms of comparable quality. Their ranges are assumptions: Android pressure on a 0–1 scale, iOS reporting no pressure, and 25% of Android devices reporting a constant 1.0 because they lack pressure sensing. Motion sensors are unavailable in 8% of mobile sessions. Treat these values as plausible rather than measured.

## 5. Fraud mechanics and signatures

| Fraud type | Mechanics | Biometric / device signature | Deliberate overlap with legitimate behaviour |
| --- | --- | --- | --- |
| `account_takeover` (1,286) | Credential variant on fraudster devices shared across victims, or a remote-access-tool (RAT) variant controlling the victim's own Android phone or desktop; drains 40–90% of the balance to mules, usually in another bank | New device, foreign IP, VPN, pasted credentials, profile change; RAT shows constant pressure, almost no hand motion, network latency jitter and remote-access flag in 70% of cases | 20% are sophisticated (domestic IP, no VPN); iOS cannot be remote-controlled |
| `social_engineering_app` (783) | Coerced victim on their own device and biometric profile sends savings to a mule | Active call (75%), 2–7 long pauses, slower typing, 3–4× longer sessions, account numbers typed rather than pasted | 25% are sophisticated (messaging instead of a call); about 4% of legitimate sessions also have a call active; older users hesitate naturally |
| `mule_account` (4,032) | 336 mule accounts receive takeover and scam proceeds and forward them within 0.5–36 h: ATM cash-outs, onward cross-bank transfers or crypto exits | Herder-controlled devices shared across mules (55%), fast and pasted input | 45% of mules operate their own phone |
| `money_laundering_layering` (1,808) | Rings of 4–9 accounts across at least 2 banks: placement in structured cash deposits of 7,000–9,900, fan-in, cross-bank fan-out, exit | Scripted sessions (40%) with near-zero timing variance and no motion; shared controller devices | Deposits happen at branches and ATMs, so they carry no biometrics |
| `synthetic_identity` (1,526) | 630 fabricated identities, 1–3 banks each, operated by device farms; 70% bust out across all their banks within days | Emulators (detected about 60% of the time), farm devices shared across identities, scripted or fluent input | Build-up transactions are labelled 0, while the account is labelled 1 |
| `first_party` (1,401) | Genuine customer spends 2.5–8× their usual amounts, then goes dormant | None: own device and own biometrics | Legitimate customers make large purchases too (0.5% of transfers) |
| `card_not_present` (1,328) | Card testing with 2–5 micro-payments, then 1–3 larger e-commerce payments | 3-D Secure: device and IP only; behavioural data only when a challenge (OTP) is triggered | 6% of legitimate web payments are micro-payments |

**Drift in time.** The v5 app release on 2024-07-01 adds a review screen and raises biometric-login uptake. Remote-access takeovers and coercion scams grow through the second half of the year. Card testing peaks in November and December. One device farm first appears on 2024-11-01, so it is unseen until the test period. Fraud prevalence is 0.95% in train, 1.10% in validation and 1.42% in test.

## 6. Validation results

These were produced by `validate_dataset.py`: a gradient-boosted baseline trained on the train split and scored on the test split, using digital sessions only (2.68% fraud).

| Feature set | ROC-AUC | PR-AUC |
| --- | --- | --- |
| Transaction features only | 0.969 | 0.717 |
| Session + biometric features only | 0.962 | 0.762 |
| Combined | 0.989 | 0.881 |

PR-AUC by fraud type (transaction only → biometric only → combined):

| Fraud type | Transaction only | Biometric only | Combined |
| --- | --- | --- | --- |
| Account takeover | 0.45 | 0.84 | 0.88 |
| Social-engineering scam | 0.11 | 0.85 | 0.87 |
| Mule account | 0.11 | 0.35 | 0.40 |
| Synthetic identity | 0.47 | 0.30 | 0.80 |
| First-party | 0.08 | 0.00 | 0.09 |

Biometrics carry most of the signal for takeovers and scams. First-party fraud stays hard, as intended.

The leakage probe compares raw account, customer and device ID numbers against fraud labels. All AUCs are 0.50 ± 0.01 with KS p > 0.5, so identifiers carry no label information. A tree model can still memorise specific fraud devices and mule accounts that recur over time; that is a realistic denylist signal, not leakage.

The combined baseline is high. A GAT, Transformer or gate therefore needs to be compared against this baseline, not only against chance. Federated-MARTI is reported against each bank's own gradient-boosted model for this reason. The difficulty can be raised with `--sophistication 2` (more victim-mimicking fraud) or lowered with `--fraud-scale` to approach real prevalence below 0.1%.

## 7. Using it in the Federated-MARTI pipeline

- **Federated clients:** one client per `clients/BANK_XX`. Train only on that bank's `split == train` rows, validate on `val`, report on `test`.
- **GAT:** account nodes come from transactions, account→account edges from `edges.csv`, and device nodes from `account_device_edges.csv`. Boundary edges point to accounts held at other banks, which is where boundary-node embedding exchange applies.
- **Transformer:** use `session_events.npz` for a session encoder, or per-account sequences of session features keyed by `seq_txn_ids`.
- **Gate:** the three expert probabilities per transaction. Train it per client or federated, not on pooled raw data.
- **Cross-bank identity:** `identity_resolution_ground_truth.csv`. Shards cannot link customers directly, because pseudonyms and device hashes are bank-salted, so matching must come from behaviour and transaction patterns.

## 8. Limitations

- **Synthetic by necessity.** Banks cannot release keystroke, touch, motion or device data, because it is personal and often biometric data under GDPR Article 9 and comparable laws. Every distribution here is a model of reality, not a sample of it, and results on this data are an upper bound until confirmed on real data.
- **Learnable generator rules.** Signatures are parametric. A flexible model can learn the generator's rules, and real fraudsters adapt in ways no generator anticipates.
- **Assumed sensor ranges.** Only typing speed and rhythm are calibrated on published studies; pressure, touch size, swipe and motion ranges are assumptions.
- **Prevalence is higher than reality.** About 1% of transactions are fraud, against below 0.1% in the field. Negatives can be downsampled for training, but metrics should be reported at a realistic prevalence too.
- **Truncation and cold start.** One session per financial transaction; non-financial logins are not simulated. Sessions over 64 events are truncated, and pauses over 60 s are clipped. `payee_is_new` has a cold start in January.
- **Fixed geography.** Six home countries carried over from v1, with no seasonality beyond December and no network-level signals (ASN, carrier).
