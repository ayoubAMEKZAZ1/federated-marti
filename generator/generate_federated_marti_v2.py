#!/usr/bin/env python3
"""
generate_federated_marti_v2.py
==============================
Synthetic multi-bank fraud / AML dataset with device and behavioural biometrics
for training, validating and testing Federated-MARTI
(heterogeneous GAT + Transformer + dynamic gate + cross-bank identity + FedAvg clients).

Everything is synthetic and seeded. No real person, account or device is represented.

What it simulates
-----------------
* Customers with stable demographic, channel and behavioural profiles
  (typing rhythm, touch pressure/size, swipe and mouse dynamics, hand tremor,
  navigation speed, hesitation), calibrated on public typing studies:
    - desktop: mean inter-key interval 239 ms, key hold 116 ms (Dhakal et al., CHI 2018)
    - mobile: 36 WPM on average, slowing with age (Palin et al., MobileHCI 2019)
* Stable devices per customer (phone, optional desktop), legitimate phone
  replacements, family-shared phones, public computers.
* Several banks, customers holding accounts in more than one bank, cross-bank transfers.
* Fraud generated as episodes with realistic mechanics and per-type biometric signatures:
    account_takeover (credential, on fraudster device | RAT, remote control of victim device)
    social_engineering_app (coerced victim, own device: active call, hesitation)
    mule_account (rapid pass-through, shared herder devices, cross-bank)
    money_laundering_layering (structured placement, fan-in/fan-out across banks, scripted sessions)
    synthetic_identity (device farms / emulators, multi-bank bust-out)
    first_party (genuine customer, genuine biometrics, spending spree then default)
    card_not_present (card testing bots via 3-D Secure, mostly no behavioural data)
* Hard negatives: legitimate new devices, travel, VPNs, calls during sessions,
  remote support, large purchases to new payees, older users' natural hesitation.
* Drift: app v5 release on 2024-07-01 changes the screen flow and biometric-login
  uptake; the fraud mix shifts toward RAT and coercion scams in H2; a new device
  farm appears only in Q4.
* Raw per-session event sequences (keys, taps, swipes, mouse, paste, screens) and
  session features aggregated from those same events.

Outputs (in --out):
  global/            ground truth, all banks (customers, accounts, devices, transactions, edges, identity pairs)
  clients/BANK_XX/   what one bank may see: pseudonymised customers, bank-salted device hashes,
                     its own transactions, boundary edges, sequences and raw session events (.npz)
  dataset_meta.json

Usage:
  python3 generate_federated_marti_v2.py --out ./fmarti_v2 --customers 40000 --banks 8 --seed 42
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- constants
AGE_BANDS = ["18-25", "26-35", "36-50", "51-65", "65+"]
AGE_P = [0.144, 0.289, 0.303, 0.174, 0.090]
INCOME = ["low", "mid", "high"]
INCOME_P = [0.35, 0.45, 0.20]
COUNTRIES = ["NG", "KE", "ZA", "US", "GH", "GB"]
COUNTRY_P = [0.358, 0.198, 0.151, 0.120, 0.096, 0.077]
ANDROID_SHARE = {"NG": 0.86, "KE": 0.90, "ZA": 0.80, "US": 0.42, "GH": 0.87, "GB": 0.50}
FOREIGN = ["RO", "RU", "BR", "CN", "IN", "VN", "AE", "MY"]
MOBILE_IKI_BY_AGE = np.array([320.0, 350.0, 400.0, 460.0, 545.0])  # ms, from 36 WPM avg and age trend
DESKTOP_IKI, DESKTOP_HOLD = 239.0, 116.0

DAYS = 365
VAL_DAY, TEST_DAY = 244, 305      # 2024-09-01, 2024-11-01
APP_V5_DAY = 182                  # 2024-07-01
NEW_FARM_DAY = 305                # device farm that only exists in the test period
START = pd.Timestamp("2024-01-01")

# raw session event encoding
ET_KEY, ET_BACKSPACE, ET_TAP, ET_SWIPE, ET_MOUSE, ET_PASTE, ET_SCREEN, ET_AUTH = 1, 2, 3, 4, 5, 6, 7, 8
EVENT_FIELDS = ["event_type", "dt_ms", "hold_ms", "pressure", "touch_size",
                "velocity", "straightness", "accel_std", "screen_idx"]
NF = len(EVENT_FIELDS)

# fraud session modes
M_LEGIT, M_ATO_CRED, M_ATO_RAT, M_APP, M_MULE, M_RING, M_SYNTH, M_CNP, M_FP = range(9)

FRAUD_TYPES = ["account_takeover", "social_engineering_app", "mule_account",
               "money_laundering_layering", "synthetic_identity", "first_party", "card_not_present"]


def month_of(day):
    return np.minimum((np.asarray(day) / 30.42).astype(int), 11)


# ----------------------------------------------------------------------------- profiles
def make_profiles(rng, n, age_idx, lit, kind="human"):
    """Behavioural profile per actor. Units: ms, 0-1 sensor scales, px/ms, m/s^2, seconds."""
    a = age_idx.astype(float)
    L = lit

    def ln(s):
        return rng.lognormal(0.0, s, n)

    P = dict(
        iki_m=MOBILE_IKI_BY_AGE[age_idx] * ln(0.30) * (1.10 - 0.20 * L),
        iki_d=DESKTOP_IKI * ln(0.40) * (0.85 + 0.075 * a) * (1.15 - 0.30 * L),
        hold_m=95.0 * ln(0.20) * (1 + 0.05 * a),
        hold_d=DESKTOP_HOLD * ln(0.18) * (1 + 0.04 * a),
        hold_cv=rng.uniform(0.15, 0.30, n),
        iki_cv=rng.uniform(0.35, 0.60, n),
        tap_ms=90.0 * ln(0.20) * (1 + 0.06 * a),
        pressure=np.clip(rng.normal(0.45, 0.09, n), 0.15, 0.85),
        pressure_sd=rng.uniform(0.03, 0.08, n),
        tsize=np.clip(rng.normal(0.50, 0.08, n), 0.20, 0.85),
        tsize_sd=rng.uniform(0.02, 0.06, n),
        swipe_v=1.4 * ln(0.28) / (1 + 0.08 * a),
        swipe_str=np.clip(rng.normal(0.92, 0.035, n), 0.75, 0.995),
        tremor=0.22 * ln(0.35) * (1 + 0.15 * a),
        think_s=2.6 * ln(0.30) * (1 + 0.18 * a) * (1.30 - 0.50 * L),
        backspace=np.clip(rng.normal(0.05, 0.02, n) + 0.012 * a, 0.005, 0.25),
        pause_p=np.clip(0.03 + 0.04 * (1 - L) + 0.01 * a + rng.normal(0, 0.01, n), 0.005, 0.30),
        mouse_v=0.75 * ln(0.30) / (1 + 0.08 * a),
        mouse_str=np.clip(rng.normal(0.80, 0.06, n), 0.50, 0.97),
        bio_login_p=np.clip(0.25 + 0.55 * L - 0.05 * a, 0.02, 0.95),
        paste_p=np.clip(0.12 + 0.25 * L + rng.normal(0, 0.05, n), 0.02, 0.60),
        table_p=rng.uniform(0.05, 0.30, n),
        bot=np.zeros(n, bool),
    )
    if kind == "fraudster":
        P["think_s"] = P["think_s"] * 0.6
        P["pause_p"] = P["pause_p"] * 0.4
        P["paste_p"] = np.clip(P["paste_p"] + 0.35, 0, 0.92)
        P["backspace"] = P["backspace"] * 0.7
        P["bio_login_p"] = np.zeros(n)
        P["table_p"] = rng.uniform(0.3, 0.7, n)
    elif kind == "bot":
        P.update(
            iki_m=rng.uniform(55, 90, n), iki_d=rng.uniform(40, 80, n),
            hold_m=rng.uniform(40, 60, n), hold_d=rng.uniform(40, 60, n),
            hold_cv=rng.uniform(0.01, 0.04, n), iki_cv=rng.uniform(0.02, 0.06, n),
            tap_ms=rng.uniform(30, 60, n), think_s=rng.uniform(0.25, 0.6, n),
            backspace=np.zeros(n), pause_p=np.zeros(n),
            swipe_str=np.full(n, 0.999), mouse_str=np.full(n, 0.995),
            swipe_v=rng.uniform(2.5, 3.5, n), mouse_v=rng.uniform(1.5, 2.5, n),
            paste_p=np.full(n, 0.95), bio_login_p=np.zeros(n),
            bot=np.ones(n, bool), table_p=np.ones(n),
        )
    return P


def concat_profiles(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


# ----------------------------------------------------------------------------- session simulation
def make_plan(rng, ttype, platform, new_payee, login, profile_change, otp, v5, paste_payee, three_ds):
    """Ordered screens; each screen = (n_swipes, typing segments [(n_keys, kind)], paste, n_taps)."""
    if three_ds:  # 3-D Secure challenge only: OTP screen
        return [(0, [(6, "otp")], False, 1)]
    plan = []
    if login == "biometric":
        plan.append(("AUTH",))
    elif login == "autofill":
        plan.append((0, [], True, 1))
    else:
        plan.append((0, [(int(rng.integers(8, 13)), "pwd")], False, 1))
    plan.append((int(rng.integers(0, 3)), [], False, 1))                      # home
    if profile_change:
        plan.append((0, [], False, 2))                                         # settings
        plan.append((0, [(int(rng.integers(10, 19)), "text")], False, 1))    # new email / phone
        plan.append((0, [(6, "otp")], False, 1))
    if ttype == "TRANSFER":
        plan.append((0, [], False, 1))
        if new_payee:
            segs = [] if paste_payee else [(int(rng.integers(10, 17)), "digits")]
            segs.append((int(rng.integers(6, 15)), "text"))
            plan.append((0, segs, paste_payee, 1))
        else:
            plan.append((int(rng.integers(0, 3)), [], False, 1))
        ref = [(int(rng.integers(3, 13)), "text")] if rng.random() < 0.4 else []
        plan.append((0, [(int(rng.integers(2, 7)), "digits")] + ref, False, 1))
    elif ttype == "PAYMENT":
        plan.append((int(rng.integers(0, 2)), [], False, 1))
        plan.append((0, [(int(rng.integers(2, 6)), "digits")], False, 1))
    elif ttype == "CASH_IN":
        plan.append((0, [], False, 3))
        plan.append((0, [(int(rng.integers(2, 6)), "digits")], False, 1))
    else:
        plan.append((0, [(int(rng.integers(2, 6)), "digits")], False, 1))
    if v5:
        plan.append((int(rng.integers(0, 2)), [], False, 1))                   # v5 review screen
    plan.append((0, [], False, 1))                                             # confirm
    if otp:
        plan.append((0, [(6, "otp")], False, 1))
    return plan


KIND_MULT = {"pwd": 1.10, "digits": 1.25, "text": 1.0, "otp": 1.15}


def simulate_session(rng, P, a, platform, plan, mod):
    """Return (events float32 [n, NF], aggregate dict). platform: 0 android, 1 ios, 2 desktop."""
    desktop = platform == 2
    bot = bool(P["bot"][a]) or mod.get("bot", False)
    injected = mod.get("injected", False)
    types, scr, kmul = [], [], []
    bs = P["backspace"][a] * mod.get("bs_mult", 1.0)
    for s, screen in enumerate(plan):
        types.append(ET_SCREEN); scr.append(s); kmul.append(1.0)
        if screen[0] == "AUTH":
            types.append(ET_AUTH); scr.append(s); kmul.append(1.0)
            continue
        n_sw, segs, paste, n_tap = screen
        for _ in range(n_sw):
            if desktop:
                types.append(ET_MOUSE); scr.append(s); kmul.append(1.0)
            types.append(ET_SWIPE); scr.append(s); kmul.append(1.0)
        if paste:
            types.append(ET_PASTE); scr.append(s); kmul.append(1.0)
        for n_k, kind in segs:
            m = KIND_MULT[kind]
            err = rng.random(n_k) < bs
            for k in range(n_k):
                types.append(ET_KEY); scr.append(s); kmul.append(m)
                if err[k]:
                    types.append(ET_KEY); scr.append(s); kmul.append(m)
                    types.append(ET_BACKSPACE); scr.append(s); kmul.append(m)
        for _ in range(n_tap):
            if desktop:
                types.append(ET_MOUSE); scr.append(s); kmul.append(1.0)
            types.append(ET_TAP); scr.append(s); kmul.append(1.0)

    t = np.array(types, dtype=np.int16)
    n = len(t)
    kmul = np.array(kmul)
    is_key = (t == ET_KEY) | (t == ET_BACKSPACE)
    is_tap, is_sw, is_mo = t == ET_TAP, t == ET_SWIPE, t == ET_MOUSE
    is_scr, is_paste, is_auth = t == ET_SCREEN, t == ET_PASTE, t == ET_AUTH
    is_touch = (is_key | is_tap | is_sw) & (not desktop)

    iki = (P["iki_d"][a] if desktop else P["iki_m"][a]) * mod.get("iki_mult", 1.0)
    hold = (P["hold_d"][a] if desktop else P["hold_m"][a]) * mod.get("hold_mult", 1.0)
    think = P["think_s"][a] * 1000.0 * mod.get("think_mult", 1.0)
    icv, hcv = P["iki_cv"][a], P["hold_cv"][a]

    def lnorm(mu, s, size):
        return rng.lognormal(np.log(np.maximum(mu, 1e-3)) - 0.5 * s * s, s, size)

    dt = np.zeros(n)
    dt[is_key] = lnorm(iki * kmul[is_key], icv, is_key.sum())
    dt[is_tap] = lnorm(think * 0.35, 0.6, is_tap.sum())
    dt[is_sw] = lnorm(think * 0.30, 0.6, is_sw.sum())
    dt[is_mo] = lnorm(650.0 * mod.get("think_mult", 1.0), 0.5, is_mo.sum())
    dt[is_paste] = lnorm(1500.0, 0.5, is_paste.sum())
    dt[is_auth] = lnorm(900.0, 0.3, is_auth.sum())
    dt[is_scr] = lnorm(think, 0.55, is_scr.sum())
    dt[0] = 0.0
    if bot:
        dt[~is_scr] = lnorm(np.where(is_key[~is_scr], iki, 300.0), 0.04, (~is_scr).sum())
        dt[is_scr] = lnorm(think, 0.05, is_scr.sum())
        dt[0] = 0.0
    else:
        act = np.flatnonzero(~is_scr)
        pp = np.where(is_key[act], 0.3, 1.0) * P["pause_p"][a]
        hes = act[rng.random(act.size) < pp]
        dt[hes] += rng.uniform(2000, 9000, hes.size)
        n_long = mod.get("long_pauses", 0)
        if n_long and act.size:
            idx = rng.choice(act, size=min(n_long, act.size), replace=False)
            dt[idx] += rng.uniform(5000, 40000, idx.size)
    if injected:
        dt[1:] += lnorm(120.0, 0.6, n - 1)  # remote-control link latency jitter

    hold_ms = np.zeros(n)
    hold_ms[is_key] = lnorm(hold, hcv, is_key.sum())
    hold_ms[is_tap] = lnorm(P["tap_ms"][a], 0.25 if not bot else 0.03, is_tap.sum())
    hold_ms[is_auth] = lnorm(600.0, 0.3, is_auth.sum())
    if injected:
        hold_ms[is_key | is_tap] = lnorm(55.0, 0.05, (is_key | is_tap).sum())

    pressure = np.full(n, -1.0)
    size = np.full(n, -1.0)
    if is_touch.any():
        if injected or bot:
            pressure[is_touch] = 1.0 if platform == 0 else -1.0
            size[is_touch] = 1.0
        else:
            if platform == 0 and mod.get("const_pressure", False):   # no pressure sensing: constant 1.0
                pressure[is_touch] = 1.0
            elif platform == 0:  # iOS exposes no pressure on modern devices
                pressure[is_touch] = np.clip(rng.normal(P["pressure"][a], P["pressure_sd"][a], is_touch.sum()), 0.05, 1.0)
            size[is_touch] = np.clip(rng.normal(P["tsize"][a], P["tsize_sd"][a], is_touch.sum()), 0.05, 1.0)

    vel = np.full(n, -1.0)
    straight = np.full(n, -1.0)
    if desktop:
        vel[is_sw] = lnorm(2.5, 0.4, is_sw.sum())  # wheel scroll
        mv = P["mouse_v"][a] * (1.6 if injected else 1.0)
        vel[is_mo] = lnorm(mv, 0.08 if (bot or injected) else 0.35, is_mo.sum())
        ms = 0.975 if injected else P["mouse_str"][a]
        straight[is_mo] = np.clip(rng.normal(ms, 0.005 if (bot or injected) else 0.08, is_mo.sum()), 0.2, 1.0)
    else:
        vel[is_sw] = lnorm(P["swipe_v"][a], 0.05 if (bot or injected) else 0.30, is_sw.sum())
        ss = 0.995 if injected else P["swipe_str"][a]
        straight[is_sw] = np.clip(rng.normal(ss, 0.003 if (bot or injected) else 0.03, is_sw.sum()), 0.3, 1.0)

    accel = np.full(n, -1.0)
    if not desktop:
        if bot or mod.get("emulator", False):
            accel[:] = 0.0
        elif injected:
            accel[:] = lnorm(0.008, 0.4, n)
        else:
            on_table = rng.random() < P["table_p"][a]
            accel[:] = lnorm(P["tremor"][a] * (0.07 if on_table else 1.0), 0.3, n)
        if mod.get("no_motion", False):
            accel[:] = -1.0

    ev = np.stack([t.astype(float), dt, hold_ms, pressure, size, vel, straight, accel,
                   np.array(scr, dtype=float)], axis=1).astype(np.float32)

    # ---- aggregates computed from the same events
    tot = dt.sum()
    keys_dt = dt[is_key & (dt <= 2000)]   # rhythm excludes pauses, as in typing studies
    nonscr = ~is_scr
    act_idx = np.flatnonzero(nonscr)
    ag = {
        "n_events": n,
        "n_screens": int(is_scr.sum()),
        "session_duration_s": tot / 1000.0,
        "time_to_first_action_s": dt[act_idx[0]] / 1000.0 if act_idx.size else np.nan,
        "key_count": int(is_key.sum()),
        "key_hold_mean_ms": hold_ms[is_key].mean() if is_key.any() else np.nan,
        "key_hold_std_ms": hold_ms[is_key].std() if is_key.sum() > 1 else np.nan,
        "iki_mean_ms": keys_dt.mean() if keys_dt.size else np.nan,
        "iki_std_ms": keys_dt.std() if keys_dt.size > 1 else np.nan,
        "typing_speed_cps": keys_dt.size / (keys_dt.sum() / 1000.0) if keys_dt.size and keys_dt.sum() > 0 else np.nan,
        "backspace_rate": (t == ET_BACKSPACE).sum() / max(is_key.sum(), 1) if is_key.any() else np.nan,
        "paste_count": int(is_paste.sum()),
        "tap_count": int(is_tap.sum()),
        "tap_hold_mean_ms": hold_ms[is_tap].mean() if is_tap.any() else np.nan,
        "touch_pressure_mean": pressure[is_touch].mean() if (is_touch.any() and platform == 0) else np.nan,
        "touch_pressure_std": pressure[is_touch].std() if (is_touch.sum() > 1 and platform == 0) else np.nan,
        "touch_size_mean": size[is_touch].mean() if is_touch.any() else np.nan,
        "touch_size_std": size[is_touch].std() if is_touch.sum() > 1 else np.nan,
        "swipe_count": int(is_sw.sum()),
        "swipe_velocity_mean": vel[is_sw].mean() if (is_sw.any() and not desktop) else np.nan,
        "swipe_straightness_mean": straight[is_sw].mean() if (is_sw.any() and not desktop) else np.nan,
        "mouse_velocity_mean": vel[is_mo].mean() if is_mo.any() else np.nan,
        "mouse_straightness_mean": straight[is_mo].mean() if is_mo.any() else np.nan,
        "accel_std_mean": accel.mean() if (not desktop and accel[0] >= 0) else np.nan,
        "pause_count_2s": int((dt[nonscr] > 2000).sum()),
        "longest_pause_s": dt[nonscr].max() / 1000.0 if nonscr.any() else np.nan,
        "hesitation_ratio": dt[nonscr][dt[nonscr] > 2000].sum() / tot if tot > 0 else np.nan,
    }
    return ev, ag


AGG_COLS = ["n_events", "n_screens", "session_duration_s", "time_to_first_action_s", "key_count",
            "key_hold_mean_ms", "key_hold_std_ms", "iki_mean_ms", "iki_std_ms", "typing_speed_cps",
            "backspace_rate", "paste_count", "tap_count", "tap_hold_mean_ms", "touch_pressure_mean",
            "touch_pressure_std", "touch_size_mean", "touch_size_std", "swipe_count", "swipe_velocity_mean",
            "swipe_straightness_mean", "mouse_velocity_mean", "mouse_straightness_mean", "accel_std_mean",
            "pause_count_2s", "longest_pause_s", "hesitation_ratio"]


# ----------------------------------------------------------------------------- main generator
class Generator:
    def __init__(self, args):
        self.a = args
        self.rng = np.random.default_rng(args.seed)
        self.B = args.banks
        self.banks = [f"BANK_{i:02d}" for i in range(1, self.B + 1)]
        self.fs = args.fraud_scale * args.customers / 40000.0
        self.devices = []  # dicts
        self.log_t0 = time.time()

    def log(self, msg):
        print(f"[{time.time() - self.log_t0:7.1f}s] {msg}", flush=True)

    # ---------------- devices
    def new_device(self, owner_kind, owner, dtype, os_, first_day=0, emulator=False, rooted=False):
        did = len(self.devices)
        r = self.rng.random(2)
        self.devices.append(dict(device_idx=did, owner_kind=owner_kind, owner_id=owner, device_type=dtype,
                                 device_os=os_, is_emulator=int(emulator), is_rooted=int(rooted),
                                 first_day=first_day,
                                 const_pressure=int(os_ == "android" and r[0] < 0.25),
                                 emulator_detected=int(emulator and r[1] < 0.6)))
        return did

    # ---------------- customers & accounts
    def build_population(self):
        rng, a = self.rng, self.a
        C = a.customers
        S = int(round(900 * self.fs))
        N = C + S
        synth = np.zeros(N, bool)
        synth[C:] = True
        age = rng.choice(5, N, p=AGE_P)
        age[C:] = rng.choice(5, S, p=[0.20, 0.42, 0.30, 0.08, 0.0])
        inc = rng.choice(3, N, p=INCOME_P)
        ctry = rng.choice(6, N, p=COUNTRY_P)
        lit = np.clip(rng.beta(5, 2, N) - 0.10 * age + 0.05 * inc, 0.05, 0.98)
        risk = np.clip(rng.beta(2.2, 9.0, N) + 0.02 * (age == 0), 0, 1)
        peak = np.where(rng.random(N) < 0.06, rng.normal(22.5, 1.0, N), np.clip(rng.normal(13.5, 3.0, N), 7, 21))
        traveller = rng.random(N) < 0.10
        vpn_user = rng.random(N) < 0.06
        # shuffle identifiers so that ID order carries no information
        perm = rng.permutation(N) + 1
        gid = np.array([f"GC_{p:06d}" for p in perm])
        self.cust = pd.DataFrame(dict(global_customer_id=gid, age_band=np.array(AGE_BANDS)[age],
                                      income_band=np.array(INCOME)[inc], base_risk_score=risk.round(4),
                                      home_country=np.array(COUNTRIES)[ctry], digital_literacy=lit.round(3)))
        self.c_age, self.c_inc, self.c_ctry, self.c_lit = age, inc, ctry, lit
        self.c_peak, self.c_trav, self.c_vpn, self.c_synth = peak, traveller, vpn_user, synth
        self.N = N
        self.log(f"customers: {C} real + {S} synthetic identities")

        # accounts
        bank_w = rng.dirichlet(np.full(self.B, 8.0))
        rows = []
        for i in range(N):
            if synth[i]:
                k = rng.choice([1, 2, 3], p=[0.35, 0.40, 0.25])
                bks = rng.choice(self.B, size=k, replace=False, p=bank_w)
                od = int(rng.integers(0, 300))
                for j, b in enumerate(bks):
                    rows.append((i, b, j == 0, od + int(rng.integers(0, 20))))
            else:
                p = rng.choice(self.B, p=bank_w)
                rows.append((i, p, True, -int(rng.integers(30, 3000))))
                if rng.random() < 0.22:
                    k = 1 if rng.random() < 0.8 else 2
                    w = bank_w.copy(); w[p] = 0; w /= w.sum()
                    for b in rng.choice(self.B, size=k, replace=False, p=w):
                        rows.append((i, b, False, -int(rng.integers(30, 2500)) if rng.random() < 0.9 else int(rng.integers(0, 300))))
        acc = pd.DataFrame(rows, columns=["cust", "bank", "is_primary", "open_day"])
        self.acc = acc
        self.log(f"accounts: {len(acc)}  ({(acc.groupby('cust').size() > 1).mean():.1%} of customers multi-bank)")

    def assign_ids(self):
        rng = self.rng
        acc = self.acc
        ids = np.empty(len(acc), dtype=object)
        for b in range(self.B):
            m = np.flatnonzero(acc.bank.values == b)
            nums = rng.choice(np.arange(100000, 1000000), size=m.size, replace=False)
            ids[m] = [f"{self.banks[b]}_ACC_{x}" for x in nums]
        acc["local_account_id"] = ids
        acc["bank_id"] = np.array(self.banks)[acc.bank.values]
        acc["global_customer_id"] = self.cust.global_customer_id.values[acc.cust.values]
        acc["role"] = "normal"
        acc.loc[self.c_synth[acc.cust.values], "role"] = "synthetic_identity"

    def customer_devices(self):
        rng = self.rng
        N = self.N
        self.phone = np.full(N, -1)
        self.phone2 = np.full(N, -1)
        self.phone2_day = np.full(N, 10 ** 6)
        self.desktop = np.full(N, -1)
        self.shared_phone = np.full(N, -1)
        for i in np.flatnonzero(~self.c_synth):
            c = COUNTRIES[self.c_ctry[i]]
            os_ = "android" if rng.random() < ANDROID_SHARE[c] else "ios"
            self.phone[i] = self.new_device("customer", i, "phone", os_, rooted=(os_ == "android" and rng.random() < 0.02))
            if rng.random() < 0.25:
                os2 = os_ if rng.random() < 0.85 else ("ios" if os_ == "android" else "android")
                d = int(rng.integers(20, 345))
                self.phone2[i] = self.new_device("customer", i, "phone", os2, first_day=d)
                self.phone2_day[i] = d
            if rng.random() < np.clip(0.25 + 0.12 * self.c_inc[i] + 0.2 * self.c_lit[i] - 0.04 * self.c_age[i], 0.05, 0.9):
                dos = rng.choice(["windows", "macos", "linux"], p=[0.80, 0.18, 0.02])
                self.desktop[i] = self.new_device("customer", i, "desktop", dos)
        real = np.flatnonzero(~self.c_synth)
        sharers = rng.choice(real, size=int(0.08 * real.size), replace=False)
        for i in sharers:
            same = real[self.c_ctry[real] == self.c_ctry[i]]
            j = rng.choice(same)
            if j != i:
                self.shared_phone[j] = self.phone[i]
        self.log(f"customer devices: {len(self.devices)}")

    # ---------------- fraud actors
    def build_actors(self):
        rng = self.rng
        fs = self.fs
        groups = dict(ato=int(100 * fs) + 5, herder=int(60 * fs) + 3, launder=int(45 * fs) + 3,
                      carder=int(50 * fs) + 3, farm=int(15 * fs) + 2, bot=int(60 * fs) + 5)
        n_h = groups["ato"] + groups["herder"] + groups["launder"] + groups["carder"] + groups["farm"]
        age = rng.choice(5, n_h, p=[0.35, 0.40, 0.20, 0.05, 0.0])
        lit = np.clip(rng.beta(8, 2, n_h), 0.5, 0.99)
        Ph = make_profiles(rng, n_h, age, lit, "fraudster")
        Pb = make_profiles(rng, groups["bot"], np.zeros(groups["bot"], int), np.ones(groups["bot"]), "bot")
        Pc = make_profiles(rng, self.N, self.c_age, self.c_lit, "human")
        self.P = concat_profiles([Pc, Ph, Pb])
        off = self.N
        self.actor = {}
        self.actor_country = {}
        self.actor_devices = {}
        for g in ["ato", "herder", "launder", "carder", "farm"]:
            ids = list(range(off, off + groups[g]))
            off += groups[g]
            self.actor[g] = ids
            for x in ids:
                self.actor_country[x] = (rng.choice(COUNTRIES, p=COUNTRY_P) if rng.random() < 0.65 else rng.choice(FOREIGN))
                devs = []
                if g == "farm":
                    nd = int(rng.integers(4, 13))
                    for _ in range(nd):
                        emu = rng.random() < 0.7
                        devs.append(self.new_device("farm", x, "phone", "android", emulator=emu, rooted=not emu and rng.random() < 0.5))
                else:
                    nd = int(rng.integers(1, 4))
                    for _ in range(nd):
                        r = rng.random()
                        if r < 0.55:
                            devs.append(self.new_device("fraudster", x, "phone", "android", emulator=rng.random() < 0.12, rooted=rng.random() < 0.3))
                        elif r < 0.65:
                            devs.append(self.new_device("fraudster", x, "phone", "ios"))
                        else:
                            devs.append(self.new_device("fraudster", x, "desktop", rng.choice(["windows", "linux"], p=[0.8, 0.2])))
                self.actor_devices[x] = devs
        self.actor["bot"] = list(range(off, off + groups["bot"]))
        # Q4-only device farm (unseen until the test period)
        x = self.actor["farm"][-1]
        self.actor_devices[x] = [self.new_device("farm", x, "phone", "android", first_day=NEW_FARM_DAY, emulator=True)
                                 for _ in range(int(rng.integers(6, 12)))]
        self.log(f"fraud actors: {groups}")

    # ---------------- roles chosen before legit generation
    def assign_roles(self):
        rng, acc = self.rng, self.acc
        real = np.flatnonzero(~self.c_synth)
        # mules
        n_m = int(480 * self.fs)
        w = np.array([4.0, 2.0, 1.0, 0.4, 0.1])[self.c_age[real]] * np.array([1.5, 1.0, 0.6])[self.c_inc[real]]
        mule_c = rng.choice(real, size=n_m, replace=False, p=w / w.sum())
        new_rows = []
        mule_acc = []
        for c in mule_c:
            mine = np.flatnonzero(acc.cust.values == c)
            if rng.random() < 0.30:
                have = set(acc.bank.values[mine])
                cand = [b for b in range(self.B) if b not in have]
                if cand:
                    new_rows.append((c, int(rng.choice(cand)), False, int(rng.integers(0, 320))))
                    continue
            mule_acc.append(int(rng.choice(mine)))
        if new_rows:
            add = pd.DataFrame(new_rows, columns=["cust", "bank", "is_primary", "open_day"])
            start = len(acc)
            self.acc = acc = pd.concat([acc, add], ignore_index=True)
            mule_acc += list(range(start, start + len(add)))
        self.mule_acc = np.array(mule_acc)
        self.mule_day = np.maximum(acc.open_day.values[self.mule_acc], 0) + rng.integers(0, 60, self.mule_acc.size)
        self.mule_day = np.minimum(self.mule_day, 340)
        herders = self.actor["herder"]
        self.mule_herder = np.where(rng.random(self.mule_acc.size) < 0.55, rng.choice(herders, self.mule_acc.size), -1)
        # synthetic identities -> farms
        synth_c = np.flatnonzero(self.c_synth)
        farms = self.actor["farm"][:-1]
        self.synth_farm = {c: int(rng.choice(farms)) for c in synth_c}
        late = synth_c[rng.random(synth_c.size) < 0.08]
        for c in late:   # identities created by the Q4 farm
            self.synth_farm[c] = self.actor["farm"][-1]
        # first-party
        n_fp = int(400 * self.fs)
        prim = acc.index[(acc.is_primary.values) & (~self.c_synth[acc.cust.values])].values
        prim = np.setdiff1d(prim, self.mule_acc)
        self.fp_acc = rng.choice(prim, size=n_fp, replace=False)
        self.fp_day = rng.integers(20, 350, n_fp)
        self.log(f"roles: {self.mule_acc.size} mule accounts, {synth_c.size} synthetic identities, {n_fp} first-party")

    # ---------------- legitimate activity
    def legit_transactions(self):
        rng, acc = self.rng, self.acc
        A = len(acc)
        cust = acc.cust.values
        prim = acc.is_primary.values
        synth = self.c_synth[cust]
        open_day = np.maximum(acc.open_day.values, 0)
        end_day = np.full(A, DAYS - 1)
        fp_end = dict(zip(self.fp_acc, self.fp_day))
        for k, d in fp_end.items():
            end_day[k] = d + 5
        act = np.where(prim, 15.0, 6.0) * np.array([0.9, 1.0, 1.3])[self.c_inc[cust]]
        act = np.where(synth, 6.0, act)
        is_mule = np.zeros(A, bool); is_mule[self.mule_acc] = True
        act = np.where(is_mule, act * 0.4, act)
        span = (end_day - open_day + 1) / DAYS
        n_rand = rng.poisson(act * rng.lognormal(0, 0.45, A) * span)
        self.acc_init_bal = np.round(rng.lognormal(np.log(np.array([400, 1500, 6000])[self.c_inc[cust]]), 0.9), 2)

        parts = []
        # random activity
        idx = np.repeat(np.arange(A), n_rand)
        lo, hi = open_day[idx], end_day[idx]
        day = lo + np.floor(rng.random(idx.size) * (hi - lo + 1)).astype(int)
        ttype = rng.choice(["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"], idx.size,
                           p=[0.38, 0.22, 0.28, 0.07, 0.05])
        parts.append(pd.DataFrame(dict(acc=idx, day=day, ttype=ttype)))
        # salary (primary, 60%) and standing orders (70%) monthly
        sal = np.flatnonzero(prim & ~synth & (rng.random(A) < 0.45))
        sal_day = rng.integers(0, 28, A)
        months = np.arange(12)
        mstart = (months * 30.42).astype(int)
        rs = np.repeat(sal, 12); rm = np.tile(mstart, sal.size) + sal_day[rs]
        ok = (rm >= open_day[rs]) & (rm <= end_day[rs])
        parts.append(pd.DataFrame(dict(acc=rs[ok], day=rm[ok], ttype="CASH_IN", recurring="salary")))
        so = np.flatnonzero(~synth & (rng.random(A) < np.where(prim, 0.35, 0.10)))
        so_day = rng.integers(0, 28, A)
        rs = np.repeat(so, 12); rm = np.tile(mstart, so.size) + so_day[rs]
        ok = (rm >= open_day[rs]) & (rm <= end_day[rs])
        parts.append(pd.DataFrame(dict(acc=rs[ok], day=rm[ok], ttype="DEBIT", recurring="standing_order")))
        L = pd.concat(parts, ignore_index=True)
        L["recurring"] = L["recurring"].fillna("")
        n = len(L)
        a_ = L.acc.values
        c_ = cust[a_]
        ag, li = self.c_age[c_], self.c_lit[c_]
        # time of day
        hour = np.clip(rng.normal(self.c_peak[c_], 3.5), 0, 23.99)
        hour = np.where(L.recurring.values != "", rng.uniform(0, 6, n), hour)   # batch-processed at night
        L["t_sec"] = (L.day.values * 86400 + hour * 3600 + rng.uniform(0, 60, n)).astype(np.int64)
        # channel
        mob = np.clip(0.85 - 0.13 * ag + 0.15 * (li - 0.5), 0.05, 0.95)
        u = rng.random(n)
        tt = L.ttype.values
        ch = np.empty(n, dtype=object)
        m = tt == "TRANSFER"
        ch[m] = np.where(u[m] < mob[m], "mobile", np.where(u[m] < mob[m] + (1 - mob[m]) * 0.75, "web", "branch"))
        m = tt == "PAYMENT"
        ch[m] = np.where(u[m] < 0.48, "pos", np.where(u[m] < 0.48 + 0.30, "web", "mobile"))
        m = tt == "CASH_OUT"
        ch[m] = np.where(u[m] < 0.85, "atm", "branch")
        m = tt == "CASH_IN"
        ch[m] = np.where(u[m] < 0.35, "branch", np.where(u[m] < 0.6, "atm", "mobile"))
        ch[tt == "DEBIT"] = "system"
        ch[L.recurring.values == "salary"] = "system"
        # mobile share drifts up over the year
        drift = (L.day.values > 200) & (ch == "branch") & (tt == "TRANSFER") & (rng.random(n) < 0.3)
        ch[drift] = "mobile"
        L["channel"] = ch
        # amounts
        med = np.array([45.0, 95.0, 220.0])[self.c_inc[c_]]
        mult = pd.Series(tt).map({"TRANSFER": 1.6, "PAYMENT": 0.7, "CASH_OUT": 1.0, "CASH_IN": 1.8, "DEBIT": 0.9}).values
        amt = med * mult * rng.lognormal(0, 0.9, n)
        salary_amt = np.array([600.0, 1800.0, 5000.0])[self.c_inc[cust]] * rng.lognormal(0, 0.25, A)
        so_amt = salary_amt * rng.uniform(0.10, 0.35, A)
        rec = L.recurring.values
        amt = np.where(rec == "salary", salary_amt[a_] * rng.normal(1, 0.02, n), amt)
        amt = np.where(rec == "standing_order", so_amt[a_], amt)
        big = (tt == "TRANSFER") & (rng.random(n) < 0.005)                     # car, rent deposit...
        amt = np.where(big, amt * rng.uniform(8, 25, n), amt)
        micro = (tt == "PAYMENT") & (ch == "web") & (rng.random(n) < 0.06)
        amt = np.where(micro, rng.choice([0.99, 1.49, 1.99, 2.99, 3.99, 4.99], n), amt)
        cash = tt == "CASH_OUT"
        amt = np.where(cash, np.maximum(20, np.round(amt / 20) * 20), amt)
        amt = np.where(cash & (ch == "atm"), np.minimum(amt, 1000), amt)
        L["amount"] = np.round(amt, 2)
        L["big_purchase"] = big
        # destinations
        self._assign_legit_dest(L, salary_amt)
        L["is_fraud"] = 0
        L["fraud_type"] = "none"
        L["episode"] = -1
        L["mode"] = M_LEGIT
        L["actor"] = c_
        L["soph"] = 0
        # synthetic identities' build-up is operated by their farm
        sy = self.c_synth[c_]
        L.loc[sy, "actor"] = [self.synth_farm[c] for c in c_[sy]]
        L.loc[sy, "mode"] = M_SYNTH
        self.legit = L
        self.log(f"legit transactions: {n}")

    def _assign_legit_dest(self, L, salary_amt):
        rng, acc = self.rng, self.acc
        A = len(acc)
        n = len(L)
        bank = acc.bank.values
        same_bank_pool = [np.flatnonzero(bank == b) for b in range(self.B)]
        # payee book per account: up to 8 payees
        k = rng.integers(3, 9, A)
        book = np.full((A, 8), -1, dtype=np.int64)
        book_ext = np.zeros((A, 8), bool)
        def rand_in_bank(bks):
            out = np.empty(bks.size, dtype=np.int64)
            for b in range(self.B):
                m = bks == b
                if m.any():
                    out[m] = rng.choice(same_bank_pool[b], m.sum())
            return out

        def rand_other_bank(bks):
            out = rng.integers(0, A, bks.size)
            bad = bank[out] == bks
            while bad.any():
                out[bad] = rng.integers(0, A, bad.sum())
                bad = bank[out] == bks
            return out

        for j in range(8):
            r = rng.random(A)
            tgt = np.where(r < 0.50, rand_in_bank(bank), rand_other_bank(bank))
            book[:, j] = tgt
            book_ext[:, j] = r >= 0.80
        dest_acc = np.full(n, -1, dtype=np.int64)
        dest = np.empty(n, dtype=object)
        a_ = L.acc.values
        tt = L.ttype.values
        m = np.flatnonzero(tt == "TRANSFER")
        rep = rng.random(m.size) < np.where(L.big_purchase.values[m], 0.1, 0.8)
        slot = (rng.random(m.size) * k[a_[m]]).astype(int)
        dsel = book[a_[m], slot]
        ext = book_ext[a_[m], slot]
        # new (non-book) payees
        r = rng.random(m.size)
        sb = bank[a_[m]]
        new_t = np.where(r < 0.5, rand_in_bank(sb), rand_other_bank(sb))
        new_ext = (~rep) & (r >= 0.8)
        final = np.where(rep, np.where(ext, -1, dsel), new_t)
        is_ext = np.where(rep, ext, new_ext)
        final[final == a_[m]] = -1
        is_ext |= final < 0
        dest_acc[m] = np.where(is_ext, -1, final)
        ext_id = np.where(rep, a_[m] * 8 + slot, 10 ** 7 + rng.integers(0, 10 ** 6, m.size))
        dest[m] = [f"EXT_P{x:08d}" for x in ext_id]
        m2 = np.flatnonzero(tt == "PAYMENT")
        merch = np.minimum(rng.zipf(1.4, m2.size), 6000)
        dest[m2] = [f"EXT_M{x:05d}" for x in merch]
        m3 = np.flatnonzero(tt == "CASH_OUT")
        dest[m3] = [f"EXT_ATM_{bank[x]:02d}_{y:03d}" for x, y in zip(a_[m3], rng.integers(0, 300, m3.size))]
        m4 = np.flatnonzero(tt == "CASH_IN")
        dest[m4] = [f"EXT_DEP_{bank[x]:02d}" for x in a_[m4]]
        sal = L.recurring.values == "salary"
        dest[sal] = [f"EXT_EMP{(x * 7919) % 4000:04d}" for x in a_[sal]]
        m5 = np.flatnonzero(tt == "DEBIT")
        dest[m5] = [f"EXT_B{(x * 104729) % 900:03d}" for x in a_[m5]]
        dest_acc_str = dest_acc >= 0
        L["dest_acc"] = dest_acc
        L["dest_ext"] = dest
        L.loc[dest_acc_str, "dest_ext"] = None

    # ---------------- fraud episodes
    def _row(self, rows, acc, t_sec, ttype, amount, channel, dest_acc, dest_ext, ftype, ep, mode, actor,
             device=-1, soph=0, flags=0):
        rows.append((acc, int(t_sec), ttype, round(float(amount), 2), channel, int(dest_acc), dest_ext,
                     1, ftype, ep, mode, int(actor), int(device), soph, flags))

    def _pick_mule(self, day, not_bank=None, prefer_other=0.7):
        ok = self.mule_day <= day
        idx = np.flatnonzero(ok)
        if idx.size == 0:
            return None
        if not_bank is not None and self.rng.random() < prefer_other:
            ob = idx[self.acc.bank.values[self.mule_acc[idx]] != not_bank]
            if ob.size:
                idx = ob
        return int(self.rng.choice(idx))

    def _mule_forward(self, rows, mi, amount, t_sec, ep, depth=0):
        rng = self.rng
        acc_i = int(self.mule_acc[mi])
        herder = self.mule_herder[mi]
        cust = int(self.acc.cust.values[acc_i])
        actor = herder if herder >= 0 else cust
        dev = int(rng.choice(self.actor_devices[herder])) if herder >= 0 else -2   # -2: own phone
        keep = amount * (1 - rng.uniform(0.05, 0.10))
        n_out = int(rng.integers(1, 3))
        shares = rng.dirichlet(np.ones(n_out)) * keep
        t = t_sec
        for s in shares:
            t = t + int(rng.uniform(0.5, 36) * 3600)
            if t >= DAYS * 86400:
                return
            r = rng.random()
            if r < 0.5:
                left = s
                while left > 20:
                    chunk = min(left, rng.choice([1000, 1500, 2000]))
                    chunk = max(20, np.floor(chunk / 20) * 20)
                    self._row(rows, acc_i, t, "CASH_OUT", chunk, "atm", -1,
                              f"EXT_ATM_{int(self.acc.bank.values[acc_i]):02d}_{int(rng.integers(0, 300)):03d}",
                              "mule_account", ep, M_MULE, actor, dev)
                    left -= chunk
                    t += int(rng.uniform(60, 1800))
            elif r < 0.70 and depth < 1:
                nxt = self._pick_mule(t // 86400, not_bank=int(self.acc.bank.values[acc_i]), prefer_other=0.8)
                if nxt is None or nxt == mi:
                    continue
                self._row(rows, acc_i, t, "TRANSFER", s, "mobile" if dev != -2 or rng.random() < 0.7 else "web",
                          int(self.mule_acc[nxt]), None, "mule_account", ep, M_MULE, actor, dev)
                self._mule_forward(rows, nxt, s, t, ep, depth + 1)
            else:
                self._row(rows, acc_i, t, "TRANSFER", s, "mobile", -1, f"EXT_X{int(rng.integers(0, 60)):03d}",
                          "mule_account", ep, M_MULE, actor, dev)

    def _start_day(self, weights_by_month):
        w = np.asarray(weights_by_month, float)
        m = self.rng.choice(12, p=w / w.sum())
        return int(min(DAYS - 2, m * 30.42 + self.rng.uniform(0, 30)))

    def fraud_episodes(self):
        rng, acc = self.rng, self.acc
        fs = self.fs
        rows = []
        ep = 0
        cust = acc.cust.values
        bank = acc.bank.values
        real_acc = np.flatnonzero(~self.c_synth[cust] & (acc.role.values == "normal"))
        mule_set = set(self.mule_acc.tolist())
        excl = mule_set | set(self.fp_acc.tolist())
        real_acc = np.array([x for x in real_acc if x not in excl])
        age_w = np.array([0.6, 0.8, 1.0, 1.6, 2.4])

        def victim(weight_age=True):
            w = age_w[self.c_age[cust[real_acc]]] if weight_age else np.ones(real_acc.size)
            return int(rng.choice(real_acc, p=w / w.sum()))

        def hour_t(day, peak=None):
            h = np.clip(rng.normal(peak if peak is not None else 14, 4), 0, 23.9)
            return int(day * 86400 + h * 3600 + rng.uniform(0, 60))

        # --- account takeover
        n_ato = int(750 * fs)
        for _ in range(n_ato):
            v = victim()
            c = int(cust[v])
            day = self._start_day(np.linspace(1.2, 0.8, 12))
            variant_rat = rng.random() < (0.20 + 0.25 * (day > 180))
            soph = int(rng.random() < min(0.95, 0.2 * self.a.sophistication))
            crew = int(rng.choice(self.actor["ato"]))
            if variant_rat:
                pdev = self.phone[c]
                if self.devices[pdev]["device_os"] == "ios":   # iOS cannot be remote-controlled
                    pdev = self.desktop[c] if self.desktop[c] >= 0 else -1
                if pdev < 0:
                    variant_rat = False
            mode = M_ATO_RAT if variant_rat else M_ATO_CRED
            dev = pdev if variant_rat else int(rng.choice(self.actor_devices[crew]))
            ch = "web" if self.devices[dev]["device_type"] == "desktop" else "mobile"
            bal = self.acc_init_bal[v]
            total = max(300.0, bal * rng.uniform(0.4, 0.9))
            k = int(rng.integers(1, 5))
            t = hour_t(day, 2 if rng.random() < 0.4 else 14)
            for j, s in enumerate(rng.dirichlet(np.ones(k)) * total):
                mi = self._pick_mule(day, not_bank=int(bank[v]))
                flags = 1 if (j == 0 and rng.random() < (0.3 if soph else 0.5)) else 0   # profile change
                if mi is None:
                    self._row(rows, v, t, "TRANSFER", s, ch, -1, f"EXT_X{int(rng.integers(0, 60)):03d}",
                              "account_takeover", ep, mode, crew, dev, soph, flags)
                else:
                    self._row(rows, v, t, "TRANSFER", s, ch, int(self.mule_acc[mi]), None,
                              "account_takeover", ep, mode, crew, dev, soph, flags)
                    self._mule_forward(rows, mi, s, t, ep)
                t += int(rng.uniform(3, 90) * 60)
            ep += 1

        # --- social engineering / authorised push payment (coerced victim)
        n_app = int(700 * fs)
        for _ in range(n_app):
            v = victim()
            c = int(cust[v])
            day = self._start_day(np.linspace(0.6, 1.6, 12))
            soph = int(rng.random() < min(0.95, 0.25 * self.a.sophistication))
            bal = self.acc_init_bal[v]
            k = int(rng.choice([1, 2, 3], p=[0.55, 0.30, 0.15]))
            t = hour_t(day, self.c_peak[c])
            for s in rng.dirichlet(np.ones(k)) * max(250.0, bal * rng.uniform(0.3, 0.95)):
                mi = self._pick_mule(t // 86400, not_bank=int(bank[v]), prefer_other=0.6)
                ch = "mobile" if rng.random() < 0.75 else "web"
                if mi is None:
                    self._row(rows, v, t, "TRANSFER", s, ch, -1, f"EXT_P{int(rng.integers(10 ** 7, 2 * 10 ** 7)):08d}",
                              "social_engineering_app", ep, M_APP, c, -2, soph)
                else:
                    self._row(rows, v, t, "TRANSFER", s, ch, int(self.mule_acc[mi]), None,
                              "social_engineering_app", ep, M_APP, c, -2, soph)
                    self._mule_forward(rows, mi, s, t, ep)
                t += int(rng.uniform(0.3, 30) * 3600)
            ep += 1

        # --- card-not-present (card testing)
        n_cnp = int(350 * fs)
        for _ in range(n_cnp):
            v = victim(weight_age=False)
            day = self._start_day([0.8] * 9 + [1.0, 1.6, 2.0])
            carder = int(rng.choice(self.actor["carder"]))
            botp = int(rng.choice(self.actor["bot"]))
            dev = int(rng.choice(self.actor_devices[carder]))
            t = hour_t(day, 3)
            for _ in range(int(rng.integers(2, 6))):
                self._row(rows, v, t, "PAYMENT", rng.uniform(0.5, 4.99), "web", -1,
                          f"EXT_M{int(rng.integers(1, 6000)):05d}", "card_not_present", ep, M_CNP, botp, dev)
                t += int(rng.uniform(1, 12) * 60)
            t += int(rng.uniform(0.5, 20) * 3600)
            for _ in range(int(rng.integers(1, 4))):
                self._row(rows, v, t, "PAYMENT", rng.uniform(80, 900), "web", -1,
                          f"EXT_M{int(rng.integers(1, 6000)):05d}", "card_not_present", ep, M_CNP, carder, dev)
                t += int(rng.uniform(5, 120) * 60)
            ep += 1

        # --- first-party (genuine customer spree before default)
        for v, d in zip(self.fp_acc, self.fp_day):
            c = int(cust[v])
            med = np.array([45.0, 95.0, 220.0])[self.c_inc[c]]
            t = hour_t(int(d), self.c_peak[c])
            for _ in range(int(rng.integers(3, 8))):
                if rng.random() < 0.6:
                    self._row(rows, v, t, "PAYMENT", med * rng.uniform(2.5, 8),
                              rng.choice(["pos", "web", "mobile"], p=[0.4, 0.35, 0.25]), -1,
                              f"EXT_M{int(rng.zipf(1.4)) % 6000:05d}", "first_party", ep, M_FP, c, -2)
                else:
                    self._row(rows, v, t, "TRANSFER", med * rng.uniform(2.5, 8), "mobile", -1,
                              f"EXT_P{int(rng.integers(10 ** 7, 2 * 10 ** 7)):08d}", "first_party", ep, M_FP, c, -2)
                t += int(rng.uniform(2, 48) * 3600)
            ep += 1

        # --- synthetic identity bust-out (coordinated across that identity's banks)
        sy_acc = pd.Series(np.flatnonzero(self.c_synth[cust])).groupby(cust[self.c_synth[cust]]).apply(list)
        self.ring_pool = []
        for c, accs in sy_acc.items():
            if rng.random() > 0.70:
                self.ring_pool += accs
                continue
            farm = self.synth_farm[c]
            od = int(max(acc.open_day.values[accs].max(), 0))
            lo = od + 60
            if farm == self.actor["farm"][-1]:
                lo = max(lo, NEW_FARM_DAY)
            if lo >= DAYS - 3:
                self.ring_pool += accs
                continue
            d = int(rng.integers(lo, DAYS - 2))
            for a_ in accs:
                t = hour_t(d + int(rng.integers(0, 3)), 20)
                for _ in range(int(rng.integers(1, 4))):
                    r = rng.random()
                    if r < 0.45:
                        self._row(rows, a_, t, "PAYMENT", rng.uniform(300, 3000), rng.choice(["web", "pos"]), -1,
                                  f"EXT_M{int(rng.integers(1, 400)):05d}", "synthetic_identity", ep, M_SYNTH, farm, -3)
                    elif r < 0.75:
                        self._row(rows, a_, t, "CASH_OUT", float(rng.choice([500, 800, 1000])), "atm", -1,
                                  f"EXT_ATM_{int(bank[a_]):02d}_{int(rng.integers(0, 300)):03d}",
                                  "synthetic_identity", ep, M_SYNTH, farm, -3)
                    else:
                        self._row(rows, a_, t, "TRANSFER", rng.uniform(500, 4000), "mobile", -1,
                                  f"EXT_X{int(rng.integers(0, 60)):03d}", "synthetic_identity", ep, M_SYNTH, farm, -3)
                    t += int(rng.uniform(0.2, 8) * 3600)
            ep += 1

        # --- layering rings (structured placement, fan-in, cross-bank fan-out, exit)
        n_ring = int(60 * self.fs)
        recruit = rng.choice(real_acc, size=n_ring * 3, replace=False)
        self.ring_members = []
        pool = list(self.ring_pool)
        rng.shuffle(pool)
        for r in range(n_ring):
            size = int(rng.integers(4, 10))
            mem = [pool.pop() for _ in range(min(len(pool), size // 2))]
            mem += list(recruit[r * 3:(r + 1) * 3])
            mem = list(dict.fromkeys(mem))[:size]
            if len(set(bank[mem])) < 2 or len(mem) < 3:
                continue
            self.ring_members += mem
            ctrl = int(rng.choice(self.actor["launder"]))
            scripted = rng.random() < 0.4
            bot = int(rng.choice(self.actor["bot"]))
            act_ = bot if scripted else ctrl
            cdev = int(rng.choice(self.actor_devices[ctrl]))
            for _ in range(int(rng.integers(2, 6))):
                day = int(rng.integers(5, DAYS - 6))
                t = hour_t(day)
                placed = []
                for _ in range(int(rng.integers(2, 6))):
                    m = int(rng.choice(mem))
                    amt = rng.uniform(7000, 9900)
                    self._row(rows, m, t, "CASH_IN", amt, rng.choice(["branch", "atm"], p=[0.7, 0.3]), -1,
                              f"EXT_DEP_{int(bank[m]):02d}", "money_laundering_layering", ep, M_RING, ctrl)
                    placed.append((m, amt))
                    t += int(rng.uniform(1, 20) * 3600)
                coll = int(rng.choice(mem))
                total = 0.0
                for m, amt in placed:
                    t += int(rng.uniform(2, 30) * 3600)
                    if m != coll:
                        x = amt * (1 - rng.uniform(0.005, 0.03))
                        self._row(rows, m, t, "TRANSFER", x, rng.choice(["mobile", "web"]), coll, None,
                                  "money_laundering_layering", ep, M_RING, act_, cdev)
                        total += x
                    else:
                        total += amt
                outs = [m for m in mem if m != coll and bank[m] != bank[coll]] or [m for m in mem if m != coll]
                k = int(min(len(outs), rng.integers(2, 5)))
                for m, s in zip(rng.choice(outs, size=k, replace=False), rng.dirichlet(np.ones(k)) * total * 0.98):
                    t += int(rng.uniform(2, 30) * 3600)
                    self._row(rows, coll, t, "TRANSFER", s, rng.choice(["mobile", "web"]), int(m), None,
                              "money_laundering_layering", ep, M_RING, act_, cdev)
                    t2 = t + int(rng.uniform(2, 30) * 3600)
                    if rng.random() < 0.5:
                        self._row(rows, int(m), t2, "CASH_OUT", min(9900, s * 0.99), "branch", -1,
                                  f"EXT_DEP_{int(bank[m]):02d}", "money_laundering_layering", ep, M_RING, ctrl)
                    else:
                        self._row(rows, int(m), t2, "TRANSFER", s * 0.99, rng.choice(["mobile", "web"]), -1,
                                  f"EXT_C{int(rng.integers(0, 300)):04d}", "money_laundering_layering", ep, M_RING, act_, cdev)
                ep += 1
        F = pd.DataFrame(rows, columns=["acc", "t_sec", "ttype", "amount", "channel", "dest_acc", "dest_ext",
                                        "is_fraud", "fraud_type", "episode", "mode", "actor", "device", "soph", "flags"])
        F = F[F.t_sec < DAYS * 86400].reset_index(drop=True)
        F["day"] = F.t_sec // 86400
        F["recurring"] = ""
        F["big_purchase"] = False
        self.fraud = F
        ra = self.acc.role.values.copy()
        ra[self.mule_acc] = "mule"
        ra[np.array(self.ring_members, dtype=int)] = np.where(ra[np.array(self.ring_members, dtype=int)] == "synthetic_identity",
                                                              "synthetic_identity_ring", "laundering_ring")
        ra[self.fp_acc] = "first_party"
        self.acc["role"] = ra
        self.log(f"fraud rows: {len(F)}  by type: {F.fraud_type.value_counts().to_dict()}")

    # ---------------- assemble, ledger and causal transaction features
    def assemble(self):
        L = self.legit.copy()
        L["device"] = -1
        L["flags"] = 0
        cols = ["acc", "t_sec", "day", "ttype", "amount", "channel", "dest_acc", "dest_ext", "is_fraud",
                "fraud_type", "episode", "mode", "actor", "device", "soph", "flags", "recurring", "big_purchase"]
        T = pd.concat([L[cols], self.fraud[cols]], ignore_index=True)
        T = T.sort_values(["t_sec", "acc"], kind="mergesort").reset_index(drop=True)
        acc = self.acc
        T["bank"] = acc.bank.values[T.acc.values]
        T["cust"] = acc.cust.values[T.acc.values]
        da = T.dest_acc.values
        T["dest_bank"] = np.where(da >= 0, acc.bank.values[np.maximum(da, 0)], -1)
        T["is_cross_bank"] = ((da >= 0) & (T.dest_bank.values != T.bank.values)).astype(int)
        T["dest_account_id"] = np.where(da >= 0, acc.local_account_id.values[np.maximum(da, 0)], T.dest_ext.values)
        # ledger: initiator debit/credit + internal beneficiary credit
        sign = np.where(T.ttype.values == "CASH_IN", 1.0, -1.0)
        led_acc = np.concatenate([T.acc.values, da[da >= 0]])
        led_t = np.concatenate([T.t_sec.values, T.t_sec.values[da >= 0] + 1])
        led_d = np.concatenate([sign * T.amount.values, T.amount.values[da >= 0]])
        led_row = np.concatenate([np.arange(len(T)), np.full((da >= 0).sum(), -1)])
        o = np.lexsort((led_t, led_acc))
        la, ld, lr = led_acc[o], led_d[o], led_row[o]
        csum = np.cumsum(ld)
        first = np.r_[True, la[1:] != la[:-1]]
        grp_start = np.maximum.accumulate(np.where(first, np.arange(la.size), 0))
        base = np.where(grp_start > 0, csum[grp_start - 1], 0.0)
        run = csum - base                       # balance relative to init, after each entry
        minrun = pd.Series(run).groupby(la).min()
        init = self.acc_init_bal.copy()
        exempt = np.isin(acc.role.values, ["synthetic_identity", "synthetic_identity_ring", "first_party"])
        need = np.zeros(len(acc))
        need[minrun.index.values] = -minrun.values
        raise_to = need + self.rng.uniform(10, 200, len(acc))
        init = np.where(~exempt & (init < raise_to), raise_to, init)
        new_bal = init[la] + run
        m = lr >= 0
        nb = np.empty(len(T)); nb[lr[m]] = new_bal[m]
        T["new_balance"] = np.round(nb, 2)
        T["old_balance"] = np.round(nb - sign * T.amount.values, 2)
        self.acc_init_bal = init
        # time features (causal)
        T = T.sort_values(["acc", "t_sec"], kind="mergesort").reset_index(drop=True)
        key = T.acc.values.astype(np.int64) * 10 ** 9 + T.t_sec.values
        i = np.arange(len(T))
        T["velocity_1h"] = i - np.searchsorted(key, key - 3600, side="left")
        T["velocity_24h"] = i - np.searchsorted(key, key - 86400, side="left")
        prev = np.r_[-1, T.t_sec.values[:-1]]
        newacc = np.r_[True, T.acc.values[1:] != T.acc.values[:-1]]
        T["time_since_last_hrs"] = np.where(newacc, 0, (T.t_sec.values - prev) / 3600.0).round(2)
        g = T.groupby("acc").amount
        prev_mean = (g.cumsum() - T.amount) / g.cumcount().replace(0, np.nan)
        T["amount_to_avg_ratio"] = (T.amount / prev_mean).round(3)
        T["payee_is_new"] = (~T.duplicated(["acc", "dest_account_id"])).astype(int)
        T.loc[~T.ttype.isin(["TRANSFER"]), "payee_is_new"] = 0
        self.T = T
        self.log(f"assembled {len(T)} transactions; fraud rate {T.is_fraud.mean():.3%}")

    # ---------------- sessions, devices, raw events
    def sessions(self):
        rng, T, P = self.rng, self.T, self.P
        n = len(T)
        ch, tt = T.channel.values, T.ttype.values
        mode = T["mode"].values
        has_session = np.isin(ch, ["mobile", "web"])
        self.max_len = self.a.seq_len
        dev_out = np.full(n, -1)
        stype = np.full(n, "none", dtype=object)
        login = np.full(n, "none", dtype=object)
        behav = np.zeros(n, np.int8)
        ctx = {k: np.zeros(n) for k in ["is_emulator", "is_rooted", "remote_access_flag", "active_call_flag",
                                         "vpn_flag", "profile_change_in_session"]}
        geo = np.full(n, np.nan)
        ipc = np.full(n, "", dtype=object)
        payee_method = np.full(n, "none", dtype=object)
        agg = np.full((n, len(AGG_COLS)), np.nan)
        events = {}
        acc_cust = T.cust.values
        days = T.day.values
        home = np.array(COUNTRIES)[self.c_ctry]
        a_actor, a_dev, a_soph, a_flags = T.actor.values, T.device.values, T["soph"].values, T["flags"].values
        a_pnew, a_amt = T.payee_is_new.values, T.amount.values
        n_done = 0
        for i in np.flatnonzero(has_session):
            c = int(acc_cust[i]); mo = int(mode[i]); actor = int(a_actor[i]); d = int(days[i])
            mod = {}
            # ---- device
            dv = int(a_dev[i])
            if mo in (M_LEGIT, M_APP, M_FP) or (mo == M_MULE and dv == -2):
                ph = self.phone2[c] if (self.phone2[c] >= 0 and d >= self.phone2_day[c]) else self.phone[c]
                if self.shared_phone[c] >= 0 and rng.random() < 0.25:
                    ph = self.shared_phone[c]
                if ch[i] == "web":
                    if rng.random() < 0.01:
                        dv = self.new_device("public", c, "desktop", "windows", first_day=d)
                    elif self.desktop[c] >= 0 and rng.random() < 0.75:
                        dv = self.desktop[c]
                    else:
                        dv = ph
                else:
                    dv = ph
            elif mo in (M_SYNTH,) or dv == -3:
                devs = [x for x in self.actor_devices[self.synth_farm[c]] if self.devices[x]["first_day"] <= d] \
                    or self.actor_devices[self.synth_farm[c]]
                dv = int(devs[c % len(devs)]) if rng.random() < 0.8 else int(rng.choice(devs))
            dev = self.devices[dv]
            dev_out[i] = dv
            dtype_ = dev["device_type"]
            platform = 2 if dtype_ == "desktop" else (0 if dev["device_os"] == "android" else 1)
            # ---- session type and context
            three_ds = (tt[i] == "PAYMENT") and ch[i] == "web"
            stype[i] = "3ds" if three_ds else ("app" if ch[i] == "mobile" else "online_banking")
            ctry = home[c] if c < self.N else "NG"
            vpn = (self.c_vpn[c] and rng.random() < 0.7) if c < self.N else False
            abroad = self.c_trav[c] and rng.random() < 0.15
            ip = rng.choice(FOREIGN + COUNTRIES) if abroad else ctry
            g = rng.uniform(500, 9000) if abroad else rng.lognormal(np.log(8), 0.9)
            call = rng.random() < 0.03 + 0.01 * self.c_age[c]
            remote = rng.random() < 0.002
            pchange = rng.random() < 0.004
            if mo in (M_ATO_CRED, M_CNP, M_RING) or (mo == M_MULE and a_dev[i] >= 0):
                fc = self.actor_country.get(int(a_actor[i]), None)
                if fc is None:  # bot actor: use controller-ish geography
                    fc = rng.choice(FOREIGN + COUNTRIES)
                if a_soph[i]:
                    ip, g, vpn = ctry, rng.lognormal(np.log(25), 1.0), rng.random() < 0.2
                else:
                    ip = fc
                    g = rng.lognormal(np.log(150), 1.0) if fc == ctry else rng.uniform(2000, 9000)
                    vpn = rng.random() < 0.6
            if mo == M_ATO_RAT:
                remote = rng.random() < 0.7
                mod["injected"] = True
            if mo == M_APP:
                call = rng.random() < (0.35 if a_soph[i] else 0.75)
                mod.update(iki_mult=rng.uniform(1.2, 1.6), hold_mult=1.1, think_mult=rng.uniform(1.5, 3.0),
                           bs_mult=1.5, long_pauses=int(rng.integers(0, 3) if a_soph[i] else rng.integers(2, 8)))
            if mo == M_MULE:
                mod.update(think_mult=0.7)
            if a_flags[i] & 1:
                pchange = True
            if dev["is_emulator"]:
                mod["emulator"] = True
            if dev["const_pressure"]:
                mod["const_pressure"] = True
            if platform != 2 and rng.random() < 0.08:
                mod["no_motion"] = True
            if mo == M_SYNTH and self.P["bot"][actor]:
                mod["bot"] = True
            ctx["is_emulator"][i] = dev["emulator_detected"]
            ctx["is_rooted"][i] = dev["is_rooted"]
            ctx["remote_access_flag"][i] = remote
            ctx["active_call_flag"][i] = call and platform != 2
            ctx["vpn_flag"][i] = vpn
            ctx["profile_change_in_session"][i] = pchange
            ipc[i] = ip
            geo[i] = g
            # ---- behavioural capture
            if three_ds:
                challenge = rng.random() < (0.15 if mo == M_LEGIT else 0.4)
                if not challenge:
                    login[i] = "none"
                    continue
            behav[i] = 1
            new_payee = bool(a_pnew[i]) and tt[i] == "TRANSFER"
            own_device = dev["owner_kind"] == "customer" and dev["owner_id"] == c
            bio_p = P["bio_login_p"][actor] + (0.15 if d >= APP_V5_DAY else 0)
            if three_ds:
                lg = "otp"
            elif own_device and platform != 2 and rng.random() < bio_p and not mod.get("injected"):
                lg = "biometric"
            elif platform == 2 and own_device and rng.random() < 0.3:
                lg = "autofill"
            elif mo in (M_ATO_CRED, M_RING, M_SYNTH) and rng.random() < 0.5:
                lg = "autofill"   # pasted stolen / managed credentials
            else:
                lg = "password"
            login[i] = lg
            paste_payee = new_payee and rng.random() < (0.15 if mo == M_APP else P["paste_p"][actor])
            payee_method[i] = ("pasted" if paste_payee else "typed") if new_payee else ("saved" if tt[i] == "TRANSFER" else "none")
            otp = (new_payee and a_amt[i] > 200) or pchange
            plan = make_plan(rng, tt[i], platform, new_payee, lg, pchange, otp, d >= APP_V5_DAY and ch[i] == "mobile",
                             paste_payee, three_ds)
            ev, ag = simulate_session(rng, P, actor, platform, plan, mod)
            agg[i] = [ag[k] for k in AGG_COLS]
            events[i] = ev
            n_done += 1
            if n_done % 100000 == 0:
                self.log(f"  sessions simulated: {n_done}")
        T["device_idx"] = dev_out
        T["session_type"] = stype
        T["login_method"] = login
        T["has_behavioural"] = behav
        for k, v in ctx.items():
            T[k] = v.astype(np.int8)
        T["ip_country"] = np.where(has_session, ipc, "")
        T["ip_country_mismatch"] = (has_session & (ipc != home[np.minimum(acc_cust, self.N - 1)])).astype(np.int8)
        T["geo_distance_km"] = np.round(geo, 1)
        T["payee_entry_method"] = payee_method
        for j, k in enumerate(AGG_COLS):
            T[k] = agg[:, j]
        self.events = events
        self.log(f"sessions: {has_session.sum()} digital, {behav.sum()} with behavioural capture")

    def device_features(self):
        T = self.T
        D = pd.DataFrame(self.devices)
        has = T.device_idx.values >= 0
        T["device_type"] = np.where(has, D.device_type.values[np.maximum(T.device_idx.values, 0)], "none")
        T["device_os"] = np.where(has, D.device_os.values[np.maximum(T.device_idx.values, 0)], "none")
        T = T.sort_values(["t_sec", "acc"], kind="mergesort").reset_index(drop=True)
        d = T.device_idx.values
        has = d >= 0
        first_acc_dev = ~T.assign(_d=d).duplicated(["acc", "_d"])
        dd = np.maximum(d, 0)
        owner_ok = (D.owner_kind.values[dd] == "customer") & (D.owner_id.values[dd] == T.cust.values)
        prereg = has & owner_ok & (D.first_day.values[dd] == 0) & (self.acc.open_day.values[T.acc.values] < 0)
        T["is_new_device_for_account"] = np.where(has, (first_acc_dev.values & ~prereg).astype(int), -1)
        first_day = T.assign(_d=d).groupby(["bank", "_d"]).day.transform("min").values
        reg_age = self.rng.integers(30, 900, len(D))   # how long a pre-2024 device had been registered
        first_day = np.where(prereg, np.minimum(first_day, -reg_age[dd]), first_day)
        T["device_age_days_at_bank"] = np.where(has, T.day.values - first_day, -1)
        newpair = first_acc_dev.values & has
        cum = pd.Series(newpair.astype(int)).groupby([T.bank.values, d]).cumsum().values
        T["device_n_accounts_at_bank"] = np.where(has, cum, -1)
        # behaviour deviation vs the account's own past sessions on the same device type (causal)
        feats = ["key_hold_mean_ms", "iki_mean_ms", "touch_pressure_mean", "touch_size_mean",
                 "swipe_velocity_mean", "accel_std_mean", "tap_hold_mean_ms", "mouse_velocity_mean"]
        T = T.sort_values(["acc", "t_sec"], kind="mergesort").reset_index(drop=True)
        grp = [T.acc.values, T.device_type.values]
        zs = []
        for f in feats:
            x = T[f]
            s1 = x.fillna(0).groupby(grp).cumsum() - x.fillna(0)
            s2 = (x.fillna(0) ** 2).groupby(grp).cumsum() - x.fillna(0) ** 2
            nprev = x.notna().astype(int).groupby(grp).cumsum() - x.notna().astype(int)
            mu = s1 / nprev.replace(0, np.nan)
            sd = np.sqrt(np.maximum(s2 / nprev.replace(0, np.nan) - mu ** 2, 0))
            z = (x - mu).abs() / (sd + 1e-6 + 0.05 * mu.abs())
            z[nprev < 3] = np.nan
            zs.append(z.clip(upper=20))
        T["behaviour_deviation_score"] = pd.concat(zs, axis=1).mean(axis=1).round(3)
        self.T = T
        self.D = D

    # ---------------- outputs
    def write(self):
        out = self.a.out
        os.makedirs(os.path.join(out, "global"), exist_ok=True)
        T, acc, D = self.T, self.acc, self.D
        T = T.sort_values(["t_sec", "acc"], kind="mergesort").reset_index(drop=True)
        T["txn_id"] = [f"TXN_{i:08d}" for i in range(1, len(T) + 1)]
        ts = START + pd.to_timedelta(T.t_sec.values, unit="s")
        T["timestamp"] = ts.strftime("%Y-%m-%dT%H:%M:%S")
        T["step"] = T.day
        T["hour_of_day"] = ts.hour
        T["day_of_week"] = ts.dayofweek
        T["split"] = np.where(T.day < VAL_DAY, "train", np.where(T.day < TEST_DAY, "val", "test"))
        T["app_version"] = np.where(T.session_type == "app", np.where(T.day >= APP_V5_DAY, "v5", "v4"),
                                    np.where(T.session_type == "none", "", "web"))
        T["bank_id"] = np.array(self.banks)[T.bank.values]
        T["local_account_id"] = acc.local_account_id.values[T.acc.values]
        T["global_customer_id"] = acc.global_customer_id.values[T.acc.values]
        T["dest_bank_id"] = np.where(T.dest_bank.values >= 0, np.array(self.banks)[np.maximum(T.dest_bank.values, 0)], "EXTERNAL")
        D["device_id"] = ["DEV_" + hashlib.sha1(f"{self.a.seed}-{i}".encode()).hexdigest()[:12] for i in D.device_idx]
        T["device_id"] = np.where(T.device_idx.values >= 0, D.device_id.values[np.maximum(T.device_idx.values, 0)], "")
        T["session_id"] = np.where(T.session_type.values != "none", "S" + T.txn_id.str[4:], "")
        T["fraud_episode_id"] = np.where(T.episode.values >= 0, ["EP_%05d" % e for e in T.episode.values], "")
        T.rename(columns={"ttype": "txn_type"}, inplace=True)
        T["amount"] = T.amount.round(2)
        for c in ["session_duration_s", "time_to_first_action_s", "key_hold_mean_ms", "key_hold_std_ms",
                  "iki_mean_ms", "iki_std_ms", "typing_speed_cps", "backspace_rate", "tap_hold_mean_ms",
                  "touch_pressure_mean", "touch_pressure_std", "touch_size_mean", "touch_size_std",
                  "swipe_velocity_mean", "swipe_straightness_mean", "mouse_velocity_mean",
                  "mouse_straightness_mean", "accel_std_mean", "longest_pause_s", "hesitation_ratio"]:
            T[c] = T[c].round(4)

        base_cols = ["txn_id", "timestamp", "step", "split", "bank_id", "local_account_id", "global_customer_id",
                     "dest_account_id", "dest_bank_id", "is_cross_bank", "amount", "txn_type", "old_balance",
                     "new_balance", "channel", "hour_of_day", "day_of_week", "time_since_last_hrs", "velocity_1h",
                     "velocity_24h", "amount_to_avg_ratio", "payee_is_new"]
        sess_cols = ["session_id", "session_type", "app_version", "device_id", "device_type", "device_os",
                     "login_method", "has_behavioural", "is_new_device_for_account", "device_age_days_at_bank",
                     "device_n_accounts_at_bank", "is_emulator", "is_rooted", "remote_access_flag",
                     "active_call_flag", "vpn_flag", "ip_country", "ip_country_mismatch", "geo_distance_km",
                     "profile_change_in_session", "payee_entry_method"] + AGG_COLS + ["behaviour_deviation_score"]
        label_cols = ["is_fraud", "fraud_type", "fraud_episode_id"]
        cols = base_cols + sess_cols + label_cols
        T[cols].to_csv(os.path.join(out, "global", "transactions.csv.gz"), index=False, compression="gzip")
        self.log("wrote global/transactions.csv.gz")

        # customers, accounts, devices, profiles (ground truth)
        cust = self.cust.copy()
        cust["is_synthetic_identity"] = self.c_synth.astype(int)
        cust.to_csv(os.path.join(out, "global", "global_customers.csv"), index=False)
        afl = T.groupby("local_account_id").is_fraud.max()
        A = acc.copy()
        A["account_open_days_ago"] = (DAYS - A.open_day).astype(int)
        A["account_fraud_label"] = A.local_account_id.map(afl).fillna(0).astype(int)
        A.loc[A.role != "normal", "account_fraud_label"] = 1
        A["account_role"] = A.role
        A[["global_customer_id", "bank_id", "local_account_id", "account_open_days_ago", "is_primary",
           "account_role", "account_fraud_label"]].rename(columns={"is_primary": "is_primary_bank"}).to_csv(
            os.path.join(out, "global", "account_identity_map.csv"), index=False)
        D["owner"] = [self.cust.global_customer_id.values[o] if k in ("customer", "public") else f"ACTOR_{o:05d}"
                      for k, o in zip(D.owner_kind, D.owner_id)]
        D[["device_id", "owner_kind", "owner", "device_type", "device_os", "is_emulator", "is_rooted",
           "first_day"]].to_csv(os.path.join(out, "global", "devices_ground_truth.csv"), index=False)
        prof = pd.DataFrame({k: np.round(v[:self.N], 4) for k, v in self.P.items() if k != "bot"})
        prof.insert(0, "global_customer_id", self.cust.global_customer_id.values)
        prof.to_csv(os.path.join(out, "global", "customer_behaviour_profiles_ground_truth.csv"), index=False)

        # graph edges
        tr = T[T.dest_bank_id != "EXTERNAL"]
        E = tr.groupby(["local_account_id", "dest_account_id"]).agg(
            src_bank=("bank_id", "first"), dst_bank=("dest_bank_id", "first"), n_txns=("txn_id", "size"),
            total_amount=("amount", "sum"), first_ts=("timestamp", "min"), last_ts=("timestamp", "max"),
            is_fraud_edge=("is_fraud", "max")).reset_index().rename(
            columns={"local_account_id": "src_account", "dest_account_id": "dst_account"})
        E["is_cross_bank"] = (E.src_bank != E.dst_bank).astype(int)
        E["edge_type"] = "transfer"
        E["total_amount"] = E.total_amount.round(2)
        E.to_csv(os.path.join(out, "global", "graph_edges.csv"), index=False)
        DE = T[T.device_id != ""].groupby(["local_account_id", "device_id"]).agg(
            bank_id=("bank_id", "first"), n_sessions=("txn_id", "size"), first_ts=("timestamp", "min"),
            last_ts=("timestamp", "max")).reset_index()
        DE.to_csv(os.path.join(out, "global", "graph_account_device_edges.csv"), index=False)

        # identity resolution pairs (positives + hard and random negatives)
        rng = self.rng
        A2 = A.merge(self.cust[["global_customer_id", "home_country", "age_band"]], on="global_customer_id")
        pos = []
        for gid, g in A2.groupby("global_customer_id"):
            if len(g) > 1:
                r = g.to_dict("records")
                for x in range(len(r)):
                    for y in range(x + 1, len(r)):
                        pos.append((r[x]["bank_id"], r[x]["local_account_id"], r[y]["bank_id"], r[y]["local_account_id"],
                                    gid, 1, "same_customer"))
        neg = []
        byk = {k: g for k, g in A2.groupby(["home_country", "age_band"])}
        idx = A2.set_index("local_account_id")
        for p in pos:
            a_row = idx.loc[p[1]]
            g = byk[(a_row.home_country, a_row.age_band)]
            g = g[(g.bank_id == p[2]) & (g.global_customer_id != p[4])]
            if len(g):
                b = g.iloc[int(rng.integers(0, len(g)))]
                neg.append((p[0], p[1], p[2], b.local_account_id, "", 0, "hard_negative"))
            if rng.random() < 0.5:
                b = A2.iloc[int(rng.integers(0, len(A2)))]
                if b.global_customer_id != p[4] and b.bank_id != p[0]:
                    neg.append((p[0], p[1], b.bank_id, b.local_account_id, "", 0, "random_negative"))
        IR = pd.DataFrame(pos + neg, columns=["account_a_bank", "account_a_local", "account_b_bank", "account_b_local",
                                              "global_customer_id", "is_same_customer", "pair_type"])
        IR.sample(frac=1, random_state=self.a.seed).to_csv(os.path.join(out, "global", "identity_resolution_ground_truth.csv"), index=False)

        # account sequences
        def seqs(df):
            g = df.sort_values("timestamp").groupby("local_account_id")
            S = g.agg(bank_id=("bank_id", "first"), global_customer_id=("global_customer_id", "first"),
                      n_txns=("txn_id", "size"), fraud_any=("is_fraud", "max"), total_amount=("amount", "sum"),
                      avg_amount=("amount", "mean"), seq_steps=("step", list), seq_amounts=("amount", list),
                      seq_types=("txn_type", list), seq_fraud=("is_fraud", list), seq_txn_ids=("txn_id", list)).reset_index()
            S["total_amount"] = S.total_amount.round(2)
            S["avg_amount"] = S.avg_amount.round(2)
            for c in ["seq_steps", "seq_amounts", "seq_types", "seq_fraud", "seq_txn_ids"]:
                S[c] = S[c].apply(json.dumps)
            return S
        seqs(T).to_csv(os.path.join(out, "global", "account_sequences.csv.gz"), index=False, compression="gzip")
        self.log("wrote global files")

        # per-bank client shards
        for b, bid in enumerate(self.banks):
            d = os.path.join(out, "clients", bid)
            os.makedirs(d, exist_ok=True)
            salt = hashlib.sha1(f"{bid}-{self.a.seed}".encode()).hexdigest()

            def h(x, p, n=12):
                return p + hashlib.sha1((salt + x).encode()).hexdigest()[:n] if x else ""
            Tb = T[T.bank_id == bid].copy()
            Tb["local_customer_id"] = [h(x, "LC_", 10) for x in Tb.global_customer_id]
            Tb["device_hash"] = [h(x, "DH_") for x in Tb.device_id]
            ccols = [c for c in cols if c not in ("global_customer_id", "device_id", "fraud_episode_id")]
            ccols.insert(ccols.index("local_account_id") + 1, "local_customer_id")
            ccols.insert(ccols.index("session_type") + 2, "device_hash")
            Tb[ccols].to_csv(os.path.join(d, "transactions.csv.gz"), index=False, compression="gzip")
            Eb = E[(E.src_bank == bid) | (E.dst_bank == bid)].copy()
            Eb["edge_scope"] = np.where(Eb.is_cross_bank == 1, "boundary", "internal")
            Eb.to_csv(os.path.join(d, "edges.csv"), index=False)
            DEb = DE[DE.bank_id == bid].copy()
            DEb["device_hash"] = [h(x, "DH_") for x in DEb.device_id]
            DEb.drop(columns=["device_id"]).to_csv(os.path.join(d, "account_device_edges.csv"), index=False)
            Sb = seqs(Tb.assign(global_customer_id=Tb.local_customer_id)).rename(columns={"global_customer_id": "local_customer_id"})
            Sb.to_csv(os.path.join(d, "sequences.csv.gz"), index=False, compression="gzip")
            # raw session events
            if not self.a.no_events:
                beh = Tb[Tb.has_behavioural == 1]
                Lm = self.max_len
                arr = np.zeros((len(beh), Lm, NF), dtype=np.float16)
                lens = np.zeros(len(beh), dtype=np.int16)
                for j, oi in enumerate(beh["_orig"].values):
                    ev = self.events[int(oi)]
                    ev = ev[:Lm] if len(ev) <= Lm else np.concatenate([ev[:Lm // 2], ev[-(Lm - Lm // 2):]])
                    ev = ev.copy()
                    ev[:, 1] = np.minimum(ev[:, 1], 60000)
                    arr[j, :len(ev)] = ev.astype(np.float16)
                    lens[j] = min(len(self.events[int(oi)]), Lm)
                np.savez_compressed(os.path.join(d, "session_events.npz"), events=arr, lengths=lens,
                                    session_id=np.asarray(beh.session_id.tolist(), dtype="U16"),
                                    txn_id=np.asarray(beh.txn_id.tolist(), dtype="U16"),
                                    split=np.asarray(beh.split.tolist(), dtype="U8"),
                                    is_fraud=beh.is_fraud.values.astype(np.int8),
                                    fields=np.array(EVENT_FIELDS))
            self.log(f"wrote clients/{bid}: {len(Tb)} txns")

        meta = dict(
            description="Synthetic multi-bank fraud/AML dataset with device and behavioural biometrics for Federated-MARTI (v2)",
            generator="generate_federated_marti_v2.py", seed=self.a.seed, n_banks=self.B,
            n_global_customers=int(self.N), n_synthetic_identities=int(self.c_synth.sum()),
            n_accounts=int(len(A)), n_transactions=int(len(T)), fraud_rate=float(T.is_fraud.mean()),
            fraud_counts=T.fraud_type.value_counts().to_dict(),
            customers_in_multiple_banks=int((A.groupby("global_customer_id").size() > 1).sum()),
            n_devices=int(len(D)), splits=dict(train="2024-01-01..2024-08-31", val="2024-09-01..2024-10-31",
                                                test="2024-11-01..2024-12-30"),
            split_counts=T.split.value_counts().to_dict(),
            raw_event_sequence=dict(max_len=self.max_len, fields=EVENT_FIELDS,
                                    event_types=dict(pad=0, key=1, backspace=2, tap_or_click=3, swipe_or_scroll=4,
                                                     mouse_move=5, paste=6, screen_change=7, biometric_auth=8),
                                    missing_value=-1, dt_clipped_ms=60000,
                                    truncation="sessions longer than max_len keep the first and last max_len/2 events"),
            drift=dict(app_v5_release="2024-07-01", q4_device_farm="first seen 2024-11-01",
                       fraud_mix="RAT takeovers and coercion scams grow through H2; card testing peaks Nov-Dec"),
        )
        with open(os.path.join(out, "dataset_meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=int)
        self.log("done")

    def run(self):
        self.build_population()
        self.assign_ids()
        self.customer_devices()
        self.build_actors()
        self.assign_roles()
        self.assign_ids_new()
        self.legit_transactions()
        self.fraud_episodes()
        self.assemble()
        self.T["_orig"] = np.arange(len(self.T))
        self.sessions()
        self.device_features()
        self.write()

    def assign_ids_new(self):
        """IDs for accounts created in assign_roles (new mule accounts)."""
        acc = self.acc
        miss = acc.local_account_id.isna().values if "local_account_id" in acc else np.ones(len(acc), bool)
        if not miss.any():
            return
        used = set(acc.local_account_id.dropna())
        rng = self.rng
        for i in np.flatnonzero(miss):
            b = int(acc.bank.values[i])
            while True:
                x = f"{self.banks[b]}_ACC_{int(rng.integers(100000, 1000000))}"
                if x not in used:
                    used.add(x)
                    break
            acc.at[i, "local_account_id"] = x
        acc["bank_id"] = np.array(self.banks)[acc.bank.values]
        acc["global_customer_id"] = self.cust.global_customer_id.values[acc.cust.values]
        acc["role"] = acc["role"].fillna("normal")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="./fmarti_v2")
    ap.add_argument("--customers", type=int, default=40000)
    ap.add_argument("--banks", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fraud-scale", type=float, default=0.7, help="multiplies the number of fraud episodes")
    ap.add_argument("--sophistication", type=float, default=1.0,
                    help="multiplies the share of sophisticated takeovers/scams that mimic the victim (difficulty knob)")
    ap.add_argument("--seq-len", type=int, default=64, help="max events per session in the raw event tensors")
    ap.add_argument("--no-events", action="store_true", help="skip writing raw session event tensors")
    args = ap.parse_args()
    Generator(args).run()


if __name__ == "__main__":
    main()
