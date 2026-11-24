# Sample data

Generated with:

```
python generator/generate_federated_marti_v2.py --out sample_data --customers 2000 --banks 8 --seed 42
```

2,032 customers (32 synthetic identities), 2,634 accounts, 56,983 transactions, 1.05% fraud. Same structure as the full benchmark (see `docs/DATA_CARD.md`): `global/` holds the ground truth, `clients/BANK_XX/` what each bank may see. Copyright © 2026 Cyprus International University, licensed under CC BY 4.0.
