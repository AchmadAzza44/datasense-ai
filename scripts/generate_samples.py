"""Generate reproducible sample datasets for DataSense AI (data SINTETIS).

Run from the project root:
    python scripts/generate_samples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config import SAMPLES_DIR, ensure_runtime_dirs  # noqa: E402

RANDOM_SEED = 2024


def build_customer_churn(rows: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    tenure = rng.integers(1, 72, rows)
    monthly_charge = np.round(rng.uniform(15, 120, rows), 2)
    support_calls = rng.poisson(1.4, rows)
    contract = rng.choice(["Bulanan", "Tahunan", "Dua Tahun"], rows, p=[0.55, 0.3, 0.15])
    internet = rng.choice(["Fiber", "DSL", "Tidak Ada"], rows, p=[0.45, 0.4, 0.15])
    satisfaction = np.clip(rng.normal(6.5, 1.8, rows), 1, 10).round(1)
    age = rng.integers(18, 75, rows)

    contract_risk = pd.Series(contract).map({"Bulanan": 1.1, "Tahunan": -0.2, "Dua Tahun": -0.9}).to_numpy()
    logit = (
        -0.9
        + contract_risk
        + 0.018 * monthly_charge
        + 0.28 * support_calls
        - 0.045 * tenure
        - 0.32 * (satisfaction - 6.5)
        + rng.normal(0, 0.7, rows)
    )
    probability = 1 / (1 + np.exp(-logit))
    churn = np.where(rng.random(rows) < probability, "Ya", "Tidak")

    frame = pd.DataFrame(
        {
            "id_pelanggan": [f"CUST-{index:05d}" for index in range(1, rows + 1)],
            "usia": age,
            "masa_berlangganan_bulan": tenure,
            "tagihan_bulanan": monthly_charge,
            "jumlah_panggilan_support": support_calls,
            "jenis_kontrak": contract,
            "jenis_internet": internet,
            "skor_kepuasan": satisfaction,
            "churn": churn,
        }
    )

    dirty_index = rng.choice(rows, size=max(3, rows // 25), replace=False)
    frame.loc[dirty_index, "tagihan_bulanan"] = np.nan
    frame.loc[dirty_index[:3], "skor_kepuasan"] = np.nan
    frame = pd.concat([frame, frame.sample(8, random_state=RANDOM_SEED)], ignore_index=True)
    return frame


def build_sales_regression(rows: int = 480) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED + 1)
    advertising = rng.uniform(5, 90, rows).round(2)
    price = rng.uniform(8, 40, rows).round(2)
    footfall = rng.integers(100, 2500, rows)
    promotion = rng.choice(["Tidak", "Diskon 10%", "Diskon 25%", "Bundling"], rows, p=[0.4, 0.25, 0.2, 0.15])
    store_type = rng.choice(["Ritel", "Online", "Grosir"], rows, p=[0.5, 0.3, 0.2])

    promo_effect = pd.Series(promotion).map(
        {"Tidak": 0.0, "Diskon 10%": 6.0, "Diskon 25%": 11.0, "Bundling": 8.0}
    ).to_numpy()
    type_effect = pd.Series(store_type).map({"Ritel": 0.0, "Online": 5.5, "Grosir": 9.0}).to_numpy()

    sales = (
        40
        + 0.85 * advertising
        + 0.012 * footfall
        - 0.9 * price
        + promo_effect
        + type_effect
        + rng.normal(0, 6.5, rows)
    ).round(2)

    frame = pd.DataFrame(
        {
            "tanggal": pd.date_range("2022-01-01", periods=rows, freq="D").strftime("%Y-%m-%d"),
            "toko": rng.choice([f"Toko {chr(65 + index)}" for index in range(8)], rows),
            "jenis_toko": store_type,
            "biaya_iklan_juta": advertising,
            "harga_rata_rata": price,
            "jumlah_pengunjung": footfall,
            "promosi": promotion,
            "penjualan_juta": sales,
        }
    )
    frame.loc[rng.choice(rows, 5, replace=False), "biaya_iklan_juta"] = np.nan
    frame = pd.concat([frame, frame.sample(4, random_state=RANDOM_SEED)], ignore_index=True)
    return frame


LOCATIONS = [
    ("Jakarta Pusat", -6.1862, 106.8340),
    ("Bandung", -6.9175, 107.6191),
    ("Semarang", -6.9932, 110.4203),
    ("Surabaya", -7.2575, 112.7521),
    ("Yogyakarta", -7.7956, 110.3695),
    ("Medan", 3.5952, 98.6722),
    ("Makassar", -5.1477, 119.4327),
    ("Balikpapan", -1.2379, 116.8529),
    ("Denpasar", -8.6705, 115.2126),
    ("Palembang", -2.9761, 104.7754),
]


def main() -> None:
    ensure_runtime_dirs()
    outputs = {
        "customer_churn.csv": build_customer_churn(),
        "sales_regression.csv": build_sales_regression(),
    }
    for name, frame in outputs.items():
        path = SAMPLES_DIR / name
        frame.to_csv(path, index=False)
        print(f"{path.relative_to(ROOT)}  ({frame.shape[0]} baris x {frame.shape[1]} kolom)")

    excel_path = SAMPLES_DIR / "sales_regression.xlsx"
    outputs["sales_regression.csv"].to_excel(excel_path, index=False, engine="openpyxl")
    print(f"{excel_path.relative_to(ROOT)}  (format XLSX)")


if __name__ == "__main__":
    main()
