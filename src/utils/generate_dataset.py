
"""
Generate a synthetic enterprise-style marketing dataset.

This dataset is inspired by real-world Marketing Mix Modeling workflows
but contains no proprietary company data.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# 25 US DMA-style markets
dmas = [
    "New York","Los Angeles","Chicago","Philadelphia","Dallas",
    "San Francisco","Boston","Atlanta","Houston","Washington",
    "Phoenix","Seattle","Detroit","Miami","Minneapolis",
    "Denver","St. Louis","Tampa","Orlando","Charlotte",
    "Portland","Pittsburgh","Cleveland","Nashville","Austin"
]

# Fictional brands
brands = ["Brand Alpha", "Brand Beta"]

months = pd.date_range("2023-01-01", "2024-12-01", freq="MS")

rows = []

for dma in dmas:
    market_size = np.random.uniform(0.8, 1.4)

    for brand in brands:
        brand_strength = np.random.uniform(0.9, 1.2)

        for month in months:

            seasonality = 1 + 0.15 * np.sin((month.month / 12) * 2 * np.pi)

            tv = np.random.randint(80000, 400000)
            digital = np.random.randint(40000, 180000)
            pr = np.random.randint(10000, 70000)
            calls = np.random.randint(600, 4000)
            email = np.random.randint(5000, 40000)
            hcp_reach = np.random.randint(1000, 12000)

            sales = (
                25000 * market_size * brand_strength * seasonality
                + 0.035 * tv
                + 0.05 * digital
                + 1.8 * calls
                + 0.025 * email
                + np.random.normal(0, 5000)
            )

            rows.append({
                "dma": dma,
                "brand": brand,
                "year_month": month.strftime("%Y-%m"),
                "tv_spend": tv,
                "digital_spend": digital,
                "pr_spend": pr,
                "sales_calls": calls,
                "email_engagement": email,
                "hcp_reach": hcp_reach,
                "sales": round(sales, 2)
            })

df = pd.DataFrame(rows)

df.to_csv("data/synthetic/marketing_mmm_data.csv", index=False)

print("Dataset created successfully!")
print(df.head())
print(f"Rows: {len(df)}")
