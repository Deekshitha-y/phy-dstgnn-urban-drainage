"""
Hyderabad Dynamic Land Cover & Impervious Surface Processor (15-Year Evolution: 2010-2025).
Models the rapid urbanization and concrete surface expansion across Greater Hyderabad.
"""

import json
import logging
import os
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Zone-wise Urbanization Growth Dynamics (Imperviousness % in 2010 vs 2025)
HYDERABAD_URBANIZATION_GROWTH = {
    "Cyberabad - Hitec City Zone": {"imperv_2010": 42.0, "imperv_2025": 86.0},
    "Kukatpally Nala Basin": {"imperv_2010": 52.0, "imperv_2025": 82.0},
    "Balkapur Nala & Central Zone": {"imperv_2010": 68.0, "imperv_2025": 88.0},
    "Hussain Sagar Surplus & Ashok Nagar": {"imperv_2010": 72.0, "imperv_2025": 90.0},
    "Murki Nala & Old City Zone": {"imperv_2010": 78.0, "imperv_2025": 89.0}
}


class HyderabadLandCoverProcessor:
    """
    Computes dynamic subcatchment imperviousness percentages and soil infiltration
    parameters across 15 years for Greater Hyderabad.
    """

    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_15year_landcover_progression(
        self,
        nodes_df: pd.DataFrame,
        start_year: int = 2010,
        end_year: int = 2025
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        Generates annual imperviousness surface ratios and Horton infiltration parameters
        for each drainage junction across 2010-2025.
        """
        logger.info(f"Modeling 15-year land-use concrete expansion for Hyderabad ({start_year}-{end_year})...")

        years = list(range(start_year, end_year + 1))
        records = []

        np.random.seed(42)

        # Base soil parameters for Hyderabad (Red Sandy Loam / Deccan Trap Basalt)
        # Horton Infiltration: f_0 (initial capacity mm/hr), f_inf (decayed capacity mm/hr), k (decay constant 1/hr)
        for _, node in nodes_df.iterrows():
            zone = node.get("zone", "Kukatpally Nala Basin")
            growth = HYDERABAD_URBANIZATION_GROWTH.get(
                zone, {"imperv_2010": 55.0, "imperv_2025": 85.0}
            )

            # S-curve (logistic) urban growth model between 2010 and 2025
            val_2010 = growth["imperv_2010"] + np.random.uniform(-3.0, 3.0)
            val_2025 = growth["imperv_2025"] + np.random.uniform(-2.0, 2.0)

            for yr in years:
                # Progress factor between 0.0 (2010) and 1.0 (2025)
                t_norm = (yr - start_year) / (end_year - start_year)
                # Logistic sigmoid growth
                growth_factor = 1.0 / (1.0 + np.exp(-6.0 * (t_norm - 0.5)))
                imperv_pct = val_2010 + growth_factor * (val_2025 - val_2010)
                imperv_pct = np.clip(imperv_pct, 10.0, 98.0)

                # Higher imperviousness -> lower infiltration, higher runoff coefficient
                runoff_coeff = 0.15 + (imperv_pct / 100.0) * 0.75  # C between 0.20 and 0.90
                
                # Horton parameters
                f_0 = 75.0 * (1.0 - imperv_pct / 100.0) + 10.0  # mm/hr
                f_inf = 10.0 * (1.0 - imperv_pct / 100.0) + 2.0  # mm/hr
                decay_k = 3.5  # 1/hr

                records.append({
                    "year": yr,
                    "node_id": node["node_id"],
                    "zone": zone,
                    "impervious_pct": round(imperv_pct, 2),
                    "pervious_pct": round(100.0 - imperv_pct, 2),
                    "runoff_coefficient": round(runoff_coeff, 3),
                    "horton_f0_mm_hr": round(f_0, 2),
                    "horton_finf_mm_hr": round(f_inf, 2),
                    "horton_decay_k": decay_k
                })

        df_timeseries = pd.DataFrame(records)

        # Add latest 2025 imperviousness directly to static nodes table
        latest_imperv = df_timeseries[df_timeseries["year"] == end_year].set_index("node_id")["impervious_pct"].to_dict()
        latest_c = df_timeseries[df_timeseries["year"] == end_year].set_index("node_id")["runoff_coefficient"].to_dict()

        nodes_updated = nodes_df.copy()
        nodes_updated["impervious_pct_current"] = nodes_updated["node_id"].map(latest_imperv)
        nodes_updated["runoff_coefficient"] = nodes_updated["node_id"].map(latest_c)
        nodes_updated.to_csv(os.path.join(self.output_dir, "nodes.csv"), index=False)

        # Save complete timeseries
        timeseries_path = os.path.join(self.output_dir, "landcover_15year_timeseries.csv")
        df_timeseries.to_csv(timeseries_path, index=False)

        # Summary statistics
        annual_mean_imperv = df_timeseries.groupby("year")["impervious_pct"].mean().round(2).to_dict()
        summary = {
            "time_range": f"{start_year}-{end_year}",
            "city": "Greater Hyderabad (GHMC)",
            "annual_mean_impervious_percentage": annual_mean_imperv,
            "overall_impervious_expansion": {
                f"mean_{start_year}": annual_mean_imperv[start_year],
                f"mean_{end_year}": annual_mean_imperv[end_year],
                "net_urban_increase_pct": round(annual_mean_imperv[end_year] - annual_mean_imperv[start_year], 2)
            }
        }

        with open(os.path.join(self.output_dir, "landcover_summary.json"), "w") as f:
            json.dump(summary, f, indent=4)

        logger.info(f"Land cover 15-year progression saved to {timeseries_path}. Net urban impervious increase: +{summary['overall_impervious_expansion']['net_urban_increase_pct']}%")

        return nodes_updated, df_timeseries, summary


if __name__ == "__main__":
    extractor_nodes = pd.read_csv("data/processed/nodes.csv")
    lc_proc = HyderabadLandCoverProcessor()
    nodes_up, ts_df, summary = lc_proc.generate_15year_landcover_progression(extractor_nodes)
    print("Land Cover Summary:")
    print(json.dumps(summary, indent=2))
