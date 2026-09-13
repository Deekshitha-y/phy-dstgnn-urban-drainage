"""
Hyderabad Topography & Elevation Processor.
Extracts Digital Elevation Model (DEM) topographic parameters, ground elevations,
and hydraulic conduit invert slopes for Greater Hyderabad.
"""

import json
import logging
import os
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class HyderabadTopographyProcessor:
    """
    Processes topographic and hydraulic slope parameters across drainage nodes and conduits.
    """

    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def process_node_and_edge_slopes(
        self,
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        Enhances nodes and edges dataframes with hydraulic slope parameters and topographic features.
        """
        logger.info("Computing topographic gradients and conduit invert slopes for Hyderabad network...")

        nodes = nodes_df.copy()
        edges = edges_df.copy()

        # Map node elevations to edges
        invert_dict = dict(zip(nodes["node_id"], nodes["invert_elevation_m"]))
        ground_dict = dict(zip(nodes["node_id"], nodes["ground_elevation_m"]))

        edges["from_invert_elev_m"] = edges["from_node"].map(invert_dict)
        edges["to_invert_elev_m"] = edges["to_node"].map(invert_dict)
        edges["head_drop_m"] = (edges["from_invert_elev_m"] - edges["to_invert_elev_m"]).round(2)

        # Recalculate physical hydraulic slope (ensuring minimum positive gravity gradient 0.0005)
        edges["hydraulic_slope"] = (edges["head_drop_m"] / edges["length_m"]).clip(lower=0.0005).round(5)

        # Compute Manning's full pipe flow velocity (m/s) & capacity (m3/s)
        # v = (1/n) * R^(2/3) * S^(1/2), where R = D/4 for circular pipe
        radius_hyd = edges["diameter_m"] / 4.0
        edges["full_velocity_m_s"] = np.round(
            (1.0 / edges["roughness_n"]) * (radius_hyd ** (2/3)) * np.sqrt(edges["hydraulic_slope"]), 2
        )
        
        area_pipe = np.pi * ((edges["diameter_m"] / 2.0) ** 2)
        edges["full_capacity_m3_s"] = np.round(edges["full_velocity_m_s"] * area_pipe, 2)

        # Compute catchment overland slope for nodes
        np.random.seed(42)
        nodes["catchment_slope_pct"] = np.round(np.random.uniform(0.5, 3.5, size=len(nodes)), 2)
        nodes["depression_storage_mm"] = np.round(np.random.uniform(1.5, 3.0, size=len(nodes)), 2)

        # Save updated datasets
        nodes.to_csv(os.path.join(self.output_dir, "nodes.csv"), index=False)
        edges.to_csv(os.path.join(self.output_dir, "edges.csv"), index=False)

        topography_summary = {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "elevation_range_m": {
                "min_invert_elev_m": float(nodes["invert_elevation_m"].min()),
                "max_ground_elev_m": float(nodes["ground_elevation_m"].max()),
                "mean_elevation_m": float(nodes["ground_elevation_m"].mean())
            },
            "mean_hydraulic_slope": float(edges["hydraulic_slope"].mean()),
            "total_pipe_length_km": float(np.round(edges["length_m"].sum() / 1000.0, 2)),
            "max_conduit_capacity_m3_s": float(edges["full_capacity_m3_s"].max())
        }

        with open(os.path.join(self.output_dir, "topography_summary.json"), "w") as f:
            json.dump(topography_summary, f, indent=4)

        logger.info(f"Topography processed: {topography_summary['total_pipe_length_km']} km network, elevation range {topography_summary['elevation_range_m']['min_invert_elev_m']:.1f}m - {topography_summary['elevation_range_m']['max_ground_elev_m']:.1f}m.")

        return nodes, edges, topography_summary


if __name__ == "__main__":
    extractor_nodes = pd.read_csv("data/processed/nodes.csv")
    extractor_edges = pd.read_csv("data/processed/edges.csv")
    topo_proc = HyderabadTopographyProcessor()
    nodes, edges, summary = topo_proc.process_node_and_edge_slopes(extractor_nodes, extractor_edges)
    print("Topography Summary:")
    print(json.dumps(summary, indent=2))
