"""
Master Orchestrator for Phase 1: Hyderabad Geospatial & 15-Year Climate Data Acquisition.
Executes the full pipeline and generates unified artifacts for the subsequent SWMM simulation & Deep Learning phases.
"""

import json
import logging
import os
import sys
from datetime import datetime

from src.data.rainfall_engine import HyderabadRainfallEngine
from src.data.hyetograph_generator import ChicagoHyetographGenerator
from src.data.drainage_topology_extractor import HyderabadDrainageExtractor
from src.data.elevation_topography import HyderabadTopographyProcessor
from src.data.landcover_processor import HyderabadLandCoverProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_phase1_pipeline():
    """
    Executes all Phase 1 data ingestion, modeling, and processing modules.
    """
    start_time = datetime.now()
    logger.info("==========================================================================")
    logger.info("STARTING PHASE 1: HYDERABAD 15-YEAR CLIMATE & DRAINAGE INGESTION PIPELINE")
    logger.info("==========================================================================")

    # 1. 15-Year Climate & Rainfall Time-Series (2010-2025)
    logger.info("\n--- [1/5] Executing 15-Year Rainfall Engine ---")
    rain_engine = HyderabadRainfallEngine(output_dir="data/rainfall_raw")
    rainfall_df = rain_engine.fetch_or_generate_15year_series(start_year=2010, end_year=2025)
    monsoon_df = rain_engine.extract_monsoon_seasons(rainfall_df)

    # 2. Synthetic Design Storm Matrix (Chicago & Huff Hyetographs)
    logger.info("\n--- [2/5] Executing Chicago Hyetograph Generator ---")
    hyeto_gen = ChicagoHyetographGenerator(output_dir="data/synthetic_storms")
    synthetic_storms = hyeto_gen.generate_all_benchmark_storms()

    # 3. Drainage Network Topology Extraction
    logger.info("\n--- [3/5] Extracting Greater Hyderabad Drainage Network Topology ---")
    extractor = HyderabadDrainageExtractor(output_dir="data/hyderabad_gis")
    nodes_df, edges_df, geojson_data = extractor.generate_hyderabad_network_topology(n_junctions_per_zone=15)

    # 4. Topography & Elevation Gradient Processor
    logger.info("\n--- [4/5] Processing Topography & Conduit Invert Slopes ---")
    topo_proc = HyderabadTopographyProcessor(output_dir="data/processed")
    nodes_df, edges_df, topo_summary = topo_proc.process_node_and_edge_slopes(nodes_df, edges_df)

    # 5. 15-Year Dynamic Land Cover & Imperviousness Progression
    logger.info("\n--- [5/5] Processing 15-Year Urbanization & Impervious Expansion ---")
    lc_proc = HyderabadLandCoverProcessor(output_dir="data/processed")
    nodes_df, lc_timeseries_df, lc_summary = lc_proc.generate_15year_landcover_progression(
        nodes_df, start_year=2010, end_year=2025
    )

    # Generate Unified Phase 1 Master Summary
    end_time = datetime.now()
    duration_sec = (end_time - start_time).total_seconds()

    phase1_master_summary = {
        "status": "SUCCESS",
        "timestamp": datetime.now().isoformat(),
        "execution_time_seconds": round(duration_sec, 2),
        "study_area": {
            "city": "Greater Hyderabad (GHMC), Telangana, India",
            "center_coordinates": [17.3850, 78.4867],
            "zones_included": [
                "Cyberabad - Hitec City Zone",
                "Kukatpally Nala Basin",
                "Balkapur Nala & Central Zone",
                "Hussain Sagar Surplus & Ashok Nagar",
                "Murki Nala & Old City Zone"
            ]
        },
        "datasets_generated": {
            "rainfall_15year": {
                "file": "data/rainfall_raw/hyderabad_hourly_rainfall_2010_2025.csv",
                "total_hours": len(rainfall_df),
                "monsoon_hours": len(monsoon_df),
                "years_covered": "2010-2025",
                "key_benchmark_included": "October 13-14, 2020 Historic Cloudburst (320mm/24h)"
            },
            "synthetic_storms": {
                "directory": "data/synthetic_storms/",
                "num_scenarios": len(synthetic_storms),
                "return_periods_yr": [2, 5, 10, 25, 50, 100],
                "durations_hr": [1.0, 2.0, 6.0, 24.0]
            },
            "network_topology": {
                "nodes_file": "data/processed/nodes.csv",
                "edges_file": "data/processed/edges.csv",
                "geojson_file": "data/hyderabad_gis/hyderabad_drainage_network.geojson",
                "total_manholes_junctions": len(nodes_df),
                "total_conduits_nalas": len(edges_df),
                "total_network_length_km": topo_summary["total_pipe_length_km"]
            },
            "urbanization_dynamics": {
                "timeseries_file": "data/processed/landcover_15year_timeseries.csv",
                "net_urban_impervious_growth_pct": lc_summary["overall_impervious_expansion"]["net_urban_increase_pct"]
            }
        }
    }

    summary_file = "data/processed/phase1_summary.json"
    with open(summary_file, "w") as f:
        json.dump(phase1_master_summary, f, indent=4)

    logger.info("==========================================================================")
    logger.info("PHASE 1 EXECUTION COMPLETE!")
    logger.info(f"Summary JSON saved to {summary_file}")
    logger.info(f"Total Execution Time: {duration_sec:.2f} seconds")
    logger.info("==========================================================================")

    return phase1_master_summary


if __name__ == "__main__":
    run_phase1_pipeline()
