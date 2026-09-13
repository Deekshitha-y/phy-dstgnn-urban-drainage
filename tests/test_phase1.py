"""
Unit and Integration Test Suite for Phase 1 Data Pipeline.
"""

import json
import os
import unittest
import numpy as np
import pandas as pd

from src.data.rainfall_engine import HyderabadRainfallEngine, HISTORICAL_BENCHMARK_STORMS
from src.data.hyetograph_generator import ChicagoHyetographGenerator
from src.data.drainage_topology_extractor import HyderabadDrainageExtractor
from src.data.elevation_topography import HyderabadTopographyProcessor
from src.data.landcover_processor import HyderabadLandCoverProcessor


class TestPhase1DataPipeline(unittest.TestCase):
    """
    Test suite verifying correctness, continuity, physical bounds,
    and integrity of Phase 1 datasets.
    """

    @classmethod
    def setUpClass(cls):
        """Set up and run pipeline components for testing."""
        cls.rain_engine = HyderabadRainfallEngine(output_dir="data/rainfall_raw")
        cls.rainfall_df = cls.rain_engine.fetch_or_generate_15year_series(2010, 2025)

        cls.hyeto_gen = ChicagoHyetographGenerator(output_dir="data/synthetic_storms")
        cls.storms = cls.hyeto_gen.generate_all_benchmark_storms()

        cls.extractor = HyderabadDrainageExtractor(output_dir="data/hyderabad_gis")
        cls.nodes_df, cls.edges_df, cls.geojson_data = cls.extractor.generate_hyderabad_network_topology()

        cls.topo_proc = HyderabadTopographyProcessor(output_dir="data/processed")
        cls.nodes_df, cls.edges_df, cls.topo_summary = cls.topo_proc.process_node_and_edge_slopes(
            cls.nodes_df, cls.edges_df
        )

        cls.lc_proc = HyderabadLandCoverProcessor(output_dir="data/processed")
        cls.nodes_df, cls.lc_ts_df, cls.lc_summary = cls.lc_proc.generate_15year_landcover_progression(
            cls.nodes_df, 2010, 2025
        )

    def test_rainfall_15year_integrity(self):
        """Verify 15-year hourly rainfall dataset has no missing values and correct span."""
        self.assertFalse(self.rainfall_df.empty)
        self.assertGreaterEqual(len(self.rainfall_df), 130000)  # ~140,000 hours
        self.assertEqual(self.rainfall_df["precipitation_mm_hr"].isna().sum(), 0)
        self.assertGreaterEqual(self.rainfall_df["precipitation_mm_hr"].min(), 0.0)
        
        # Check monsoon season tagging
        monsoons = self.rainfall_df[self.rainfall_df["is_monsoon"] == 1]
        self.assertFalse(monsoons.empty)
        self.assertTrue(set(monsoons["month"].unique()).issubset({6, 7, 8, 9, 10}))

    def test_october_2020_historic_event(self):
        """Verify the October 13-14, 2020 historic super-flood is correctly represented."""
        oct_event = self.rainfall_df[
            (self.rainfall_df["timestamp"] >= "2020-10-13 00:00:00") &
            (self.rainfall_df["timestamp"] <= "2020-10-14 23:00:00")
        ]
        total_24h = oct_event["precipitation_mm_hr"].sum()
        self.assertGreaterEqual(total_24h, 300.0, f"Expected >= 300mm for Oct 2020 event, got {total_24h}")

    def test_chicago_hyetographs(self):
        """Verify synthetic storm hyetographs satisfy hydrological properties."""
        self.assertIn("hyderabad_chicago_Tr100yr_2hr", self.storms)
        self.assertIn("hyderabad_chicago_Tr2yr_2hr", self.storms)

        storm_100 = self.storms["hyderabad_chicago_Tr100yr_2hr"]
        storm_2 = self.storms["hyderabad_chicago_Tr2yr_2hr"]

        # 100-yr storm must produce strictly higher peak intensity and total rainfall than 2-yr storm
        self.assertGreater(storm_100["intensity_mm_hr"].max(), storm_2["intensity_mm_hr"].max())
        self.assertGreater(storm_100["cumulative_depth_mm"].iloc[-1], storm_2["cumulative_depth_mm"].iloc[-1])

    def test_drainage_network_topology(self):
        """Verify network graph topology, node-edge connectivity, and attribute validity."""
        self.assertGreaterEqual(len(self.nodes_df), 50)
        self.assertGreaterEqual(len(self.edges_df), 50)

        # Check all edges refer to valid node IDs
        node_ids = set(self.nodes_df["node_id"])
        for _, edge in self.edges_df.iterrows():
            self.assertIn(edge["from_node"], node_ids)
            self.assertIn(edge["to_node"], node_ids)
            self.assertGreater(edge["length_m"], 0.0)
            self.assertGreater(edge["diameter_m"], 0.0)

    def test_topography_and_slopes(self):
        """Verify ground/invert elevations and positive hydraulic slopes."""
        self.assertTrue((self.nodes_df["ground_elevation_m"] > self.nodes_df["invert_elevation_m"]).all())
        self.assertTrue((self.edges_df["hydraulic_slope"] > 0.0).all())
        self.assertGreater(self.topo_summary["total_pipe_length_km"], 5.0)

    def test_landcover_progression(self):
        """Verify 15-year dynamic impervious surface expansion."""
        self.assertFalse(self.lc_ts_df.empty)
        # Check values are valid percentages
        self.assertTrue((self.lc_ts_df["impervious_pct"] >= 0.0).all())
        self.assertTrue((self.lc_ts_df["impervious_pct"] <= 100.0).all())

        # Check net urbanization growth from 2010 to 2025 is positive
        mean_2010 = self.lc_ts_df[self.lc_ts_df["year"] == 2010]["impervious_pct"].mean()
        mean_2025 = self.lc_ts_df[self.lc_ts_df["year"] == 2025]["impervious_pct"].mean()
        self.assertGreater(mean_2025, mean_2010)


if __name__ == "__main__":
    unittest.main()
