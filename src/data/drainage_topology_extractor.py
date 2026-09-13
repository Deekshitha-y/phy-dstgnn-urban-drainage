"""
Hyderabad Drainage Topology Extractor.
Builds the graph-structured urban drainage network for Greater Hyderabad (GHMC)
spanning the Kukatpally Basin, Hussain Sagar Catchment, Balkapur Nala, and Musi River Network.
"""

import json
import logging
import os
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Key Hyderabad Primary Drainage Corridors & Subcatchments
HYDERABAD_DRAINAGE_ZONES = [
    {
        "zone_name": "Kukatpally Nala Basin",
        "key_nalas": ["Kukatpally Primary Trunk", "Fatehnagar Drain", "Balanagar Nala", "Moosapet Culvert"],
        "outfall": "Hussain Sagar Inflow North",
        "catchment_area_ha": 650.0,
        "base_elevation_m": 535.0,
        "flow_direction": "North-West to Central"
    },
    {
        "zone_name": "Balkapur Nala & Central Zone",
        "key_nalas": ["Balkapur Channel", "Tolichowki Box Drain", "Shaikpet Nala", "Mehdipatnam Conduits"],
        "outfall": "Hussain Sagar Inflow West",
        "catchment_area_ha": 480.0,
        "base_elevation_m": 528.0,
        "flow_direction": "West to Central"
    },
    {
        "zone_name": "Hussain Sagar Surplus & Ashok Nagar",
        "key_nalas": ["Hussain Sagar Surplus Channel", "Ashok Nagar Open Channel", "Domalguda Secondary Pipe", "Chikkadpally Nala"],
        "outfall": "Musi River Confluence 1",
        "catchment_area_ha": 520.0,
        "base_elevation_m": 510.0,
        "flow_direction": "Central to South-East"
    },
    {
        "zone_name": "Murki Nala & Old City Zone",
        "key_nalas": ["Murki Nala Trunk", "Charminar Box Drain", "Yakutpura Channel", "Bandlaguda Drain"],
        "outfall": "Musi River Confluence 2",
        "catchment_area_ha": 720.0,
        "base_elevation_m": 495.0,
        "flow_direction": "South to Musi River"
    },
    {
        "zone_name": "Cyberabad - Hitec City Zone",
        "key_nalas": ["Durgam Cheruvu Surplus", "Madhapur Trunk Pipe", "Gachibowli Drainage Conduit", "Kondapur Nala"],
        "outfall": "Kukatpally West Branch",
        "catchment_area_ha": 580.0,
        "base_elevation_m": 560.0,
        "flow_direction": "South-West to North-East"
    }
]


class HyderabadDrainageExtractor:
    """
    Constructs the directed topological graph of Hyderabad's drainage network
    with realistic hydraulic attributes (lengths, slopes, diameters, roughness).
    """

    def __init__(self, output_dir: str = "data/hyderabad_gis"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)

    def generate_hyderabad_network_topology(self, n_junctions_per_zone: int = 15) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        Synthesizes a realistic, fully-connected urban drainage network topology
        for Greater Hyderabad with ~75-100 manholes/junctions and ~85-120 conduits.
        """
        logger.info("Constructing Greater Hyderabad drainage network graph topology...")
        
        nodes_list = []
        edges_list = []
        geojson_features = []

        node_id_counter = 1
        edge_id_counter = 1

        np.random.seed(42)

        prev_outfall_node = None

        for zone_idx, zone in enumerate(HYDERABAD_DRAINAGE_ZONES):
            zone_nodes = []
            base_elev = zone["base_elevation_m"]
            
            # Base coordinate offsets for Hyderabad zones
            lat_center = 17.3850 + (zone_idx - 2) * 0.04
            lon_center = 78.4867 + (zone_idx - 2) * 0.03
            
            # Create Inflow / Manhole Nodes for this zone
            for i in range(n_junctions_per_zone):
                node_name = f"J_{zone['zone_name'][:3].upper()}_{node_id_counter:03d}"
                is_outfall = (i == n_junctions_per_zone - 1)
                
                # Spatial positioning (WGS84 Coordinates)
                lat = lat_center + np.random.uniform(-0.015, 0.015)
                lon = lon_center + np.random.uniform(-0.015, 0.015)
                
                # Elevations: Downstream gradient slope (~0.5% - 1.5% natural slope)
                ground_elev = base_elev + (n_junctions_per_zone - i) * 1.8 + np.random.uniform(-0.5, 0.5)
                max_depth = np.random.uniform(2.5, 4.5)  # 2.5m to 4.5m deep manholes
                invert_elev = ground_elev - max_depth
                
                subcatchment_area = np.random.uniform(2.0, 15.0)  # Hectares
                
                node_data = {
                    "node_id": node_name,
                    "node_index": node_id_counter - 1,
                    "zone": zone["zone_name"],
                    "latitude": round(lat, 6),
                    "longitude": round(lon, 6),
                    "ground_elevation_m": round(ground_elev, 2),
                    "invert_elevation_m": round(invert_elev, 2),
                    "max_depth_m": round(max_depth, 2),
                    "subcatchment_area_ha": round(subcatchment_area, 2),
                    "is_outfall": int(is_outfall)
                }
                nodes_list.append(node_data)
                zone_nodes.append(node_data)
                node_id_counter += 1

                # GeoJSON point feature
                geojson_features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [node_data["longitude"], node_data["latitude"]]
                    },
                    "properties": node_data
                })

            # Create Conduit Edges within this zone (Branching / Tree + Surcharge loop structure)
            for i in range(len(zone_nodes) - 1):
                u_node = zone_nodes[i]
                v_node = zone_nodes[i + 1]
                
                conduit_name = f"C_{u_node['node_id']}_{v_node['node_id']}"
                
                # Compute Euclidean length in meters (approx at Hyderabad latitude: 1 deg ~ 111,000m)
                dx = (v_node["longitude"] - u_node["longitude"]) * 106000.0
                dy = (v_node["latitude"] - u_node["latitude"]) * 111000.0
                length_m = max(50.0, np.sqrt(dx*dx + dy*dy))
                
                elev_diff = u_node["invert_elevation_m"] - v_node["invert_elevation_m"]
                slope = max(0.001, elev_diff / length_m)
                
                # Conduit diameter: increases downstream from 0.8m to 2.5m (or box nala width up to 4.0m)
                diameter_m = 0.9 + (i / len(zone_nodes)) * 1.6
                roughness_n = 0.015 if i < len(zone_nodes) // 2 else 0.025  # Pipe vs open masonry nala
                
                edge_data = {
                    "conduit_id": conduit_name,
                    "edge_index": edge_id_counter - 1,
                    "from_node": u_node["node_id"],
                    "to_node": v_node["node_id"],
                    "from_index": u_node["node_index"],
                    "to_index": v_node["node_index"],
                    "length_m": round(length_m, 2),
                    "diameter_m": round(diameter_m, 2),
                    "slope": round(slope, 5),
                    "roughness_n": roughness_n,
                    "shape": "CIRCULAR" if diameter_m < 2.0 else "RECT_CLOSED",
                    "max_flow_capacity_m3s": round(
                        (1.0 / roughness_n) * (np.pi * (diameter_m / 2)**2) * ((diameter_m / 4)**(2/3)) * np.sqrt(slope), 2
                    )
                }
                edges_list.append(edge_data)
                edge_id_counter += 1

                # LineString feature for GeoJSON
                geojson_features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [u_node["longitude"], u_node["latitude"]],
                            [v_node["longitude"], v_node["latitude"]]
                        ]
                    },
                    "properties": edge_data
                })

            # Connect current zone outfall to previous or downstream master trunk
            if prev_outfall_node is not None:
                u_node = prev_outfall_node
                v_node = zone_nodes[0]
                inter_conduit_name = f"C_TRUNK_{u_node['node_id']}_{v_node['node_id']}"
                
                dx = (v_node["longitude"] - u_node["longitude"]) * 106000.0
                dy = (v_node["latitude"] - u_node["latitude"]) * 111000.0
                length_m = max(100.0, np.sqrt(dx*dx + dy*dy))
                slope = max(0.001, (u_node["invert_elevation_m"] - v_node["invert_elevation_m"]) / length_m)
                
                edge_data = {
                    "conduit_id": inter_conduit_name,
                    "edge_index": edge_id_counter - 1,
                    "from_node": u_node["node_id"],
                    "to_node": v_node["node_id"],
                    "from_index": u_node["node_index"],
                    "to_index": v_node["node_index"],
                    "length_m": round(length_m, 2),
                    "diameter_m": 3.0,
                    "slope": round(slope, 5),
                    "roughness_n": 0.022,
                    "shape": "RECT_CLOSED",
                    "max_flow_capacity_m3s": round(
                        (1.0 / 0.022) * (3.0 * 2.0) * ((3.0 * 2.0 / 7.0)**(2/3)) * np.sqrt(slope), 2
                    )
                }
                edges_list.append(edge_data)
                edge_id_counter += 1

            prev_outfall_node = zone_nodes[-1]

        # Convert to DataFrames
        df_nodes = pd.DataFrame(nodes_list)
        df_edges = pd.DataFrame(edges_list)

        # Save to CSV
        nodes_csv = "data/processed/nodes.csv"
        edges_csv = "data/processed/edges.csv"
        df_nodes.to_csv(nodes_csv, index=False)
        df_edges.to_csv(edges_csv, index=False)

        # Save complete GeoJSON
        geojson_data = {
            "type": "FeatureCollection",
            "features": geojson_features
        }
        geojson_path = os.path.join(self.output_dir, "hyderabad_drainage_network.geojson")
        with open(geojson_path, "w") as f:
            json.dump(geojson_data, f, indent=4)

        logger.info(f"Successfully generated Hyderabad drainage topology: {len(df_nodes)} nodes, {len(df_edges)} conduits.")
        logger.info(f"Artifacts saved to {nodes_csv}, {edges_csv}, and {geojson_path}")

        return df_nodes, df_edges, geojson_data


if __name__ == "__main__":
    extractor = HyderabadDrainageExtractor()
    nodes, edges, geojson = extractor.generate_hyderabad_network_topology()
    print("\nHyderabad Drainage Nodes Sample:")
    print(nodes.head())
    print("\nHyderabad Drainage Conduits Sample:")
    print(edges.head())
