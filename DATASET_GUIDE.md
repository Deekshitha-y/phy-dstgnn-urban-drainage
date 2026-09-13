# 15-Year Hyderabad Dataset Acquisition & Preprocessing Guide (2010–2025)

---

## 1. Study Area: Greater Hyderabad Municipal Corporation (GHMC)
* **Geographic Extent (Bounding Box):**
  * Min Lat: `17.20° N`, Max Lat: `17.60° N`
  * Min Lon: `78.20° E`, Max Lon: `78.65° E`
* **Key Catchments & Drainage Basins:**
  1. **Kukatpally Nala Basin** (Highly urbanized, rapid runoff).
  2. **Hussain Sagar Surplus Drain & Balkapur Nala**.
  3. **Murki Nala & Central Zone Drainage Network**.
  4. **Musi River Basin & Downstream Outfalls**.

---

## 2. Public Dataset Sources & Acquisition Steps

### 2.1 15-Year Weather & Precipitation (2010–2025)
1. **ECMWF ERA5-Land Hourly Reanalysis:**
   * **Parameter:** `total_precipitation`, `2m_temperature`, `surface_pressure`, `total_evaporation`.
   * **Resolution:** $0.1^\circ \times 0.1^\circ$ ($\sim 9\text{ km}$), Hourly intervals.
   * **Access:** Free via Copernicus Climate Data Store (CDS) API (`cdsapi` Python package).
2. **Telangana State Development Planning Society (TSDPS) & IMD Stations:**
   * **Stations:** Begumpet AWS, Shamshabad Airport, Hakeempet, Charminar, Serilingampally.
   * **Resolution:** 1-hour / 15-minute precipitation.
3. **NASA GPM IMERG Final Run:**
   * Calibrated satellite precipitation at $0.1^\circ$, 30-minute resolution for cloudburst peak identification.

### 2.2 Digital Elevation Model (DEM)
* **Dataset:** **Copernicus DEM GLO-30** (30m) or **ISRO Bhuvan CartoDEM** (10m/30m).
* **Processing:**
  * Extract manhole rim surface elevation $Z_{\text{ground}}$ for every drainage junction.
  * Compute terrain slope map ($S_{\text{surface}}$) using `rasterio` / `richdem`.

### 2.3 15-Year Dynamic Land Use & Impervious Surface Mapping (2010–2025)
* **Datasets:**
  * **ESRI 10m Annual Land Use Time Series (2017–2024)** (Sentinel-2 derived).
  * **ESA WorldCover 10m** (2020, 2021).
  * **Copernicus Global Land Cover (2010–2019)**.
* **Impervious Calculation:**
  $$\text{Impervious \%} = \frac{\text{Built-up Area} + \text{Roads} + \text{Pavement}}{\text{Total Subcatchment Area}} \times 100$$
  * Captures the sharp increase in impervious concrete surfaces across Hitec City, Gachibowli, and Kukatpally from 2010 to 2025.

### 2.4 Drainage Network Topology (GIS Shapefiles)
* **OpenStreetMap (OSM) Overpass API:**
  * Tags: `waterway=drain`, `waterway=canal`, `waterway=ditch`, `man_made=manhole`.
* **GHMC Strategic Nala Development Programme (SNDP):**
  * Primary nalas, secondary stormwater drains, box culverts, and underground concrete pipes.

---

## 3. Major Historical Flood Benchmark Events for Testing

| Date | Max 24h Rainfall | Impacted Hyderabad Zones | Role in Testing |
|---|---|---|---|
| **October 13–14, 2020** | **320 mm** (Historic Cloudburst) | Begumpet, Nadeem Colony, Tolichowki, Musi Banks | **Primary Extreme Benchmark (Test Set)** |
| **September 21–23, 2016** | **170 mm** | Alwal, Nizampet, Quthbullapur, Begumpet | Historical Inundation Validation |
| **August 15, 2019** | **125 mm** | Khairatabad, Ameerpet, Secunderabad | Validation Set Event |
| **July 23–25, 2022** | **145 mm** | Serilingampally, Chandanagar, Kukatpally | Generalization Test Event |
| **July & Sept 2024** | **110 mm** | Gachibowli, Madhapur, Jubilee Hills | Recent Real-World Test Event |

---

## 4. Dataset Directory Organization

```
data/
├── hyderabad_gis/
│   ├── dem_30m_hyderabad.tif         # Copernicus DEM 30m
│   ├── nalas_and_pipes.geojson       # Extracted OSM/GHMC Drainage Network
│   └── landcover_timeseries/         # Annual LULC rasters (2010–2025)
├── rainfall_raw/
│   ├── era5_hyderabad_2010_2025.nc   # 15-Year Hourly ERA5 Precipitation
│   └── imd_station_rainfall.csv      # Ground station rain gauge data
├── swmm_models/
│   └── hyderabad_drainage_master.inp # Calibrated EPA-SWMM network
└── processed_tensors/
    ├── train_dataset.h5              # 2010–2019 Monsoons
    ├── val_dataset.h5                # 2020–2021 Monsoons (includes Oct 2020)
    └── test_dataset.h5               # 2022–2025 Monsoons + Synthetic Extreme Storms
```
