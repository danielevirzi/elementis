"""
Vigilance Data Tool Functions

Comprehensive set of callable functions for querying Italian environmental monitoring data:
- Fire Hotspots (FIRMS NASA with ISTAT enrichment)
- Flood Bulletins (DPC with risk classification)

Based on vigilance module examples and schemas.
"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd
import pandas as pd
from pyproj import CRS
from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)

# Standard CRS for GeoJSON (WGS84)
GEOJSON_CRS_URN = "urn:ogc:def:crs:OGC::CRS84"
GEOJSON_CRS = CRS.from_string(GEOJSON_CRS_URN)


class VigilanceTools:
    """
    Comprehensive toolkit for querying Italian environmental monitoring data
    """
    
    def __init__(self, vigilance_path: Optional[str] = None):
        """
        Initialize vigilance tools
        
        Args:
            vigilance_path: Path to vigilance data directory (default: ./data/vigilance)
        """
        self.vigilance_path = Path(vigilance_path or os.getenv("VIGILANCE_PATH", "./data/vigilance"))
        self.hotspots_path = self.vigilance_path / "hotspots"
        self.floods_path = self.vigilance_path / "floods"
        
        logger.info(f"Initialized VigilanceTools with path: {self.vigilance_path}")
    
    # ============================================================================
    # UTILITY FUNCTIONS
    # ============================================================================
    
    def _load_hotspots(self, file_path: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Load fire hotspots GeoJSON
        
        Args:
            file_path: Specific file path, or None to load latest
            
        Returns:
            GeoDataFrame with hotspot data
        """
        if file_path is None:
            # Load the default latest 24h file
            file_path = self.hotspots_path / "last_24h_hotspot_Italy.geojson"
        else:
            file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Hotspot file not found: {file_path}")
        
        gdf = gpd.read_file(file_path)
        
        # Ensure correct CRS
        if gdf.crs is None:
            gdf = gdf.set_crs(GEOJSON_CRS_URN)
        elif not gdf.crs.equals(GEOJSON_CRS):
            gdf = gdf.to_crs(GEOJSON_CRS)
        else:
            gdf = gdf.set_crs(GEOJSON_CRS_URN, allow_override=True)
        
        return gdf
    
    def _load_bulletin(self, file_path: Optional[str] = None, bulletin_type: str = "oggi") -> gpd.GeoDataFrame:
        """
        Load flood bulletin GeoJSON
        
        Args:
            file_path: Specific file path, or None to load latest
            bulletin_type: 'oggi' (today) or 'domani' (tomorrow)
            
        Returns:
            GeoDataFrame with bulletin data
        """
        if file_path is None:
            # Find the most recent bulletin of specified type
            pattern = f"*_{bulletin_type}.geojson"
            files = sorted(self.floods_path.glob(pattern))
            
            if not files:
                raise FileNotFoundError(f"No {bulletin_type} bulletin files found")
            
            file_path = files[-1]  # Most recent
        else:
            file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Bulletin file not found: {file_path}")
        
        gdf = gpd.read_file(file_path)
        
        # Ensure correct CRS
        if gdf.crs is None:
            gdf = gdf.set_crs(GEOJSON_CRS_URN)
        elif not gdf.crs.equals(GEOJSON_CRS):
            gdf = gdf.to_crs(GEOJSON_CRS)
        else:
            gdf = gdf.set_crs(GEOJSON_CRS_URN, allow_override=True)
        
        return gdf
    
    # ============================================================================
    # FIRE HOTSPOTS QUERY FUNCTIONS
    # ============================================================================
    
    def get_hotspots_by_region(self, region_name: str) -> Dict[str, Any]:
        """
        Get all fire hotspots in a specific region
        
        Args:
            region_name: Italian region name (e.g., 'Sicilia', 'Calabria', 'Lombardia')
            
        Returns:
            Dictionary with filtered hotspots and metadata
        """
        try:
            gdf = self._load_hotspots()
            filtered = gdf[gdf['regione'] == region_name]
            
            return {
                "success": True,
                "region": region_name,
                "total_hotspots": len(filtered),
                "hotspots": filtered.drop(columns=['geometry']).to_dict('records'),
                "bounds": filtered.total_bounds.tolist() if len(filtered) > 0 else None
            }
        except Exception as e:
            logger.error(f"Error in get_hotspots_by_region: {e}")
            return {"success": False, "error": str(e)}
    
    def get_hotspots_by_province(self, province_name: str) -> Dict[str, Any]:
        """
        Get all fire hotspots in a specific province
        
        Args:
            province_name: Italian province name (e.g., 'Salerno', 'Palermo')
            
        Returns:
            Dictionary with filtered hotspots and metadata
        """
        try:
            gdf = self._load_hotspots()
            filtered = gdf[gdf['provincia'] == province_name]
            
            return {
                "success": True,
                "province": province_name,
                "total_hotspots": len(filtered),
                "hotspots": filtered.drop(columns=['geometry']).to_dict('records'),
                "bounds": filtered.total_bounds.tolist() if len(filtered) > 0 else None
            }
        except Exception as e:
            logger.error(f"Error in get_hotspots_by_province: {e}")
            return {"success": False, "error": str(e)}
    
    def get_hotspots_by_municipality(self, municipality_name: str) -> Dict[str, Any]:
        """
        Get all fire hotspots in a specific municipality
        
        Args:
            municipality_name: Italian municipality name (e.g., 'Roma', 'Milano')
            
        Returns:
            Dictionary with filtered hotspots and metadata
        """
        try:
            gdf = self._load_hotspots()
            filtered = gdf[gdf['comune'] == municipality_name]
            
            return {
                "success": True,
                "municipality": municipality_name,
                "total_hotspots": len(filtered),
                "hotspots": filtered.drop(columns=['geometry']).to_dict('records'),
                "bounds": filtered.total_bounds.tolist() if len(filtered) > 0 else None
            }
        except Exception as e:
            logger.error(f"Error in get_hotspots_by_municipality: {e}")
            return {"success": False, "error": str(e)}
    
    def get_hotspots_by_intensity(self, intensity_level: str) -> Dict[str, Any]:
        """
        Get fire hotspots by intensity level
        
        Args:
            intensity_level: 'Bassa', 'Media', 'Alta', 'Molto Alta', 'Sconosciuta'
            
        Returns:
            Dictionary with filtered hotspots and metadata
        """
        try:
            gdf = self._load_hotspots()
            filtered = gdf[gdf['intensita_incendio'] == intensity_level]
            
            return {
                "success": True,
                "intensity_level": intensity_level,
                "total_hotspots": len(filtered),
                "hotspots": filtered.drop(columns=['geometry']).to_dict('records'),
                "bounds": filtered.total_bounds.tolist() if len(filtered) > 0 else None
            }
        except Exception as e:
            logger.error(f"Error in get_hotspots_by_intensity: {e}")
            return {"success": False, "error": str(e)}
    
    def get_hotspots_by_confidence(self, confidence_category: str) -> Dict[str, Any]:
        """
        Get fire hotspots by confidence category
        
        Args:
            confidence_category: 'Alta', 'Media', 'Bassa', 'Sconosciuta'
            
        Returns:
            Dictionary with filtered hotspots and metadata
        """
        try:
            gdf = self._load_hotspots()
            filtered = gdf[gdf['categoria_confidenza'] == confidence_category]
            
            return {
                "success": True,
                "confidence_category": confidence_category,
                "total_hotspots": len(filtered),
                "hotspots": filtered.drop(columns=['geometry']).to_dict('records'),
                "bounds": filtered.total_bounds.tolist() if len(filtered) > 0 else None
            }
        except Exception as e:
            logger.error(f"Error in get_hotspots_by_confidence: {e}")
            return {"success": False, "error": str(e)}
    
    def get_hotspots_by_sensor(self, sensor_type: str) -> Dict[str, Any]:
        """
        Get fire hotspots detected by specific sensor
        
        Args:
            sensor_type: 'MODIS' or 'VIIRS'
            
        Returns:
            Dictionary with filtered hotspots and metadata
        """
        try:
            gdf = self._load_hotspots()
            filtered = gdf[gdf['strumento'] == sensor_type]
            
            return {
                "success": True,
                "sensor_type": sensor_type,
                "total_hotspots": len(filtered),
                "hotspots": filtered.drop(columns=['geometry']).to_dict('records'),
                "bounds": filtered.total_bounds.tolist() if len(filtered) > 0 else None
            }
        except Exception as e:
            logger.error(f"Error in get_hotspots_by_sensor: {e}")
            return {"success": False, "error": str(e)}
    
    def get_hotspots_by_satellite(self, satellite_name: str) -> Dict[str, Any]:
        """
        Get fire hotspots from specific satellite
        
        Args:
            satellite_name: 'Terra', 'Aqua', 'N' (NOAA S-NPP), 'J1' (NOAA-20), 'J2' (NOAA-21)
            
        Returns:
            Dictionary with filtered hotspots and metadata
        """
        try:
            gdf = self._load_hotspots()
            filtered = gdf[gdf['satellite'] == satellite_name]
            
            return {
                "success": True,
                "satellite": satellite_name,
                "total_hotspots": len(filtered),
                "hotspots": filtered.drop(columns=['geometry']).to_dict('records'),
                "bounds": filtered.total_bounds.tolist() if len(filtered) > 0 else None
            }
        except Exception as e:
            logger.error(f"Error in get_hotspots_by_satellite: {e}")
            return {"success": False, "error": str(e)}
    
    def get_hotspots_by_date(self, date: str) -> Dict[str, Any]:
        """
        Get fire hotspots for a specific date
        
        Args:
            date: Date in format 'YYYY-MM-DD'
            
        Returns:
            Dictionary with filtered hotspots and metadata
        """
        try:
            gdf = self._load_hotspots()
            filtered = gdf[gdf['data_acquisizione'] == date]
            
            return {
                "success": True,
                "date": date,
                "total_hotspots": len(filtered),
                "hotspots": filtered.drop(columns=['geometry']).to_dict('records'),
                "bounds": filtered.total_bounds.tolist() if len(filtered) > 0 else None
            }
        except Exception as e:
            logger.error(f"Error in get_hotspots_by_date: {e}")
            return {"success": False, "error": str(e)}
    
    def get_hotspots_by_time_of_day(self, time_period: str) -> Dict[str, Any]:
        """
        Get fire hotspots by time of day
        
        Args:
            time_period: 'Giorno' (day) or 'Notte' (night)
            
        Returns:
            Dictionary with filtered hotspots and metadata
        """
        try:
            gdf = self._load_hotspots()
            filtered = gdf[gdf['giorno_notte'] == time_period]
            
            return {
                "success": True,
                "time_period": time_period,
                "total_hotspots": len(filtered),
                "hotspots": filtered.drop(columns=['geometry']).to_dict('records'),
                "bounds": filtered.total_bounds.tolist() if len(filtered) > 0 else None
            }
        except Exception as e:
            logger.error(f"Error in get_hotspots_by_time_of_day: {e}")
            return {"success": False, "error": str(e)}
    
    def get_hotspots_within_distance(
        self, 
        latitude: float, 
        longitude: float, 
        radius_km: float
    ) -> Dict[str, Any]:
        """
        Get fire hotspots within distance from a center point
        
        Args:
            latitude: Center point latitude
            longitude: Center point longitude
            radius_km: Radius in kilometers
            
        Returns:
            Dictionary with filtered hotspots and metadata
        """
        try:
            gdf = self._load_hotspots()
            
            # Convert to projected CRS for accurate distance calculation
            gdf_projected = gdf.to_crs('EPSG:3857')  # Web Mercator
            center_point = Point(longitude, latitude)
            center_projected = gpd.GeoSeries([center_point], crs=GEOJSON_CRS_URN).to_crs('EPSG:3857')[0]
            
            # Create buffer (radius in meters)
            buffer = center_projected.buffer(radius_km * 1000)
            
            # Find hotspots within buffer
            within_buffer = gdf_projected.geometry.within(buffer)
            filtered = gdf[within_buffer]
            
            return {
                "success": True,
                "center": {"latitude": latitude, "longitude": longitude},
                "radius_km": radius_km,
                "total_hotspots": len(filtered),
                "hotspots": filtered.drop(columns=['geometry']).to_dict('records'),
                "bounds": filtered.total_bounds.tolist() if len(filtered) > 0 else None
            }
        except Exception as e:
            logger.error(f"Error in get_hotspots_within_distance: {e}")
            return {"success": False, "error": str(e)}
    
    def get_hotspots_in_bounding_box(
        self,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float
    ) -> Dict[str, Any]:
        """
        Get fire hotspots within a bounding box
        
        Args:
            min_lon: Minimum longitude
            min_lat: Minimum latitude
            max_lon: Maximum longitude
            max_lat: Maximum latitude
            
        Returns:
            Dictionary with filtered hotspots and metadata
        """
        try:
            gdf = self._load_hotspots()
            
            bbox = Polygon([
                (min_lon, min_lat), 
                (max_lon, min_lat),
                (max_lon, max_lat), 
                (min_lon, max_lat)
            ])
            
            filtered = gdf[gdf.geometry.within(bbox)]
            
            return {
                "success": True,
                "bounding_box": {
                    "min_lon": min_lon,
                    "min_lat": min_lat,
                    "max_lon": max_lon,
                    "max_lat": max_lat
                },
                "total_hotspots": len(filtered),
                "hotspots": filtered.drop(columns=['geometry']).to_dict('records'),
                "bounds": filtered.total_bounds.tolist() if len(filtered) > 0 else None
            }
        except Exception as e:
            logger.error(f"Error in get_hotspots_in_bounding_box: {e}")
            return {"success": False, "error": str(e)}
    
    def get_hotspots_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about fire hotspots
        
        Returns:
            Dictionary with statistical summary
        """
        try:
            gdf = self._load_hotspots()
            
            # Regional statistics
            region_stats = gdf.groupby('regione').size().to_dict()
            
            # Intensity statistics
            intensity_stats = gdf.groupby('intensita_incendio').size().to_dict()
            
            # Sensor statistics
            sensor_stats = gdf.groupby('strumento').size().to_dict()
            
            # Confidence statistics
            confidence_stats = gdf.groupby('categoria_confidenza').size().to_dict()
            
            # Time statistics
            day_night_stats = gdf.groupby('giorno_notte').size().to_dict()
            
            return {
                "success": True,
                "total_hotspots": len(gdf),
                "by_region": region_stats,
                "by_intensity": intensity_stats,
                "by_sensor": sensor_stats,
                "by_confidence": confidence_stats,
                "by_time_of_day": day_night_stats,
                "date_range": {
                    "min": gdf['data_acquisizione'].min(),
                    "max": gdf['data_acquisizione'].max()
                },
                "bounds": gdf.total_bounds.tolist()
            }
        except Exception as e:
            logger.error(f"Error in get_hotspots_statistics: {e}")
            return {"success": False, "error": str(e)}
    
    # ============================================================================
    # FLOOD BULLETINS QUERY FUNCTIONS
    # ============================================================================
    
    def get_flood_zones_by_region(self, region_name: str, bulletin_type: str = "oggi") -> Dict[str, Any]:
        """
        Get all flood zones in a specific region
        
        Args:
            region_name: Italian region name (e.g., 'Emilia-Romagna', 'Toscana')
            bulletin_type: 'oggi' (today) or 'domani' (tomorrow)
            
        Returns:
            Dictionary with filtered zones and metadata
        """
        try:
            gdf = self._load_bulletin(bulletin_type=bulletin_type)
            filtered = gdf[gdf['regione'] == region_name]
            
            return {
                "success": True,
                "region": region_name,
                "bulletin_type": bulletin_type,
                "total_zones": len(filtered),
                "zones": filtered.drop(columns=['geometry']).to_dict('records'),
                "bounds": filtered.total_bounds.tolist() if len(filtered) > 0 else None
            }
        except Exception as e:
            logger.error(f"Error in get_flood_zones_by_region: {e}")
            return {"success": False, "error": str(e)}
    
    def get_flood_zones_by_risk_level(self, risk_level: str, bulletin_type: str = "oggi") -> Dict[str, Any]:
        """
        Get flood zones by risk level text
        
        Args:
            risk_level: 'Verde', 'Giallo', 'Arancione', 'Rosso'
            bulletin_type: 'oggi' (today) or 'domani' (tomorrow)
            
        Returns:
            Dictionary with filtered zones and metadata
        """
        try:
            gdf = self._load_bulletin(bulletin_type=bulletin_type)
            
            # Check in Criticita column
            if 'Criticita' in gdf.columns:
                filtered = gdf[gdf['Criticita'].str.contains(risk_level, na=False, case=False)]
            elif 'criticita' in gdf.columns:
                filtered = gdf[gdf['criticita'].str.contains(risk_level, na=False, case=False)]
            else:
                return {
                    "success": False,
                    "error": "No risk level column found in bulletin"
                }
            
            return {
                "success": True,
                "risk_level": risk_level,
                "bulletin_type": bulletin_type,
                "total_zones": len(filtered),
                "zones": filtered.drop(columns=['geometry']).to_dict('records'),
                "bounds": filtered.total_bounds.tolist() if len(filtered) > 0 else None
            }
        except Exception as e:
            logger.error(f"Error in get_flood_zones_by_risk_level: {e}")
            return {"success": False, "error": str(e)}
    
    def get_flood_zones_by_risk_class(
        self,
        risk_class: int,
        risk_type: str = "criticita_class",
        bulletin_type: str = "oggi"
    ) -> Dict[str, Any]:
        """
        Get flood zones by numeric risk class
        
        Args:
            risk_class: Numeric risk level (0=Verde, 1=Giallo, 2=Arancione, 3=Rosso)
            risk_type: 'criticita_class', 'idrogeo_class', 'temporali_class', 'idraulico_class'
            bulletin_type: 'oggi' (today) or 'domani' (tomorrow)
            
        Returns:
            Dictionary with filtered zones and metadata
        """
        try:
            gdf = self._load_bulletin(bulletin_type=bulletin_type)
            
            if risk_type not in gdf.columns:
                return {
                    "success": False,
                    "error": f"Column '{risk_type}' not found in bulletin"
                }
            
            filtered = gdf[gdf[risk_type] == risk_class]
            
            risk_names = {0: 'Verde', 1: 'Giallo', 2: 'Arancione', 3: 'Rosso'}
            
            return {
                "success": True,
                "risk_class": risk_class,
                "risk_name": risk_names.get(risk_class, 'Unknown'),
                "risk_type": risk_type,
                "bulletin_type": bulletin_type,
                "total_zones": len(filtered),
                "zones": filtered.drop(columns=['geometry']).to_dict('records'),
                "bounds": filtered.total_bounds.tolist() if len(filtered) > 0 else None
            }
        except Exception as e:
            logger.error(f"Error in get_flood_zones_by_risk_class: {e}")
            return {"success": False, "error": str(e)}
    
    def get_flood_zones_by_minimum_risk_class(
        self,
        min_risk_class: int,
        risk_type: str = "criticita_class",
        bulletin_type: str = "oggi"
    ) -> Dict[str, Any]:
        """
        Get flood zones with risk class >= minimum level
        
        Args:
            min_risk_class: Minimum risk level (0-3)
            risk_type: 'criticita_class', 'idrogeo_class', 'temporali_class', 'idraulico_class'
            bulletin_type: 'oggi' (today) or 'domani' (tomorrow)
            
        Returns:
            Dictionary with filtered zones and metadata
        """
        try:
            gdf = self._load_bulletin(bulletin_type=bulletin_type)
            
            if risk_type not in gdf.columns:
                return {
                    "success": False,
                    "error": f"Column '{risk_type}' not found in bulletin"
                }
            
            filtered = gdf[gdf[risk_type] >= min_risk_class]
            
            return {
                "success": True,
                "min_risk_class": min_risk_class,
                "risk_type": risk_type,
                "bulletin_type": bulletin_type,
                "total_zones": len(filtered),
                "zones": filtered.drop(columns=['geometry']).to_dict('records'),
                "bounds": filtered.total_bounds.tolist() if len(filtered) > 0 else None
            }
        except Exception as e:
            logger.error(f"Error in get_flood_zones_by_minimum_risk_class: {e}")
            return {"success": False, "error": str(e)}
    
    def get_flood_zones_by_zone_code(self, zone_code: str, bulletin_type: str = "oggi") -> Dict[str, Any]:
        """
        Get specific flood zone by code
        
        Args:
            zone_code: Zone code (e.g., "Abru-A", "Sici-B")
            bulletin_type: 'oggi' (today) or 'domani' (tomorrow)
            
        Returns:
            Dictionary with zone data
        """
        try:
            gdf = self._load_bulletin(bulletin_type=bulletin_type)
            
            if 'Zona_all' in gdf.columns:
                filtered = gdf[gdf['Zona_all'] == zone_code]
            elif 'codice' in gdf.columns:
                filtered = gdf[gdf['codice'] == zone_code]
            else:
                return {
                    "success": False,
                    "error": "No zone code column found in bulletin"
                }
            
            return {
                "success": True,
                "zone_code": zone_code,
                "bulletin_type": bulletin_type,
                "total_zones": len(filtered),
                "zones": filtered.drop(columns=['geometry']).to_dict('records'),
                "bounds": filtered.total_bounds.tolist() if len(filtered) > 0 else None
            }
        except Exception as e:
            logger.error(f"Error in get_flood_zones_by_zone_code: {e}")
            return {"success": False, "error": str(e)}
    
    def get_flood_zones_with_name_pattern(
        self,
        pattern: str,
        bulletin_type: str = "oggi"
    ) -> Dict[str, Any]:
        """
        Get flood zones with names matching a pattern
        
        Args:
            pattern: Text pattern to search for
            bulletin_type: 'oggi' (today) or 'domani' (tomorrow)
            
        Returns:
            Dictionary with filtered zones and metadata
        """
        try:
            gdf = self._load_bulletin(bulletin_type=bulletin_type)
            
            if 'Nome_zona' not in gdf.columns:
                return {
                    "success": False,
                    "error": "No zone name column found in bulletin"
                }
            
            filtered = gdf[gdf['Nome_zona'].str.contains(pattern, na=False, case=False)]
            
            return {
                "success": True,
                "pattern": pattern,
                "bulletin_type": bulletin_type,
                "total_zones": len(filtered),
                "zones": filtered.drop(columns=['geometry']).to_dict('records'),
                "bounds": filtered.total_bounds.tolist() if len(filtered) > 0 else None
            }
        except Exception as e:
            logger.error(f"Error in get_flood_zones_with_name_pattern: {e}")
            return {"success": False, "error": str(e)}
    
    def get_flood_zones_within_distance(
        self,
        latitude: float,
        longitude: float,
        radius_km: float,
        bulletin_type: str = "oggi"
    ) -> Dict[str, Any]:
        """
        Get flood zones within distance from a center point
        
        Args:
            latitude: Center point latitude
            longitude: Center point longitude
            radius_km: Radius in kilometers
            bulletin_type: 'oggi' (today) or 'domani' (tomorrow)
            
        Returns:
            Dictionary with filtered zones and metadata
        """
        try:
            gdf = self._load_bulletin(bulletin_type=bulletin_type)
            
            # Convert to projected CRS for accurate distance calculation
            gdf_projected = gdf.to_crs('EPSG:3857')  # Web Mercator
            center_point = Point(longitude, latitude)
            center_projected = gpd.GeoSeries([center_point], crs=GEOJSON_CRS_URN).to_crs('EPSG:3857')[0]
            
            # Create buffer (radius in meters)
            buffer = center_projected.buffer(radius_km * 1000)
            
            # Find zones within buffer
            within_buffer = gdf_projected.geometry.within(buffer)
            filtered = gdf[within_buffer]
            
            return {
                "success": True,
                "center": {"latitude": latitude, "longitude": longitude},
                "radius_km": radius_km,
                "bulletin_type": bulletin_type,
                "total_zones": len(filtered),
                "zones": filtered.drop(columns=['geometry']).to_dict('records'),
                "bounds": filtered.total_bounds.tolist() if len(filtered) > 0 else None
            }
        except Exception as e:
            logger.error(f"Error in get_flood_zones_within_distance: {e}")
            return {"success": False, "error": str(e)}
    
    def get_flood_zones_in_bounding_box(
        self,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
        bulletin_type: str = "oggi"
    ) -> Dict[str, Any]:
        """
        Get flood zones within a bounding box
        
        Args:
            min_lon: Minimum longitude
            min_lat: Minimum latitude
            max_lon: Maximum longitude
            max_lat: Maximum latitude
            bulletin_type: 'oggi' (today) or 'domani' (tomorrow)
            
        Returns:
            Dictionary with filtered zones and metadata
        """
        try:
            gdf = self._load_bulletin(bulletin_type=bulletin_type)
            
            bbox = Polygon([
                (min_lon, min_lat),
                (max_lon, min_lat),
                (max_lon, max_lat),
                (min_lon, max_lat)
            ])
            
            filtered = gdf[gdf.geometry.within(bbox)]
            
            return {
                "success": True,
                "bounding_box": {
                    "min_lon": min_lon,
                    "min_lat": min_lat,
                    "max_lon": max_lon,
                    "max_lat": max_lat
                },
                "bulletin_type": bulletin_type,
                "total_zones": len(filtered),
                "zones": filtered.drop(columns=['geometry']).to_dict('records'),
                "bounds": filtered.total_bounds.tolist() if len(filtered) > 0 else None
            }
        except Exception as e:
            logger.error(f"Error in get_flood_zones_in_bounding_box: {e}")
            return {"success": False, "error": str(e)}
    
    def get_flood_statistics(self, bulletin_type: str = "oggi") -> Dict[str, Any]:
        """
        Get comprehensive statistics about flood bulletins
        
        Args:
            bulletin_type: 'oggi' (today) or 'domani' (tomorrow)
            
        Returns:
            Dictionary with statistical summary
        """
        try:
            gdf = self._load_bulletin(bulletin_type=bulletin_type)
            
            # Regional statistics
            region_stats = gdf.groupby('regione').size().to_dict()
            
            # Risk class statistics
            stats = {
                "success": True,
                "bulletin_type": bulletin_type,
                "total_zones": len(gdf),
                "by_region": region_stats,
                "bounds": gdf.total_bounds.tolist()
            }
            
            # Add risk class statistics if available
            if 'criticita_class' in gdf.columns:
                stats['by_criticita_class'] = gdf.groupby('criticita_class').size().to_dict()
            
            if 'idrogeo_class' in gdf.columns:
                stats['by_idrogeo_class'] = gdf.groupby('idrogeo_class').size().to_dict()
            
            if 'temporali_class' in gdf.columns:
                stats['by_temporali_class'] = gdf.groupby('temporali_class').size().to_dict()
            
            if 'idraulico_class' in gdf.columns:
                stats['by_idraulico_class'] = gdf.groupby('idraulico_class').size().to_dict()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error in get_flood_statistics: {e}")
            return {"success": False, "error": str(e)}
    
    def compare_flood_bulletins(self) -> Dict[str, Any]:
        """
        Compare today's and tomorrow's flood bulletins
        
        Returns:
            Dictionary with comparison statistics
        """
        try:
            gdf_oggi = self._load_bulletin(bulletin_type="oggi")
            gdf_domani = self._load_bulletin(bulletin_type="domani")
            
            # Compare risk levels by zone
            changes = []
            
            if 'Zona_all' in gdf_oggi.columns and 'Zona_all' in gdf_domani.columns:
                for zone_code in gdf_oggi['Zona_all'].unique():
                    oggi_row = gdf_oggi[gdf_oggi['Zona_all'] == zone_code]
                    domani_row = gdf_domani[gdf_domani['Zona_all'] == zone_code]
                    
                    if len(oggi_row) > 0 and len(domani_row) > 0:
                        if 'criticita_class' in gdf_oggi.columns:
                            oggi_risk = oggi_row.iloc[0]['criticita_class']
                            domani_risk = domani_row.iloc[0]['criticita_class']
                            
                            if oggi_risk != domani_risk:
                                changes.append({
                                    "zone_code": zone_code,
                                    "region": oggi_row.iloc[0].get('regione', 'Unknown'),
                                    "today_risk_class": int(oggi_risk),
                                    "tomorrow_risk_class": int(domani_risk),
                                    "change": int(domani_risk - oggi_risk)
                                })
            
            return {
                "success": True,
                "today_total_zones": len(gdf_oggi),
                "tomorrow_total_zones": len(gdf_domani),
                "zones_with_changes": len(changes),
                "changes": changes,
                "increased_risk_zones": len([c for c in changes if c['change'] > 0]),
                "decreased_risk_zones": len([c for c in changes if c['change'] < 0])
            }
            
        except Exception as e:
            logger.error(f"Error in compare_flood_bulletins: {e}")
            return {"success": False, "error": str(e)}
    
    # ============================================================================
    # GENERAL UTILITY FUNCTIONS
    # ============================================================================
    
    def list_available_files(self) -> Dict[str, Any]:
        """
        List all available data files
        
        Returns:
            Dictionary with lists of available files
        """
        try:
            hotspot_files = [f.name for f in self.hotspots_path.iterdir() if f.is_file() and not f.name.startswith('.')]
            flood_files = [f.name for f in self.floods_path.iterdir() if f.is_file() and not f.name.startswith('.')]
            
            return {
                "success": True,
                "hotspots": hotspot_files,
                "floods": flood_files,
                "total_files": len(hotspot_files) + len(flood_files)
            }
        except Exception as e:
            logger.error(f"Error in list_available_files: {e}")
            return {"success": False, "error": str(e)}
    
    def get_available_regions(self, data_type: str = "hotspots") -> Dict[str, Any]:
        """
        Get list of available regions in dataset
        
        Args:
            data_type: 'hotspots' or 'floods'
            
        Returns:
            Dictionary with list of regions
        """
        try:
            if data_type == "hotspots":
                gdf = self._load_hotspots()
            else:
                gdf = self._load_bulletin()
            
            regions = sorted(gdf['regione'].unique().tolist())
            
            return {
                "success": True,
                "data_type": data_type,
                "total_regions": len(regions),
                "regions": regions
            }
        except Exception as e:
            logger.error(f"Error in get_available_regions: {e}")
            return {"success": False, "error": str(e)}
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get overall summary of all vigilance data
        
        Returns:
            Dictionary with comprehensive summary
        """
        try:
            # Get hotspots summary
            hotspots_summary = self.get_hotspots_statistics()
            
            # Get floods summary for both bulletins
            floods_oggi_summary = self.get_flood_statistics(bulletin_type="oggi")
            floods_domani_summary = self.get_flood_statistics(bulletin_type="domani")
            
            # Get file listings
            files = self.list_available_files()
            
            return {
                "success": True,
                "hotspots": hotspots_summary,
                "floods_oggi": floods_oggi_summary,
                "floods_domani": floods_domani_summary,
                "files": files
            }
        except Exception as e:
            logger.error(f"Error in get_data_summary: {e}")
            return {"success": False, "error": str(e)}
