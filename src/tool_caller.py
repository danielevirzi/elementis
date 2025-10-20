"""
Tool caller for reading local GeoJSON and Parquet files

Integrates with VigilanceTools for comprehensive environmental data querying.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import geopandas as gpd

from tool import VigilanceTools

logger = logging.getLogger(__name__)


class ToolCaller:
    """
    Tool caller for accessing local data files
    
    Provides methods to read and analyze GeoJSON and Parquet files
    containing fire hotspot and flood data.
    
    Integrates with VigilanceTools for comprehensive querying capabilities.
    """
    
    def __init__(self):
        """Initialize tool caller"""
        logger.info("Initializing ToolCaller...")
        
        # Data paths
        self.vigilance_path = Path(os.getenv("VIGILANCE_PATH", "./data/vigilance"))
        self.hotspots_path = self.vigilance_path / "hotspots"
        self.floods_path = self.vigilance_path / "floods"
        
        # Ensure directories exist
        self.hotspots_path.mkdir(parents=True, exist_ok=True)
        self.floods_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize vigilance tools
        self.vigilance_tools = VigilanceTools(str(self.vigilance_path))
        
        # Available tools - now includes comprehensive vigilance functions
        self.tools = {
            # Legacy tools (for backward compatibility)
            "read_fire_data": self.read_fire_data,
            "read_flood_data": self.read_flood_data,
            "list_available_data": self.list_available_data,
            "get_data_summary": self.get_data_summary,
            "filter_by_date": self.filter_by_date,
            "get_statistics": self.get_statistics,
            
            # Fire Hotspots Query Tools
            "get_hotspots_by_region": self.vigilance_tools.get_hotspots_by_region,
            "get_hotspots_by_province": self.vigilance_tools.get_hotspots_by_province,
            "get_hotspots_by_municipality": self.vigilance_tools.get_hotspots_by_municipality,
            "get_hotspots_by_intensity": self.vigilance_tools.get_hotspots_by_intensity,
            "get_hotspots_by_confidence": self.vigilance_tools.get_hotspots_by_confidence,
            "get_hotspots_by_sensor": self.vigilance_tools.get_hotspots_by_sensor,
            "get_hotspots_by_satellite": self.vigilance_tools.get_hotspots_by_satellite,
            "get_hotspots_by_date": self.vigilance_tools.get_hotspots_by_date,
            "get_hotspots_by_time_of_day": self.vigilance_tools.get_hotspots_by_time_of_day,
            "get_hotspots_within_distance": self.vigilance_tools.get_hotspots_within_distance,
            "get_hotspots_in_bounding_box": self.vigilance_tools.get_hotspots_in_bounding_box,
            "get_hotspots_statistics": self.vigilance_tools.get_hotspots_statistics,
            
            # Flood Bulletins Query Tools
            "get_flood_zones_by_region": self.vigilance_tools.get_flood_zones_by_region,
            "get_flood_zones_by_risk_level": self.vigilance_tools.get_flood_zones_by_risk_level,
            "get_flood_zones_by_risk_class": self.vigilance_tools.get_flood_zones_by_risk_class,
            "get_flood_zones_by_minimum_risk_class": self.vigilance_tools.get_flood_zones_by_minimum_risk_class,
            "get_flood_zones_by_zone_code": self.vigilance_tools.get_flood_zones_by_zone_code,
            "get_flood_zones_with_name_pattern": self.vigilance_tools.get_flood_zones_with_name_pattern,
            "get_flood_zones_within_distance": self.vigilance_tools.get_flood_zones_within_distance,
            "get_flood_zones_in_bounding_box": self.vigilance_tools.get_flood_zones_in_bounding_box,
            "get_flood_statistics": self.vigilance_tools.get_flood_statistics,
            "compare_flood_bulletins": self.vigilance_tools.compare_flood_bulletins,
            
            # General Utility Tools
            "list_available_files": self.vigilance_tools.list_available_files,
            "get_available_regions": self.vigilance_tools.get_available_regions,
            "get_vigilance_data_summary": self.vigilance_tools.get_data_summary,
        }
        
        logger.info(f"ToolCaller initialized successfully with {len(self.tools)} tools")
    
    def execute(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """
        Execute a tool by name
        
        Args:
            tool_name: Name of tool to execute
            params: Parameters for the tool
            
        Returns:
            Tool execution result
        """
        logger.info(f"Executing tool: {tool_name}")
        
        if tool_name not in self.tools:
            logger.error(f"Unknown tool: {tool_name}")
            available_tools = self.list_tools()
            return {
                "error": f"Tool '{tool_name}' not found",
                "available_tools": available_tools
            }
        
        try:
            result = self.tools[tool_name](**params)
            return result
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {"error": f"Error executing {tool_name}: {str(e)}"}
    
    def list_tools(self) -> Dict[str, str]:
        """
        List all available tools with descriptions
        
        Returns:
            Dictionary mapping tool names to descriptions
        """
        tool_descriptions = {
            # Legacy Tools
            "read_fire_data": "Read raw fire hotspot data files",
            "read_flood_data": "Read raw flood bulletin data files",
            "list_available_data": "List all available data files",
            "get_data_summary": "Get summary of data category",
            "filter_by_date": "Filter data by date range",
            "get_statistics": "Get statistical analysis of data",
            
            # Fire Hotspots Tools
            "get_hotspots_by_region": "Get fire hotspots in a specific Italian region",
            "get_hotspots_by_province": "Get fire hotspots in a specific province",
            "get_hotspots_by_municipality": "Get fire hotspots in a specific municipality",
            "get_hotspots_by_intensity": "Get fire hotspots by intensity level (Bassa/Media/Alta/Molto Alta)",
            "get_hotspots_by_confidence": "Get fire hotspots by confidence category (Alta/Media/Bassa)",
            "get_hotspots_by_sensor": "Get fire hotspots by sensor type (MODIS/VIIRS)",
            "get_hotspots_by_satellite": "Get fire hotspots by satellite (Terra/Aqua/N/J1/J2)",
            "get_hotspots_by_date": "Get fire hotspots for a specific date",
            "get_hotspots_by_time_of_day": "Get fire hotspots by time (Giorno/Notte)",
            "get_hotspots_within_distance": "Get fire hotspots within radius from a point",
            "get_hotspots_in_bounding_box": "Get fire hotspots within geographic bounding box",
            "get_hotspots_statistics": "Get comprehensive statistics about fire hotspots",
            
            # Flood Bulletins Tools
            "get_flood_zones_by_region": "Get flood zones in a specific Italian region",
            "get_flood_zones_by_risk_level": "Get flood zones by risk level text (Verde/Giallo/Arancione/Rosso)",
            "get_flood_zones_by_risk_class": "Get flood zones by numeric risk class (0-3)",
            "get_flood_zones_by_minimum_risk_class": "Get flood zones with risk >= minimum level",
            "get_flood_zones_by_zone_code": "Get specific flood zone by code",
            "get_flood_zones_with_name_pattern": "Get flood zones matching name pattern",
            "get_flood_zones_within_distance": "Get flood zones within radius from a point",
            "get_flood_zones_in_bounding_box": "Get flood zones within geographic bounding box",
            "get_flood_statistics": "Get comprehensive statistics about flood bulletins",
            "compare_flood_bulletins": "Compare today's and tomorrow's flood bulletins",
            
            # General Utility Tools
            "list_available_files": "List all available vigilance data files",
            "get_available_regions": "Get list of available regions in dataset",
            "get_vigilance_data_summary": "Get overall summary of all vigilance data",
        }
        
        return tool_descriptions
    
    def read_fire_data(self, data_type: str = "hotspots", file_pattern: str = "*.geojson") -> Dict[str, Any]:
        """
        Read fire hotspot data
        
        Args:
            data_type: Type of fire data
            file_pattern: File pattern to match
            
        Returns:
            Dictionary with data and metadata
        """
        logger.info(f"Reading fire data: {data_type}")
        
        try:
            # Find files
            files = list(self.hotspots_path.glob(file_pattern))
            
            if not files:
                return {"error": f"No fire data files found matching {file_pattern}"}
            
            # Read the most recent file
            latest_file = max(files, key=lambda f: f.stat().st_mtime)
            
            # Determine file type and read
            if latest_file.suffix == '.geojson':
                gdf = gpd.read_file(latest_file)
            elif latest_file.suffix == '.parquet':
                gdf = gpd.read_parquet(latest_file)
            else:
                return {"error": f"Unsupported file type: {latest_file.suffix}"}
            
            # Prepare summary
            result = {
                "file": latest_file.name,
                "total_records": len(gdf),
                "columns": list(gdf.columns),
                "sample_data": gdf.head(10).to_dict('records'),
                "geometry_type": gdf.geometry.type.unique().tolist() if hasattr(gdf, 'geometry') else None,
                "bounds": gdf.total_bounds.tolist() if hasattr(gdf, 'geometry') else None
            }
            
            logger.info(f"Read {len(gdf)} fire records from {latest_file.name}")
            return result
            
        except Exception as e:
            logger.error(f"Error reading fire data: {e}")
            return {"error": str(e)}
    
    def read_flood_data(self, data_type: str = "floods", file_pattern: str = "*.geojson") -> Dict[str, Any]:
        """
        Read flood monitoring data
        
        Args:
            data_type: Type of flood data
            file_pattern: File pattern to match
            
        Returns:
            Dictionary with data and metadata
        """
        logger.info(f"Reading flood data: {data_type}")
        
        try:
            # Find files
            files = list(self.floods_path.glob(file_pattern))
            
            if not files:
                return {"error": f"No flood data files found matching {file_pattern}"}
            
            # Read the most recent file
            latest_file = max(files, key=lambda f: f.stat().st_mtime)
            
            # Determine file type and read
            if latest_file.suffix == '.geojson':
                gdf = gpd.read_file(latest_file)
            elif latest_file.suffix == '.parquet':
                gdf = gpd.read_parquet(latest_file)
            else:
                return {"error": f"Unsupported file type: {latest_file.suffix}"}
            
            # Prepare summary
            result = {
                "file": latest_file.name,
                "total_records": len(gdf),
                "columns": list(gdf.columns),
                "sample_data": gdf.head(10).to_dict('records'),
                "geometry_type": gdf.geometry.type.unique().tolist() if hasattr(gdf, 'geometry') else None,
                "bounds": gdf.total_bounds.tolist() if hasattr(gdf, 'geometry') else None
            }
            
            logger.info(f"Read {len(gdf)} flood records from {latest_file.name}")
            return result
            
        except Exception as e:
            logger.error(f"Error reading flood data: {e}")
            return {"error": str(e)}
    
    def list_available_data(self) -> Dict[str, List[str]]:
        """
        List all available data files
        
        Returns:
            Dictionary with lists of available files
        """
        logger.info("Listing available data files")
        
        result = {
            "hotspots": [f.name for f in self.hotspots_path.iterdir() if f.is_file()],
            "floods": [f.name for f in self.floods_path.iterdir() if f.is_file()]
        }
        
        logger.info(f"Found {len(result['hotspots'])} hotspot files and {len(result['floods'])} flood files")
        return result
    
    def get_data_summary(self, data_category: str) -> Dict[str, Any]:
        """
        Get summary statistics for a data category
        
        Args:
            data_category: 'hotspots' or 'floods'
            
        Returns:
            Summary statistics
        """
        logger.info(f"Getting summary for: {data_category}")
        
        path = self.hotspots_path if data_category == "hotspots" else self.floods_path
        
        files = list(path.glob("*.*"))
        
        return {
            "category": data_category,
            "total_files": len(files),
            "file_types": list(set(f.suffix for f in files)),
            "total_size_mb": sum(f.stat().st_size for f in files) / (1024 * 1024),
            "files": [f.name for f in files]
        }
    
    def filter_by_date(self, data_category: str, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        Filter data by date range
        
        Args:
            data_category: 'hotspots' or 'floods'
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            
        Returns:
            Filtered data summary
        """
        logger.info(f"Filtering {data_category} by date: {start_date} to {end_date}")
        
        # Placeholder implementation
        return {
            "category": data_category,
            "start_date": start_date,
            "end_date": end_date,
            "message": "Date filtering not yet fully implemented"
        }
    
    def get_statistics(self, data_category: str) -> Dict[str, Any]:
        """
        Get statistical analysis of data
        
        Args:
            data_category: 'hotspots' or 'floods'
            
        Returns:
            Statistical summary
        """
        logger.info(f"Computing statistics for: {data_category}")
        
        try:
            # Read data
            if data_category == "hotspots":
                result = self.read_fire_data()
            else:
                result = self.read_flood_data()
            
            if "error" in result:
                return result
            
            return {
                "category": data_category,
                "total_records": result["total_records"],
                "columns": result["columns"],
                "message": "Basic statistics computed"
            }
            
        except Exception as e:
            logger.error(f"Error computing statistics: {e}")
            return {"error": str(e)}
