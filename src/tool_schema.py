"""
LangChain Tool Schemas using Pydantic

Defines structured input schemas for all Vigilance tools using Pydantic
for automatic validation and OpenAI function calling compatibility.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ============================================================================
# FIRE HOTSPOTS TOOL SCHEMAS
# ============================================================================

class HotspotsByRegionInput(BaseModel):
    """Input schema for getting fire hotspots by region"""
    region_name: str = Field(
        description="Nome della regione italiana (es: 'Sicilia', 'Lombardia', 'Calabria')"
    )


class HotspotsByProvinceInput(BaseModel):
    """Input schema for getting fire hotspots by province"""
    province_name: str = Field(
        description="Nome della provincia italiana (es: 'Palermo', 'Milano', 'Roma')"
    )


class HotspotsByMunicipalityInput(BaseModel):
    """Input schema for getting fire hotspots by municipality"""
    municipality_name: str = Field(
        description="Nome del comune italiano (es: 'Roma', 'Milano', 'Napoli')"
    )


class HotspotsByIntensityInput(BaseModel):
    """Input schema for getting fire hotspots by intensity"""
    intensity_level: Literal["Bassa", "Media", "Alta", "Molto Alta", "Sconosciuta"] = Field(
        description="Livello di intensità del fuoco: 'Bassa', 'Media', 'Alta', 'Molto Alta', 'Sconosciuta'"
    )


class HotspotsByConfidenceInput(BaseModel):
    """Input schema for getting fire hotspots by confidence"""
    confidence_category: Literal["Alta", "Media", "Bassa", "Sconosciuta"] = Field(
        description="Categoria di confidenza: 'Alta', 'Media', 'Bassa', 'Sconosciuta'"
    )


class HotspotsBySensorInput(BaseModel):
    """Input schema for getting fire hotspots by sensor"""
    sensor_type: Literal["MODIS", "VIIRS"] = Field(
        description="Tipo di sensore satellitare: 'MODIS' o 'VIIRS'"
    )


class HotspotsBySatelliteInput(BaseModel):
    """Input schema for getting fire hotspots by satellite"""
    satellite_name: Literal["Terra", "Aqua", "N", "J1", "J2"] = Field(
        description="Nome del satellite: 'Terra', 'Aqua', 'N' (NOAA S-NPP), 'J1' (NOAA-20), 'J2' (NOAA-21)"
    )


class HotspotsByDateInput(BaseModel):
    """Input schema for getting fire hotspots by date"""
    date: str = Field(
        description="Data in formato 'YYYY-MM-DD' (es: '2025-10-15')",
        pattern=r"^\d{4}-\d{2}-\d{2}$"
    )


class HotspotsByTimeOfDayInput(BaseModel):
    """Input schema for getting fire hotspots by time of day"""
    time_period: Literal["Giorno", "Notte"] = Field(
        description="Periodo del giorno: 'Giorno' o 'Notte'"
    )


class HotspotsWithinDistanceInput(BaseModel):
    """Input schema for getting fire hotspots within distance"""
    latitude: float = Field(
        description="Latitudine del punto centrale (es: 41.9028 per Roma)",
        ge=-90.0,
        le=90.0
    )
    longitude: float = Field(
        description="Longitudine del punto centrale (es: 12.4964 per Roma)",
        ge=-180.0,
        le=180.0
    )
    radius_km: float = Field(
        description="Raggio in chilometri (es: 50)",
        gt=0,
        le=1000
    )


class HotspotsInBoundingBoxInput(BaseModel):
    """Input schema for getting fire hotspots in bounding box"""
    min_lon: float = Field(description="Longitudine minima", ge=-180.0, le=180.0)
    min_lat: float = Field(description="Latitudine minima", ge=-90.0, le=90.0)
    max_lon: float = Field(description="Longitudine massima", ge=-180.0, le=180.0)
    max_lat: float = Field(description="Latitudine massima", ge=-90.0, le=90.0)


# ============================================================================
# FLOOD BULLETINS TOOL SCHEMAS
# ============================================================================

class FloodZonesByRegionInput(BaseModel):
    """Input schema for getting flood zones by region"""
    region_name: str = Field(
        description="Nome della regione italiana (es: 'Emilia-Romagna', 'Toscana')"
    )
    bulletin_type: Literal["oggi", "domani"] = Field(
        default="oggi",
        description="Tipo di bollettino: 'oggi' (oggi) o 'domani' (domani)"
    )


class FloodZonesByRiskLevelInput(BaseModel):
    """Input schema for getting flood zones by risk level"""
    risk_level: Literal["Verde", "Giallo", "Arancione", "Rosso"] = Field(
        description="Livello di rischio: 'Verde' (nessuno), 'Giallo' (ordinario), 'Arancione' (moderato), 'Rosso' (elevato)"
    )
    bulletin_type: Literal["oggi", "domani"] = Field(
        default="oggi",
        description="Tipo di bollettino: 'oggi' o 'domani'"
    )


class FloodZonesByRiskClassInput(BaseModel):
    """Input schema for getting flood zones by numeric risk class"""
    risk_class: int = Field(
        description="Classe di rischio numerica: 0 (Verde), 1 (Giallo), 2 (Arancione), 3 (Rosso)",
        ge=0,
        le=3
    )
    risk_type: Literal["criticita_class", "idrogeo_class", "temporali_class", "idraulico_class"] = Field(
        default="criticita_class",
        description="Tipo di rischio: 'criticita_class' (generale), 'idrogeo_class' (idrogeologico), 'temporali_class' (temporali), 'idraulico_class' (idraulico)"
    )
    bulletin_type: Literal["oggi", "domani"] = Field(
        default="oggi",
        description="Tipo di bollettino: 'oggi' o 'domani'"
    )


class FloodZonesByMinimumRiskClassInput(BaseModel):
    """Input schema for getting flood zones with minimum risk level"""
    min_risk_class: int = Field(
        description="Classe di rischio minima: 0-3 (restituisce tutte le zone >= questo valore)",
        ge=0,
        le=3
    )
    risk_type: Literal["criticita_class", "idrogeo_class", "temporali_class", "idraulico_class"] = Field(
        default="criticita_class",
        description="Tipo di rischio da filtrare"
    )
    bulletin_type: Literal["oggi", "domani"] = Field(
        default="oggi",
        description="Tipo di bollettino: 'oggi' o 'domani'"
    )


class FloodZonesByZoneCodeInput(BaseModel):
    """Input schema for getting specific flood zone by code"""
    zone_code: str = Field(
        description="Codice della zona (es: 'Abru-A', 'Sici-B', 'Lomb-A')"
    )
    bulletin_type: Literal["oggi", "domani"] = Field(
        default="oggi",
        description="Tipo di bollettino: 'oggi' o 'domani'"
    )


class FloodZonesWithNamePatternInput(BaseModel):
    """Input schema for getting flood zones by name pattern"""
    pattern: str = Field(
        description="Pattern di testo da cercare nel nome della zona (case-insensitive)"
    )
    bulletin_type: Literal["oggi", "domani"] = Field(
        default="oggi",
        description="Tipo di bollettino: 'oggi' o 'domani'"
    )


class FloodZonesWithinDistanceInput(BaseModel):
    """Input schema for getting flood zones within distance"""
    latitude: float = Field(
        description="Latitudine del punto centrale",
        ge=-90.0,
        le=90.0
    )
    longitude: float = Field(
        description="Longitudine del punto centrale",
        ge=-180.0,
        le=180.0
    )
    radius_km: float = Field(
        description="Raggio in chilometri",
        gt=0,
        le=1000
    )
    bulletin_type: Literal["oggi", "domani"] = Field(
        default="oggi",
        description="Tipo di bollettino: 'oggi' o 'domani'"
    )


class FloodZonesInBoundingBoxInput(BaseModel):
    """Input schema for getting flood zones in bounding box"""
    min_lon: float = Field(description="Longitudine minima", ge=-180.0, le=180.0)
    min_lat: float = Field(description="Latitudine minima", ge=-90.0, le=90.0)
    max_lon: float = Field(description="Longitudine massima", ge=-180.0, le=180.0)
    max_lat: float = Field(description="Latitudine massima", ge=-90.0, le=90.0)
    bulletin_type: Literal["oggi", "domani"] = Field(
        default="oggi",
        description="Tipo di bollettino: 'oggi' o 'domani'"
    )


class FloodStatisticsInput(BaseModel):
    """Input schema for getting flood statistics"""
    bulletin_type: Literal["oggi", "domani"] = Field(
        default="oggi",
        description="Tipo di bollettino: 'oggi' o 'domani'"
    )


# ============================================================================
# UTILITY TOOL SCHEMAS
# ============================================================================

class GetAvailableRegionsInput(BaseModel):
    """Input schema for getting available regions"""
    data_type: Literal["hotspots", "floods"] = Field(
        description="Tipo di dati: 'hotspots' (incendi) o 'floods' (alluvioni)"
    )


# Empty schemas for tools with no parameters
class EmptyInput(BaseModel):
    """Empty input schema for tools with no parameters"""
    pass
