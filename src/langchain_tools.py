"""
LangChain Tools for Vigilance Environmental Monitoring

Wraps Vigilance tools with LangChain's @tool decorator and Pydantic schemas
for seamless integration with LangChain agents and OpenAI function calling.
"""

import logging
from typing import Any, Dict, List

from langchain.tools import tool
from langchain_core.tools import StructuredTool

from tool import VigilanceTools
from tool_schema import (
    # Fire Hotspots
    HotspotsByRegionInput,
    HotspotsByProvinceInput,
    HotspotsByMunicipalityInput,
    HotspotsByIntensityInput,
    HotspotsByConfidenceInput,
    HotspotsBySensorInput,
    HotspotsBySatelliteInput,
    HotspotsByDateInput,
    HotspotsByTimeOfDayInput,
    HotspotsWithinDistanceInput,
    HotspotsInBoundingBoxInput,
    # Flood Bulletins
    FloodZonesByRegionInput,
    FloodZonesByRiskLevelInput,
    FloodZonesByRiskClassInput,
    FloodZonesByMinimumRiskClassInput,
    FloodZonesByZoneCodeInput,
    FloodZonesWithNamePatternInput,
    FloodZonesWithinDistanceInput,
    FloodZonesInBoundingBoxInput,
    FloodStatisticsInput,
    # Utilities
    GetAvailableRegionsInput,
    EmptyInput,
)

logger = logging.getLogger(__name__)


# Initialize Vigilance Tools
vigilance = VigilanceTools()


# ============================================================================
# FIRE HOTSPOTS TOOLS
# ============================================================================

@tool(args_schema=HotspotsByRegionInput)
def get_hotspots_by_region(region_name: str) -> Dict[str, Any]:
    """
    Ottieni tutti i punti caldi di incendio in una specifica regione italiana.
    
    Usa questo tool quando l'utente chiede informazioni su incendi in una regione specifica
    come Sicilia, Calabria, Lombardia, etc.
    
    Args:
        region_name: Nome della regione italiana (es: 'Sicilia', 'Lombardia')
    
    Returns:
        Dizionario con hotspots e metadati della regione
    """
    return vigilance.get_hotspots_by_region(region_name)


@tool(args_schema=HotspotsByProvinceInput)
def get_hotspots_by_province(province_name: str) -> Dict[str, Any]:
    """
    Ottieni tutti i punti caldi di incendio in una specifica provincia italiana.
    
    Usa questo tool per query a livello provinciale come Milano, Palermo, Roma.
    
    Args:
        province_name: Nome della provincia italiana
    
    Returns:
        Dizionario con hotspots e metadati della provincia
    """
    return vigilance.get_hotspots_by_province(province_name)


@tool(args_schema=HotspotsByMunicipalityInput)
def get_hotspots_by_municipality(municipality_name: str) -> Dict[str, Any]:
    """
    Ottieni tutti i punti caldi di incendio in un specifico comune italiano.
    
    Usa questo tool per query a livello comunale (più specifico di provincia).
    
    Args:
        municipality_name: Nome del comune italiano
    
    Returns:
        Dizionario con hotspots e metadati del comune
    """
    return vigilance.get_hotspots_by_municipality(municipality_name)


@tool(args_schema=HotspotsByIntensityInput)
def get_hotspots_by_intensity(intensity_level: str) -> Dict[str, Any]:
    """
    Ottieni incendi filtrati per livello di intensità.
    
    Livelli disponibili: 'Bassa', 'Media', 'Alta', 'Molto Alta', 'Sconosciuta'
    Usa per trovare incendi particolarmente intensi o gravi.
    
    Args:
        intensity_level: Livello di intensità del fuoco
    
    Returns:
        Dizionario con hotspots filtrati per intensità
    """
    return vigilance.get_hotspots_by_intensity(intensity_level)


@tool(args_schema=HotspotsByConfidenceInput)
def get_hotspots_by_confidence(confidence_category: str) -> Dict[str, Any]:
    """
    Ottieni incendi filtrati per categoria di confidenza del rilevamento.
    
    Categorie: 'Alta', 'Media', 'Bassa', 'Sconosciuta'
    Usa per avere dati più affidabili (Alta confidenza).
    
    Args:
        confidence_category: Categoria di confidenza
    
    Returns:
        Dizionario con hotspots filtrati per confidenza
    """
    return vigilance.get_hotspots_by_confidence(confidence_category)


@tool(args_schema=HotspotsBySensorInput)
def get_hotspots_by_sensor(sensor_type: str) -> Dict[str, Any]:
    """
    Ottieni incendi rilevati da un sensore satellitare specifico.
    
    Sensori disponibili:
    - 'MODIS': Sensore classico, risoluzione ~1km
    - 'VIIRS': Sensore più moderno, risoluzione ~375m
    
    Args:
        sensor_type: Tipo di sensore ('MODIS' o 'VIIRS')
    
    Returns:
        Dizionario con hotspots del sensore specificato
    """
    return vigilance.get_hotspots_by_sensor(sensor_type)


@tool(args_schema=HotspotsBySatelliteInput)
def get_hotspots_by_satellite(satellite_name: str) -> Dict[str, Any]:
    """
    Ottieni incendi rilevati da un satellite specifico.
    
    Satelliti disponibili:
    - 'Terra', 'Aqua': Satelliti MODIS NASA
    - 'N': NOAA S-NPP (VIIRS)
    - 'J1': NOAA-20 (VIIRS)
    - 'J2': NOAA-21 (VIIRS)
    
    Args:
        satellite_name: Nome del satellite
    
    Returns:
        Dizionario con hotspots del satellite specificato
    """
    return vigilance.get_hotspots_by_satellite(satellite_name)


@tool(args_schema=HotspotsByDateInput)
def get_hotspots_by_date(date: str) -> Dict[str, Any]:
    """
    Ottieni incendi per una data specifica.
    
    Formato data: 'YYYY-MM-DD' (es: '2025-10-15')
    Usa per analisi storiche o confronti temporali.
    
    Args:
        date: Data in formato ISO 'YYYY-MM-DD'
    
    Returns:
        Dizionario con hotspots della data specificata
    """
    return vigilance.get_hotspots_by_date(date)


@tool(args_schema=HotspotsByTimeOfDayInput)
def get_hotspots_by_time_of_day(time_period: str) -> Dict[str, Any]:
    """
    Ottieni incendi per periodo del giorno (diurno/notturno).
    
    Periodi: 'Giorno' o 'Notte'
    Utile per analisi di quando si verificano più incendi.
    
    Args:
        time_period: 'Giorno' o 'Notte'
    
    Returns:
        Dizionario con hotspots del periodo specificato
    """
    return vigilance.get_hotspots_by_time_of_day(time_period)


@tool(args_schema=HotspotsWithinDistanceInput)
def get_hotspots_within_distance(latitude: float, longitude: float, radius_km: float) -> Dict[str, Any]:
    """
    Ottieni incendi entro una certa distanza da un punto (ricerca radiale).
    
    Usa per trovare incendi vicini a una città o punto di interesse.
    Esempi coordinate:
    - Roma: 41.9028, 12.4964
    - Milano: 45.4642, 9.1900
    - Palermo: 38.1157, 13.3615
    
    Args:
        latitude: Latitudine del punto centrale (-90 a 90)
        longitude: Longitudine del punto centrale (-180 a 180)
        radius_km: Raggio di ricerca in chilometri
    
    Returns:
        Dizionario con hotspots entro il raggio specificato
    """
    return vigilance.get_hotspots_within_distance(latitude, longitude, radius_km)


@tool(args_schema=HotspotsInBoundingBoxInput)
def get_hotspots_in_bounding_box(min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> Dict[str, Any]:
    """
    Ottieni incendi all'interno di un'area rettangolare (bounding box).
    
    Usa per analisi su aree geografiche definite come regioni, macroaree.
    Esempio Italia Nord: min_lon=6.5, min_lat=44.0, max_lon=13.5, max_lat=47.0
    
    Args:
        min_lon: Longitudine minima (ovest)
        min_lat: Latitudine minima (sud)
        max_lon: Longitudine massima (est)
        max_lat: Latitudine massima (nord)
    
    Returns:
        Dizionario con hotspots nell'area specificata
    """
    return vigilance.get_hotspots_in_bounding_box(min_lon, min_lat, max_lon, max_lat)


@tool(args_schema=EmptyInput)
def get_hotspots_statistics() -> Dict[str, Any]:
    """
    Ottieni statistiche complete sugli incendi rilevati.
    
    Include:
    - Totale hotspots
    - Distribuzione per regione
    - Distribuzione per intensità
    - Distribuzione per sensore/satellite
    - Distribuzione per confidenza
    - Distribuzione giorno/notte
    - Range temporale
    - Confini geografici
    
    Usa per avere una panoramica generale degli incendi in Italia.
    
    Returns:
        Dizionario con statistiche complete
    """
    return vigilance.get_hotspots_statistics()


# ============================================================================
# FLOOD BULLETINS TOOLS
# ============================================================================

@tool(args_schema=FloodZonesByRegionInput)
def get_flood_zones_by_region(region_name: str, bulletin_type: str = "oggi") -> Dict[str, Any]:
    """
    Ottieni zone di allerta alluvionale per una specifica regione italiana.
    
    Usa per trovare il rischio alluvione in una regione specifica.
    
    Args:
        region_name: Nome della regione italiana (es: 'Emilia-Romagna', 'Toscana')
        bulletin_type: 'oggi' per bollettino odierno, 'domani' per previsione domani
    
    Returns:
        Dizionario con zone di allerta della regione
    """
    return vigilance.get_flood_zones_by_region(region_name, bulletin_type)


@tool(args_schema=FloodZonesByRiskLevelInput)
def get_flood_zones_by_risk_level(risk_level: str, bulletin_type: str = "oggi") -> Dict[str, Any]:
    """
    Ottieni zone di alluvione per livello di rischio (testo).
    
    Livelli di rischio:
    - 'Verde': Nessuna allerta (fenomeni assenti)
    - 'Giallo': Allerta ordinaria (criticità ordinaria)
    - 'Arancione': Allerta moderata (criticità moderata)
    - 'Rosso': Allerta elevata (criticità elevata)
    
    Args:
        risk_level: Livello di rischio testuale
        bulletin_type: 'oggi' o 'domani'
    
    Returns:
        Dizionario con zone al livello di rischio specificato
    """
    return vigilance.get_flood_zones_by_risk_level(risk_level, bulletin_type)


@tool(args_schema=FloodZonesByRiskClassInput)
def get_flood_zones_by_risk_class(risk_class: int, risk_type: str = "criticita_class", bulletin_type: str = "oggi") -> Dict[str, Any]:
    """
    Ottieni zone di alluvione per classe di rischio numerica (più preciso).
    
    Classi di rischio:
    - 0: Verde (nessuna allerta)
    - 1: Giallo (allerta ordinaria)
    - 2: Arancione (allerta moderata)
    - 3: Rosso (allerta elevata)
    
    Tipi di rischio:
    - 'criticita_class': Criticità generale
    - 'idrogeo_class': Rischio idrogeologico (frane, dissesto)
    - 'temporali_class': Rischio temporali
    - 'idraulico_class': Rischio idraulico (esondazioni fiumi)
    
    Args:
        risk_class: Classe numerica 0-3
        risk_type: Tipo di rischio da filtrare
        bulletin_type: 'oggi' o 'domani'
    
    Returns:
        Dizionario con zone della classe di rischio specificata
    """
    return vigilance.get_flood_zones_by_risk_class(risk_class, risk_type, bulletin_type)


@tool(args_schema=FloodZonesByMinimumRiskClassInput)
def get_flood_zones_by_minimum_risk_class(min_risk_class: int, risk_type: str = "criticita_class", bulletin_type: str = "oggi") -> Dict[str, Any]:
    """
    Ottieni zone con rischio >= livello minimo (per trovare tutte le allerte attive).
    
    Esempi:
    - min_risk_class=1: Tutte le allerte (giallo, arancione, rosso)
    - min_risk_class=2: Solo allerte gravi (arancione, rosso)
    - min_risk_class=3: Solo allerte rosse
    
    Args:
        min_risk_class: Livello minimo 0-3
        risk_type: Tipo di rischio (criticita_class, idrogeo_class, etc.)
        bulletin_type: 'oggi' o 'domani'
    
    Returns:
        Dizionario con zone che hanno almeno il livello di rischio specificato
    """
    return vigilance.get_flood_zones_by_minimum_risk_class(min_risk_class, risk_type, bulletin_type)


@tool(args_schema=FloodZonesByZoneCodeInput)
def get_flood_zones_by_zone_code(zone_code: str, bulletin_type: str = "oggi") -> Dict[str, Any]:
    """
    Ottieni una specifica zona di allerta per codice.
    
    I codici zona sono nel formato: Regione-Lettera (es: 'Lomb-A', 'Sici-B')
    Usa quando conosci il codice specifico della zona.
    
    Args:
        zone_code: Codice della zona (es: 'Lomb-A', 'Emil-B1')
        bulletin_type: 'oggi' o 'domani'
    
    Returns:
        Dizionario con i dati della zona specificata
    """
    return vigilance.get_flood_zones_by_zone_code(zone_code, bulletin_type)


@tool(args_schema=FloodZonesWithNamePatternInput)
def get_flood_zones_with_name_pattern(pattern: str, bulletin_type: str = "oggi") -> Dict[str, Any]:
    """
    Cerca zone di allerta per nome (ricerca testuale).
    
    La ricerca è case-insensitive e cerca il pattern nel nome della zona.
    Utile per trovare zone che includono specifiche località o fiumi.
    
    Args:
        pattern: Testo da cercare nel nome della zona
        bulletin_type: 'oggi' o 'domani'
    
    Returns:
        Dizionario con zone che corrispondono al pattern
    """
    return vigilance.get_flood_zones_with_name_pattern(pattern, bulletin_type)


@tool(args_schema=FloodZonesWithinDistanceInput)
def get_flood_zones_within_distance(latitude: float, longitude: float, radius_km: float, bulletin_type: str = "oggi") -> Dict[str, Any]:
    """
    Ottieni zone di allerta alluvione entro una distanza da un punto.
    
    Usa per trovare allerte vicine a città o punti di interesse.
    Esempi coordinate:
    - Bologna: 44.4949, 11.3426
    - Firenze: 43.7696, 11.2558
    
    Args:
        latitude: Latitudine del punto centrale
        longitude: Longitudine del punto centrale
        radius_km: Raggio di ricerca in chilometri
        bulletin_type: 'oggi' o 'domani'
    
    Returns:
        Dizionario con zone di allerta nel raggio specificato
    """
    return vigilance.get_flood_zones_within_distance(latitude, longitude, radius_km, bulletin_type)


@tool(args_schema=FloodZonesInBoundingBoxInput)
def get_flood_zones_in_bounding_box(min_lon: float, min_lat: float, max_lon: float, max_lat: float, bulletin_type: str = "oggi") -> Dict[str, Any]:
    """
    Ottieni zone di allerta all'interno di un'area rettangolare.
    
    Usa per analisi su macroaree geografiche.
    
    Args:
        min_lon: Longitudine minima (ovest)
        min_lat: Latitudine minima (sud)
        max_lon: Longitudine massima (est)
        max_lat: Latitudine massima (nord)
        bulletin_type: 'oggi' o 'domani'
    
    Returns:
        Dizionario con zone di allerta nell'area specificata
    """
    return vigilance.get_flood_zones_in_bounding_box(min_lon, min_lat, max_lon, max_lat, bulletin_type)


@tool(args_schema=FloodStatisticsInput)
def get_flood_statistics(bulletin_type: str = "oggi") -> Dict[str, Any]:
    """
    Ottieni statistiche complete sulle allerte alluvionali.
    
    Include:
    - Totale zone
    - Distribuzione per regione
    - Distribuzione per classe di rischio (generale, idrogeo, temporali, idraulico)
    - Confini geografici
    
    Usa per avere una panoramica generale delle allerte in Italia.
    
    Args:
        bulletin_type: 'oggi' per bollettino odierno, 'domani' per previsione
    
    Returns:
        Dizionario con statistiche complete
    """
    return vigilance.get_flood_statistics(bulletin_type)


@tool(args_schema=EmptyInput)
def compare_flood_bulletins() -> Dict[str, Any]:
    """
    Confronta il bollettino di oggi con quello di domani per trovare cambiamenti nel rischio.
    
    Identifica zone dove il rischio:
    - Aumenta (es: da giallo a arancione)
    - Diminuisce (es: da arancione a giallo)
    
    Utile per capire l'evoluzione della situazione alluvionale.
    
    Returns:
        Dizionario con confronto tra bollettini e zone con cambiamenti
    """
    return vigilance.compare_flood_bulletins()


# ============================================================================
# UTILITY TOOLS
# ============================================================================

@tool(args_schema=EmptyInput)
def list_available_files() -> Dict[str, Any]:
    """
    Elenca tutti i file di dati disponibili nel sistema.
    
    Mostra:
    - File hotspots (incendi)
    - File floods (alluvioni)
    - Totale file
    
    Returns:
        Dizionario con lista di file disponibili
    """
    return vigilance.list_available_files()


@tool(args_schema=GetAvailableRegionsInput)
def get_available_regions(data_type: str) -> Dict[str, Any]:
    """
    Ottieni la lista delle regioni italiane disponibili nei dati.
    
    Usa per sapere quali regioni hanno dati disponibili.
    
    Args:
        data_type: 'hotspots' per incendi, 'floods' per alluvioni
    
    Returns:
        Dizionario con lista di regioni disponibili
    """
    return vigilance.get_available_regions(data_type)


@tool(args_schema=EmptyInput)
def get_vigilance_data_summary() -> Dict[str, Any]:
    """
    Ottieni un riepilogo completo di tutti i dati Vigilance disponibili.
    
    Include:
    - Statistiche incendi (hotspots)
    - Statistiche alluvioni oggi
    - Statistiche alluvioni domani
    - File disponibili
    
    Usa per avere una vista d'insieme completa del sistema.
    
    Returns:
        Dizionario con riepilogo completo dei dati
    """
    return vigilance.get_data_summary()


# ============================================================================
# TOOL COLLECTION
# ============================================================================

def get_all_vigilance_tools() -> List[StructuredTool]:
    """
    Get all Vigilance tools as LangChain StructuredTool objects.
    
    Returns:
        List of all available tools for LangChain agents
    """
    tools = [
        # Fire Hotspots Tools (12)
        get_hotspots_by_region,
        get_hotspots_by_province,
        get_hotspots_by_municipality,
        get_hotspots_by_intensity,
        get_hotspots_by_confidence,
        get_hotspots_by_sensor,
        get_hotspots_by_satellite,
        get_hotspots_by_date,
        get_hotspots_by_time_of_day,
        get_hotspots_within_distance,
        get_hotspots_in_bounding_box,
        get_hotspots_statistics,
        
        # Flood Bulletins Tools (10)
        get_flood_zones_by_region,
        get_flood_zones_by_risk_level,
        get_flood_zones_by_risk_class,
        get_flood_zones_by_minimum_risk_class,
        get_flood_zones_by_zone_code,
        get_flood_zones_with_name_pattern,
        get_flood_zones_within_distance,
        get_flood_zones_in_bounding_box,
        get_flood_statistics,
        compare_flood_bulletins,
        
        # Utility Tools (3)
        list_available_files,
        get_available_regions,
        get_vigilance_data_summary,
    ]
    
    logger.info(f"Initialized {len(tools)} LangChain Vigilance tools")
    return tools


def get_tool_names() -> List[str]:
    """
    Get list of all tool names.
    
    Returns:
        List of tool names
    """
    tools = get_all_vigilance_tools()
    return [tool.name for tool in tools]


def get_tool_descriptions() -> Dict[str, str]:
    """
    Get mapping of tool names to descriptions.
    
    Returns:
        Dictionary mapping tool names to descriptions
    """
    tools = get_all_vigilance_tools()
    return {tool.name: tool.description for tool in tools}
