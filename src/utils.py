"""
Utility functions for Elementis
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml

# Define custom MEMORY log level (between INFO and DEBUG)
MEMORY_LEVEL = 15
logging.addLevelName(MEMORY_LEVEL, "MEMORY")

def memory(self, message, *args, **kwargs):
    """Custom logging method for memory-related logs"""
    if self.isEnabledFor(MEMORY_LEVEL):
        self._log(MEMORY_LEVEL, message, args, **kwargs)

# Add the memory method to Logger class
logging.Logger.memory = memory


def setup_logging(log_level: str = None) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger
    """
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO")
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "elementis.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("elementis")
    logger.info(f"Logging initialized at {log_level} level")
    
    return logger


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML files
    
    Args:
        config_path: Path to config file (optional)
        
    Returns:
        Configuration dictionary
    """
    config = {}
    
    config_dir = Path("config")
    
    if config_path:
        # Load specific config file
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Load all config files
        if config_dir.exists():
            for config_file in config_dir.glob("*.yaml"):
                with open(config_file, 'r') as f:
                    file_config = yaml.safe_load(f) or {}
                    config[config_file.stem] = file_config
    
    return config


def save_config(config: Dict[str, Any], config_name: str):
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        config_name: Name of config file
    """
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    config_path = config_dir / f"{config_name}.yaml"
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def ensure_directories():
    """Ensure all required directories exist"""
    directories = [
        "data/documents",
        "data/processed",
        "data/vigilance/hotspots",
        "data/vigilance/floods",
        "data/vector_db",
        "config",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def parse_date_range(date_string: str) -> tuple:
    """
    Parse date range string
    
    Args:
        date_string: Date range string (e.g., "2024-01-01 to 2024-12-31")
        
    Returns:
        Tuple of (start_date, end_date)
    """
    parts = date_string.split("to")
    if len(parts) == 2:
        start = datetime.fromisoformat(parts[0].strip())
        end = datetime.fromisoformat(parts[1].strip())
        return start, end
    
    return None, None


def get_project_root() -> Path:
    """Get project root directory"""
    return Path(__file__).parent.parent



    
    # Check .env file
    if not Path(".env").exists():
        errors.append(".env file not found (copy from .env.example)")
    
    return errors
