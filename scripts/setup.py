"""
Initial environment setup script for Elementis
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import ensure_directories, validate_environment, setup_logging


def main():
    """Run initial setup"""
    print("=" * 60)
    print("Elementis - Initial Setup")
    print("=" * 60)
    
    # Setup logging
    logger = setup_logging()
    
    # Create required directories
    print("\n1. Creating directory structure...")
    try:
        ensure_directories()
        print("   [SUCCESS] Directories created successfully")
    except Exception as e:
        print(f"   [ERROR] Error creating directories: {e}")
        return 1
    
    # Create .gitkeep files for empty directories
    print("\n2. Creating .gitkeep files...")
    gitkeep_dirs = [
        "data/documents",
        "data/processed",
        "data/vigilance/hotspots",
        "data/vigilance/floods",
        "data/vector_db"
    ]
    
    for dir_path in gitkeep_dirs:
        gitkeep_file = Path(dir_path) / ".gitkeep"
        gitkeep_file.touch()
    
    print("   [SUCCESS] .gitkeep files created")
    
    # Check for .env file
    print("\n3. Checking environment configuration...")
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists():
        if env_example.exists():
            print("   [WARNING] .env file not found")
            print("   [INFO] Please copy .env.example to .env and configure it")
            print(f"      Command: copy {env_example} {env_file}")
        else:
            print("   [ERROR] Neither .env nor .env.example found")
            return 1
    else:
        print("   [SUCCESS] .env file exists")
    
    # Validate environment
    print("\n4. Validating environment...")
    errors = validate_environment()
    
    if errors:
        print("   [WARNING] Found issues:")
        for error in errors:
            print(f"      - {error}")
    else:
        print("   [SUCCESS] Environment validation passed")
    
    # Summary
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Configure .env file with your settings")
    print("2. Run: python scripts/download_models.py")
    print("3. Add PDF documents to data/documents/")
    print("4. Run: python scripts/extract_markdown.py")
    print("   Then clean markdown files manually")
    print("5. Run: python scripts/build_from_markdown.py")
    print("5. Start the app: python src/app.py")
    print("\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
