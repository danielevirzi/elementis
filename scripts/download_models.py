"""
Script to download required models from HuggingFace
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import setup_logging
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def main():
    """Download required models"""
    print("=" * 60)
    print("Elementis - Download Models from HuggingFace")
    print("=" * 60)
    
    # Load environment
    load_dotenv()
    
    # Setup logging
    logger = setup_logging()
    
    # HuggingFace settings
    model_name = os.getenv("HUGGINGFACE_MODEL", "microsoft/Phi-4-mini-instruct")
    hf_token = os.getenv("HUGGINGFACE_TOKEN", None)
    
    print(f"\nModel to download: {model_name}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Check cache
    print("\n1. Checking HuggingFace cache...")
    cache_dir = Path.home() / ".cache" / "huggingface"
    if cache_dir.exists():
        print(f"   Cache directory: {cache_dir}")
    else:
        print("   Cache will be created on first download")
    
    # Download tokenizer
    print(f"\n2. Downloading tokenizer for {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
            trust_remote_code=True
        )
        print("   [SUCCESS] Tokenizer downloaded successfully")
    except Exception as e:
        print(f"   [ERROR] Error downloading tokenizer: {e}")
        logger.error(f"Failed to download tokenizer: {e}")
        
        if "gated" in str(e).lower():
            print("\n   This model requires access approval:")
            print(f"   1. Visit: https://huggingface.co/{model_name}")
            print("   2. Accept the terms of use")
            print("   3. Generate a token at: https://huggingface.co/settings/tokens")
            print("   4. Add to .env: HUGGINGFACE_TOKEN=your_token_here")
        
        return 1
    
    # Download model
    print(f"\n3. Downloading model {model_name}...")
    print("   This may take several minutes depending on model size and internet speed...")
    print("   Model will be cached for future use.")
    
    try:
        # Determine if using quantization
        if torch.cuda.is_available():
            print("   Using 4-bit quantization for GPU (saves memory)")
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                token=hf_token,
                trust_remote_code=True
            )
        else:
            print("   Downloading for CPU (no quantization)")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float32,
                token=hf_token,
                trust_remote_code=True
            )
        
        print("   [SUCCESS] Model downloaded successfully")
        
    except Exception as e:
        print(f"\n   [ERROR] Error downloading model: {e}")
        logger.error(f"Failed to download model: {e}")
        return 1
    
    # Test model
    print("\n4. Testing model...")
    try:
        inputs = tokenizer("Hello, how are you?", return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   Model response: {response[:50]}...")
        print("   [SUCCESS] Model is working")
        
    except Exception as e:
        print(f"   [WARNING]  Error testing model: {e}")
    
    print("\n" + "=" * 60)
    print("Model setup complete!")
    print("=" * 60)
    print(f"\nModel {model_name} is ready to use.")
    print(f"Model size on disk: {cache_dir / 'hub'}")
    
    # Suggest other models
    print("\nRecommended models for different tasks:")
    print("  - microsoft/Phi-4-mini-instruct (4B - Current default, best balance)")
    print("  - Qwen/Qwen3-1.7B (1.7B - Router/reasoning tasks)")
    print("  - meta-llama/Llama-3.2-1B-Instruct (1B - Very fast)")
    print("  - google/gemma-2-2b-it (2B - Good balance)")
    print("  - mistralai/Mistral-7B-Instruct-v0.3 (7B - More capable)")
    print("\nTo change model: Edit HUGGINGFACE_MODEL in .env")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nDownload cancelled.")
        sys.exit(1)
