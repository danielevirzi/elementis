"""
LLM model management using vLLM for efficient inference
"""

import logging
import os
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils import load_config

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manage LLM models and inference via vLLM
    
    vLLM provides:
    - Efficient GPU memory management with PagedAttention
    - Faster inference than HuggingFace transformers
    - Native support for quantized models
    - Batched inference for better throughput
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize model manager with vLLM
        
        Args:
            model_name: Model to load (defaults to config/env)
        """
        logger.info("Initializing ModelManager with vLLM...")
        
        # Load configuration
        self.config = load_config()
        model_config = self.config.get("models", {})
        vllm_config = self.config.get("vllm", {})
        
        # Model settings
        self.model_name = (
            model_name or 
            os.getenv("HUGGINGFACE_MODEL", "unsloth/Phi-4-mini-instruct-bnb-4bit")
        )
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN", None)
        
        # Generation parameters (deterministic by default)
        self.temperature = model_config.get("default", {}).get("temperature", 0)
        self.max_tokens = model_config.get("default", {}).get("max_tokens", 150)
        self.top_p = model_config.get("default", {}).get("top_p", 0.9)
        self.top_k = model_config.get("default", {}).get("top_k", 40)
        
        # vLLM settings
        self.gpu_memory_utilization = vllm_config.get("gpu_memory_utilization", 0.85)
        self.tensor_parallel_size = vllm_config.get("tensor_parallel_size", 1)
        self.dtype = vllm_config.get("dtype", "half")
        self.max_model_len = vllm_config.get("max_model_len", 4096)
        self.enforce_eager = vllm_config.get("enforce_eager", False)
        
        # Check GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.device == "cpu":
            logger.warning("GPU not available! vLLM requires CUDA GPU for optimal performance")
            raise RuntimeError("vLLM requires CUDA GPU. Please ensure CUDA is available.")
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Loading model: {self.model_name}")
        
        # Load model with vLLM
        try:
            self._load_model()
            logger.info(f"✓ Model {self.model_name} loaded successfully with vLLM")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        logger.info("✓ ModelManager initialized successfully")
    
    def _load_model(self):
        """Load model using vLLM"""
        logger.info("Loading model with vLLM...")
        logger.info(f"  GPU memory utilization: {self.gpu_memory_utilization}")
        logger.info(f"  Tensor parallel size: {self.tensor_parallel_size}")
        logger.info(f"  Data type: {self.dtype}")
        logger.info(f"  Max model length: {self.max_model_len}")
        
        # Load tokenizer separately for preprocessing
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self.hf_token,
            trust_remote_code=True
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize vLLM engine
        # vLLM automatically handles quantized models like unsloth bnb-4bit
        self.llm = LLM(
            model=self.model_name,
            tokenizer=self.model_name,
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=self.tensor_parallel_size,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            trust_remote_code=True,
            enforce_eager=self.enforce_eager,
            download_dir=os.getenv("HF_HOME", None),
        )
        
        logger.info("✓ vLLM model loaded")
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        Generate text using vLLM
        
        Args:
            prompt: User prompt
            temperature: Sampling temperature (overrides default)
            max_tokens: Maximum tokens to generate (overrides default)
            system_prompt: Optional system prompt
            stop: Stop sequences
            
        Returns:
            Generated text
        """
        logger.debug(f"Generating response for prompt: {prompt[:100]}...")
        
        # Use provided values or defaults
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            # Prepare messages
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            # Apply chat template if available
            if hasattr(self.tokenizer, 'apply_chat_template'):
                messages = [{"role": "user", "content": full_prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                formatted_prompt = full_prompt
            
            # Configure sampling parameters
            sampling_params = SamplingParams(
                temperature=temp,
                max_tokens=max_tok,
                top_p=self.top_p,
                top_k=self.top_k,
                stop=stop,
            )
            
            # Generate with vLLM
            outputs = self.llm.generate([formatted_prompt], sampling_params)
            
            # Extract generated text
            generated_text = outputs[0].outputs[0].text
            
            logger.debug(f"Generated {len(generated_text)} characters")
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error generating response: {str(e)}"
    
    def generate_batch(
        self,
        prompts: List[str],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None
    ) -> List[str]:
        """
        Generate text for multiple prompts in batch (efficient with vLLM)
        
        Args:
            prompts: List of prompts
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt
            
        Returns:
            List of generated texts
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            # Prepare prompts
            formatted_prompts = []
            for prompt in prompts:
                if system_prompt:
                    full_prompt = f"{system_prompt}\n\n{prompt}"
                else:
                    full_prompt = prompt
                
                if hasattr(self.tokenizer, 'apply_chat_template'):
                    messages = [{"role": "user", "content": full_prompt}]
                    formatted = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    formatted = full_prompt
                
                formatted_prompts.append(formatted)
            
            # Configure sampling
            sampling_params = SamplingParams(
                temperature=temp,
                max_tokens=max_tok,
                top_p=self.top_p,
                top_k=self.top_k,
            )
            
            # Batch generate
            outputs = self.llm.generate(formatted_prompts, sampling_params)
            
            # Extract results
            results = [output.outputs[0].text.strip() for output in outputs]
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch generation: {e}")
            return [f"Error: {str(e)}"] * len(prompts)
    
    def generate_with_context(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate response with provided context
        
        Args:
            query: User query
            context: Context to include
            system_prompt: Optional system prompt
            
        Returns:
            Generated response
        """
        # Build prompt with context
        prompt = f"""Context:
{context}

Question: {query}

Answer:"""
        
        return self.generate(prompt, system_prompt=system_prompt)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about current model"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "engine": "vLLM",
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "tensor_parallel_size": self.tensor_parallel_size,
            "dtype": self.dtype,
            "max_model_len": self.max_model_len
        }
    
    def get_available_models(self) -> List[str]:
        """Get list of recommended quantized models"""
        return [
            "unsloth/Phi-4-mini-instruct-bnb-4bit",
            "unsloth/Phi-4-mini-reasoning-bnb-4bit",
            "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
            "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
            "unsloth/Mistral-7B-Instruct-v0.3-bnb-4bit",
        ]
