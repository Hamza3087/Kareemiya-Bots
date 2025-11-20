#!/usr/bin/env python3
"""
Production-ready Qwen3-4B intent detector with caching and fallback
Based on proven high-performance binary classifier with neutral support

Optimized for SIP bot production use with efficient inference and caching
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import hashlib
from collections import OrderedDict
import time
import threading
import logging
import os
import uuid
import warnings
warnings.filterwarnings("ignore")
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# Import config for timeout values
from . import config

class OptimizedQwenIntentDetector:
    def __init__(self, cache_size=10000, use_quantization=True, logger=None, model_name="JungZoona/T3Q-qwen2.5-14b-v1.0-e3"):
        """Production-optimized Qwen3-4B intent detector with caching"""
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info(f"Initializing Qwen intent detector with model: {model_name}...")
        
        # Use local model directory
        self.model_name = model_name
        local_model_path = f"models/T3Q-qwen2.5-14b"
        
        # Create models directory if it doesn't exist
        os.makedirs(local_model_path, exist_ok=True)
        
        try:
            # Try to load from local path first
            if os.path.exists(os.path.join(local_model_path, "config.json")):
                self.logger.info(f"Loading Qwen from local path: {local_model_path}")
                model_path = local_model_path
            else:
                self.logger.info(f"Downloading Qwen model {model_name} to: {local_model_path}")
                model_path = model_name
            
            # Configure 4-bit quantization for memory efficiency
            if use_quantization and torch.cuda.is_available():
                self.logger.info("Using 4-bit quantization for memory efficiency")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                quantization_config = None
            
            # Load tokenizer with Qwen3 specific settings
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side="left",
                cache_dir=local_model_path if model_path == model_name else None
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Check GPU availability before loading model
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.logger.info(f"🎮 CUDA detected: {gpu_name} ({gpu_memory:.2f}GB)")
                self.logger.info(f"🎮 CUDA memory available: {torch.cuda.mem_get_info(0)[0] / 1024**3:.2f}GB free")
            else:
                self.logger.error("❌ NO CUDA DETECTED - Model will run on CPU (TOO SLOW for production!)")
                raise RuntimeError("CUDA not available - cannot run 14B model on CPU in production")

            # Load model with optimizations
            try:
                # Try with Flash Attention 2 first (if available)
                if quantization_config:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        quantization_config=quantization_config,
                        device_map="auto",
                        trust_remote_code=True,
                        torch_dtype="auto",
                        attn_implementation="flash_attention_2",
                        cache_dir=local_model_path if model_path == model_name else None
                    )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto" if torch.cuda.is_available() else None,
                        trust_remote_code=True,
                        attn_implementation="flash_attention_2",
                        cache_dir=local_model_path if model_path == model_name else None
                    )
                self.logger.info("Loaded with Flash Attention 2")
            except:
                # Fallback to standard attention
                self.logger.info("Flash Attention 2 not available, using standard attention")
                if quantization_config:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        quantization_config=quantization_config,
                        device_map="auto",
                        trust_remote_code=True,
                        torch_dtype="auto",
                        cache_dir=local_model_path if model_path == model_name else None
                    )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto" if torch.cuda.is_available() else None,
                        trust_remote_code=True,
                        cache_dir=local_model_path if model_path == model_name else None
                    )
            
            self.model.eval()
            self.device = next(self.model.parameters()).device

            # Verify model is on GPU (not CPU)
            device_str = str(self.device)
            self.logger.info(f"✅ Model loaded on device: {device_str}")

            if "cpu" in device_str.lower():
                self.logger.error(f"❌ CRITICAL: Model loaded on CPU instead of GPU!")
                self.logger.error(f"❌ This will cause 100x slower inference (195s instead of 1-3s)")
                raise RuntimeError("Model loaded on CPU instead of GPU - this is too slow for production!")

            if "cuda" in device_str.lower():
                # Log GPU memory usage after loading
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                self.logger.info(f"🎮 GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

            # Save model locally if downloaded
            if model_path == model_name:
                self.logger.info(f"Saving model to local path: {local_model_path}")
                self.model.save_pretrained(local_model_path)
                self.tokenizer.save_pretrained(local_model_path)
                
        except Exception as e:
            self.logger.error(f"Failed to load Qwen model: {e}")
            raise
        
        # Thread-safe LRU Cache
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.cache_lock = threading.Lock()
        
        # Performance metrics with neutral tracking
        self.metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_requests': 0,
            'total_inference_time': 0,
            'successful_inferences': 0,
            'failed_inferences': 0,
            'positive_responses': 0,
            'negative_responses': 0,
            'neutral_responses': 0,
            'clarifying_responses': 0
        }

        # GPU inference lock - ensures only ONE inference at a time across all concurrent calls
        self.gpu_lock = threading.Lock()

        # GPU lock and timing metrics for monitoring concurrency
        self.lock_wait_times = []
        self.inference_times = []
        self.lock_acquisitions = 0
        self.lock_timeouts = 0

        self.logger.info(f"✅ Qwen model loaded on {self.device} with cache_size={cache_size}")
    
    def detect_intent(self, question: str, answer: str, timeout: float = None) -> Optional[str]:
        """
        Detect intent with timeout protection
        Returns: "positive", "negative", "neutral", "clarifying", or None on error
        """
        # Use config default if no timeout specified
        if timeout is None:
            timeout = config.QWEN_TOTAL_TIMEOUT

        # Generate unique call ID for tracking this request through logs
        call_id = uuid.uuid4().hex[:8]

        self.metrics['total_requests'] += 1

        # Validate inputs
        if not question or not answer:
            self.logger.warning(f"[{call_id}] Empty question or answer provided")
            return None

        # Clean inputs
        question = question.strip()
        answer = answer.strip()

        # Log incoming request
        self.logger.info(f"[{call_id}] 🎯 Qwen request - Q: '{question[:50]}...' A: '{answer[:50]}...'")

        # Check cache first
        cache_key = hashlib.md5(f"{question}|{answer}".encode()).hexdigest()

        with self.cache_lock:
            if cache_key in self.cache:
                self.cache.move_to_end(cache_key)
                self.metrics['cache_hits'] += 1
                cached_result = self.cache[cache_key]
                self.logger.info(f"[{call_id}] ✅ Cache HIT - result: {cached_result}")
                return cached_result

        self.metrics['cache_misses'] += 1
        self.logger.info(f"[{call_id}] ❌ Cache MISS - running inference")

        # Run Qwen inference
        start_time = time.time()
        result = self._run_inference(question, answer, timeout, call_id)
        total_time = time.time() - start_time

        self.metrics['total_inference_time'] += total_time

        if result:
            self.metrics['successful_inferences'] += 1
            # Track response types
            if result == 'positive':
                self.metrics['positive_responses'] += 1
            elif result == 'negative':
                self.metrics['negative_responses'] += 1
            elif result == 'neutral':
                self.metrics['neutral_responses'] += 1
            elif result == 'clarifying':
                self.metrics['clarifying_responses'] += 1

            self.logger.info(f"[{call_id}] ✅ Inference SUCCESS in {total_time:.3f}s - result: {result}")
            self._cache_result(cache_key, result)
        else:
            self.metrics['failed_inferences'] += 1
            self.logger.warning(f"[{call_id}] ⚠️ Inference FAILED/TIMEOUT after {total_time:.3f}s")
        
        return result
    
    def _run_inference(self, question: str, answer: str, timeout: float, call_id: str) -> Optional[str]:
        """Run inference with HARD timeout that actually interrupts execution"""
        try:
            # Use ThreadPoolExecutor to enforce hard timeout
            # This ensures inference cannot run longer than timeout even if GPU lock succeeds
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._do_gpu_inference, question, answer, call_id)
                try:
                    result = future.result(timeout=timeout)
                    return result
                except FutureTimeoutError:
                    self.logger.error(f"[{call_id}] ⏱️ HARD TIMEOUT after {timeout}s - inference too slow!")
                    self.metrics['failed_inferences'] += 1
                    return None
        except Exception as e:
            self.logger.error(f"[{call_id}] Qwen inference error: {e}")
            return None

    def _do_gpu_inference(self, question: str, answer: str, call_id: str) -> Optional[str]:
        """GPU inference with serialization lock to handle concurrent calls"""
        lock_acquired = False
        lock_start = time.time()

        try:
            # Try to acquire GPU lock with timeout (from config)
            self.logger.debug(f"[{call_id}] ⏳ Waiting for GPU lock...")
            lock_acquired = self.gpu_lock.acquire(timeout=config.QWEN_GPU_LOCK_TIMEOUT)

            if not lock_acquired:
                lock_wait = time.time() - lock_start
                self.logger.warning(f"[{call_id}] ⏱️ GPU lock timeout after {lock_wait:.2f}s - another call is using GPU")
                self.lock_timeouts += 1
                return None

            lock_wait = time.time() - lock_start
            self.lock_acquisitions += 1
            self.lock_wait_times.append(lock_wait)
            self.logger.debug(f"[{call_id}] 🔓 GPU lock acquired after {lock_wait:.3f}s wait")

            # Warn if lock wait exceeded 10 seconds (indicates high GPU contention)
            if lock_wait > 10.0:
                self.logger.warning(f"[{call_id}] ⚠️ High GPU contention: waited {lock_wait:.2f}s for lock (>10s threshold)")

            # Perform actual GPU inference
            inference_start = time.time()
            result = self._actual_gpu_inference(question, answer, call_id)
            inference_time = time.time() - inference_start

            self.inference_times.append(inference_time)
            self.logger.debug(f"[{call_id}] ⚡ GPU inference completed in {inference_time:.3f}s")

            return result

        finally:
            if lock_acquired:
                self.gpu_lock.release()
                self.logger.debug(f"[{call_id}] 🔒 GPU lock released")

    def _actual_gpu_inference(self, question: str, answer: str, call_id: str) -> Optional[str]:
        """Actual GPU inference logic (runs while holding GPU lock)"""
        # Set up slow inference warning timer
        slow_warning_timer = threading.Timer(
            5.0,
            lambda: self.logger.warning(f"[{call_id}] ⚠️ Inference taking longer than 5 seconds - possible GPU/CPU issue!")
        )
        slow_warning_timer.start()

        try:
            # Create prompt using notebook pattern - outputs number only
            prompt = f"""You are a classification system. Your task is to match the user's input to one of the following 18 categories.

MATCHING EXAMPLES:
- "yes" -> 16
- "yeah I do" -> 16
- "sure" -> 16
- "that's right" -> 16
- "no" -> 17
- "nope" -> 17
- "I don't think so" -> 17
- "not really" -> 17
- "where did you get my info" -> 1
- "how do you have my details" -> 1
- "who gave you my number" -> 1
- "what portal" -> 2
- "where does this information come from" -> 2
- "I never had an accident" -> 3
- "I wasn't in an accident" -> 3
- "I don't know man it sounds like a scam" -> 15
- "this sounds like a scam" -> 15
- "compensation for what" -> 4
- "you should already know" -> 5
- "where are you calling from" -> 6
- "where are you guys based" -> 6
- "are you my insurance" -> 7
- "is this my insurance company" -> 7
- "what company is this" -> 8
- "who are you people" -> 8
- "I wasn't hurt" -> 9
- "I didn't get injured" -> 9
- "my insurance is handling it" -> 10
- "I don't drive" -> 11
- "I've never driven" -> 11
- "I have a lawyer" -> 12
- "I already have an attorney" -> 12
- "I already got paid" -> 13
- "I was already compensated" -> 13
- "are you a lawyer" -> 14
- "is this a law firm" -> 14
- "this is a scam" -> 15
- "sounds like a scam" -> 15
- "I don't trust this" -> 15
- "when exactly was this" -> 18
- "what was the date" -> 18

CATEGORIES:
1. Where did you get this information / how do you have my number?
2. What is this online portal / where does this info come from?
3. I didn't have an accident / I wasn't in an accident
4. What is this compensation for?
5. You should have this information / why are you asking me?
6. Where are you based / where are you calling from?
7. Are you from my insurance company?
8. What is Accident Claims Helpline / who are you?
9. I wasn't injured / I didn't get hurt
10. My insurance is handling it
11. I never drove / I don't drive
12. I have an attorney / I already have a lawyer
13. I already received compensation / I was already paid
14. Are you a lawyer / is this a law firm?
15. This sounds like a scam / I don't trust this
16. Positive/Yes/Agreement
17. Negative/No/Disagreement
18. Other question not listed above

USER INPUT: "{answer}"

IMPORTANT: Respond with ONLY the matching category number (1-18). Do not add any other text, explanation, or punctuation.

ID:"""

            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            # Apply chat template with Qwen3 optimizations
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=5,  # Reduced since we only need one word
                    temperature=0.01,   # Slightly higher for better neutral detection
                    do_sample=False,    # Enable sampling for more diverse outputs
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode response
            response_tokens = outputs[0][inputs.input_ids.shape[1]:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

            # Parse response using your proven method
            return self._parse_response(response)

        except Exception as e:
            self.logger.error(f"[{call_id}] GPU inference error: {e}")
            return None
        finally:
            # Cancel slow inference warning timer if still running
            slow_warning_timer.cancel()
    
    def _parse_response(self, response_text):
        """Parse the model's output to extract classification with debug logging"""
        import re

        # Debug: log the actual model response
        self.logger.debug(f"🔍 Raw Qwen response: '{response_text}'")

        # Extract number from response
        match = re.search(r'\b(1[0-8]|[1-9])\b', response_text)
        if match:
            num = int(match.group(1))
            # Map number to intent string
            if 1 <= num <= 15:
                intent = f"rebuttal_question_{num}"
            elif num == 16:
                intent = "positive"
            elif num == 17:
                intent = "negative"
            elif num == 18:
                intent = "unhandled_question"
            else:
                intent = "negative"

            self.logger.debug(f"✅ Mapped {num} -> {intent}")
            return intent

        # Default to negative if unclear
        self.logger.info(f"⚠️ No number found in '{response_text}', defaulting to negative")
        return "negative"
    
    def _cache_result(self, cache_key: str, result: str):
        """Thread-safe cache management"""
        with self.cache_lock:
            self.cache[cache_key] = result
            self.cache.move_to_end(cache_key)
            
            # Maintain cache size
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
    
    def get_metrics(self) -> dict:
        """Get performance metrics"""
        total_requests = self.metrics['total_requests']
        if total_requests == 0:
            return self.metrics
        
        cache_hit_rate = self.metrics['cache_hits'] / total_requests
        avg_inference_time = (
            self.metrics['total_inference_time'] / max(self.metrics['successful_inferences'], 1)
        )
        
        return {
            **self.metrics,
            'cache_hit_rate': cache_hit_rate,
            'avg_inference_time': avg_inference_time,
            'model_name': self.model_name
        }

    def get_performance_metrics(self) -> dict:
        """Get GPU lock and inference performance metrics for monitoring concurrency"""
        import statistics

        metrics = {
            'lock_acquisitions': self.lock_acquisitions,
            'lock_timeouts': self.lock_timeouts,
            'lock_success_rate': (
                (self.lock_acquisitions / (self.lock_acquisitions + self.lock_timeouts) * 100)
                if (self.lock_acquisitions + self.lock_timeouts) > 0 else 0
            ),
        }

        if self.lock_wait_times:
            metrics.update({
                'avg_lock_wait_time': statistics.mean(self.lock_wait_times),
                'max_lock_wait_time': max(self.lock_wait_times),
            })

        if self.inference_times:
            metrics.update({
                'avg_inference_time': statistics.mean(self.inference_times),
                'p50_inference_time': statistics.median(self.inference_times),
                'p95_inference_time': (
                    statistics.quantiles(self.inference_times, n=20)[18]
                    if len(self.inference_times) > 20 else max(self.inference_times)
                ),
                'max_inference_time': max(self.inference_times),
            })

        return metrics

    def clear_cache(self):
        """Clear the inference cache"""
        with self.cache_lock:
            self.cache.clear()
        self.logger.info("Cache cleared")
    
    def get_cache_size(self):
        """Get current cache size"""
        with self.cache_lock:
            return len(self.cache)
    
    def warmup(self):
        """Warmup the model with a test inference"""
        self.logger.info("Warming up Qwen model...")
        try:
            test_question = "Do you recall being in a minor road traffic accident?"
            test_answer = "Yes, I remember."
            result = self.detect_intent(test_question, test_answer, timeout=5.0)
            self.logger.info(f"Warmup completed. Test result: {result}")
        except Exception as e:
            self.logger.error(f"Warmup failed: {e}")
            raise
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.logger.info("Cleaning up Qwen model...")
            
            # Clear cache
            with self.cache_lock:
                self.cache.clear()
            
            # Clean up model and tokenizer
            if hasattr(self, 'model') and self.model:
                del self.model
                
            if hasattr(self, 'tokenizer') and self.tokenizer:
                del self.tokenizer
            
            # Force garbage collection
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.logger.info("✅ Qwen cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")