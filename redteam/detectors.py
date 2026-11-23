"""
Detector wrappers for the red team loop.

- Lynceus: local inference via subprocess (unlimited, free)
- GPTZero: web API (rate-limited, needs API key)
- StealthWriter: web API (rate-limited, needs API key)
"""

import json
import os
import subprocess
import time
import urllib.request
import urllib.error
import warnings
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DetectionResult:
    """Result from a single detection run."""
    detector: str
    ai_probability: float  # 0.0 = definitely human, 1.0 = definitely AI
    raw_response: dict = field(default_factory=dict)
    error: Optional[str] = None
    latency_ms: float = 0.0


class LynceusDetector:
    """Run Lynceus locally via subprocess. Unlimited, free, no API key."""
    
    def __init__(
        self,
        model: str = "tmr",
        device: str = "cpu",
        batch_size: int = 32,
        lynceus_bin: str = "uv run lynceus",
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.lynceus_bin = lynceus_bin
    
    def detect(self, text: str) -> DetectionResult:
        """Score text using local Lynceus."""
        start = time.time()
        try:
            # Write text to a temp file to avoid shell escaping issues
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(text)
                tmp_path = f.name
            
            try:
                cmd = (
                    f"{self.lynceus_bin} --json "
                    f"--model {self.model} "
                    f"--device {self.device} "
                    f"--batch-size {self.batch_size} "
                    f"--file {tmp_path}"
                )
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, timeout=120
                )
                
                if result.returncode != 0:
                    return DetectionResult(
                        detector="lynceus",
                        ai_probability=-1.0,
                        error=f"Exit code {result.returncode}: {result.stderr[:500]}",
                        latency_ms=(time.time() - start) * 1000,
                    )
                
                # Parse JSON output
                data = json.loads(result.stdout)
                return DetectionResult(
                    detector="lynceus",
                    ai_probability=data.get("overall_score", -1.0),
                    raw_response=data,
                    latency_ms=(time.time() - start) * 1000,
                )
            finally:
                os.unlink(tmp_path)
                
        except subprocess.TimeoutExpired:
            return DetectionResult(
                detector="lynceus",
                ai_probability=-1.0,
                error="Timeout (120s)",
                latency_ms=(time.time() - start) * 1000,
            )
        except json.JSONDecodeError as e:
            return DetectionResult(
                detector="lynceus",
                ai_probability=-1.0,
                error=f"JSON parse error: {e}",
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return DetectionResult(
                detector="lynceus",
                ai_probability=-1.0,
                error=str(e),
                latency_ms=(time.time() - start) * 1000,
            )


class GPTZeroDetector:
    """GPTZero API detector. Rate-limited, needs API key."""
    
    API_URL = "https://api.gptzero.me/v2/predict/text"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GPTZERO_API_KEY", "")
        self._last_call = 0.0
        self._min_interval = 6.0  # seconds between calls (rate limit safety)
    
    def detect(self, text: str) -> DetectionResult:
        """Score text using GPTZero API."""
        if not self.api_key:
            return DetectionResult(
                detector="gptzero",
                ai_probability=-1.0,
                error="No GPTZERO_API_KEY set",
            )
        
        # Rate limiting
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        
        start = time.time()
        try:
            payload = json.dumps({"text": text}).encode("utf-8")
            req = urllib.request.Request(
                self.API_URL,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": self.api_key,
                },
                method="POST",
            )
            
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            
            self._last_call = time.time()
            
            # GPTZero returns documents[0].average_generated_prob
            avg_prob = (
                data.get("documents", [{}])[0]
                .get("average_generated_prob", -1.0)
            )
            
            return DetectionResult(
                detector="gptzero",
                ai_probability=avg_prob,
                raw_response=data,
                latency_ms=(time.time() - start) * 1000,
            )
            
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:500]
            return DetectionResult(
                detector="gptzero",
                ai_probability=-1.0,
                error=f"HTTP {e.code}: {body}",
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return DetectionResult(
                detector="gptzero",
                ai_probability=-1.0,
                error=str(e),
                latency_ms=(time.time() - start) * 1000,
            )


class StealthWriterDetector:
    """StealthWriter API detector. Rate-limited, needs API key.

    .. deprecated:: 7.3
       This wrapper is kept for CLI backward compatibility ONLY. The endpoint
       is a guess (originally "adjust based on actual API shape"), the vendor
       API is unstable, and KB 03 §5.1 notes GPTZero catches StealthWriter
       output at 100% AI — a dead oracle rather than a useful detection signal.
       Do not add it to the default detector set; enabling it via the
       ``--detectors stealthwriter`` flag emits a deprecation warning.
    """

    # StealthWriter's public API endpoint -- GUESSED, unverified.
    API_URL = "https://www.stealthwriter.ai/api/check"

    def __init__(self, api_key: Optional[str] = None):
        warnings.warn(
            "StealthWriterDetector is deprecated: the endpoint is guessed and "
            "KB 03 finds GPTZero flags StealthWriter text at 100% AI. Prefer "
            "LynceusDetector / GPTZeroDetector.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.api_key = api_key or os.environ.get("STEALTHWRITER_API_KEY", "")
        self._last_call = 0.0  # conservative rate limit
    
    def detect(self, text: str) -> DetectionResult:
        """Score text using StealthWriter API."""
        if not self.api_key:
            return DetectionResult(
                detector="stealthwriter",
                ai_probability=-1.0,
                error="No STEALTHWRITER_API_KEY set",
            )
        
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        
        start = time.time()
        try:
            payload = json.dumps({"text": text}).encode("utf-8")
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            req = urllib.request.Request(
                self.API_URL, data=payload, headers=headers, method="POST"
            )
            
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            
            self._last_call = time.time()
            
            # Parse response — adjust based on actual API shape
            ai_prob = data.get("ai_probability", data.get("score", -1.0))
            
            return DetectionResult(
                detector="stealthwriter",
                ai_probability=ai_prob,
                raw_response=data,
                latency_ms=(time.time() - start) * 1000,
            )
            
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:500]
            return DetectionResult(
                detector="stealthwriter",
                ai_probability=-1.0,
                error=f"HTTP {e.code}: {body}",
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return DetectionResult(
                detector="stealthwriter",
                ai_probability=-1.0,
                error=str(e),
                latency_ms=(time.time() - start) * 1000,
            )


def get_detectors() -> dict[str, object]:
    """Instantiate all available detectors."""
    return {
        "lynceus": LynceusDetector(),
        "gptzero": GPTZeroDetector(),
        "stealthwriter": StealthWriterDetector(),
    }
