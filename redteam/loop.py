#!/usr/bin/env python3
"""
Red team loop for AI text detection.

Takes AI-generated text, applies transformation strategies,
and checks if they evade detection. Logs everything for analysis.

Usage:
    # Basic loop with local Lynceus only
    python redteam/loop.py --input redteam/samples/ai_text.txt

    # With external detectors (rate-limited)
    python redteam/loop.py --input redteam/samples/ai_text.txt --detectors lynceus,gptzero

    # Specify max iterations
    python redteam/loop.py --input redteam/samples/ai_text.txt --max-iter 50

    # Custom target score
    python redteam/loop.py --input redteam/samples/ai_text.txt --target 0.5
"""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent dir to path so we can import transforms/detectors
sys.path.insert(0, str(Path(__file__).parent.parent))

from redteam.transforms import (
    ALL_TRANSFORMS,
    SAMPLING_TRANSFORM_MAP,
    apply_all_transforms,
    apply_random_transforms,
    apply_specific_transforms,
)
from redteam.detectors import (
    DetectionResult,
    LynceusDetector,
    GPTZeroDetector,
    StealthWriterDetector,
)


@dataclass
class IterationRecord:
    """One iteration of the red team loop."""
    iteration: int
    strategy: str
    applied_transforms: list[str]
    ai_probability: dict[str, float]  # detector_name → score
    readability_ok: bool
    text_preview: str  # first 200 chars
    timestamp: str
    latency_ms: float
    passed: bool  # True if ALL detectors below target


@dataclass
class RedTeamConfig:
    """Configuration for the red team loop."""
    target_score: float = 0.5       # Target AI probability (below this = pass)
    max_iterations: int = 100
    iterations_per_strategy: int = 5  # How many times to try each strategy
    detectors: list[str] = field(default_factory=lambda: ["lynceus"])
    device: str = "cpu"
    model: str = "tmr"
    ollama_model: str = "qwen3:1.7b"  # Model for LLM-based rewrites
    log_dir: str = "redteam/logs"
    seed: Optional[int] = None


class RedTeamLoop:
    """Main red team loop orchestrator."""
    
    def __init__(self, config: RedTeamConfig):
        self.config = config
        self.detectors = self._init_detectors()
        self.results: list[IterationRecord] = []
        self.best_score = 1.0
        self.best_text = ""
        self.best_strategy = ""
        self.iteration = 0
        
        # Create log directory
        os.makedirs(config.log_dir, exist_ok=True)
        self.log_path = os.path.join(
            config.log_dir,
            f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
    
    def _init_detectors(self) -> dict:
        """Initialize requested detectors."""
        dets = {}
        if "lynceus" in self.config.detectors:
            dets["lynceus"] = LynceusDetector(
                model=self.config.model,
                device=self.config.device,
            )
        if "gptzero" in self.config.detectors:
            dets["gptzero"] = GPTZeroDetector()
        if "stealthwriter" in self.config.detectors:
            dets["stealthwriter"] = StealthWriterDetector()
        return dets
    
    def detect(self, text: str) -> dict[str, DetectionResult]:
        """Run all detectors on text."""
        results = {}
        for name, detector in self.detectors.items():
            results[name] = detector.detect(text)
        return results
    
    def _lynceus_score(self, text: str) -> float:
        """Run only the local Lynceus detector and return its AI probability.

        Returns 1.0 (worst) on any detector error / missing detector so
        injected score callables never raise on a broken local back end."""
        res = self.detect(text).get("lynceus")
        if res is None or res.error or res.ai_probability < 0:
            return 1.0
        return res.ai_probability

    def check_readability(self, text: str) -> bool:
        """Quick readability check: not too short, not gibberish."""
        words = text.split()
        # Minimum word count
        if len(words) < 20:
            return False
        # Check for excessive repetition (gibberish indicator)
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                return False
        # Check sentence structure
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) < 2:
            return False
        return True
    
    def log_iteration(self, record: IterationRecord):
        """Append record to JSONL log."""
        with open(self.log_path, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")
        self.results.append(record)
    
    def get_strategy_text(self, strategy: str, text: str) -> tuple[str, list[str]]:
        """Apply a named strategy to text.

        Phase-7 specific strategies (sampling attacks, combo_sample_all,
        paraphrase_verify, guided_optimize) are matched here BEFORE the generic
        ``single_``/``combo_`` prefix branches so e.g. ``combo_sample_all`` is
        not accidentally routed as a ``combo_<name>`` transform combo.
        """
        # --- Phase 7: sampling-level attacks (KB 03 3) ---
        if strategy in SAMPLING_TRANSFORM_MAP:
            return SAMPLING_TRANSFORM_MAP[strategy](
                text, model=self.config.ollama_model
            ), [strategy]
        elif strategy == "combo_sample_all":
            from redteam.transforms import combo_sample_all
            result = combo_sample_all(text, model=self.config.ollama_model)
            return result, ["combo_sample_all"]
        elif strategy == "paraphrase_verify":
            from redteam.transforms import paraphrase_with_verify
            result, _score, _rounds = paraphrase_with_verify(
                text, model=self.config.ollama_model, detect_fn=self._lynceus_score,
            )
            return result, ["paraphrase_verify"]
        elif strategy == "guided_optimize":
            from redteam.transforms import guided_optimize
            result, _score, _used = guided_optimize(
                text, model=self.config.ollama_model, score_fn=self._lynceus_score,
            )
            return result, ["guided_optimize"]
        # --- classic strategies ---
        elif strategy == "all_transforms":
            return apply_all_transforms(text)
        elif strategy == "random_3":
            return apply_random_transforms(text, n=3)
        elif strategy == "random_5":
            return apply_random_transforms(text, n=5)
        elif strategy == "random_7":
            return apply_random_transforms(text, n=7)
        elif strategy == "llm_rewrite":
            from redteam.transforms import llm_rewrite
            result = llm_rewrite(text, model=self.config.ollama_model)
            return result, ["llm_rewrite"]
        elif strategy == "llm_aggressive":
            from redteam.transforms import llm_rewrite_aggressive
            result = llm_rewrite_aggressive(text, model=self.config.ollama_model)
            return result, ["llm_rewrite_aggressive"]
        elif strategy == "llm_paraphrase":
            from redteam.transforms import llm_rewrite_paraphrase
            result = llm_rewrite_paraphrase(text, model=self.config.ollama_model)
            return result, ["llm_rewrite_paraphrase"]
        elif strategy == "llm+regex":
            from redteam.transforms import llm_rewrite
            llm_result = llm_rewrite(text, model=self.config.ollama_model)
            return apply_all_transforms(llm_result)
        elif strategy.startswith("single_"):
            # Apply a single transform
            name = strategy.replace("single_", "")
            return apply_specific_transforms(text, [name])
        elif strategy.startswith("combo_"):
            # Apply a specific regex transform combo
            names = strategy.replace("combo_", "").split("+")
            return apply_specific_transforms(text, names)
        else:
            return apply_random_transforms(text, n=3)

    def run(self, input_text: str):
        """Main loop."""
        if self.config.seed is not None:
            random.seed(self.config.seed)
        
        # Define strategies
        single_transforms = [f"single_{name}" for name, _ in ALL_TRANSFORMS]
        combos = [
            "combo_replace_ai_filler+replace_ai_openers+remove_hedging",
            "combo_replace_ai_filler+split_long_sentences+vary_sentence_length",
            "combo_remove_hedging+vary_sentence_length+inject_personal_touch",
            "combo_replace_ai_openers+remove_hedging+lowercase_after_connector",
            "combo_replace_ai_filler+replace_ai_openers+remove_hedging+lowercase_after_connector",
            "combo_replace_ai_filler+split_long_sentences+vary_sentence_length+inject_personal_touch",
        ]
        
        all_strategies = (
            ["llm_rewrite", "llm_aggressive", "llm_paraphrase", "llm+regex",
             "paraphrase_verify", "guided_optimize",
             "sample_temperature", "sample_rep_penalty", "sample_top_p",
             "combo_sample_all"] +
            ["all_transforms"] +
            ["random_3", "random_5", "random_7"] +
            single_transforms +
            combos
        )
        
        print(f"🎯 Red Team Loop")
        print(f"   Target: AI probability < {self.config.target_score}")
        print(f"   Detectors: {', '.join(self.config.detectors)}")
        print(f"   Max iterations: {self.config.max_iterations}")
        print(f"   Strategies: {len(all_strategies)}")
        print(f"   Log: {self.log_path}")
        print()
        
        # First, score the original text
        print("📝 Scoring original text...")
        orig_results = self.detect(input_text)
        orig_score = orig_results.get("lynceus", DetectionResult("lynceus", -1.0)).ai_probability
        print(f"   Original AI probability: {orig_score:.3f}")
        print()
        
        self.best_score = orig_score
        self.best_text = input_text
        self.best_strategy = "original"
        
        # Iterate through strategies
        strategies_tried = 0
        while self.iteration < self.config.max_iterations:
            # Pick a strategy (cycle through, with some randomness)
            strategy = all_strategies[strategies_tried % len(all_strategies)]
            strategies_tried += 1
            
            # Apply strategy
            transformed, applied = self.get_strategy_text(strategy, input_text)
            
            # Check readability
            readable = self.check_readability(transformed)
            if not readable:
                record = IterationRecord(
                    iteration=self.iteration,
                    strategy=strategy,
                    applied_transforms=applied,
                    ai_probability={},
                    readability_ok=False,
                    text_preview=transformed[:200],
                    timestamp=datetime.now().isoformat(),
                    latency_ms=0,
                    passed=False,
                )
                self.log_iteration(record)
                self.iteration += 1
                continue
            
            # Detect
            start = time.time()
            results = self.detect(transformed)
            latency = (time.time() - start) * 1000
            
            # Check scores
            scores = {}
            all_pass = True
            for name, result in results.items():
                if result.error:
                    print(f"   ⚠️  {name} error: {result.error}")
                    scores[name] = -1.0
                    all_pass = False
                else:
                    scores[name] = result.ai_probability
                    if result.ai_probability > self.config.target_score:
                        all_pass = False
            
            # Track best
            lynceus_score = scores.get("lynceus", 1.0)
            if 0 <= lynceus_score < self.best_score:
                self.best_score = lynceus_score
                self.best_text = transformed
                self.best_strategy = strategy
                print(f"   🏆 New best: {lynceus_score:.3f} (strategy: {strategy})")
            
            # Log
            record = IterationRecord(
                iteration=self.iteration,
                strategy=strategy,
                applied_transforms=applied,
                ai_probability=scores,
                readability_ok=readable,
                text_preview=transformed[:200],
                timestamp=datetime.now().isoformat(),
                latency_ms=latency,
                passed=all_pass,
            )
            self.log_iteration(record)
            
            # Progress
            status = "✅ PASS" if all_pass else "❌ FAIL"
            score_str = " | ".join(f"{k}={v:.3f}" for k, v in scores.items() if v >= 0)
            print(f"  [{self.iteration:3d}] {status}  {score_str}  strategy={strategy}")
            
            if all_pass:
                print(f"\n🎉 Passed all detectors! Strategy: {strategy}")
                break
            
            self.iteration += 1
        
        # Summary
        self._print_summary()
        self._save_best()
    
    def _print_summary(self):
        """Print run summary."""
        print(f"\n{'='*60}")
        print(f"📊 Red Team Summary")
        print(f"   Iterations: {self.iteration + 1}")
        print(f"   Original score: {self.results[0].ai_probability.get('lynceus', -1):.3f}" if self.results else "")
        print(f"   Best score: {self.best_score:.3f}")
        print(f"   Best strategy: {self.best_strategy}")
        print(f"   Best text preview: {self.best_text[:150]}...")
        print(f"   Log file: {self.log_path}")
        
        # Strategy effectiveness
        strategy_scores: dict[str, list[float]] = {}
        for r in self.results:
            s = r.ai_probability.get("lynceus", -1.0)
            if s >= 0:
                strategy_scores.setdefault(r.strategy, []).append(s)
        
        if strategy_scores:
            print(f"\n   Strategy effectiveness (avg AI probability):")
            for strat, scores in sorted(strategy_scores.items(), key=lambda x: min(x[1])):
                avg = sum(scores) / len(scores)
                best = min(scores)
                print(f"     {strat:40s}  avg={avg:.3f}  best={best:.3f}  runs={len(scores)}")
        print(f"{'='*60}\n")
    
    def _save_best(self):
        """Save the best text to a file."""
        if self.best_text:
            best_path = os.path.join(self.config.log_dir, "best_text.txt")
            with open(best_path, "w") as f:
                f.write(self.best_text)
            print(f"💾 Best text saved to: {best_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Red team loop for AI text detection evasion"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Input text file to transform"
    )
    parser.add_argument(
        "--target", "-t", type=float, default=0.5,
        help="Target AI probability (below this = pass). Default: 0.5"
    )
    parser.add_argument(
        "--max-iter", type=int, default=100,
        help="Maximum iterations. Default: 100"
    )
    parser.add_argument(
        "--detectors", "-d", default="lynceus",
        help="Comma-separated detector names. Default: lynceus"
    )
    parser.add_argument(
        "--model", "-m", default="tmr",
        help="Lynceus model. Default: tmr"
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help="Compute device. Default: cpu"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--log-dir", default="redteam/logs",
        help="Log directory. Default: redteam/logs"
    )
    parser.add_argument(
        "--ollama-model", default="qwen3:1.7b",
        help="Ollama model for LLM rewrites. Default: qwen3:1.7b"
    )
    
    args = parser.parse_args()
    
    # Read input
    with open(args.input, "r") as f:
        input_text = f.read()
    
    config = RedTeamConfig(
        target_score=args.target,
        max_iterations=args.max_iter,
        detectors=[d.strip() for d in args.detectors.split(",")],
        model=args.model,
        device=args.device,
        seed=args.seed,
        log_dir=args.log_dir,
        ollama_model=args.ollama_model,
    )
    
    loop = RedTeamLoop(config)
    loop.run(input_text)


if __name__ == "__main__":
    main()
