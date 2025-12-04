"""Functional evaluation for model quality.

Checks if a model produces coherent, non-degenerate outputs.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

# Test prompts with expected patterns/keywords
DEFAULT_TEST_PROMPTS = [
    {
        "prompt": "Once upon a time",
        "min_length": 20,
        "description": "Story continuation",
    },
    {
        "prompt": "The quick brown fox",
        "min_length": 15,
        "description": "Common phrase continuation",
    },
    {
        "prompt": "In the beginning",
        "min_length": 20,
        "description": "Narrative opening",
    },
    {
        "prompt": "She walked into the",
        "min_length": 15,
        "description": "Action continuation",
    },
    {
        "prompt": "Hello, my name is",
        "min_length": 10,
        "description": "Introduction pattern",
    },
]


@dataclass
class EvalResult:
    """Results from functional evaluation."""

    passed: bool
    coherence_score: float  # 0-1
    repetition_score: float  # 0-1 (1 = no repetition)
    completion_score: float  # 0-1
    overall_score: float  # 0-1

    # Thresholds used
    thresholds: Dict[str, float] = field(default_factory=dict)

    # Detailed results
    samples: List[Dict[str, Any]] = field(default_factory=list)
    failure_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for registry storage."""
        return {
            "passed": self.passed,
            "coherence_score": round(self.coherence_score, 3),
            "repetition_score": round(self.repetition_score, 3),
            "completion_score": round(self.completion_score, 3),
            "overall_score": round(self.overall_score, 3),
            "failure_reasons": self.failure_reasons,
        }


class FunctionalEvaluator:
    """Evaluates model output quality through functional tests.

    Checks:
    1. Coherence: Output forms words/sentences, uses diverse vocabulary
    2. Repetition: Output doesn't degenerate into loops
    3. Completion: Output is contextually relevant to prompts

    Usage:
        evaluator = FunctionalEvaluator()
        result = evaluator.evaluate(model, tokenizer, device)
        if result.passed:
            print("Model passed functional tests")
    """

    def __init__(
        self,
        test_prompts: Optional[List[Dict[str, Any]]] = None,
        coherence_threshold: float = 0.5,
        repetition_threshold: float = 0.6,
        completion_threshold: float = 0.5,
        gen_length: int = 100,
    ):
        """Initialize evaluator.

        Args:
            test_prompts: List of test prompts with metadata
            coherence_threshold: Minimum coherence score to pass (0-1)
            repetition_threshold: Minimum repetition score to pass (0-1)
            completion_threshold: Minimum completion score to pass (0-1)
            gen_length: Number of tokens to generate per prompt
        """
        self.test_prompts = test_prompts or DEFAULT_TEST_PROMPTS
        self.coherence_threshold = coherence_threshold
        self.repetition_threshold = repetition_threshold
        self.completion_threshold = completion_threshold
        self.gen_length = gen_length

    def evaluate(
        self,
        generate_fn: Callable[[str, int], str],
        verbose: bool = True,
    ) -> EvalResult:
        """Run functional evaluation.

        Args:
            generate_fn: Function that takes (prompt, max_tokens) and returns generated text
            verbose: Print progress and results

        Returns:
            EvalResult with scores and pass/fail status
        """
        samples = []
        coherence_scores = []
        repetition_scores = []
        completion_scores = []
        failure_reasons = []

        for i, test in enumerate(self.test_prompts):
            prompt = test["prompt"]
            min_length = test.get("min_length", 10)

            if verbose:
                print(f"  [{i+1}/{len(self.test_prompts)}] Testing: {prompt!r}")

            # Generate completion
            try:
                output = generate_fn(prompt, self.gen_length)
                completion = output[len(prompt) :].strip()
            except Exception as e:
                if verbose:
                    print(f"    ERROR: {e}")
                failure_reasons.append(f"Generation failed for '{prompt}': {e}")
                continue

            # Analyze completion
            coherence = self._score_coherence(completion)
            repetition = self._score_repetition(completion)
            context_score = self._score_completion(prompt, completion, min_length)

            coherence_scores.append(coherence)
            repetition_scores.append(repetition)
            completion_scores.append(context_score)

            sample = {
                "prompt": prompt,
                "completion": completion[:200],  # Truncate for storage
                "coherence": round(coherence, 3),
                "repetition": round(repetition, 3),
                "completion_score": round(context_score, 3),
            }
            samples.append(sample)

            if verbose:
                preview = completion[:60].replace("\n", " ")
                print(f"    Output: {preview}...")
                print(
                    f"    Scores: coherence={coherence:.2f} "
                    f"repetition={repetition:.2f} completion={context_score:.2f}"
                )

        # Calculate overall scores
        if not coherence_scores:
            return EvalResult(
                passed=False,
                coherence_score=0.0,
                repetition_score=0.0,
                completion_score=0.0,
                overall_score=0.0,
                failure_reasons=["No successful generations"],
                samples=[],
            )

        avg_coherence = sum(coherence_scores) / len(coherence_scores)
        avg_repetition = sum(repetition_scores) / len(repetition_scores)
        avg_completion = sum(completion_scores) / len(completion_scores)
        overall = (avg_coherence + avg_repetition + avg_completion) / 3

        # Check thresholds
        passed = True
        if avg_coherence < self.coherence_threshold:
            passed = False
            failure_reasons.append(
                f"Coherence {avg_coherence:.2f} < {self.coherence_threshold}"
            )
        if avg_repetition < self.repetition_threshold:
            passed = False
            failure_reasons.append(
                f"Repetition {avg_repetition:.2f} < {self.repetition_threshold}"
            )
        if avg_completion < self.completion_threshold:
            passed = False
            failure_reasons.append(
                f"Completion {avg_completion:.2f} < {self.completion_threshold}"
            )

        return EvalResult(
            passed=passed,
            coherence_score=avg_coherence,
            repetition_score=avg_repetition,
            completion_score=avg_completion,
            overall_score=overall,
            thresholds={
                "coherence": self.coherence_threshold,
                "repetition": self.repetition_threshold,
                "completion": self.completion_threshold,
            },
            samples=samples,
            failure_reasons=failure_reasons,
        )

    def _score_coherence(self, text: str) -> float:
        """Score text coherence (0-1).

        Checks:
        - Has actual words (not just random characters)
        - Vocabulary diversity
        - Reasonable word lengths
        """
        if not text or len(text) < 5:
            return 0.0

        # Extract words
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        if len(words) < 3:
            return 0.1

        # Vocabulary diversity (unique / total)
        vocab_diversity = len(set(words)) / len(words)

        # Average word length (should be 3-8 for English)
        avg_len = sum(len(w) for w in words) / len(words)
        len_score = 1.0 if 3 <= avg_len <= 8 else max(0, 1 - abs(avg_len - 5.5) / 10)

        # Has punctuation (indicates sentence structure)
        has_punct = 1.0 if re.search(r"[.!?,;:]", text) else 0.5

        # Combine scores
        return (vocab_diversity * 0.4 + len_score * 0.3 + has_punct * 0.3)

    def _score_repetition(self, text: str) -> float:
        """Score for repetition (1 = no repetition, 0 = degenerate).

        Detects:
        - Same token repeated many times
        - Same phrase repeated
        - Stuck in loops
        """
        if not text or len(text) < 10:
            return 0.5

        # Check character-level repetition
        char_counts = Counter(text)
        max_char_ratio = max(char_counts.values()) / len(text)
        if max_char_ratio > 0.5:  # One char is >50% of output
            return 0.1

        # Check word-level repetition
        words = text.lower().split()
        if len(words) < 3:
            return 0.5

        word_counts = Counter(words)
        max_word_ratio = max(word_counts.values()) / len(words)

        # Check for repeated phrases (n-grams)
        ngram_repetition = self._check_ngram_repetition(words)

        # Score: 1 means no repetition
        word_score = 1.0 - min(1.0, max_word_ratio * 2)
        ngram_score = 1.0 - ngram_repetition

        return (word_score * 0.5 + ngram_score * 0.5)

    def _check_ngram_repetition(self, words: List[str], n: int = 3) -> float:
        """Check for repeated n-grams. Returns repetition ratio (0-1)."""
        if len(words) < n * 2:
            return 0.0

        ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        ngram_counts = Counter(ngrams)

        if not ngrams:
            return 0.0

        # Count how many ngrams appear more than once
        repeated = sum(1 for count in ngram_counts.values() if count > 1)
        return repeated / len(ngrams)

    def _score_completion(
        self, prompt: str, completion: str, min_length: int
    ) -> float:
        """Score contextual relevance of completion (0-1).

        Checks:
        - Minimum length requirement
        - Doesn't just repeat the prompt
        - Forms coherent continuation
        """
        if not completion:
            return 0.0

        score = 0.0

        # Length check
        words = completion.split()
        if len(words) >= min_length:
            score += 0.4
        elif len(words) >= min_length // 2:
            score += 0.2

        # Not just repeating prompt
        prompt_words = set(prompt.lower().split())
        completion_words = set(completion.lower().split())
        if completion_words - prompt_words:  # Has new words
            score += 0.3

        # Continuation flows from prompt (simple check: starts lowercase or with common continuations)
        first_char = completion[0] if completion else ""
        if first_char.islower() or first_char in ",.!?;:":
            score += 0.15
        elif completion.split()[0].lower() in ["and", "the", "a", "but", "so", "then", "there", "in", "on", "was", "were", "had", "who", "that"]:
            score += 0.15

        # Has sentence structure
        if re.search(r"[.!?]", completion):
            score += 0.15

        return min(1.0, score)
