"""Functional evaluation for model quality.

Checks if a model produces coherent, non-degenerate outputs.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from wordfreq import word_frequency

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
        - Real word ratio: % of words that exist in English dictionary
        - Word frequency: average frequency of words (common words = more coherent)
        - Sentence structure: punctuation and capitalization patterns
        """
        if not text or len(text) < 5:
            return 0.0

        # Extract words (letters only, lowercase)
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        if len(words) < 3:
            return 0.1

        # Filter to meaningful words (3+ chars) for real word check
        # Short tokens like "a", "I" are valid but "tw", "cl" are noise
        meaningful_words = [w for w in words if len(w) >= 3]
        if not meaningful_words:
            return 0.1

        # Real word ratio - most important metric
        # Use minimum frequency threshold (1e-7) to filter rare abbreviations
        min_freq = 1e-7
        real_word_count = sum(
            1 for w in meaningful_words if word_frequency(w, "en") >= min_freq
        )
        real_word_ratio = real_word_count / len(meaningful_words)

        # Average word frequency (higher = more common/natural words)
        frequencies = [word_frequency(w, "en") for w in meaningful_words]
        avg_freq = sum(frequencies) / len(frequencies)
        # Common words have freq ~1e-3, rare ~1e-6, gibberish = 0
        # Scale so avg_freq of 1e-4 (typical English text) -> ~1.0
        freq_score = min(1.0, avg_freq * 10000)

        # Sentence structure indicators
        has_punct = 1.0 if re.search(r"[.!?]", text) else 0.5
        # Check for proper capitalization after sentence endings
        has_caps = 1.0 if re.search(r"[.!?]\s+[A-Z]", text) else 0.7

        # Combine scores - real word ratio is dominant
        # Use squared ratio to penalize mixed gibberish more heavily
        # (60% real words -> 0.36, 90% -> 0.81, 100% -> 1.0)
        real_word_score = real_word_ratio ** 2
        return (
            real_word_score * 0.7  # 70% weight on real words (squared)
            + freq_score * 0.1  # 10% on word frequency
            + has_punct * 0.1  # 10% on punctuation
            + has_caps * 0.1  # 10% on capitalization
        )

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
        - Minimum length with real words
        - Has new meaningful words (not just prompt repetition)
        - Grammatical flow from prompt
        """
        if not completion:
            return 0.0

        score = 0.0

        # Extract meaningful words (3+ chars, real English)
        min_freq = 1e-7
        words = re.findall(r"\b[a-zA-Z]+\b", completion.lower())
        real_words = [
            w for w in words
            if len(w) >= 3 and word_frequency(w, "en") >= min_freq
        ]

        # Length check based on real words
        if len(real_words) >= min_length:
            score += 0.4
        elif len(real_words) >= min_length // 2:
            score += 0.2

        # Has new real words (not just repeating prompt)
        prompt_words = set(re.findall(r"\b[a-zA-Z]+\b", prompt.lower()))
        new_real_words = set(real_words) - prompt_words
        if len(new_real_words) >= 3:
            score += 0.3
        elif new_real_words:
            score += 0.15

        # Continuation flows grammatically
        first_char = completion[0] if completion else ""
        if first_char.islower() or first_char in ",.!?;:":
            score += 0.15
        elif words and words[0] in [
            "and", "the", "a", "but", "so", "then", "there",
            "in", "on", "was", "were", "had", "who", "that",
            "she", "he", "it", "they", "her", "his", "one",
        ]:
            score += 0.15

        # Has complete sentences
        if re.search(r"[.!?]", completion):
            score += 0.15

        return min(1.0, score)
