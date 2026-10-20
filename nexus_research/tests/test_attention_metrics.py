import math
import unittest

from nexus_research.attention_metrics import Observation, summarize_attention


class AttentionMetricsTests(unittest.TestCase):
    def test_two_tracks_one_zone(self):
        samples = [
            Observation("a", 0.0, "sweets", 0.9),
            Observation("a", 0.1, "sweets", 0.8),
            Observation("b", 0.0, "sweets", 0.95),
        ]
        result = summarize_attention(samples, sample_interval_s=0.1)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].estimated_attention_s, 0.3)
        self.assertEqual(result[0].unique_tracks, 2)

    def test_unknown_and_low_confidence_abstain(self):
        samples = [
            Observation("a", 0.0, None, 0.99),
            Observation("a", 0.1, "snacks", 0.2),
            Observation("a", 0.2, "snacks", 0.8),
        ]
        result = summarize_attention(samples, sample_interval_s=0.1)
        self.assertEqual(result[0].accepted_observations, 1)
        self.assertAlmostEqual(result[0].estimated_attention_s, 0.1)

    def test_duplicate_timestamp_does_not_double_count(self):
        samples = [
            Observation("a", 0.0, "coffee", 0.9),
            Observation("a", 0.0, "coffee", 0.9),
        ]
        result = summarize_attention(samples, sample_interval_s=0.1)
        self.assertAlmostEqual(result[0].estimated_attention_s, 0.1)

    def test_large_gap_not_backfilled(self):
        samples = [
            Observation("a", 0.0, "coffee", 0.9),
            Observation("a", 10.0, "coffee", 0.9),
        ]
        result = summarize_attention(samples, sample_interval_s=0.1)
        self.assertAlmostEqual(result[0].estimated_attention_s, 0.1)

    def test_invalid_parameters(self):
        for interval in (0, -1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                summarize_attention([], sample_interval_s=interval)

    def test_invalid_observation_confidence(self):
        with self.assertRaises(ValueError):
            summarize_attention([Observation("a", 0.0, "a", math.nan)], sample_interval_s=0.1)

    def test_no_samples(self):
        self.assertEqual(summarize_attention([], sample_interval_s=0.1), [])


if __name__ == "__main__":
    unittest.main()
