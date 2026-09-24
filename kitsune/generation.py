"""Shared generation utilities for the teacher/second-opinion passes."""
import torch
from transformers.generation import StoppingCriteria


class RepetitionStop(StoppingCriteria):
    """Stop a row once its last `window` generated tokens are periodic (period <= max_period): a decoder loop.
    Without this one runaway row pins the whole batch to max_new_tokens, and the decode loop is latency-bound."""

    def __init__(self, prompt_len: int, window: int = 24, max_period: int = 12):
        self.prompt_len, self.window, self.max_period = prompt_len, window, max_period

    def __call__(self, input_ids, scores, **kwargs):
        gen = input_ids[:, self.prompt_len:]
        done = torch.zeros(gen.shape[0], dtype=torch.bool, device=gen.device)
        if gen.shape[1] < self.window:
            return done
        last = gen[:, -self.window:]
        for p in range(1, self.max_period + 1):
            done |= (last[:, p:] == last[:, :-p]).all(dim=1)
        return done
