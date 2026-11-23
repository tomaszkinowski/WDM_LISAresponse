"""Choosing a time-block length that fits a memory budget.

Deliberately a short file. The function turns a byte budget into a block
length using a formula its own docstring calls a conservative upper bound, so
asserting that formula would mean copying it into the test and checking the
code against itself - and would freeze a heuristic that is meant to be
adjustable.

What is worth pinning is the behaviour a caller depends on: the result is a
usable block length, it moves the right way with the budget, and it never
returns zero.
"""

import pytest

from WDM_LISAresponse.code.utils.utils import resolve_time_chunk_size

NUM_SAMPLES = 100_000


@pytest.mark.parametrize("budget", [1.0, 2 ** 20, 2 ** 30, 2 ** 40, 1e18])
def test_the_result_is_always_a_usable_block_length(budget):
    """Whatever the budget, the answer is between one sample and the record.

    Callers loop over the record in blocks of this size, so zero would spin
    forever and a value beyond the record would index off the end.
    """
    chunk = resolve_time_chunk_size(NUM_SAMPLES, 1, 6,
                                    memory_budget_bytes=budget)

    assert 1 <= chunk <= NUM_SAMPLES


def test_a_bigger_budget_never_gives_a_smaller_block():
    """More memory buys blocks at least as long, never shorter."""
    budgets = [2 ** 16, 2 ** 20, 2 ** 24, 2 ** 28, 2 ** 32]

    chunks = [resolve_time_chunk_size(NUM_SAMPLES, 4, 6,
                                      memory_budget_bytes=budget)
              for budget in budgets]

    assert chunks == sorted(chunks)


@pytest.mark.parametrize("field", ["batch", "links"])
def test_more_work_per_sample_gives_shorter_blocks(field):
    """Blocks shrink as each sample costs more.

    Both the source batch and the link count multiply the per-sample working
    set, so either growing must not lengthen the block.
    """
    budget = 2 ** 24
    small = dict(batch_size=1, num_links=1)
    large = dict(batch_size=32, num_links=1) if field == "batch" \
        else dict(batch_size=1, num_links=6)

    assert (resolve_time_chunk_size(NUM_SAMPLES, memory_budget_bytes=budget,
                                    **large)
            <= resolve_time_chunk_size(NUM_SAMPLES,
                                       memory_budget_bytes=budget, **small))


def test_an_impossible_budget_still_returns_one_sample():
    """A budget too small for a single sample clamps to one, not zero.

    This is the guard that stops the caller's loop making no progress. It is
    the only branch of the function that cannot be reached by any sensible
    input, which is exactly why it is worth a test.
    """
    chunk = resolve_time_chunk_size(NUM_SAMPLES, 1000, 6,
                                    memory_budget_bytes=1.0)

    assert chunk == 1


def test_a_short_record_is_never_over_blocked():
    """A generous budget gives the whole record in one block, not more."""
    chunk = resolve_time_chunk_size(17, 1, 6, memory_budget_bytes=2 ** 40)

    assert chunk == 17
