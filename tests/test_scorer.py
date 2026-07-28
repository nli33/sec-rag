"""Unit tests for secrag/eval/scorer.py's numeric matching (no network/CLI calls)."""
from secrag.eval.scorer import normalize_numeric, numeric_match


def test_basic_numeric_match():
    assert numeric_match("Revenue was $100 million.", "$100 million") is True


def test_numeric_mismatch():
    assert numeric_match("Revenue was $100 million.", "$50 million") is False


def test_gold_not_numeric_returns_none():
    assert numeric_match("Revenue was $100 million.", "It depends on the segment.") is None


def test_matches_final_answer_after_intermediate_dollar_figures():
    # Regression: the scorer used to grab the first "$"-prefixed number it found (an
    # intermediate calculation figure) instead of checking the model's actual final answer.
    predicted = (
        "Calculation: $116M / $6,489M = 1.79%; $131M / $7,500M = 1.75%. "
        "3-year average ≈ **1.9%**"
    )
    assert numeric_match(predicted, "1.9%") is True


def test_matches_final_percent_after_dollar_growth_calc():
    predicted = (
        "Total net sales grew from $135,987 million (2016) to $177,866 million (2017). "
        "Change = (177,866 − 135,987) / 135,987 × 100 = **30.8%** increase."
    )
    assert numeric_match(predicted, "30.8%") is True


def test_ticker_3m_not_parsed_as_scale_word():
    # Regression: "3M" (the ticker/company name) was parsed as "3 million" by the bare
    # single-letter scale-word regex, with no way to distinguish it from a real "3M" dollar
    # figure written without a "$" prefix.
    assert normalize_numeric("The quick ratio for 3M was 0.96 by Jun'23 close") == (0.96, False)


def test_dollar_prefixed_bare_letter_scale_still_parsed():
    assert normalize_numeric("Capex was $116M this year") == (116_000_000.0, True)


def test_normalize_numeric_no_number_returns_none():
    assert normalize_numeric("no digits here") is None


def test_bare_fiscal_year_not_treated_as_the_answer():
    # Regression: a gold answer whose only "number" is a fiscal-year reference (e.g. "flat
    # in FY 2023 vs FY 2022") isn't really a numeric-answer question at all — the year is
    # incidental context repeated in nearly every answer, so treating it as the target value
    # caused coincidental false-positive matches against unrelated predicted text.
    gold = "The Real Growth was flat in FY 2023 vs FY 2022."
    assert normalize_numeric(gold) is None
    # A predicted answer about a completely different, wrong fact but mentioning the same
    # fiscal year should NOT spuriously match now that bare years are excluded.
    unrelated_pred = "Revenue grew significantly in FY2023 compared to FY2022."
    assert numeric_match(unrelated_pred, gold) is None
