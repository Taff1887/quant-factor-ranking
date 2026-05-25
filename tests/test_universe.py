"""Correctness tests for point-in-time S&P 500 membership reconstruction.

These guard the single most important methodological control in the project:
that we never grant a stock index membership before it actually joined.
"""

import pandas as pd

from qfr.data.universe import membership_panel, parse_changes, reconstruct_membership


def test_reconstruct_basic_add_remove():
    today = {"A", "B"}
    changes = parse_changes([{"date": "2021-06-01", "symbol": "B", "removedTicker": "X"}])

    # Before the change: B had not yet joined, X was still a member.
    before = reconstruct_membership(today, changes, pd.Timestamp("2021-01-01"))
    assert before == {"A", "X"}

    # After the change: back to today's set.
    after = reconstruct_membership(today, changes, pd.Timestamp("2021-07-01"))
    assert after == {"A", "B"}


def test_reverse_chronological_add_then_remove():
    # B was added (2021) then removed (2022) -> not a current member.
    today = {"A"}
    changes = parse_changes(
        [
            {"date": "2021-01-01", "symbol": "B", "removedTicker": ""},
            {"date": "2022-01-01", "symbol": "", "removedTicker": "B"},
        ]
    )
    # Before both changes B was NOT in the index.
    assert reconstruct_membership(today, changes, pd.Timestamp("2020-01-01")) == {"A"}
    # Between the two changes B WAS in the index.
    assert reconstruct_membership(today, changes, pd.Timestamp("2021-06-01")) == {"A", "B"}


def test_membership_panel_shape_and_content():
    today = {"A", "B"}
    changes = parse_changes([{"date": "2021-06-01", "symbol": "B", "removedTicker": "X"}])
    dates = [pd.Timestamp("2021-01-31"), pd.Timestamp("2021-12-31")]

    panel = membership_panel(today, changes, dates)

    assert set(panel.columns) == {"date", "symbol"}
    jan = set(panel.loc[panel["date"] == "2021-01-31", "symbol"])
    dec = set(panel.loc[panel["date"] == "2021-12-31", "symbol"])
    assert jan == {"A", "X"}
    assert dec == {"A", "B"}
