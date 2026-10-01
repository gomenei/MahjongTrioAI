from analyze_ppo_history import analyze


def history_rows(ranks, points):
    return [
        {
            "update": (index + 1) * 20,
            "total_hands": (index + 1) * 640,
            "evaluation": {
                "average_rank": rank,
                "average_points": point,
                "first_rate": 0.33,
                "third_rate": 0.33,
            },
        }
        for index, (rank, point) in enumerate(zip(ranks, points))
    ]


def test_analyze_detects_consistent_improvement():
    result = analyze(
        history_rows(
            [2.10, 2.08, 2.06, 2.04, 2.02, 2.00],
            [-1000, -700, -400, -100, 200, 500],
        )
    )
    assert result["verdict"] == "strong_internal_support"
    assert result["rank_trend"]["slope_per_100_updates"] < 0
    assert result["point_trend"]["slope_per_100_updates"] > 0


def test_analyze_rejects_flat_or_worsening_curve():
    result = analyze(
        history_rows(
            [2.00, 2.01, 2.00, 2.02, 2.01, 2.03],
            [100, 50, 80, 20, 30, -10],
        )
    )
    assert result["verdict"] == "not_supported_or_plateau"
