from spot_check.plan import infer_csv_plan_tag, rt_plan_label_from_csv_stem


def test_infer_csv_plan_tag() -> None:
    stem = "15186535_T0G40_ic256-45-9018-data acquisition-2026-05-06-16-19-12"
    assert infer_csv_plan_tag(stem) == "T0G40"
    assert rt_plan_label_from_csv_stem(stem) == "T0G40"
