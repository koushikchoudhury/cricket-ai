from cricket_ai.faithfulness import is_faithful, numbers, unsupported_numbers


def test_numbers():
    assert numbers("Scored 1,234 runs at 152.3, 4th time, 2016") == ["1234", "152.3", "4", "2016"]
    assert numbers("V Kohli") == []


def test_rounding_is_allowed_but_invention_is_not():
    evidence = "Win probability for the chasing side: 83.9%. Needs 10 off 12 balls."
    assert is_faithful("About 84% with 10 needed off 12 balls.", evidence)
    assert unsupported_numbers("Needs 10 off 102 balls.", evidence) == ["102"]


def test_question_counts_as_evidence():
    q = "How many runs did YK Pathan score in the 2016 season?"
    assert is_faithful("In the 2016 season YK Pathan scored 361 runs.", q, "scored 361 runs")
