"""Compare recorded evidence; missing values never establish compatibility."""


def compare_identity_evidence(left, right):
    missing, different = [], []
    for key in sorted(left.keys() | right.keys()):
        a, b = left.get(key), right.get(key)
        if a is None or a == "" or b is None or b == "":
            missing.append(key)
        elif a != b:
            different.append(key)
    reasons = []
    if different:
        reasons.append("mismatch:" + ",".join(different))
    if missing:
        reasons.append("insufficient_evidence:" + ",".join(missing))
    return (
        "mismatch" if different else "insufficient_evidence" if missing else "passed",
        "; ".join(reasons) or None,
    )
