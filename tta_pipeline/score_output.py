# Project-authored portions: GNU GPL version 3 only.
# See LICENSE, LICENSING.md and DEPENDENCY_EXCEPTION.md.
# Third-party portions retain their original notices and licenses.
"""Small, dependency-free score events and terminal formatting."""
import json
import math


EVENT_PREFIX = "TTA_SCORE_EVENT "


def valid_score(row):
    try:
        return (not row.get("error") and row.get("status") == "ok"
                and row.get("strict_eligible") is True
                and all(math.isfinite(float(row[k])) for k in ("lse_d", "lse_c")))
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def emit_score(row, cached=False):
    score = {key: row[key] for key in
             ("clip", "identity", "lse_d", "lse_c", "status", "strict_eligible")}
    print(EVENT_PREFIX + json.dumps(dict(event="score", cached=cached, score=score),
                                   ensure_ascii=False, allow_nan=False), flush=True)


def emit_failure(row):
    print(EVENT_PREFIX + json.dumps(dict(event="failure", **row),
                                   ensure_ascii=False, allow_nan=False), flush=True)


class ScorePrinter:
    def __init__(self, expected=None):
        self.expected = expected
        self.seen = set()
        self.failed = set()

    def score(self, row, cached=False, source=""):
        key = row["clip"], row["identity"]
        if self.expected is not None and key not in self.expected:
            raise ValueError("Unexpected score event: %s / %s" % key)
        if not valid_score(row):
            raise ValueError("Invalid score event: %s / %s" % key)
        if key in self.seen:
            return
        label = "cached" if cached else "scored"
        if source:
            label += ": " + source
        print("[%s] %s / %s  D=%.9f  C=%.9f" %
              (label, key[0], key[1], float(row["lse_d"]), float(row["lse_c"])), flush=True)
        self.seen.add(key)

    def failure(self, row):
        key = row.get("clip", "?"), row.get("identity", "?")
        marker = key + (str(row.get("error", "Evaluation failed")),)
        if marker not in self.failed:
            print("[failed] %s / %s: %s" % marker, flush=True)
            self.failed.add(marker)

    def event(self, event):
        if event.get("event") == "score":
            self.score(event["score"], cached=event.get("cached") is True)
        elif event.get("event") == "failure":
            self.failure(event)


def print_summary(summary, no_average=False, source=""):
    complete = summary["state"] == "complete"
    scope = "full fixed cohort" if summary.get("full_fixed_cohort") else "requested selection"
    print("%s: %d/%d cells (%s%s)." %
          ("Complete" if complete else "Partial", summary["cells"],
           summary["expected_cells"], scope, "; " + source if source else ""), flush=True)
    if not complete:
        print("Missing cells: %d; failed workers: %d." %
              (summary["expected_cells"] - summary["cells"],
               len(summary.get("failures", []))), flush=True)
    if not no_average and summary["cells"]:
        print("%s mean: D=%.9f  C=%.9f (%d/%d cells%s)." %
              ("Final" if complete else "Partial", summary["lse_d"], summary["lse_c"],
               summary["cells"], summary["expected_cells"],
               "; " + source if source else ""), flush=True)
