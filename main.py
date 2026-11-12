#!/usr/bin/env python3
"""roblox-up-and-coming-forecast-01 entry: kernel forecaster.

The chart is a slow random walk with no exploitable momentum: playing
growth barely predicts rank change (corr ~0.02 on 3,021 read pairs) and
extrapolating rank velocity makes point forecasts WORSE at every horizon.
What is predictable is the DISTRIBUTION: how often a game that closes at
rank r occupies each place over the next 72 hours. This program estimates
that transition kernel from the history itself (every cut point, every
forward read within 72h, pooled by rank band) and forecasts with it, so
each place's probability mass follows the measured survival of the chart
(top-10 overlap falls 8.7 -> 7.2 -> 5.6 -> 4.0 over ~6/19/37/62h).
The mass left unnamed covers games not yet on the closing chart, which no
entry can name. Deterministic: no randomness, no clock.
"""
import json
import math
import os
from datetime import datetime, timedelta
from pathlib import Path

# Rank bands for kernel pooling (closing rank -> band index).
BAND_EDGES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 17, 22, 28, 35, 45, 60, 80, 10**9]
WINDOW_HOURS = 72.0
MAX_CUTS = 40
MAX_FWD_PER_CUT = 12


def band_of(rank):
    for i in range(len(BAND_EDGES) - 1):
        if BAND_EDGES[i] <= rank < BAND_EDGES[i + 1]:
            return i
    return len(BAND_EDGES) - 2


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_history(input_dir):
    reads = []
    with open(Path(input_dir) / "history.ndjson", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                reads.append(json.loads(line))
    reads.sort(key=lambda r: r["at"])
    return reads


def estimate_kernel(reads, places):
    """kernel[place][band] = share of forward reads where the game at that
    place closed at a rank in that band. new_share[place] = share where the
    game was not on the closing chart at all."""
    n_bands = len(BAND_EDGES) - 1
    counts = [[0.0] * n_bands for _ in range(places)]
    new_counts = [0.0] * places
    totals = [0.0] * places
    window = timedelta(hours=WINDOW_HOURS)
    # Cut points spaced >= 6h apart, each with forward reads inside 72h.
    cut_idx = []
    last_cut_time = None
    for i, r in enumerate(reads):
        t = parse_time(r["at"])
        if last_cut_time is not None and (t - last_cut_time).total_seconds() < 6 * 3600:
            continue
        fwd = [j for j in range(i + 1, len(reads))
               if timedelta(0) < parse_time(reads[j]["at"]) - t <= window]
        if len(fwd) >= 2:
            cut_idx.append(i)
            last_cut_time = t
    cut_idx = cut_idx[-MAX_CUTS:]
    for i in cut_idx:
        closing = {g["universe_id"]: g["rank"] for g in reads[i]["up_and_coming"]}
        fwd = [j for j in range(i + 1, len(reads))
               if timedelta(0) < parse_time(reads[j]["at"]) - parse_time(reads[i]["at"]) <= window]
        if len(fwd) > MAX_FWD_PER_CUT:
            step = len(fwd) / MAX_FWD_PER_CUT
            fwd = [fwd[int(k * step)] for k in range(MAX_FWD_PER_CUT)]
        for j in fwd:
            chart = sorted(reads[j]["up_and_coming"], key=lambda g: g["rank"])
            for k in range(min(places, len(chart))):
                uid = chart[k]["universe_id"]
                totals[k] += 1.0
                if uid in closing:
                    counts[k][band_of(closing[uid])] += 1.0
                else:
                    new_counts[k] += 1.0
    kernel = []
    new_share = []
    for k in range(places):
        tot = totals[k] or 1.0
        kernel.append([c / tot for c in counts[k]])
        new_share.append(new_counts[k] / tot)
    return kernel, new_share


def build_forecast(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    task = json.loads((input_dir / "task.json").read_text(encoding="utf-8"))
    places = int(task["places"])
    reads = load_history(input_dir)
    last = sorted(reads[-1]["up_and_coming"], key=lambda g: g["rank"])
    closing = {g["universe_id"]: g["rank"] for g in last}
    kernel, new_share = estimate_kernel(reads, places)

    probabilities = []
    expected_place = {}
    for k in range(places):
        # Unnamed mass = measured share of genuinely new games at this place.
        unnamed = min(0.30, max(0.05, new_share[k]))
        named_mass = 1.0 - unnamed
        # Raw mass per game: the band's kernel share SPLIT EQUALLY among the
        # closing chart's games in that band (a band's value is the chance
        # the place is held by SOME game of the band, not by each of them).
        band_size = {}
        for r in closing.values():
            band_size[band_of(r)] = band_size.get(band_of(r), 0) + 1
        raw = {}
        for uid, r in closing.items():
            p = kernel[k][band_of(r)] / band_size[band_of(r)]
            if p > 0:
                raw[uid] = p
        total_raw = sum(raw.values()) or 1.0
        rows = []
        for uid, p in raw.items():
            q = p / total_raw * named_mass
            if q >= 0.003:
                rows.append((uid, min(q, 0.45)))
        # Renormalise after the cap and the floor so the place sums to named_mass.
        s = sum(q for _, q in rows)
        if s > 0:
            rows = [(uid, q / s * named_mass) for uid, q in rows]
        for uid, q in rows:
            probabilities.append({"universe_id": uid, "place": k + 1, "p": round(q, 6)})
            expected_place[uid] = expected_place.get(uid, 0.0) + q * (k + 1)
    # top10: highest total top-10 probability, ordered by expected place.
    totals = {}
    for row in probabilities:
        totals[row["universe_id"]] = totals.get(row["universe_id"], 0.0) + row["p"]
    ranked = sorted(totals, key=lambda u: (-totals[u], expected_place.get(u, 99), closing.get(u, 999)))
    top10 = ranked[:places]
    # Order the top10 by expected place, then closing rank.
    top10.sort(key=lambda u: (expected_place.get(u, 99) / max(totals[u], 1e-9), closing.get(u, 999)))
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / task["output"], "w", encoding="utf-8") as f:
        f.write(json.dumps({"probabilities": probabilities, "top10": top10}) + "\n")
    return {"places": places, "rows": len(probabilities), "top10": top10}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    info = build_forecast(input_dir, output_dir)
    print(f"roblox: {info['rows']} probability rows, top10 {info['top10']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
