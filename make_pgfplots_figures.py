"""Generate Section 5 figures from batch summaries, with no fitted curve."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from artifact_io import ROOT, output_directory
from analyze_results import PARAMETERS, SWEEPS, wilson_interval
from sparse_lwe import theorem_predictor

COLORS = dict(m="155EEF", n="D92D20", k="0E9384", q="F79009", sigma="7A5AF8")


def normalized_rows(rows: list[dict]) -> list[dict]:
    """Derive all plotted values from parameters and counts, never legacy Gamma."""
    result = []
    seen = set()
    for row in rows:
        sweep, point = row["sweep"], int(row["point_index"])
        if sweep not in SWEEPS or (sweep, point) in seen:
            raise ValueError("invalid or duplicate batch")
        seen.add((sweep, point))
        parameters = {k: float(row[k]) if k == "sigma" else int(row[k]) for k in PARAMETERS}
        count, successes = int(row["trials"]), int(row["successes"])
        low, high = wilson_interval(successes, count)
        result.append(dict(sweep=sweep, point_index=point, value=parameters[sweep],
                           **parameters, trials=count, successes=successes,
                           success_rate=successes / count, wilson95_low=low,
                           wilson95_high=high, **theorem_predictor(**parameters)))
    if {r["sweep"] for r in result} != set(SWEEPS):
        raise ValueError("the five-panel figure requires all five sweeps")
    return result


def plot_tables(rows: list[dict]) -> tuple[dict[str, list], dict[str, list]]:
    first, second = {}, {}
    for sweep in SWEEPS:
        group = [r for r in rows if r["sweep"] == sweep]
        first[sweep] = [(r["value"], r["success_rate"],
                         max(0., r["wilson95_high"] - r["success_rate"]),
                         max(0., r["success_rate"] - r["wilson95_low"]))
                        for r in sorted(group, key=lambda r: r["value"])]
        second[sweep] = [(r["gamma"], r["success_rate"])
                         for r in sorted(group, key=lambda r: r["gamma"])]
    return first, second


def build(rows: list[dict]) -> str:
    rows = normalized_rows(rows)
    first, second = plot_tables(rows)
    lines = ["% Generated from counts and current Section 5 Gamma; do not edit by hand."]
    for sweep, color in COLORS.items():
        lines.append(r"\definecolor{sweep" + sweep + "}{HTML}{" + color + "}")
    lines.extend([r"\pgfplotsset{sweepaxis/.style={width=\linewidth,height=0.58\linewidth,",
                  r"ymin=0,ymax=1.05,grid=major,grid style={draw=gray!25},",
                  r"tick label style={font=\small},label style={font=\small}}}",
                  r"\begin{figure}[H]", r"\centering"])
    for index, sweep in enumerate(SWEEPS):
        label = r"\sigma" if sweep == "sigma" else sweep
        ylabel = ",ylabel={success rate}" if index % 2 == 0 else ""
        lines.extend([r"\begin{subfigure}[b]{0.48\textwidth}", r"\centering",
                      r"\begin{tikzpicture}", r"\begin{axis}[sweepaxis,xlabel={$" + label + "$}" + ylabel + "]",
                      r"\addplot+[color=sweep" + sweep + r",mark=*,thick,error bars/.cd,y dir=both,y explicit]",
                      r"table[x=x,y=y,y error plus=ep,y error minus=em] {", "x y ep em"])
        for x, y, ep, em in first[sweep]:
            lines.append(f"{x:.8g} {y:.8g} {ep:.4f} {em:.4f}")
        lines.extend(["};", r"\end{axis}", r"\end{tikzpicture}",
                      r"\caption{Vary $" + label + r"$.}\label{fig:sweep-" + sweep + "}",
                      r"\end{subfigure}"])
        if index in (0, 2):
            lines.append(r"\hfill")
        elif index in (1, 3):
            lines.extend(["", r"\medskip"])
    counts = sorted({r["trials"] for r in rows})
    description = (f"Each point represents {counts[0]} independent instances; " if len(counts) == 1
                   else "Trial counts are recorded in the summary; ")
    lines.extend([r"\caption{Full-recovery rates in five one-parameter sweeps. " + description
                  + r"error bars are Wilson 95\% confidence intervals.}",
                  r"\label{fig:sweep-all}", r"\end{figure}", "",
                  r"\begin{figure}[H]", r"\centering", r"\begin{tikzpicture}",
                  r"\begin{axis}[width=0.80\textwidth,height=0.42\textwidth,",
                  r"xmode=log,log basis x=10,ymin=0,ymax=1.05,xlabel={$\Gamma$},",
                  r"ylabel={success rate},grid=major,grid style={draw=gray!25},",
                  r"legend columns=3,legend style={font=\small,draw=none,at={(0.5,-0.24)},anchor=north}]"])
    for sweep in SWEEPS:
        label = r"\sigma" if sweep == "sigma" else sweep
        lines.extend([r"\addplot+[color=sweep" + sweep + r",mark=*,thick,only marks] table[x=x,y=y] {", "x y"])
        for x, y in second[sweep]:
            lines.append(f"{x:.5f} {y:.8g}")
        lines.extend(["};", r"\addlegendentry{$" + label + "$}"])
    lines.extend([r"\end{axis}", r"\end{tikzpicture}",
                  r"\caption{Full-recovery rates from all " + str(len(rows))
                  + r" experimental batches plotted against $\Gamma$.}",
                  r"\label{fig:sweep-combined}", r"\end{figure}"])
    return "\n".join(lines) + "\n"


def generate(summary: Path, destination: Path) -> None:
    destination = output_directory(destination)
    with summary.open(newline="", encoding="utf-8") as stream:
        fragment = build(list(csv.DictReader(stream)))
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "figures.tex").write_text(fragment, encoding="utf-8")
    (destination / "preview.tex").write_text(
        "\n".join([r"\documentclass{article}", r"\usepackage[margin=25mm]{geometry}",
                    r"\usepackage{amsmath,subcaption,float,pgfplots}",
                    r"\pgfplotsset{compat=1.18}", r"\begin{document}",
                    r"\input{figures.tex}", r"\end{document}", ""]), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ROOT / "results/paper/summary_results.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/paper")
    args = parser.parse_args()
    generate(args.summary, args.output_dir)
    print(f"Generated figures.tex and preview.tex in {args.output_dir}.")


if __name__ == "__main__":
    main()
