#!/usr/bin/env python
"""Build result figures used by the STREAM manuscript."""

from __future__ import annotations

import argparse
from pathlib import Path

from build_project_slide_deck import (
    ROOT,
    collect_legacy_panel_summary,
    collect_mouse_summary,
    collect_zebrafish_summary,
    save_legacy_heatmap,
    save_mouse_full_plot,
    save_zebrafish_plot,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="docs/figures")
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mouse_summary = collect_mouse_summary()
    legacy_summary = collect_legacy_panel_summary(mouse_summary)
    zfish_summary = collect_zebrafish_summary()

    save_mouse_full_plot(mouse_summary, out_dir / "mouse_full_panel_heldout.png")
    save_legacy_heatmap(legacy_summary, out_dir / "mouse_legacy_panel_heldout.png")
    save_zebrafish_plot(zfish_summary, out_dir / "zebrafish_transfer_learned.png")

    mouse_summary.to_csv(out_dir / "mouse_heldout_summary.csv", index=False)
    legacy_summary.to_csv(out_dir / "mouse_legacy_panel_summary.csv", index=False)
    zfish_summary.to_csv(out_dir / "zebrafish_transfer_learned_summary.csv", index=False)
    print(f"Wrote manuscript figures to {out_dir}")


if __name__ == "__main__":
    main()
