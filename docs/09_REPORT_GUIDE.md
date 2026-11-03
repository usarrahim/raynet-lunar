# 09 · Final report and presentation

Write in simple, clear English (A2/B1 level, short sentences), in the owner's voice, the same
style as the HW1 proposal. Claude Code may draft sections, but every number must come from
`results/` and every claim about the method must match the code.

## Report outline (about 8-10 pages)
1. **Abstract** (150-200 words): problem, idea, main numbers (val), main finding, demo link.
2. **Introduction:** why lunar coverage maps matter; ray tracing is slow; RadioUNet is fast
   but weak in shadows; our question and hypotheses H1-H3.
3. **Related work:** radio map networks on Earth (RadioUNet, PMNet, physics-aware models),
   lunar work (RadioLunaDiff, the dataset), polar transforms. Reuse the HW1 literature table.
4. **Data:** dataset, splits, masks, our dev split, the diffraction-disabled fact, licence.
5. **Method:** physics prior (formulas), polar view and horizon scan, RayNet architecture
   (one simple figure), training (budget table), ONNX export for the demo.
6. **Experimental setup:** E0-E6 table, equal budget rule, seeds, metrics, regions, statistics.
7. **Results:** main table (val), training curves, example maps.
8. **Analysis:** H1 (numbers + p-values), H2 (region shares), H3, error by distance, coverage
   accuracy, speed. What surprised us.
9. **Demo:** screenshot + link, how it works in the browser.
10. **Discussion and limits:** synthetic terrain, no diffraction in ground truth, single
    antenna type, limited GPU budget, results only on val (test is hidden).
11. **Conclusion and next steps:** real lunar DTMs, more transmitters, optimisation of TX positions.
12. **AI-use statement** (see below). 13. **References.**

## AI-use statement (suggested text, edit in your own words)
> I used Claude (Anthropic) as an assistant in two ways. In the proposal stage it helped me
> search and summarize the literature and make my plan clear. In the implementation stage I
> used Claude Code as a coding assistant: it wrote code following my task list and design, and
> I reviewed, ran and tested it. The experiment design, the decisions, the training runs and
> the interpretation of the results are mine. All numbers come from my own runs, and the code is
> public on GitHub.

## Slides (10-12 slides, 10 minutes)
1. Title + one demo screenshot. 2. Problem (one picture: terrain + coverage). 3. Why it is hard
(shadows, voids, speed). 4. Baseline and its errors by region (from P1.6). 5. Idea: three parts.
6. Horizon scan picture (`docs/img/physics_features.png` style). 7. RayNet diagram (simple).
8. Experiments table. 9. Results: main table. 10. Where the gain comes from (region bars).
11. Live demo (keep a 30-second screen recording as a backup). 12. Limits, next steps, thanks.

## Questions to be ready for
- Why polar coordinates? What happens near the transmitter?
- Why is the knife-edge term only a feature here? (Diffraction is disabled in the data.)
- How do you know the gain is not just more parameters? (E5b.)
- Why not use the frozen validation split for early stopping? (Dev split, last checkpoint.)
- How fast is it, and on what hardware?
- What part is your own contribution?
