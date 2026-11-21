# Agent instructions

## Devlog (required)

Keep `docs/devlog.md` up to date. After every change to the study design, the
data pipeline, the paper's claims or numbers, or any decision the authors make,
add an entry before finishing the task:

- newest entry first, under a dated heading (YYYY-MM-DD);
- what changed, why, and which files were touched;
- if a decision reverses an earlier one, say which entry it replaces.

A task is not finished until the devlog reflects it.

## Paper

- `paper/main.tex` is the conference paper. Every number, table and figure comes
  from `paper/scripts/`; never type a result by hand, add it to a script instead.
- Rebuild with `cd paper && latexmk -pdf main.tex` and check `main.log` for
  undefined references and overfull boxes.
- Titles use one plain phrase, never "X: Y".
- The project is a series of conference papers, not a journal.

## Frozen layers

Layers listed as frozen in the devlog are not changed without the authors'
explicit decision, recorded in the devlog.
