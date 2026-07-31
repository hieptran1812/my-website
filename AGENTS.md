# Project rules for paper-reading posts

- Treat the linked paper or local PDF as the source of truth. Before editing a paper analysis, extract/read the paper text and verify every equation and reported number against it.
- Preserve LaTeX commands literally in Markdown. Never write bare forms such as `frac`, `sum`, `in`, `mathbb`, `left`, `right`, `mid`, `prod`, `lambda`, or `pi` where a LaTeX command is intended.
- After editing equations, scan for control characters (especially form-feed, backspace, and tab) and malformed math commands, then run the repository's build or blog validation before handoff.
- When a formula is an explanatory abstraction rather than an equation stated by the paper, label it explicitly as such; do not present an inferred objective as the paper's exact formula.
