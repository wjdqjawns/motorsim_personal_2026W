# Introduction to Control Theory

```
Book ID: CTS-001
Title: Introduction to Control Theory
Author: Beomjun Chung
Updated: 2026.07
```

## About

This book is an academic overview of control theory — not an undergraduate "automatic control" course companion, but a map of the discipline itself. It is the foundation volume of the Control Theory Series: every later volume expands one region of the territory this book surveys.

The book is organized around four pillars:

1. **Modeling** — how we represent dynamic systems
2. **Analysis** — how we understand their behavior
3. **Control** — how we shape their behavior by designing inputs
4. **Estimation** — how we infer models and states from data

The part structure of this volume mirrors the structure of the series, so the table of contents doubles as a map of the whole project.

## Structure of This Volume

| Part | Title | Role |
|------|-------|------|
| I | Introduction | What control theory is, its history, and the series roadmap |
| II | System Theory for Modeling | Mathematical models; classification and properties of systems |
| III | System Theory for Analysis | Time-domain, frequency-domain, state-space analysis; stability theory |
| IV | Control Theory | Feedback theory, classical controllers, controller design |
| V | Estimation Theory | Model estimation (system ID, calibration); deterministic and stochastic state estimation |
| VI | Advanced Topics | Surveys of nonlinear systems and control, advanced control methods, learning based control, and FDI/FTC — previews of the later volumes |

Scope: this volume is written in continuous time. Discrete-time and hybrid systems are introduced conceptually and developed fully from Volume 2 onward. Lyapunov stability appears at an introductory level; the full treatment belongs to Volume 3.

Part VI chapters follow a uniform survey format (Motivation / Concepts / Representative Methods / Applications / Further Reading) rather than full technical development.

## Repository Layout

```
introduction_to_control_theory/
├── main.tex          # document root: cover, revision history, abstract, TOC,
│                     # parts/chapters, bibliography, appendix
├── chapters/         # one .tex file per chapter
├── figures/
├── references.bib
└── README.md
```

Adjust to the shared report template (cover → revision history → abstract → TOC → parts → bibliography → appendix).

## Build

```
latexmk -pdf
```

## Status

- [x] Table of contents finalized (6 parts, 15 chapters)
- [ ] Part I — Introduction (written last)
- [ ] Part II — System Theory for Modeling
- [ ] Part III — System Theory for Analysis
- [ ] Part IV — Control Theory
- [ ] Part V — Estimation Theory
- [ ] Part VI — Advanced Topics