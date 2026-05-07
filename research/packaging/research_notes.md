# Toolkit Packaging — Research Dossier (Item 9)

For Item 9: package the spatial-confounding-audit pipeline as `pip install spatial-confounding-audit`.

---

## 1. Modern Python Research Package Layout (2025)

### Standard for new ML/stats research packages
- **src layout** (not flat). Recommended by PyPA, adopted by scikit-learn, statsmodels, JAX. Ensures tests run against *installed* package. https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/
- **pyproject.toml as single source of truth** (PEP 621, PEP 660). Drop setup.py/setup.cfg.
- **Build backend: Hatchling.** https://hatch.pypa.io/. Lightest-weight, PEP-517 compliant. Poetry remains popular but its dependency resolver and non-standard lockfile are increasingly seen as friction; many 2024–2025 scientific packages have migrated off Poetry.
- **uv as installer/resolver/env manager.** https://github.com/astral-sh/uv. **De-facto standard for fast resolution and reproducible environments** in 2025. Used by HuggingFace `datasets`, `lerobot`, scientific-python cookie template. Generates universal `uv.lock`.
- **Testing: pytest + hypothesis.** Coverage via `coverage.py` reported through `pytest-cov`.
- **Linting/formatting: ruff + ruff-format** (replaces flake8/black/isort). https://docs.astral.sh/ruff/.
- **Type checking: mypy --strict** for public API; researchers frequently scope mypy to top-level only.

### Reference template
**scientific-python/cookie** (https://github.com/scientific-python/cookie) — most current and widely vetted scaffold (cookiecutter for scientific-python packages, maintained by Henry Schreiner et al.). Bakes in src layout, hatchling, ruff, pytest, mypy, pre-commit, GH Actions, Sphinx.

---

## 2. Documentation Tooling

### Survey of comparable packages
- **scikit-learn:** Sphinx with sphinx-gallery for executable examples
- **statsmodels:** Sphinx + nbsphinx
- **EconML:** Sphinx with notebook gallery
- **CausalML (Uber):** Sphinx + nbsphinx
- **DoWhy:** Recently migrated to **Sphinx + PyData theme** (formerly Sphinx + RTD theme)
- **DoubleML:** Sphinx for Python, pkgdown for R

### Dominant pattern in causal-inference research packages
**Sphinx + sphinx-gallery + PyData/Furo theme.** MkDocs Material is excellent for general-purpose libraries (FastAPI, pydantic) but the causal-inference field uses Sphinx because of `sphinx-gallery`'s tight integration with Jupyter-style examples and `autodoc`'s superior handling of NumPy-style docstrings.

### Recommendation
**Sphinx + Furo theme** (cleaner than RTD) + `myst-parser` (Markdown support) + `sphinx-gallery` for executable tutorials + `numpydoc` for API reference. Build on Read the Docs (free, standard).

### Tutorials should include
1. End-to-end example reproducing small slice of paper
2. Standalone NMI computation
3. Standalone frozen-probe diagnostic
4. Confounder escalation walk-through

Each as Jupyter notebook executed by sphinx-gallery on every doc build.

---

## 3. JOSS Submission

JOSS (https://joss.theoj.org/) — accepts research software meeting:
- **Substantial scholarly effort** (> few hundred lines, or clear novel contribution)
- **Open-source license** (OSI-approved)
- **Documentation covering installation, example usage, API, tests**
- **Automated tests with CI**
- **Short paper.md (~250–1000 words)** describing statement of need, summary, citations

### Recent accepted causal-inference packages
- **DoubleML** — Bach et al. 2022, JOSS 7(74):4108. https://doi.org/10.21105/joss.04108
- **Causal Curve** — Kobrosly 2020, JOSS 5(52):2523. https://doi.org/10.21105/joss.02523
- **dowhy** — JOSS-published 2022. https://joss.theoj.org/papers/10.21105/joss.03533
- **CausalML** — Chen et al. 2020, arXiv:2002.11631
- **pgmpy** — Ankan & Panda 2015+

### What gets rejected
- Thin wrappers around single function
- Lack of independent tests
- No statement of need distinguishing from existing tools
- Authors unresponsive to reviewer feedback

**Our toolkit clears the bar provided:**
- Document distinct contribution vs DoWhy/EconML — spatial-confounding audit + frozen-probe diagnostic + escalation test is novel
- Ship tests with >70% coverage
- Include JOSS paper.md

### Strategy
**Submit JOSS in parallel with JBES paper revision.** JOSS gives citable software DOI strengthening JBES "reproducibility" claim; JOSS review process catches packaging issues before JBES reviewers see code.

---

## 4. Reproducibility Artifact

JBES doesn't have ACM-style artifact badging, but reviewers increasingly expect AEA-style replication packages.

### Standards
- **AEA Data Editor.** https://aeadataeditor.github.io/. Required: README documenting all data sources, code to reproduce every figure/table from raw data, computational environment specification, expected runtime.
- **ACM Artifact Review and Badging v1.1.** https://www.acm.org/publications/policies/artifact-review-and-badging-current. Three badges: Available, Functional, Reproducible. Useful template.

### Recommended environment specification
- **`uv.lock`** for Python deps + **`Dockerfile`** pinning OS and CUDA version
- Conda-lock acceptable but uv has become 2025 standard and is faster
- Reproducible Research notebooks: `notebooks/` with one notebook per main-paper table/figure, executed end-to-end with cached intermediate artifacts (`dvc` or plain pickle)
- **Zenodo deposit** for release tag → citable DOI for exact code version reviewed

### Concrete artifact bundle for JBES
1. Source repo on GitHub with tagged release
2. `uv.lock` + `pyproject.toml` for exact deps
3. `Dockerfile` + `docker-compose.yml`
4. `make reproduce` target that downloads data, runs analysis, regenerates every paper figure into `outputs/`
5. Zenodo DOI for release
6. README with hardware specs, total runtime, data licensing

---

## 5. Continuous Integration

scientific-python/cookie template provides current GitHub Actions workflow.

### Recommended jobs
- **`test.yml`** — On push/PR: matrix over Python 3.10/3.11/3.12 and ubuntu/macos/windows. pytest + coverage. Upload to Codecov.
- **`docs.yml`** — On push to main and on tag: build Sphinx, deploy to GitHub Pages or RTD.
- **`publish.yml`** — On release tag: build wheels via `cibuildwheel` if needed (our package is pure Python so just `python -m build`), publish to PyPI via OIDC trusted publishing (no API tokens). https://docs.pypi.org/trusted-publishers/
- **`pre-commit.yml`** — Run pre-commit on PRs (ruff, mypy, codespell)
- **`reproduce.yml`** — Optional but high-impact: weekly cron job that runs full reproduction pipeline on tiny slice of data, fails if outputs change. **Strong reviewer signal.**

Sample skeleton: https://github.com/scientific-python/cookie/tree/main/%7B%7Bcookiecutter.project_name%7D%7D/.github/workflows

---

## 6. Naming and Discoverability

### Three candidates evaluated
- **`spatial-confounding-audit`** — Descriptive, exact-match SEO for paper title, but long. PyPI namespace clean as of 2026-04. ✓
- **`sca-toolkit`** — Short but ambiguous (sca = "side-channel attack" in security; "supply-chain audit" elsewhere). **Avoid.**
- **`text-spatial-causal`** — Descriptive but undersells audit/diagnostic angle (the novel contribution).

### Recommendation
**`spatial-confounding-audit`** as PyPI distribution name with **`import spatialaudit`** as import name (short, memorable, no hyphens). Pattern established by:
- `scikit-learn` → `sklearn`
- `python-dateutil` → `dateutil`
- `beautifulsoup4` → `bs4`

### Discoverability boosters
1. Include keywords `causal-inference`, `confounding`, `domain-adaptation`, `nlp`, `spatial-statistics`, `econometrics` in `pyproject.toml`
2. Add `pywhy`, `econml`, `causalml` topic tags on GitHub
3. Cross-reference from DoWhy's "related projects" page (PR friendly)
4. `awesome-causal-inference` PR

---

## Proposed Repo Skeleton

```
spatial-confounding-audit/
  pyproject.toml                  # hatchling backend, PEP 621 metadata
  uv.lock                         # universal lock
  README.md                       # quickstart + paper citation + JOSS DOI
  LICENSE                         # MIT or BSD-3
  CITATION.cff                    # for GitHub citation widget
  CHANGELOG.md
  .pre-commit-config.yaml         # ruff, mypy, codespell
  Dockerfile
  docker-compose.yml
  Makefile                        # `make test`, `make docs`, `make reproduce`

  src/spatialaudit/
    __init__.py                   # __version__, top-level re-exports
    nmi.py                        # NMI computation
    location_probe.py             # Location-classification probe
    backdoor_dr.py                # Doubly-robust backdoor estimator
    dml.py                        # Double machine learning wrapper
    adversarial.py                # Gradient-reversal deconfounding
    frozen_probe.py               # Frozen-encoder probe diagnostic (headline)
    escalation.py                 # Confounder escalation test
    diagnostics.py                # Unified Audit() class
    datasets/                     # Toy datasets for tutorials
    _internal/                    # Private utilities

  tests/
    test_nmi.py
    test_location_probe.py
    test_backdoor_dr.py
    test_dml.py
    test_adversarial.py
    test_frozen_probe.py          # Includes synthetic Gaussian example from Prop 1
    test_escalation.py
    test_integration.py           # End-to-end on toy dataset
    fixtures/

  docs/
    conf.py                       # Sphinx + Furo + myst + sphinx-gallery
    index.md
    installation.md
    quickstart.md
    api/                          # autodoc-generated
    tutorials/                    # sphinx-gallery executable notebooks
      01_nmi_basics.py
      02_frozen_probe_diagnostic.py
      03_full_audit_pipeline.py
      04_reproducing_paper_table_3.py
    theory/
      frozen_probe_theorem.md     # Proposition 2 in plain prose
      backdoor_assumptions.md
    references.bib

  paper/                          # JOSS submission
    paper.md
    paper.bib

  notebooks/                      # Reproducibility (mirror of paper)
    figure1_nmi_heatmap.ipynb
    figure2_frozen_probe.ipynb
    table3_dr_estimates.ipynb

  scripts/
    download_data.py
    reproduce_all.py

  .github/workflows/
    test.yml
    docs.yml
    publish.yml
    pre-commit.yml
    reproduce.yml
```

### Top-level API target (single-import experience)

```python
from spatialaudit import Audit
audit = Audit(text=texts, location=locs, treatment=t, outcome=y)
audit.nmi()                       # returns NMI table
audit.location_probe()            # returns classifier accuracy
audit.adversarial_deconfound()    # returns deconfounded encoder
audit.frozen_probe()              # returns recovered-confounder accuracy + p-value
audit.escalation()                # returns probe-capacity ladder
audit.report(out="audit.html")    # one-page HTML diagnostic
```

**The `Audit` facade is the single most important UX decision:** a JBES reviewer running `pip install spatial-confounding-audit` and pasting six lines from README must reproduce headline diagnostic in under five minutes on CPU laptop. Everything else serves that goal.
