---
name: research-documentation
description: >
  Unified documentation lookup and academic research skill for quant trading
  workflows. Covers API/library documentation retrieval, Context7 integration
  for versioned docs, and arXiv paper search for market microstructure,
  forecasting, and strategy research.
version: 1.0.0
metadata:
  consolidates:
    - documentation-lookup-1
    - context7-documentation-lookup-2
    - arxiv-research-search
---

# Research & Documentation Lookup

Unified skill for finding, citing, and applying information from library
documentation, versioned API references, and academic research papers.
Tailored for a quantitative trading research system where correct API usage,
up-to-date library behaviour, and awareness of recent academic work are all
critical.

---

## 1. When to Use

Activate this skill when the task involves any of the following:

- Looking up the correct API signature, parameters, or return types for a
  library used in the trading stack (pandas, NumPy, scikit-learn,
  NautilusTrader, etc.).
- Producing a small, correct, self-contained code example that demonstrates
  library usage.
- Retrieving version-specific documentation via Context7 to ensure
  compatibility with pinned dependency versions.
- Searching arXiv for papers on market microstructure, alpha signals,
  portfolio optimisation, execution algorithms, risk modelling, or ML methods
  relevant to trading.
- Triaging and summarising academic papers to decide whether they are worth
  a full read.

---

## 2. Inputs to Gather

| Input | Details |
|-------|---------|
| **Library / API** | Name, version, specific module or function of interest. |
| **Query** | What you need to know -- behaviour, edge cases, migration notes. |
| **Context** | How the result will be used (backtest, live engine, notebook). |
| **arXiv topic** | Keywords, categories (q-fin.TR, cs.LG, stat.ML), date range. |
| **Constraints** | Must match a pinned version, must work offline, etc. |

---

## 3. API & Library Documentation Lookup

### 3.1 Workflow

1. **Identify** the library, version, and specific symbol (function, class, constant).
2. **Retrieve** the official documentation page or docstring.
3. **Extract** the signature, parameter descriptions, return type, and any caveats.
4. **Produce** a minimal, runnable code snippet that demonstrates correct usage.
5. **Cite** the documentation URL and version so the user can verify.

### 3.2 Best Practices

- Always confirm the **version** of the library installed in the project
  (`pip show <pkg>` or check `pyproject.toml` / `requirements.txt`).
- Prefer official docs over blog posts or Stack Overflow.
- When behaviour differs between versions, note the difference explicitly.
- For deprecated APIs, show the replacement and migration path.

### 3.3 Common Libraries in This Stack

| Library | Typical Use |
|---------|-------------|
| `pandas` | DataFrames, time-series indexing, resampling. |
| `numpy` | Vectorised math, array operations. |
| `scikit-learn` | Preprocessing, feature selection, model evaluation. |
| `nautilus_trader` | Backtest engine, order management, strategy base classes. |
| `polars` | High-performance DataFrames for large tick datasets. |
| `xgboost` / `lightgbm` / `catboost` | Gradient-boosted trees. |
| `optuna` | Hyperparameter optimisation. |
| `shap` | Feature importance / model explainability. |
| `plotly` / `matplotlib` | Visualisation. |

---

## 4. Context7 Integration

Context7 provides versioned, structured documentation lookups that are more
reliable than general web search for pinned-dependency projects.

### 4.1 Workflow

1. **Resolve library ID**: Use the Context7 `resolve-library-id` tool with the
   library name and a description of what you need. This returns a
   Context7-compatible library ID (format: `/org/project`).
2. **Query documentation**: Use the Context7 `query-docs` tool with the
   resolved library ID and a specific question. Be detailed -- e.g.,
   "How to set up purged k-fold cross-validation in scikit-learn" rather than
   just "cross validation".
3. **Apply**: Integrate the retrieved code snippets and explanations into the
   current task, verifying that they match the pinned version.

### 4.2 Tips

- Select libraries based on **name match**, **source reputation** (High/Medium),
  **snippet coverage**, and **benchmark score**.
- If a specific version is pinned, include the version in the library ID
  (`/org/project/version`) for version-accurate results.
- Limit to 3 Context7 calls per question to avoid excessive API usage.

---

## 5. Academic Paper Search (arXiv)

### 5.1 Workflow

1. **Define** the research question: what phenomenon, method, or market
   behaviour are you investigating?
2. **Construct** an arXiv search query using relevant keywords and category
   filters:
   - `q-fin.TR` -- Trading and market microstructure.
   - `q-fin.PM` -- Portfolio management.
   - `q-fin.RM` -- Risk management.
   - `q-fin.ST` -- Statistical finance.
   - `cs.LG` -- Machine learning.
   - `stat.ML` -- Statistics and machine learning.
3. **Retrieve** candidate papers (title, authors, abstract, date).
4. **Triage** each paper:
   - Is the method applicable to the current trading context?
   - Is the data regime comparable (equities vs. crypto, HFT vs. daily)?
   - Are the results reproducible with available data?
5. **Summarise** the top 3-5 papers with a one-paragraph assessment each.
6. **Cite** with full arXiv IDs (e.g., `arXiv:2301.12345`) for traceability.

### 5.2 Search Tips

- Combine domain terms with method terms:
  `"market microstructure" AND "transformer"` or
  `"pairs trading" AND "reinforcement learning"`.
- Filter by date range to focus on recent work (last 2-3 years).
- Check the "References" section of a relevant paper to find related work
  that the search may have missed.

### 5.3 Triage Template

For each candidate paper, produce:

```text
Title:    <paper title>
arXiv ID: <arXiv:YYMM.NNNNN>
Authors:  <first author et al.>
Date:     <submission date>
Relevance: [High / Medium / Low]
Summary:  <2-3 sentences on method and key finding>
Applicability: <How it maps to our trading system; data requirements>
Action:   [Read in full / Skim methods section / Skip]
```

---

## 6. Outputs

For any documentation or research task, deliver:

- **Concrete answer**: the API signature, code snippet, or paper summary.
- **Source citation**: URL, version, or arXiv ID.
- **Integration sketch**: how to use the finding in the current codebase
  (files to change, imports to add, tests to run).
- **Risk notes**: version incompatibilities, deprecated APIs, unreproducible
  paper results, or known bugs.

---

## 7. Workflow Summary

1. **Restate** the information need in precise terms.
2. **Choose channel**: library docs, Context7 versioned lookup, or arXiv search.
3. **Retrieve** the information using the appropriate tool.
4. **Verify** against the project's pinned versions and constraints.
5. **Deliver** a cited, actionable answer with a code snippet or summary.
6. **Log** the lookup in the research journal if it informs a strategy decision.
