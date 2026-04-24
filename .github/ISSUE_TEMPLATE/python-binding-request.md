---
name: Python binding request
about: Request a KAIO op be exposed from Python (`kaio-py`)
title: "[py] "
labels: ["area:kaio-py", "user-request"]
---

## Which op do you need from Python?

<!-- e.g. kaio.matmul_int8, kaio.attention_flash, etc. -->

## What are you trying to do?

<!-- One or two sentences. A real use case helps us prioritize. -->

## Which dtypes / shapes?

<!-- e.g. "f16 inputs, (batch, seq=2048, d_model=4096)" or "packed INT4
weights with f16 activations". If you don't know, say so — we'll
pick reasonable defaults. -->

## Platform

- [ ] Windows
- [ ] Linux
- [ ] Both

## Framework context

<!-- Are you coming from PyTorch? NumPy? Pure Python? Triton? Mix? -->

## Anything else

<!-- Links to reference implementations, related papers, model configs
you want to run, etc. -->

---

**Why this template exists.** The `kaio-py` crate ships as a Sprint
8.1 scaffold — one kernel (`matmul_tc`) exposed end-to-end to prove
the PyO3 path works. Broader op coverage is intentionally
user-demand-gated rather than scheduled: with a solo maintainer and
a fast-moving Rust core, scheduled Python build-out would slow the
Rust side without a matching return. Filing this template tells us
there's a concrete use case, unlocks a narrow sprint to expose the
op you need, and is the trigger for PyPI publish if it hasn't
happened yet. See
[Phase 8 master plan §4 "Scope decision"](../../docs/development/sprints/phase8/phase8_master_plan.md)
for the full rationale.
