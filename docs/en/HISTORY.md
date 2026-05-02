# Project Trajectory

This repository contains a consolidated version of a longer exploratory work. For context on how NCT reached its current form:

---

## Personal Origin

This work emerged from a symbolic framework called **NCT (Números Cuánticos Tridimensionales / 3D Quantum Numbers)** that I developed **autodidactically** over several months, **without prior formal mathematical training**.

The original intuition: work with ordered tuples over a discrete alphabet of four symbols `{+, -, 0, ~}` and binary operations between them to represent geometric states.

I arrived at this representation through **direct experimentation**, not from mathematical literature. This means many ideas in NCT have known counterparts in formal disciplines (see equivalences table in README).

**Recognizing these equivalences doesn't invalidate my path of discovery, but contextualizes it:** what is validated here is not "new mathematics", but a concrete discrete representation technique whose specific form came from NCT.

---

## Experimental Validation Process

Over months of systematic experiments, I tested each component:

| Component | Hypothesis | Result |
|-----------|------------|--------|
| Operations ⊕, ⊗ | Would provide discriminative signal | ❌ Did not exceed simple baselines |
| Quantization {+, -, 0, ~} | Would capture geometric structure | ✅ Showed consistent signal |
| Weight table | Learn state→rupture associations | ✅ Generalized in cross-validation |
| Physical unification | Equivalence with theoretical models | ❌ Not validable with this pipeline |
| AGI | Base for symbolic reasoning | ❌ No experimental evidence |
| "Phase 3-6-9" and metaphors | Decorative elements | ❌ No predictive value |

---

## What Was Discarded vs. What Survived

### Discarded (Pruning)

- **Binary operations ⊕, ⊗** between states as detection engine
- **Physical unification applications** (not validable with RGB-D benchmark)
- **AGI applications** (no experimental basis)
- **Decorative layers** like "phase 3-6-9"

### Survived Validation

| Element | Justification |
|---------|---------------|
| **Discretization into 4 states {+, -, 0, ~}** | Captures essential geometric information |
| **Weight table by 3D motif** | Enables adaptation to real data |
| **State `~` as transition marker** | Useful for identifying ambiguous zones |
| **Triangular ambiguity gate over classical delta** | Don't correct where delta already works well |

Keeping the name "NCT" in this report is a personal decision: it is the internal project brand since its inception. Equivalences with standard techniques are openly documented in the README.

**Known equivalences:**
- NCT descriptors are variants of local curvature descriptors
- The quantization is similar to LBP (Local Binary Patterns) in 3D
- The ambiguity gate is a form of attention mechanism

---

## Version Chronology

| Version | Focus | Status |
|---------|-------|--------|
| v1-v11 | Theoretical exploration and binary operations | ❌ Discarded |
| v12-v12.1 | NCT motifs on synthetic depth | Transition |
| v13 | Real RGB-D (NYU Depth V2) | ✅ **Current base** |
| v13.4 | Grouped split validation | ✅ Consolidated |
| v14.2.1 | Scene leave-one-out | ✅ Consolidated |

---

## Project Philosophy

> **"What cannot be experimentally falsified does not belong in this repository."**

This project adopts principles of:

- **Strong empiricism:** Only components that improve real metrics
- **Minimalism:** The simplest representation that captures the phenomenon
- **Honesty:** Document both positive and negative findings
- **Reproducibility:** Any claim must be verifiable by third parties

---

## Acknowledgments

The final method is the result of **discarding more than 80%** of the original ideas. This "aggressive pruning" process was crucial to arrive at a system that actually works on real data.
