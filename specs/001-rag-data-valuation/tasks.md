# Tasks: RAG Data Valuation System

**Input**: Design documents from `/specs/001-rag-data-valuation/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US4)
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 Create project structure (`src/dv/`, `src/rag/`, `src/utils/`, `tests/`)
- [x] T002 Initialize Python 3.12 project and dependencies in `requirements.txt`
- [x] T003 [P] Configure Ruff for linting and formatting standards in `pyproject.toml`
- [x] T004 [P] Setup Jupyter environment for experiment validation in `tests/notebooks/`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

- [x] T005 [P] Create base models for `Chunk`, `ValuationResult`, `ValuationReport` in `src/dv/models/entities.py`
- [x] T006 [P] Implement `Valuator`, `Signaler`, and `Judge` interfaces in `src/dv/interfaces.py`
- [x] T007 [P] Implement hashing utility for chunk identification (MD5/SHA) in `src/utils/hashing.py`
- [x] T008 [P] Implement device selection utility (MPS/CUDA) in `src/utils/torch_utils.py`

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Multi-Algorithm Valuation (Priority: P1) 🎯 MVP

**Goal**: Researcher evaluates Different DV Algorithms (LOO, Shapley, Attention) on the same RAG pipeline.

**Independent Test**: Run the valuation suite on a small dataset and verify report generation for all 3 methods.

### Implementation for User Story 1

- [x] T009 [P] [US1] Implement `LOOValuator` (Leave-One-Out) in `src/dv/algorithms/loo.py`
- [x] T010 [P] [US1] Implement `ShapleyValuator` (Exact for k≤10, Monte Carlo for k>10) in `src/dv/algorithms/shapley.py`
- [x] T011 [P] [US1] Implement surrogate model signals in `src/dv/models/surrogate.py`
- [x] T012 [P] [US1] Implement `AttentionValuator` utilizing surrogate signals in `src/dv/algorithms/attention.py`
- [x] T013 [US1] Implement main `ValuationSuite` to coordinate multiple methods in `src/dv/core.py`
- [x] T014 [US1] Implement `RagasJudge` wrapping `ragas.evaluate` in `src/dv/evaluation/judges.py`

**Checkpoint**: User Story 1 (MVP) fully functional with local and Ragas-based evaluation.

---

## Phase 4: User Story 2 - Hallucination Filtering (Priority: P2)

**Goal**: System optimizer identifies and removes chunks leading to hallucinations (negative value).

**Independent Test**: Filter chunks with negative DV scores and verify improvement in faithfulness metrics.

### Implementation for User Story 2

- [x] T015 [P] [US2] Implement `MNLIJudge` and `LLMJudge` in `src/dv/evaluation/judges.py`
- [x] T016 [US2] Implement agreement metrics (Kendall Tau and MAE) in `src/dv/evaluation/metrics.py`
- [x] T017 [US2] Implement negative-value filtering logic in `src/dv/evaluation/filtering.py`
- [x] T018 [US2] Create experiment script to validate 85% precision (SC-001) and 15% faithfulness increase (SC-004).

---

## Phase 5: User Story 4 - Chunking Comparison (Priority: P2)

**Goal**: Researcher compares standard chunking against optimized strategies.

**Independent Test**: Generate chunks using different strategies and verify different DV score distributions.

### Implementation for User Story 4

- [x] T019 [P] [US4] Implement fixed-length and recursive chunkers in `src/rag/chunking/`
- [x] T020 [P] [US4] Implement semantic chunking logic in `src/rag/chunking/semantic.py`
- [x] T021 [US4] Create comparison notebook `tests/notebooks/chunking_comparison.ipynb` for DV analysis.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements, Proxy Filter, and Final Validation.

- [x] T022 [P] Implement Proxy Filter (Regression) with MAE loss in `src/dv/algorithms/proxy.py`
- [x] T023 Optimize Shapley computation to meet <30s target for k=5 (SC-002)
- [x] T024 Validate system behavior for empty retrieval (Return Empty list)
- [x] T025 Run `quickstart.md` validation on the completed system

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies.
- **Foundational (Phase 2)**: Depends on Phase 1 completion.
- **User Stories (Phase 3-5)**: Depend on Phase 2 completion. US2 also depends on US1 algorithms.
- **Polish (Phase 6)**: Depends on all user stories completion.

### Parallel Opportunities

- T009, T010, T011, T012 (US1 Algorithms)
- T015, T016 (US2 Components)
- T019, T020 (US4 Chunkers)

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phases 1 & 2.
2. Complete Phase 3 (US1).
3. **VALIDATE**: Run valuation with `RagasJudge` and verify faithfulness reporting.

### Incremental Delivery

1. Foundation ready.
2. Add US1 → Multi-algorithm report with Ragas integration.
3. Add US2 → Quality improvement via filtering & Proxy Filter training.
4. Add US4 → Chunking scale experiments.
5. Final Polish & Performance optimization.
