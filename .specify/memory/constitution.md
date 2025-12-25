<!--
Sync Impact Report:
- Version change: 1.0.1 -> 1.1.0
- List of modified principles:
  - Added VI. Module Extensibility
  - Added VII. Experimental Benchmarking
- Added sections: N/A
- Removed sections: N/A
- Templates requiring updates (✅ updated / ⚠ pending):
  - .specify/templates/plan-template.md (⚠ pending - needs Constitution Check update)
  - .specify/templates/spec-template.md (⚠ pending)
  - .specify/templates/tasks-template.md (⚠ pending)
-->
# RAG Data Valuation Constitution

## Core Principles

### I. Pythonic Excellence & Ruff Enforcement
All code MUST be written in Python 3.12. Standard conventions and clean code practices are non-negotiable. Ruff MUST be used to enforce linting and formatting standards to maintain a consistent codebase.

### II. Test-Driven Frameworks (Pytest & Jupyter)
Logic MUST be verified using `pytest` for unit/integration testing. Final experimentation and valuation validation MUST be conducted in Jupyter notebooks to ensure reproducibility and clear visualization of results.

### III. Interface Consistency (UX)
User experience consistency across CLI tools and internal APIs is mandatory. All interfaces MUST follow predictable patterns for data input/output, ensuring that the valuation framework is easy to integrate and use.

### IV. Scalable & High-Performance Valuation
Data valuation algorithms (LOO, Shapley, etc.) MUST be optimized for performance. Use `torch` and `transformers` efficiently for local compute, and leverage external APIs (e.g., OpenAI) for cloud-based scalability when handling large RAG datasets.

### V. Iterative RAG Refinement
The ultimate goal of this framework is to refine RAG pipelines. Every change SHOULD contribute to identifying hallucination-causing or high-value data chunks, leading to measurable improvements in answer authenticity.

### VI. Module Extensibility
New valuation algorithms MUST implement the `Valuator` abstract base class from `src.dv.interfaces`. This ensures consistent integration into the `ValuationSuite` and allows for seamless comparison between different valuation methods.

### VII. Experimental Benchmarking
The `dv_real_benchmark.ipynb` and related "dv_real..." notebooks are designated as experimental environments. They MUST be used for prototyping and benchmarking new valuation methods against real-world datasets before they are promoted to stable modules.

## Development Workflow

All feature development MUST follow the `speckit` workflow:
1. **Specify**: High-level requirements and goal definition.
2. **Plan**: Detailed implementation and verification strategy.
3. **Tasking**: Granular, dependency-ordered execution steps.
4. **Implement**: Execution of tasks with frequent verification.

## Quality Gates
- 100% Ruff compliance required for PR approval.
- All `pytest` suites must pass.
- Notebooks must run top-to-bottom without errors for experiment results.

## Governance
This constitution supersedes all other documentation. Amendments require a version bump and updates to all dependent templates.

**Version**: 1.1.0 | **Ratified**: 2025-12-25 | **Last Amended**: 2025-12-25

