# Feature Specification: RAG Data Valuation System

**Feature Branch**: `001-rag-data-valuation`  
**Created**: 2025-12-25  
**Status**: Draft  
**Input**: User description: "Hệ thống RAG Data Valuation để tìm giá trị thực của dữ liệu qua các phương pháp LOO, Shapley, và Attention"

## Clarifications

### Session 2025-12-25
- Q: How should the system store the generated valuation reports and chunk scores for experimentation? → A: File-based (Local JSON/CSV files).
- Q: How should the system uniquely identify chunks to ensure valuation scores are correctly mapped across different experiment runs or chunking strategies? → A: Content Hash (MD5/SHA of text).
- Q: How should the system handle Attention-based valuation for API-based models? → A: Surrogate Model (Small local model provides signals).
- Q: Should the system be built as a standalone "Valuation Wrapper" or a fully integrated Framework? → A: Standalone Wrapper (Interfaces with external RAG).
- Q: What is the definitive scale and reference for the "Faithfulness Ground Truth"? → A: Numerical (0.0 to 1.0 continuous score).
- Q: What is the training objective for the Proxy Filter? → A: Regression (Score Prediction of continuous DV values).
- Q: Which Ragas metric is the primary signal for data valuation? → A: Faithfulness (Factuality).
- Q: How should the system handle empty retrieval for a query? → A: Return Empty list (Skip valuation gracefully).
- Q: What are the operational limits for exact Shapley calculations? → A: Exact for $k \le 10$, MC for $k > 10$.
- Q: How is "agreement" defined for the Proxy Filter (SC-003)? → A: Mean Absolute Error (MAE).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Researcher evaluates Different DV Algorithms (Priority: P1)

As a researcher, I want to compare multiple Data Valuation (DV) algorithms (LOO, Shapley, Attention) on the same RAG pipeline so that I can determine which method most accurately identifies the "true value" of retrieved chunks.

**Why this priority**: Core objective of the system is to investigate and compare DV methods.
**Independent Test**: Can be tested by running the valuation suite on a small dataset and verifying that scores are generated for each method.

**Acceptance Scenarios**:

1. **Given** a query and top-k retrieved chunks, **When** I run the valuation suite, **Then** I should receive a valuation report containing LOO, Shapley, and Attention-based scores for each chunk.
2. **Given** the generated scores, **When** I compare them against faithfulness metrics, **Then** I should see a correlation analysis indicating which DV method aligns best with "gold" evidence.

---

### User Story 2 - System Optimizer Filters Noise/Hallucinations (Priority: P2)

As a system optimizer, I want to use the best-performing DV method to identify and remove chunks that lead to hallucinations (negative value) from my knowledge base.

**Why this priority**: Practical application of DV results to improve RAG quality.
**Independent Test**: Remove chunks identified as "hallucination-inducing" and verify that LLM response faithfulness improves.

**Acceptance Scenarios**:

1. **Given** a set of chunks with negative DV scores, **When** they are filtered out from the RAG pipeline, **Then** the LLM's faithfulness score (via NLI/Judge) should increase compared to using all chunks.

---

### User Story 4 - Researcher compares Chunking Strategies (Priority: P2)

As a researcher, I want to compare standard chunking (length/token) against optimized strategies (semantic/recursive) to see how chunking granularity affects Data Valuation results.

**Why this priority**: Chunking is the foundation of RAG; valuation results depend heavily on how data is partitioned.
**Independent Test**: Run retriever on different chunking profiles and verify that valuation scores are generated for each.

**Acceptance Scenarios**:

1. **Given** a knowledge base, **When** I apply simple (fixed-length) and optimized (semantic) chunking, **Then** the valuation suite should identify different high-value chunks for the same query.
2. **Given** different chunking outputs, **When** I evaluate their DV scores, **Then** I should be able to visualize the distribution of "gold" vs. "hallucination" chunks across strategies.

---

### Edge Cases

- **Tied Scores**: How does the system handle cases where multiple chunks have identical DV scores but different semantic content?
- **Empty Retrieval**: If the retriever returns 0 chunks for a query, the system MUST return an empty list of valuation results and skip processing gracefully.
- **LLM Non-determinism**: How does the system handle variance in LLM-as-a-judge scores when evaluating faithfulness? (Expected: Multiple trials or high temperature=0).
- **Abnormal Chunk Sizes**: How does Attention-based DV handle extremely long chunks that might exceed LLM context windows during extraction?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST implement **Leave-One-Out (LOO)** valuation by measuring the quality drop when a chunk is removed.
- **FR-002**: System MUST implement **Shapley Value** valuation. Use **Exact calculation** for $k \le 10$ chunks and **Monte Carlo (MC) sampling** for $k > 10$ to ensure stability.
- **FR-003**: System MUST implement **Attention/Gradient-based** valuation by extracting internal LLM weights or gradients for retrieved chunks.
- **FR-004**: System MUST evaluate chunk values against a **Faithfulness Ground Truth** measured using NLI models or LLM-as-a-judge (e.g., GPT-4 or similar).
- **FR-005**: System MUST provide a comparison module to analyze the precision/recall of each DV method in identifying "hallucination-inducing" (negative value) vs. "gold" chunks.
- **FR-006**: System MUST support training a **Proxy Filter** as a **Regression Model** using the best-performing DV scores as continuous training labels.
- **FR-007**: System MUST implement **diverse chunking strategies**:
    - **Simple**: Fixed length (characters/tokens).
    - **Optimized**: Recursive character splitting and Semantic chunking.
- **FR-008**: System MUST provide a **comprehensive metrics suite** for evaluation:
    - **DV Metrics**: Agreement (Kendall Tau) between different DV methods.
    - **RAG Metrics**: Faithfulness, Answer Relevance, and Context Precision.
    - **Efficiency Metrics**: Latency per valuation method.
- **FR-003a**: When using API-based LLMs for generation, the system MUST use a local surrogate model (e.g., Llama-3-8B) to compute Attention/Gradient-based valuation signals.
- **FR-010a**: The hybrid architecture MUST allow concurrent initialization of a local model (for signals) and an API model (for generation/faithfulness judging).
- **FR-009**: System MUST support **General Knowledge datasets** (e.g., Wikipedia, SQuAD, or common benchmark corpora) for initial valuation experiments.
- **FR-010**: System MUST support a hybrid model architecture:
    - **Self-hosted**: Support open-weight models (Llama-3, Mistral) via HuggingFace for **Attention/Gradient-based** signals.
    - **API-based**: Support OpenAI-compatible APIs (using cheaper models like GPT-4o-mini or local LLM servers) for high-speed **LOO/Shapley** comparison and chunking evaluation.

### Key Entities *(include if feature involves data)*

- **Query**: The input question from the user.
- **Chunk**: A segment of text retrieved from the knowledge base. Uniquely identified by its content hash (e.g., MD5/SHA).
- **DV Suite**: A collection of valuation algorithms applied to chunks.
- **Faithfulness Score**: A numerical metric (0.0 to 1.0 continuous scale) representing how much an answer is supported by the chunks, used as the "Gold" ground truth.
- **Valuation Report**: The output mapping each chunk to its calculated value across different methods. (Stored as local JSON/CSV files).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The best DV method identifies hallucination-inducing chunks with at least **85% precision** compared to the Faithfulness Ground Truth.
- **SC-002**: Computation of Shapley Values for $k=5$ chunks completes in under **30 seconds** per query (optimizing combination counts).
- **SC-003**: The Proxy Filter achieves a **Mean Absolute Error (MAE) < 0.1** with the original DV method while reducing inference latency by **10x**.
- **SC-004**: Filtering negative-value chunks leads to a measurable increase (at least **15%**) in average faithfulness scores across a benchmark dataset.

## Assumptions

- We assume access to an LLM where internal states (attention/gradients) can be extracted for FR-003, or we will use a surrogate model if using an API.
- We assume the existence of a RAG pipeline (Retriever + LLM) that can be swapped or modified for valuation experiments. The system will act as a standalone wrapper/interceptor for these components.
