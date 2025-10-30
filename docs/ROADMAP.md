# MozzareLLM Development Roadmap

**Version**: 1.0  
**Last Updated**: 2025-01-30  
**Status**: Active Development

---

## Overview

This roadmap outlines the development priorities for enhancing MozzareLLM's analytical capabilities through improved Retrieval-Augmented Generation (RAG), Chain-of-Thought (CoT) reasoning, and prompt engineering. The priorities align with the design principles outlined in the [project design document](https://docs.google.com/document/d/1xTFxuFIu_-kz4m96E-u110yPIcWOOwYvaN_1AOODpzk/edit).

### Key Design Principles (from design doc):
1. **Prioritize deterministic computing** where possible
2. **Ground model outputs** in context beyond training data
3. **Trace reasoning** via Chain-of-Thought framework
4. **Scale efficiently** in time and cost
5. **Maintain lightweight architecture** - avoid heavy infrastructure dependencies

---

## Phase 1: Core RAG Infrastructure (Foundation)
**Owner**: Alexa  
**Priority**: HIGH  
**Goal**: Build robust retrieval system that grounds LLM analysis in external, verified biological knowledge

### Task 1.1: UniProt API Integration in `retrieval.py`

**Current State**: 
- Gene annotations passed as CSV/dictionary (`gene_annotations` parameter in `ClusterAnalyzer`)
- No automatic fetching of functional data

**Target State**:
- Live API calls to UniProt for gene functional annotations
- Integrated into existing `retrieval.py` module
- Fallback gracefully when API unavailable

**Design Questions**:
1. **Which UniProt fields to retrieve?**
   - Options: Function, GO terms, pathways, protein families, subcellular location, disease associations
   - Consider: Token budget vs. information value
   
2. **Caching strategy?**
   - Option A: In-memory cache (cleared between runs)
   - Option B: Persistent file cache (SQLite, JSON, or pickle)
   - Option C: No caching (always fresh, but slower)
   - Consider: Balance between freshness and API rate limits

**Implementation Notes**:
- Modify `retrieval.py::retrieve_context()` to add UniProt fetching
- New function: `_fetch_uniprot_annotations(genes: list[str]) -> dict[str, str]`
- Update `RetrievalContext` model if needed to track API sources
- Add UniProt as a citation source in retrieval metadata

**Success Criteria**:
- [ ] Can fetch gene annotations via UniProt API
- [ ] Integrates seamlessly with existing `retrieve_context()` function
- [ ] Handles API errors and rate limits gracefully
- [ ] Caching reduces redundant API calls (if implemented)
- [ ] Documentation updated with UniProt usage examples

**Complexity Estimate**: Medium (2-3 story points)

---

### Task 1.2: PubMed Pathway Knowledge File Generator

**Current State**:
- Manual creation of pathway-specific markdown files (e.g., `kinetochore.md`, `centromere.md`)
- Static knowledge base that becomes outdated
- Not scalable for diverse pathway analyses

**Target State**:
- Automated pipeline to generate pathway knowledge files from PubMed literature
- Dynamic knowledge generation based on identified pathways
- Structured markdown files suitable for RAG retrieval

**Design Questions**:
1. **When to generate pathway knowledge files?**
   - Option A: **Post-analysis** - After LLM identifies pathway, generate knowledge file, then re-analyze
   - Option B: **Pre-populated library** - Generate files for common pathways ahead of time
   - Option C: **Hybrid** - Pre-populate common pathways, generate on-demand for novel ones
   - Consider: Latency vs. freshness trade-offs

2. **PubMed query strategy?**
   - Use pathway name + established genes as search terms?
   - Filter by publication date (e.g., last 5 years only)?
   - Filter by journal impact factor or publication type?
   - How many articles to retrieve per pathway?

3. **Knowledge file structure?**
   - Should we extract abstracts, full text, or specific sections?
   - How to organize: By gene? By sub-process? Chronologically?
   - Should we include citation metadata (PMID, DOI, year)?

4. **Integration with Claude's native PubMed search?**
   - Claude (Anthropic) has built-in PubMed search capability
   - Should we use Claude's native search or build independent pipeline?
   - Consider: Simplicity vs. control/reproducibility

**Implementation Notes**:
- New module: `mozzarellm/utils/pubmed_knowledge.py`
- Key functions:
  - `generate_pathway_knowledge(pathway_name: str, genes: list[str], output_dir: str) -> str`
  - `query_pubmed(search_terms: list[str], max_results: int) -> list[dict]`
  - `format_knowledge_markdown(articles: list[dict]) -> str`
- Consider using Anthropic's PubMed tool (available to Claude models) vs. NCBI E-utilities API
- Store generated files in `data/knowledge/pathways/{pathway_name}.md`

**Success Criteria**:
- [ ] Can automatically generate pathway knowledge files from PubMed
- [ ] Generated files are structured for effective RAG retrieval
- [ ] Files include proper citations (PMID/DOI)
- [ ] Pipeline handles API rate limits and errors
- [ ] Documentation with examples of generated knowledge files

**Complexity Estimate**: Large (5-8 story points)

**Dependencies**: May benefit from Task 1.1 completion (using gene annotations to improve PubMed queries)

---

### Task 1.3: Enhanced Retrieval Algorithm

**Current State** (from `retrieval.py`):
- Simple keyword matching on local text files
- Basic scoring: term frequency + proximity bonus
- Context window extraction (±4 lines around matches)
- Relevance scoring prioritizes: annotations (100) > screen context (50) > knowledge files (capped at 90)

**Target State**:
- Improved relevance scoring that better captures semantic relationships
- Better handling of synonyms and related terms
- More sophisticated evidence ranking

**Design Questions**:
1. **Semantic search vs. keyword matching?**
   - Option A: **Keep keyword-based** (lightweight, per Design Principle 5)
   - Option B: **Add embeddings** (better semantics, but adds dependencies: sentence-transformers, FAISS/ChromaDB)
   - Option C: **Hybrid** (keyword first, semantic reranking)
   - Consider: Accuracy gain vs. infrastructure complexity

2. **Term expansion strategy?**
   - Should we expand gene symbols to include aliases/synonyms?
   - Use biological ontologies (GO, UniProt) for related terms?
   - How to weight original vs. expanded terms?

3. **Context window optimization?**
   - Current: ±4 lines around match
   - Should this be adaptive based on content type (annotations vs. papers)?
   - Should we extract at sentence/paragraph boundaries instead?

4. **Evidence diversity?**
   - Should we ensure diverse sources in top-k results?
   - Avoid over-representation from single file/source?

5. **Relevance score calibration?**
   - Are current weights (annotations=100, context=50, knowledge=90) optimal?
   - Should scores be normalized or kept as raw values?

**Implementation Notes**:
- Enhance `retrieval.py::_search_file_for_terms()`
- Consider new function: `_expand_search_terms(genes: list[str]) -> list[str]`
- Potentially add: `_rerank_by_semantic_similarity()` if going with embeddings
- Update relevance scoring in `retrieve_context()`

**Success Criteria**:
- [ ] Improved relevance of top-k retrieved snippets (measured by manual review)
- [ ] Better handling of gene synonyms and related terms
- [ ] Maintains lightweight architecture (unless semantic search is worth the trade-off)
- [ ] Documentation of retrieval algorithm and scoring methodology

**Complexity Estimate**: Medium-Large (3-5 story points, depending on semantic search decision)

**Dependencies**: Task 1.1 (UniProt) may provide gene synonyms for term expansion

---

## Phase 2: CoT Restructuring (Reasoning Improvements)
**Owner**: Alexa  
**Priority**: HIGH  
**Goal**: Implement robust chain-of-thought reasoning with validation and traceability

### Task 2.1: Restructure CoT to Reference Prompt Sections Sequentially

**Current State**:
- CoT instructions (`ENHANCED_COT_INSTRUCTIONS` or `CONCISE_COT_INSTRUCTIONS`) appended at end of prompt
- Monolithic reasoning structure that doesn't leverage the modular prompt sections (1-9)
- CoT steps are generic, not tailored to the structured analysis flow

**Target State**:
- CoT reasoning that explicitly references and utilizes prompt sections 1-9 in order
- Step-by-step reasoning that mirrors the prompt structure:
  1. Core task understanding (Section 1)
  2. Experimental context integration (Section 2)
  3. Gene classification framework (Section 3)
  4. Gene annotation review (Section 4)
  5. Evidence synthesis (Section 5)
  6. Pathway confidence assessment (Section 6)
  7. [Future] Phenotypic strength validation (Section 7)
  8. [Future] Mechanistic hypotheses (Section 8)
  9. [Future] Experiment suggestions (Section 9)

**Design Questions**:
1. **CoT instruction placement?**
   - Option A: Single comprehensive CoT section that references all prior sections
   - Option B: Mini-CoT prompts after each major section (iterative reasoning)
   - Option C: CoT instructions at the start that preview the entire analysis flow
   - Consider: Token efficiency vs. reasoning quality

2. **Section-specific reasoning prompts?**
   - Should each prompt section (1-9) have tailored reasoning questions?
   - Example: "For Section 6, consider: Do established genes represent >70% of cluster?"
   - How prescriptive vs. open-ended should these be?

3. **Balance between structure and flexibility?**
   - Too rigid: May constrain novel insights
   - Too loose: May miss critical analytical steps
   - How to find the right balance?

4. **CoT verbosity control?**
   - Current options: `ENHANCED_COT_INSTRUCTIONS` (detailed) vs. `CONCISE_COT_INSTRUCTIONS` (brief)
   - Should we add more granularity (minimal, standard, detailed, exhaustive)?
   - How does verbosity affect token usage and quality?

**Implementation Notes**:
- Modify `prompts.py` to create section-aware CoT instructions
- Update `prompt_factory.py::make_cluster_analysis_prompt()` to weave CoT throughout
- Consider new prompt component: `SECTION_BASED_COT_INSTRUCTIONS`
- May need to refactor how `cot_instructions` parameter is handled in `ClusterAnalyzer`

**Success Criteria**:
- [ ] CoT reasoning explicitly references prompt sections 1-9
- [ ] Reasoning follows logical flow of analysis stages
- [ ] Improved analytical quality (measured by validation set)
- [ ] Token usage remains reasonable (document overhead)
- [ ] Backward compatible with existing CoT implementations

**Complexity Estimate**: Medium (3-4 story points)

---

### Task 2.2: Design and Implement Reasoning Trace Capture System

**Current State**:
- Raw LLM response stored in `ClusterResult.raw_response`
- No structured capture of reasoning steps
- Difficult to audit or debug analytical decisions
- Cannot trace how LLM arrived at specific gene classifications

**Target State**:
- Structured capture of CoT reasoning for each cluster
- Traceable decision path for pathway identification and gene prioritization
- Enables post-hoc analysis and quality improvement
- Supports human oversight and intervention (Objective C from design doc)

**Design Questions**:
1. **Storage location and format?**
   - Option A: Add `reasoning_trace` field to `ClusterResult` model (JSON-serializable)
   - Option B: Separate file per cluster (`{cluster_id}_reasoning.json`)
   - Option C: Separate file per analysis run (`{output_base}_reasoning.json` with all clusters)
   - Option D: Hybrid - structured field in model + optional file export
   - Consider: Ease of access vs. file management complexity

2. **Reasoning trace structure?**
   ```python
   # Option A: Flat list of steps
   reasoning_trace = [
     {"step": 1, "action": "pathway_hypothesis", "content": "...", "timestamp": "..."},
     {"step": 2, "action": "gene_classification", "content": "...", "timestamp": "..."},
   ]
   
   # Option B: Nested by prompt section
   reasoning_trace = {
     "section_1_task": {"reasoning": "...", "key_decisions": [...]},
     "section_6_confidence": {"reasoning": "...", "evidence_cited": [...]},
   }
   
   # Option C: Timeline with evidence links
   reasoning_trace = {
     "timeline": [...],
     "evidence_used": {"snippet_1": ["step_2", "step_4"], ...},
     "decisions": {"pathway": "...", "high_priority_genes": [...]}
   }
   ```
   - Which structure best supports debugging and analysis?

3. **Extraction method?**
   - Option A: **Parse from raw response** - Extract reasoning sections using regex/LLM parsing
   - Option B: **Request structured output** - Modify output format to include explicit reasoning fields
   - Option C: **Separate reasoning call** - Make a second API call asking LLM to explain its reasoning
   - Consider: Accuracy vs. cost/latency

4. **Integration with validation (Task 2.3)?**
   - Should reasoning traces include validation checkpoints?
   - Should validator have access to original reasoning trace?
   - How to version reasoning if analysis is revised?

5. **Verbosity control?**
   - Should users control reasoning trace detail level?
   - High-detail traces may be very large for complex clusters
   - Low-detail may miss critical decision points

**Implementation Notes**:
- Modify `models.py::ClusterResult` to add reasoning capture fields
- Update `llm_analysis_utils.py::process_cluster_response()` to extract reasoning
- Consider new utility: `reasoning_parser.py` for parsing structured reasoning
- Update `analyzer.py::_analyze_single_cluster()` to save reasoning traces
- Modify output format in `prompts.py::OUTPUT_FORMAT_JSON` if using structured output approach

**Success Criteria**:
- [ ] Reasoning traces captured for all cluster analyses
- [ ] Traces are human-readable and structured
- [ ] Can trace specific decisions (e.g., "Why was GeneX priority=9?")
- [ ] Storage approach is sustainable for large-scale analyses
- [ ] Documentation with examples of reasoning trace inspection

**Complexity Estimate**: Medium-Large (4-6 story points, depending on extraction method)

**Dependencies**: Task 2.1 (section-based CoT may inform reasoning structure)

---

### Task 2.3: Implement CoT Validation/Self-Critique Mechanism

**Current State**:
- Single-pass analysis with no validation
- Quality checks are post-hoc (in `analyzer.py::_validate_cluster_result()`)
- No mechanism for LLM to catch its own errors
- No multi-turn refinement

**Target State**:
- LLM validates its own analysis before finalizing
- Catches common errors: inconsistencies, unsupported claims, missed genes
- Optionally refines analysis based on critique
- Improves reliability and reduces hallucinations (Objective B from design doc)

**Design Questions**:
1. **Validation approach?**
   - Option A: **Self-critique** - Single LLM call reviews its own output, flags issues
   - Option B: **Multi-pass** - LLM analyzes → critique → revise → final output (2-3 calls)
   - Option C: **Consistency check** - Run analysis multiple times, flag disagreements (expensive)
   - Option D: **Separate validator model** - Use different model or temperature for validation
   - Consider: Cost/latency vs. quality improvement

2. **What to validate?**
   - Pathway confidence matches established gene ratio?
   - All input genes are classified?
   - Claims are supported by evidence citations?
   - Priority scores are justified by rationales?
   - No contradictions in reasoning?
   - Other quality checks?

3. **Validation prompt design?**
   - Should validator see original prompt + analysis, or just the analysis?
   - Should validator have access to evidence/annotations?
   - How to phrase validation instructions to avoid false positives?

4. **Handling validation failures?**
   - Option A: **Auto-revise** - LLM automatically fixes flagged issues (requires multi-pass)
   - Option B: **Flag only** - Warn user, but keep original analysis
   - Option C: **Hybrid** - Auto-fix minor issues, flag major ones for human review
   - Consider: Autonomy vs. human oversight

5. **Integration with reasoning traces?**
   - Should validation be captured in reasoning trace?
   - Should we show "before" and "after" if analysis is revised?
   - How to present validation findings to users?

6. **When to validate?**
   - Every cluster (expensive but thorough)?
   - Only low-confidence or ambiguous clusters?
   - User-configurable threshold?

**Implementation Notes**:
- New module: `mozzarellm/utils/validation.py`
- Key functions:
  - `validate_cluster_analysis(cluster_result: ClusterResult, original_prompt: str) -> ValidationResult`
  - `create_validation_prompt(analysis: dict, genes: list[str]) -> str`
  - `apply_validation_fixes(cluster_result: ClusterResult, validation: ValidationResult) -> ClusterResult`
- Update `analyzer.py::_analyze_single_cluster()` to include validation step
- Add validation settings to `ClusterAnalyzer.__init__()` (e.g., `enable_validation=False`, `validation_mode="self_critique"`)
- Consider adding `ValidationResult` model to `models.py`

**Success Criteria**:
- [ ] Validation catches common analytical errors
- [ ] Measurable improvement in analysis quality (benchmark on validation set)
- [ ] Validation results are captured and reportable
- [ ] User can control validation behavior (enable/disable, mode selection)
- [ ] Documentation with examples of validation catching errors

**Complexity Estimate**: Large (5-8 story points)

**Dependencies**: Task 2.2 (reasoning traces) would enhance validation by providing decision context

---

## Phase 3: Prompt Engineering & Feature Integration
**Owner**: Matteo  
**Priority**: MEDIUM  
**Goal**: Complete the prompt structure with advanced analytical features

### Task 3.1: Implement Section 7 - Phenotypic Strength Validation

**Current State**:
- Section 7 is a placeholder in `prompts.py::PHENOTYPIC_STRENGTH_CONFIDENCE_EVALUATION`
- Phenotypic strength data exists in cluster data (e.g., `phenotypic_strength = "669/5299"`)
- No cross-validation between pathway confidence and phenotypic strength

**Target State**:
- Prompt section that cross-checks pathway confidence against phenotypic strength
- Identifies mismatches (high confidence + low strength, or vice versa)
- Prompts LLM to reconsider analysis when mismatched

**Example Input Format**:
```csv
gene_symbol,cluster,up_features,down_features,phenotypic_strength
AATF,21,"interphase_cell_correlation_dapi_tubulin,interphase_cell_correlation_tubulin_phalloidin",
"interphase_nucleus_area,interphase_cell_area",669/5299
```

**Design Questions**:
1. **Phenotypic strength normalization?**
   - Current format: `numerator/denominator` (e.g., `669/5299`)
   - Should we compute ratio (0-1)? Percentile? Z-score?
   - How to define "high" vs. "low" strength thresholds?

2. **When to flag mismatches?**
   - High confidence + Low strength: Always flag?
   - Low confidence + High strength: Always flag?
   - Medium confidence + Any strength: No flag?
   - What thresholds define "mismatch"?

3. **How to prompt reassessment?**
   - Ask LLM to explicitly justify the mismatch?
   - Trigger automatic re-analysis with stricter criteria?
   - Just flag in reasoning/summary without forcing revision?

4. **Integration with CoT validation (Task 2.3)?**
   - Should phenotypic strength check be part of validation?
   - Or a separate reasoning step before validation?

5. **Handling missing phenotypic strength?**
   - Not all clusters may have this data
   - Should section be skipped, or should LLM note the absence?

**Implementation Notes**:
- Complete `prompts.py::PHENOTYPIC_STRENGTH_CONFIDENCE_EVALUATION` placeholder
- Update `prompt_factory.py::make_cluster_analysis_prompt()` to conditionally include Section 7
- Add `phenotypic_strength` parameter to `make_cluster_analysis_prompt()`
- Update `ClusterAnalyzer.analyze()` to pass phenotypic strength from `cluster_df`
- May need to add `phenotypic_strength` field to `ClusterResult` model

**Success Criteria**:
- [ ] Section 7 prompt completed and integrated
- [ ] Phenotypic strength data parsed from cluster input
- [ ] Mismatches are identified and flagged in analysis
- [ ] Documentation with examples of mismatch detection

**Complexity Estimate**: Small-Medium (2-3 story points)

---

### Task 3.2: Implement Section 8 - Feature Directionality Mechanistic Hypotheses

**Current State**:
- Section 8 is a placeholder in `prompts.py::FEATURE_DIRECTIONALITY_HYPOTHESIS`
- Feature directionality data exists (`up_features`, `down_features` columns)
- No mechanistic hypothesis generation based on directional changes

**Target State**:
- Prompt section that uses up/down-regulated features to generate mechanistic hypotheses
- Links directional changes to specific pathway components or processes
- Frames hypotheses for experimental validation

**Example Input Format**:
```csv
up_features: "interphase_cell_correlation_dapi_tubulin,interphase_cell_correlation_tubulin_phalloidin"
down_features: "interphase_nucleus_area,interphase_cell_area,interphase_nucleus_solidity"
```

**Design Questions**:
1. **Feature interpretation strategy?**
   - Should prompt include definitions of imaging features?
   - How to help LLM connect feature changes to biological processes?
   - Should we provide feature ontology or let LLM use training knowledge?

2. **Hypothesis specificity?**
   - High-level (e.g., "Cytoskeletal reorganization is occurring")
   - Medium (e.g., "Tubulin-microtubule dynamics are altered")
   - Detailed (e.g., "Increased tubulin polymerization at centrosomes")
   - Let LLM decide based on evidence quality?

3. **Timing in analysis flow?**
   - Should come AFTER pathway identification (Section 6)
   - Should come BEFORE experiment suggestions (Section 9)
   - Should mechanistic hypotheses inform experiment design?

4. **Handling heterogeneous features?**
   - Some clusters may have many up/down features, others few
   - Should we prioritize most significant features?
   - How to handle when features don't align with identified pathway?

5. **Integration with other sections?**
   - Should feature directionality influence pathway confidence (Section 6)?
   - Should it be validated in CoT validation (Task 2.3)?
   - How to present in final output?

**Implementation Notes**:
- Complete `prompts.py::FEATURE_DIRECTIONALITY_HYPOTHESIS` placeholder
- Update `prompt_factory.py::make_cluster_analysis_prompt()` to conditionally include Section 8
- Add `up_features` and `down_features` parameters to `make_cluster_analysis_prompt()`
- Update `ClusterAnalyzer.analyze()` to parse feature columns from `cluster_df`
- Consider adding `mechanistic_hypothesis` field to `ClusterResult` model

**Success Criteria**:
- [ ] Section 8 prompt completed and integrated
- [ ] Feature directionality data parsed and formatted for prompt
- [ ] Mechanistic hypotheses generated for clusters with feature data
- [ ] Hypotheses are biologically plausible and specific
- [ ] Documentation with examples of hypothesis generation

**Complexity Estimate**: Medium (3-4 story points)

**Dependencies**: May benefit from Task 3.1 (phenotypic strength context) and Task 1.1 (UniProt data for biological context)

---

### Task 3.3: Implement Section 9 - Lab-Bounded Follow-Up Experiment Suggestions

**Current State**:
- Section 9 is a placeholder in `prompts.py::FOLLOW_UP_EXPERIMENT_SUGGESTIONS`
- No structured experiment suggestions in output
- LLM may suggest experiments in summary, but not systematically

**Target State**:
- Prompt section that generates specific, actionable experiment suggestions
- Suggestions bounded by lab capabilities (from stable reference file)
- Tailored to pathway confidence, phenotypic strength, and mechanistic hypotheses
- Prioritizes validation of high-priority genes

**Design Questions**:
1. **Lab capabilities file format?**
   - Option A: Simple list of available techniques/assays
   ```markdown
   # Available Assays
   - Immunofluorescence microscopy (IF)
   - Western blot
   - Co-immunoprecipitation (Co-IP)
   - CRISPR knockout
   - Live-cell imaging
   - Flow cytometry
   ```
   - Option B: Structured with details
   ```json
   {
     "techniques": [
       {
         "name": "Co-IP",
         "purpose": "Protein-protein interactions",
         "throughput": "medium",
         "cost": "medium"
       }
     ]
   }
   ```
   - Option C: Markdown with categories (biochemical, genetic, imaging, etc.)

2. **How specific should suggestions be?**
   - High-level (e.g., "Validate protein interactions")
   - Medium (e.g., "Co-IP GeneX with known pathway members")
   - Very specific (e.g., "Co-IP FLAG-tagged GeneX with pathway member GeneY; expect interaction if GeneX is part of Complex Z")
   - Balance between specificity and overfitting?

3. **Suggestion categories?**
   - Should we structure suggestions into types?
     - Pathway validation experiments
     - High-priority gene validation experiments
     - Mechanistic hypothesis testing experiments
     - Ambiguity resolution experiments (for low confidence clusters)

4. **How many suggestions per cluster?**
   - Fixed number (e.g., always 2-4)?
   - Adaptive based on pathway confidence (high confidence = fewer suggestions)?
   - User-configurable?

5. **Integration with previous analysis?**
   - Should experiments address phenotypic strength mismatches (Section 7)?
   - Should experiments test mechanistic hypotheses (Section 8)?
   - Should priority align with high-priority genes (priority ≥8)?

6. **Handling missing lab capabilities file?**
   - Should system provide generic suggestions?
   - Warn user and skip section?
   - Use default set of common techniques?

**Implementation Notes**:
- Complete `prompts.py::FOLLOW_UP_EXPERIMENT_SUGGESTIONS` placeholder
- Design lab capabilities file format and provide template
- Update `prompt_factory.py::make_cluster_analysis_prompt()` to include Section 9 with lab capabilities
- Add `lab_capabilities_file` parameter to `ClusterAnalyzer.__init__()`
- Load and format lab capabilities in prompt assembly
- Consider adding `suggested_experiments` field to `ClusterResult` model

**Success Criteria**:
- [ ] Section 9 prompt completed and integrated
- [ ] Lab capabilities file format defined with template/examples
- [ ] Experiment suggestions are specific, actionable, and lab-appropriate
- [ ] Suggestions align with pathway confidence and gene priorities
- [ ] Documentation with examples of experiment suggestion generation

**Complexity Estimate**: Medium (3-4 story points)

**Dependencies**: Benefits from completion of Sections 7-8 (phenotypic strength and mechanistic hypotheses inform experiments)

---

### Task 3.4: Lab Capabilities File Design and Documentation

**Current State**:
- No defined format for lab capabilities
- No template or examples

**Target State**:
- Well-defined file format (markdown, JSON, or YAML)
- Template file users can customize
- Documentation on how to create and maintain
- Example files for different lab types (imaging-focused, molecular, genetics, etc.)

**Design Questions**:
1. **File format choice?**
   - Markdown (human-friendly, easy to edit)
   - JSON (structured, easy to parse)
   - YAML (balance of human-readable and structured)

2. **What information to include per technique?**
   - Just technique name?
   - Additional context: purpose, throughput, cost, expertise required?
   - When is more detail helpful vs. overwhelming?

3. **Categorization scheme?**
   - By technique type (biochemical, genetic, imaging)?
   - By experimental goal (validation, discovery, mechanistic)?
   - By biological scale (molecular, cellular, organism)?
   - Flat list vs. hierarchical?

4. **Versioning and updates?**
   - Should users version their lab capabilities files?
   - How to handle changes in capabilities over time?
   - Should we track which version was used for each analysis?

**Implementation Notes**:
- Create `templates/lab_capabilities_template.md` (or .json/.yaml)
- Create `examples/lab_capabilities_imaging_lab.md`
- Create `examples/lab_capabilities_molecular_lab.md`
- Document in main README.md and create `docs/lab_capabilities.md` guide
- Add validation function to check lab capabilities file format

**Success Criteria**:
- [ ] Lab capabilities file format defined and documented
- [ ] Template file created for users to customize
- [ ] At least 2 example files for different lab types
- [ ] Clear documentation on creating and using lab capabilities files

**Complexity Estimate**: Small (1-2 story points)

**Dependencies**: Should be completed before or alongside Task 3.3

---

## Phase 4: System Refinements
**Owner**: Low Priority / Future Work  
**Goal**: Polish and optimize the system

### Task 4.1: Quality Metrics for RAG Effectiveness

**Description**: Develop metrics to assess RAG retrieval quality
- Relevance of retrieved snippets to final analysis
- Coverage of genes in annotations vs. knowledge base
- Citation frequency and distribution
- Evidence diversity metrics

**Design Questions**:
- How to measure "good" retrieval without ground truth?
- Should we track which evidence influenced which decisions?
- How to present metrics to users?

**Complexity Estimate**: Medium (3-4 story points)

---

### Task 4.2: Comprehensive Documentation and Examples

**Description**: Expand documentation with real-world examples and best practices
- Tutorial: Basic usage with toy dataset
- Tutorial: Advanced usage with RAG and CoT
- Tutorial: Interpreting results and reasoning traces
- Best practices guide for prompt customization
- API reference documentation

**Complexity Estimate**: Medium (3-4 story points)

---

### Task 4.3: Unit and Integration Testing

**Description**: Expand test coverage
- Unit tests for retrieval algorithms
- Integration tests for full analysis pipeline
- Mocking strategies for API calls (UniProt, PubMed)
- Regression tests for prompt changes

**Complexity Estimate**: Medium-Large (4-6 story points)

---

## Phase 5: Claude-Specific Optimizations (Future/Optional)
**Owner**: Low Priority / Interact with Claude Team  
**Goal**: Leverage Claude-specific features for efficiency

### Task 5.1: Prompt Caching Implementation

**Description**: Implement Anthropic's prompt caching to reduce costs
- Cache static prompt components (system prompt, classification rules)
- Identify cacheable vs. dynamic prompt sections
- Measure cost savings

**Design Questions**:
- Which prompt sections change least frequently?
- How often do cached components need refreshing?
- Trade-offs between cache hits and prompt flexibility?

**References**:
- [Anthropic Prompt Caching Docs](https://docs.anthropic.com/claude/docs/prompt-caching)

**Complexity Estimate**: Small-Medium (2-3 story points)

---

### Task 5.2: Extended Thinking Mode Support

**Description**: Integrate Claude's extended thinking feature for complex analyses
- Enable extended thinking for configurable cluster complexity thresholds
- Capture and display extended reasoning outputs
- Measure quality improvement vs. latency cost

**Design Questions**:
- When to automatically enable extended thinking?
- How to present extended reasoning to users?
- Is latency acceptable for batch processing?

**References**:
- [Extended Thinking Documentation](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)

**Complexity Estimate**: Medium (3-4 story points)

---

### Task 5.3: Batch Processing API Integration

**Description**: Use Anthropic's Message Batches API for large-scale analyses
- 50% cost reduction for batch processing
- Results available within ~24 hours
- Suitable for non-urgent, large dataset analyses

**Design Questions**:
- When to recommend batch vs. real-time processing?
- How to handle partial batch failures?
- UI/UX for batch job submission and monitoring?

**References**:
- [Anthropic Batch Processing Docs](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing)

**Complexity Estimate**: Medium (3-4 story points)

---

## Success Metrics

### Phase 1 (RAG Infrastructure):
- **Coverage**: >90% of genes have functional annotations (UniProt + knowledge base)
- **Relevance**: Top-5 retrieved snippets are relevant to final pathway identification (manual review)
- **Automation**: Pathway knowledge files generated in <5 minutes
- **Freshness**: PubMed-based knowledge includes recent publications (last 2 years)

### Phase 2 (CoT Reasoning):
- **Traceability**: 100% of analyses have structured reasoning traces
- **Quality**: Validation catches ≥50% of analytical errors (measured on validation set with known issues)
- **Consistency**: Multi-pass validation reduces inter-run variability by ≥30%

### Phase 3 (Prompt Features):
- **Integration**: Sections 7-9 functional for all clusters with required data
- **Specificity**: Experiment suggestions are lab-appropriate (100% match available techniques)
- **Actionability**: Suggested experiments are implementable without additional clarification

### Phase 4 (Refinements):
- **Documentation**: Users can complete tutorials without external help
- **Test Coverage**: >80% code coverage with unit/integration tests
- **Quality Metrics**: RAG effectiveness metrics tracked and reportable

### Phase 5 (Optimizations):
- **Cost Reduction**: Prompt caching reduces API costs by ≥30%
- **Throughput**: Batch processing handles 1000+ clusters efficiently
- **Quality**: Extended thinking improves analysis quality on complex clusters

---

## Dependency Graph

```
Phase 1 (RAG):
  1.1 (UniProt) → 1.3 (Enhanced Retrieval)
  1.1 (UniProt) → 1.2 (PubMed Knowledge)
  
Phase 2 (CoT):
  2.1 (CoT Restructure) → 2.2 (Reasoning Trace)
  2.2 (Reasoning Trace) → 2.3 (Validation)
  
Phase 3 (Prompts):
  1.1 (UniProt) → 3.2 (Feature Directionality)
  3.1 (Phenotypic Strength) → 3.2 (Feature Directionality)
  3.1, 3.2 → 3.3 (Experiment Suggestions)
  3.4 (Lab Capabilities) → 3.3 (Experiment Suggestions)
  
Phase 4 & 5:
  All previous phases → Documentation
  All previous phases → Testing
  Claude features independent but benefit from completed core system
```

---

## Open Questions for Design Team

These high-level questions should be discussed before starting implementation:

1. **RAG Architecture Philosophy**: Should we maintain lightweight keyword-based retrieval (Design Principle 5) or invest in semantic search for better accuracy? What's the right balance?

2. **CoT vs. Extended Thinking**: Are Claude's built-in extended thinking capabilities sufficient, or do we need custom CoT? Can we use both?

3. **Validation Aggressiveness**: How much validation is too much? Balance between catching errors and overwhelming users with caveats?

4. **Multi-model Strategy**: Should we support running analysis with multiple models and ensembling results? Or is single-model sufficient?

5. **Prompt Versioning**: As prompts evolve, how do we ensure reproducibility? Should we version prompts and track which version was used?

6. **Human-in-the-Loop**: Where should we add human review checkpoints? After pathway identification? After validation? At the end?

7. **Benchmarking Strategy**: How do we measure if improvements actually help? Do we need a gold-standard validation set? Manual review process?

---

## Next Steps

1. **Phase 1 Priority**: Start with Task 1.1 (UniProt) as foundation for other RAG improvements
2. **Design Decisions**: Resolve open questions for Tasks 1.2 and 1.3 before implementation
3. **Parallel Development**: Once Task 1.1 is stable, can begin Phase 2 (CoT) in parallel
4. **Iterative Validation**: Test each component on real datasets before moving to next phase
5. **Documentation as You Go**: Update docs with each completed task

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-01-30 | Roadmap Team | Initial roadmap with Phases 1-5, task assignments |

---

## Appendix: Related Resources

- [Design Document](https://docs.google.com/document/d/1xTFxuFIu_-kz4m96E-u110yPIcWOOwYvaN_1AOODpzk/edit)
- [Claude API Documentation](https://docs.anthropic.com/)
- [UniProt API Documentation](https://www.uniprot.org/help/programmatic_access)
- [NCBI E-utilities Documentation](https://www.ncbi.nlm.nih.gov/books/NBK25501/)
- [MozzareLLM Reference Papers](https://drive.google.com/drive/folders/1occhSlJ4PPSI2vAncscPq6ub_YXxyRm3)
