# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-LLM-Schema is an end-to-end system for verifiable synthesis of multiple, disparate Multi-LLM consensus results. It performs advanced security log analysis by orchestrating multiple LLM providers (OpenAI, Gemini, Claude) and using sophisticated consensus algorithms to generate unified security schemas.

## Development Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure API keys
# Create api_key.txt with the following format:
# OPENAI_API_KEY = "your_key_here"
# GEMINI_API_KEY = "your_key_here"
# ANTHROPIC_API_KEY = "your_key_here"
```

### Running the Pipeline
```bash
# Run the complete V3 pipeline
python V3_pipeline.py --input sample_logs.txt --output output_v3

# Run with custom config
python V3_pipeline.py --input sample_logs.txt --output output_v3 --config config.json
```

### Testing
```bash
# Run integration tests
python test_v3_pipeline_integration.py

# Run specific stage tests
python test_v3_pipeline_stage2.py  # Test stage 2 consensus
python test_v3_pipeline_stage5.py  # Test stage 5 consensus
python test_consensus_orchestratorv2.py  # Test consensus orchestrator
```

## Architecture

### V3 Pipeline (5-Stage Architecture)

The system implements a 5-stage pipeline in `V3_pipeline.py`:

1. **Stage 1 (Prompt1)**: Field extraction using multiple LLM providers
   - Executes across OpenAI, Gemini, Claude (as configured in `MODEL_CONFIG`)
   - Uses `FewShotOrchestrator` to manage LLM calls
   - Outputs: Individual JSON files per provider in `output_v3/stage1_prompt1/`

2. **Stage 2 (SEMANTIC_CORE Consensus)**: JSON1 consensus
   - Uses `SemanticBFTConsensusProcessor` with research mode enabled
   - Applies SBERT embeddings, semantic clustering, Tree Edit Distance (TED)
   - Creates unified consensus from multiple provider outputs
   - Outputs: `output_v3/stage2_semantic_core/json1_consensus.json`

3. **Stage 3 (Prompt2)**: Security analysis report generation
   - Takes JSON1 consensus and generates detailed security reports
   - Uses search-enabled providers for threat intelligence
   - Outputs: Markdown reports per provider in `output_v3/stage3_prompt2/`

4. **Stage 4 (Prompt3)**: Security pattern detection
   - Converts reports to structured JSON2 security patterns
   - Extracts attack patterns, IOCs, vulnerabilities, detection rules
   - Outputs: JSON files in `output_v3/stage4_prompt3/`

5. **Stage 5 (OPTIMIZED_CONSENSUS)**: Final unified schema
   - Uses `SemanticBFTConsensusProcessor` with optimized configuration
   - Applies architectural fixes for knowledge preservation
   - Creates final unified security schema
   - Outputs: `output_v3/stage5_optimized_consensus/final_unified_schema.json`

### Key Components

**LLM Providers** (`providers/`):
- `openai_provider.py`: OpenAI integration with search capabilities
- `gemini_provider.py`: Google Gemini integration with search
- `claude_provider.py`: Anthropic Claude integration with search
- Each provider has base and search-enabled variants

**Consensus Tools** (`tools/`):
- `consensus_orchestratorv2.py`: Primary consensus orchestrator with architectural fixes
- `universal_consensus_engine.py`: Multi-algorithm consensus engine
- `bft_consensus.py`: Byzantine Fault Tolerant consensus
- `semantic_similarity.py`: Semantic similarity calculations using SBERT
- `semantic_tree_edit_distance.py`: Tree edit distance for JSON structure comparison
- `graph_clustering.py`: Semantic clustering algorithms
- `dempster_shafer.py`: Evidence combination theory
- `mcts_optimization.py`: Monte Carlo Tree Search optimization
- `weighted_voting_reliability.py`: Weighted voting based on reliability scores
- `ice_loop_refinement.py`: Iterative Confidence Enhancement
- `muse_llm_adaptation.py`: Multi-Use Semantic Ensemble adaptation

**Core Modules**:
- `V3_pipeline.py`: Main pipeline orchestrator
- `semantic_bft_consensus.py`: Comprehensive consensus processor
- `few_shot_orchestrator.py`: LLM orchestration for multi-stage prompts
- `preprocessing.py`: Log preprocessing and cleaning
- `config.py`: Configuration including `PROMPT_TEMPLATES`, `MODEL_CONFIG`, temperature settings

**Metrics** (`metrics/`):
- `consensus_metrics.py`: Consensus quality measurement
- `uncertainty_metrics.py`: Uncertainty quantification
- `quality_metrics.py`: Output quality assessment
- `performance_metrics.py`: Performance tracking
- `cost_efficiency_tracker.py`: API cost tracking

### Configuration System

**Model Configuration** (`config.py` - `MODEL_CONFIG`):
```python
MODEL_CONFIG = {
    "prompt1": {
        "openai": "o4-mini",
        "gemini": "gemini-2.5-pro",
        "gemini_flash": "gemini-2.5-flash",
    },
    "prompt2": {
        "claude": "claude-sonnet-4-20250514",
        "openai": "o4-mini",
        "gemini": "gemini-2.5-pro"
    },
    "prompt3": {
        "gemini": "gemini-2.5-pro"
    }
}
```

**Consensus Configuration**:
- SEMANTIC_CORE (Stage 2): Field-level consensus with comprehensive algorithms
- OPTIMIZED_CONSENSUS (Stage 5): Security pattern consensus with knowledge preservation
- Both use `SemanticBFTConsensusProcessor` with different hyperparameters

**Prompt Templates** (`config.py` - `PROMPT_TEMPLATES`):
- Prompt 1: Security-aware log schema architect (field extraction)
- Prompt 2: Cyber threat analyst with search (report generation)
- Prompt 3: Security analyst AI (structured pattern detection)

### Data Flow

```
Raw Log → Preprocessing → Stage 1 (Prompt1) → Multiple JSON files
                                                      ↓
                                              Stage 2 (SEMANTIC_CORE)
                                                      ↓
                                              JSON1 Consensus
                                                      ↓
                                              Stage 3 (Prompt2) → Multiple Reports
                                                      ↓
                                              Stage 4 (Prompt3) → Multiple JSON2 files
                                                      ↓
                                              Stage 5 (OPTIMIZED_CONSENSUS)
                                                      ↓
                                              Final Unified Schema
```

### Important Implementation Details

**API Key Management**:
- Keys loaded from `api_key.txt` using regex pattern matching
- Fallback to environment variables (`OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`)
- Handled in `V3_pipeline.py::get_api_keys()`

**Consensus Algorithm Selection**:
- Research mode (`enable_research_mode=True`) activates `ConsensusOrchestratorV2`
- Consensus strength can be 'basic', 'moderate', or 'comprehensive'
- Stage 2 uses comprehensive consensus for field extraction
- Stage 5 uses comprehensive consensus with architectural fixes

**_original_content Wrapper Fix**:
- Stage 5 applies `_fix_original_content_wrapper()` to handle nested consensus structures
- Converts wrapped content to proper list/dict formats
- Adds consensus metadata (votes, sources) to merged results

**Output Directory Structure**:
```
output_v3/
├── stage1_prompt1/           # Individual LLM outputs (JSON)
├── stage2_semantic_core/     # JSON1 consensus results
├── stage3_prompt2/           # Security analysis reports (Markdown)
├── stage4_prompt3/           # Structured security patterns (JSON)
└── stage5_optimized_consensus/ # Final unified schema
```

## Common Workflows

### Adding a New LLM Provider
1. Create provider class in `providers/` inheriting from base provider
2. Implement `generate()` and search-enabled variant
3. Update `MODEL_CONFIG` in `config.py`
4. Add provider initialization in `FewShotOrchestrator._create_providers()`

### Modifying Consensus Algorithms
1. Core consensus logic is in `tools/consensus_orchestratorv2.py`
2. Individual algorithms are in separate `tools/` files
3. Configuration is managed through `SemanticBFTConsensusProcessor.config`
4. Enable/disable algorithms via config flags (e.g., `use_bft_consensus`, `use_mcts_optimization`)

### Adjusting Prompt Templates
1. Edit `PROMPT_TEMPLATES` in `config.py`
2. Templates use Python string formatting with `{variable}` placeholders
3. Test changes with integration tests before production use

### Debugging Pipeline Failures
1. Check logs for stage-specific errors
2. Examine intermediate outputs in `output_v3/stage*` directories
3. Use individual stage tests (`test_v3_pipeline_stage*.py`)
4. Verify API keys are correctly configured
5. Check provider-specific errors in `few_shot_orchestrator.py`

## Critical Configuration Notes

- **Temperature Settings**: Configured per use case in `config.py` (DEFAULT_TEMPERATURE=0.7, CRITIC_TEMPERATURE=0.3)
- **Token Limits**: Set per provider to avoid API limits (OPENAI_MAX_TOKENS=10000, ANTHROPIC_MAX_TOKENS=8000, GEMINI_MAX_TOKENS=65535)
- **Consensus Thresholds**: `similarity_threshold` and `consensus_threshold` affect how strict consensus is (lower = more inclusive)
- **Preservation Ratio**: Controls how much content is kept from original JSONs (0.8 = keep 80%)
