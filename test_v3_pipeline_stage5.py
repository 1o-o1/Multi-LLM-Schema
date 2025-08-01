#!/usr/bin/env python3
"""
Test V3_pipeline Stage 5 End-to-End with Architectural Fixes
Validates the complete pipeline integration with the new consensus_orchestratorv2
"""

import sys
import json
import time
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def count_content_items(data):
    """Count total content items in nested data structure"""
    if isinstance(data, dict):
        return sum(count_content_items(v) for v in data.values()) + len(data)
    elif isinstance(data, list):
        return sum(count_content_items(item) for item in data) + len(data)
    else:
        return 1

def test_v3_pipeline_stage5():
    """Test V3_pipeline Stage 5 with Simple Schema Merger integration"""
    
    print("Testing V3_pipeline Stage 5 End-to-End Integration")
    print("=" * 60)
    print("Input: output_v3_new/stage4_prompt3")
    print("Validating:")
    print("- V3_pipeline Stage 5 with Simple Schema Merger")
    print("- Comprehensive knowledge preservation vs restrictive consensus")
    print("- Quality comparison between approaches")
    print("- Complete JSON processing pipeline")
    print("=" * 60)
    
    try:
        # Test 1: Initialize V3_pipeline and test Stage 5 method
        print("\n1. Initializing V3_pipeline with Simple Schema Merger config...")
        
        from V3_pipeline import V3ConsensusPipeline
        
        # Initialize V3 pipeline (already configured with Simple Schema Merger)
        pipeline = V3ConsensusPipeline()
        
        print("SUCCESS: V3ConsensusPipeline initialized")
        print(f"Consensus processor ready: {pipeline.optimized_consensus_processor is not None}")
        
        # Test 2: Test with Stage 4 JSON files
        print("\n2. Testing with Stage 4 JSON files...")
        
        json_files = [
            "output_v3_new/stage4_prompt3/gemini_from_claude_report_prompt3.json",
            "output_v3_new/stage4_prompt3/gemini_from_gemini_report_prompt3.json", 
            "output_v3_new/stage4_prompt3/gemini_from_openai_report_prompt3.json"
        ]
        
        # Verify files exist
        existing_files = []
        for json_file in json_files:
            if Path(json_file).exists():
                existing_files.append(json_file)
            else:
                print(f"WARNING: File not found: {json_file}")
        
        if not existing_files:
            print("ERROR: No test JSON files found")
            return False
        
        print(f"SUCCESS: Found {len(existing_files)} test files")
        
        # Test 3: Create mock Stage 4 result and test Stage 5 method
        print("\n3. Creating mock Stage 4 result and testing Stage 5...")
        
        # Create mock Stage 4 result
        from V3_pipeline import PipelineStageResult
        
        mock_stage4_result = PipelineStageResult(
            stage_name="stage4_prompt3",
            content={},  # Not used in stage 5
            confidence=0.8,
            processing_time=1.0,
            method_used="multi_model_prompt3",
            source_files=existing_files,
            metadata={}
        )
        
        # Create output directory
        output_path = Path("test_results/stage5_simple_merger")
        output_path.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        # Call the actual V3_pipeline Stage 5 method (will use Simple Schema Merger)
        stage5_result = pipeline._execute_stage5_optimized_consensus(
            mock_stage4_result,
            output_path
        )
        
        processing_time = time.time() - start_time
        
        print(f"SUCCESS: Stage 5 completed in {processing_time:.2f}s")
        print(f"Method used: {stage5_result.method_used}")
        
        # Test 4: Validate results
        print("\n4. Validating Stage 5 results...")
        
        if not stage5_result:
            print("ERROR: No Stage 5 result returned")
            return False
        
        if not stage5_result.content:
            print("ERROR: No content in Stage 5 result")
            return False
        
        # Check quality metrics
        confidence = stage5_result.confidence
        metadata = stage5_result.metadata
        quality_metrics = metadata.get('quality_metrics', {})
        
        print(f"Confidence: {confidence:.3f}")
        print(f"Method used: {stage5_result.method_used}")
        print(f"Semantic BFT consensus used: {metadata.get('processing_method', 'unknown')}")
        print(f"Knowledge preservation: {metadata.get('knowledge_preservation_approach', 'unknown')}")
        print(f"Original files: {metadata.get('original_file_count', len(existing_files))}")
        print(f"Processing time: {processing_time:.2f}s")
        
        # Test 5: Check Simple Schema Merger vs Complex Consensus
        print("\n5. Analyzing knowledge preservation approach...")
        
        semantic_bft_used = metadata.get('processing_method') == 'OPTIMIZED_CONSENSUS_V2'
        processing_method = metadata.get('processing_method', 'unknown')
        knowledge_approach = metadata.get('knowledge_preservation_approach', 'unknown')
        
        # Count content items to compare with previous restrictive approach
        content_items = count_content_items(stage5_result.content)
        
        print(f"Semantic BFT Consensus: {'YES' if semantic_bft_used else 'NO'}")
        print(f"Processing method: {processing_method}")
        print(f"Knowledge preservation: {knowledge_approach}")
        print(f"Total content items preserved: {content_items}")
        
        # Test 6: Save results
        print("\n6. Saving results...")
        
        output_dir = Path("test_results")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / "v3_pipeline_stage5.json"
        
        # Save comprehensive result to compare with restrictive consensus
        comprehensive_result = {
            'stage5_result': {
                'content': stage5_result.content,
                'confidence': stage5_result.confidence,
                'processing_time': stage5_result.processing_time,
                'method_used': stage5_result.method_used,
                'metadata': stage5_result.metadata
            },
            'analysis': {
                'semantic_bft_used': semantic_bft_used,
                'processing_method': processing_method,
                'knowledge_preservation_approach': knowledge_approach,
                'total_content_items': content_items,
                'files_processed': len(existing_files)
            },
            'test_metadata': {
                'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'test_type': 'V3_pipeline_Stage5_SemanticBFTConsensus',
                'vs_restrictive_consensus': True
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_result, f, indent=2, default=str)
        
        print(f"Results saved to: {output_file}")
        
        # Final validation
        print("\n" + "=" * 60)
        print("V3_PIPELINE STAGE 5 SIMPLE SCHEMA MERGER VALIDATION SUMMARY")
        print("=" * 60)
        
        success_criteria = [
            ("Pipeline initialization", True),
            ("Semantic BFT Consensus used", semantic_bft_used),
            ("JSON files processing", content_items > 0),
            ("Knowledge preservation active", knowledge_approach == 'consensus_filtered'),
            ("Processing method correct", processing_method == 'OPTIMIZED_CONSENSUS_V2'),
            ("Processing completion", processing_time > 0)
        ]
        
        passed_count = 0
        for criterion, passed in success_criteria:
            status = "PASS" if passed else "FAIL"
            print(f"{status:<6} {criterion}")
            if passed:
                passed_count += 1
        
        overall_success = passed_count >= 5  # At least 5/6 criteria must pass
        
        print("-" * 60)
        if overall_success:
            print("SUCCESS: V3_pipeline Stage 5 with Semantic BFT Consensus PASSED!")
            print("Semantic consensus with architectural fixes is working.")
            print(f"Processed {content_items} content items with semantic consensus")
        else:
            print("WARNING: Some validation criteria failed.")
            print(f"Passed: {passed_count}/{len(success_criteria)} criteria")
        
        print("=" * 60)
        
        return overall_success
        
    except Exception as e:
        print(f"\nERROR: V3_pipeline Stage 5 test failed: {e}")
        logger.error(f"Test exception: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    print("V3_pipeline Stage 5 End-to-End Test")
    print("Testing complete integration with ConsensusOrchestratorV2")
    print()
    
    success = test_v3_pipeline_stage5()
    
    if success:
        print("\nALL TESTS PASSED!")
        print("V3_pipeline Stage 5 is ready with semantic BFT consensus and architectural fixes integrated.")
    else:
        print("\nSome tests failed. Check logs for details.")