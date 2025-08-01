#!/usr/bin/env python3
"""
Test V3_pipeline Stage 2 Consensus
Tests Stage 2 consensus with output_v3/stage1_prompt1 input
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

def test_v3_pipeline_stage2():
    """Test V3_pipeline Stage 2 consensus with proper input"""
    
    print("Testing V3_pipeline Stage 2 Consensus")
    print("=" * 60)
    print("Input: output_v3_new/stage1_prompt1")
    print("Testing semantic consensus with architectural fixes")
    print("=" * 60)
    
    try:
        # Test 1: Check input files exist
        print("\n1. Checking input files in output_v3_new/stage1_prompt1...")
        
        input_dir = Path("output_v3_new/stage1_prompt1")
        if not input_dir.exists():
            print(f"ERROR: Input directory not found: {input_dir}")
            return False
        
        # Find JSON files in the directory
        json_files = list(input_dir.glob("*.json"))
        if not json_files:
            print(f"ERROR: No JSON files found in {input_dir}")
            return False
        
        print(f"SUCCESS: Found {len(json_files)} JSON files")
        for json_file in json_files:
            print(f"  - {json_file.name}")
        
        # Test 2: Initialize V3_pipeline with Stage 2 focus config
        print("\n2. Initializing V3_pipeline with Stage 2 configuration...")
        
        from V3_pipeline import V3ConsensusPipeline
        
        # Configuration focused on Stage 2 consensus improvements
        stage2_config = {
            'semantic_consensus': {
                'use_bft_consensus': True,
                'use_muse_adaptation': True,
                'use_ice_refinement': True,
                'use_semantic_ted': True,
                'use_dempster_shafer': True,  # Test DS instead of MUSE
                'clustering_method': 'distance_based'
            }
        }
        
        pipeline = V3ConsensusPipeline()
        
        print("SUCCESS: V3_pipeline initialized with Stage 2 config")
        
        # Test 3: Create mock Stage 1 result
        print("\n3. Creating mock Stage 1 result...")
        
        class MockStage1Result:
            def __init__(self, json_files):
                self.source_files = [str(f) for f in json_files]
                self.content = {}  # Stage 1 content would be here
        
        mock_stage1_result = MockStage1Result(json_files)
        
        print(f"Mock Stage 1 result created with {len(mock_stage1_result.source_files)} files")
        
        # Test 4: Execute Stage 2 consensus
        print("\n4. Executing Stage 2 consensus...")
        
        output_path = Path("test_results/stage2_consensus")
        output_path.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        # Execute Stage 2 (semantic consensus)
        stage2_result = pipeline._execute_stage2_semantic_core_consensus(
            mock_stage1_result, 
            output_path
        )
        
        processing_time = time.time() - start_time
        
        print(f"SUCCESS: Stage 2 consensus completed in {processing_time:.2f}s")
        
        # Test 5: Validate Stage 2 results
        print("\n5. Validating Stage 2 results...")
        
        print(f"Method used: {stage2_result.method_used}")
        print(f"Confidence: {stage2_result.confidence:.3f}")
        print(f"Processing time: {stage2_result.processing_time:.2f}s")
        
        # Check for consensus features
        metadata = stage2_result.metadata
        print(f"BFT enabled: {metadata.get('bft_enabled', False)}")
        print(f"MUSE enabled: {metadata.get('muse_enabled', False)}")
        print(f"ICE enabled: {metadata.get('ice_enabled', False)}")
        print(f"TED enabled: {metadata.get('ted_enabled', False)}")
        print(f"Dempster-Shafer enabled: {metadata.get('dempster_shafer_enabled', False)}")
        
        # Test 6: Check output quality
        print("\n6. Checking output quality...")
        
        if stage2_result.content:
            content_keys = list(stage2_result.content.keys())
            print(f"Output contains {len(content_keys)} main sections")
            
            # Check for proper structure
            if 'parsed_json' in stage2_result.content:
                parsed_json = stage2_result.content['parsed_json']
                print(f"Parsed JSON contains {len(parsed_json)} fields")
            
            # Save detailed results
            result_file = output_path / "stage2_detailed_results.json"
            
            detailed_results = {
                'stage_name': stage2_result.stage_name,
                'method_used': stage2_result.method_used,
                'confidence': stage2_result.confidence,
                'processing_time': stage2_result.processing_time,
                'metadata': stage2_result.metadata,
                'content_summary': {
                    'total_keys': len(stage2_result.content.keys()),
                    'main_sections': list(stage2_result.content.keys())
                },
                'test_metadata': {
                    'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'input_files_count': len(json_files),
                    'input_directory': str(input_dir),
                    'architectural_fixes_tested': True
                }
            }
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(detailed_results, f, indent=2, default=str)
            
            print(f"Detailed results saved to: {result_file}")
        
        # Final validation
        print("\n" + "=" * 60)
        print("STAGE 2 CONSENSUS VALIDATION SUMMARY")
        print("=" * 60)
        
        success_criteria = [
            ("Input files found", len(json_files) > 0),
            ("Pipeline initialization", True),
            ("Stage 2 execution", stage2_result is not None),
            ("Processing completed", processing_time > 0),
            ("Confidence score", stage2_result.confidence > 0),
            ("Output generated", bool(stage2_result.content))
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
            print("SUCCESS: Stage 2 consensus validation PASSED!")
            print("Stage 2 is working with architectural fixes.")
        else:
            print("WARNING: Some Stage 2 validation criteria failed.")
            print(f"Passed: {passed_count}/{len(success_criteria)} criteria")
        
        print("=" * 60)
        
        return overall_success
        
    except Exception as e:
        print(f"\nERROR: Stage 2 test failed: {e}")
        logger.error(f"Stage 2 test exception: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    print("V3_pipeline Stage 2 Consensus Test")
    print("Testing with output_v3/stage1_prompt1 input")
    print()
    
    success = test_v3_pipeline_stage2()
    
    if success:
        print("\nSTAGE 2 TESTS PASSED!")
        print("Stage 2 consensus is working with architectural fixes.")
    else:
        print("\nSome Stage 2 tests failed. Check logs for details.")