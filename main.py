#!/usr/bin/env python3
"""
Main entry point for the ICD-10 Extraction System

This script provides a command-line interface for extracting ICD-10 codes
from clinical text using multiple AI agents.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from icd10_extractor import ICD10ExtractionSystem
from icd10_extractor.utils import load_patient_note, ensure_directory_exists

from dotenv import load_dotenv

load_dotenv()



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Extract ICD-10 codes from clinical text")
    parser.add_argument(
        "--input", "-i", 
        type=str, 
        help="Path to clinical text file (default: data/clinical_text_3.txt)"
    )
    parser.add_argument(
        "--output", "-o", 
        type=str, 
        default="output/extraction_results.csv",
        help="Output CSV file path (default: output/extraction_results.csv)"
    )
    parser.add_argument(
        "--icd10-data", 
        type=str, 
        default="data/icd10_2019.csv",
        help="Path to ICD-10 CSV data file (default: data/icd10_2019.csv)"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Setup OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("Please set the OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    # Determine input file
    input_file = args.input or "data/clinical_text_3.txt"
    
    # Ensure output directory exists
    ensure_directory_exists(os.path.dirname(args.output))
    
    try:
        # Initialize the system
        logger.info("Initializing ICD-10 extraction system...")
        system = ICD10ExtractionSystem(
            openai_api_key=openai_api_key,
            icd10_csv_path=args.icd10_data
        )
        
        # Load patient note
        logger.info(f"Loading clinical text from: {input_file}")
        patient_note = load_patient_note(input_file)
        logger.info(f"Loaded clinical text ({len(patient_note)} characters)")
        
        # Extract ICD-10 codes
        logger.info("Starting ICD-10 extraction process...")
        results = system.extract_icd10_codes(patient_note)
        
        # Print results summary
        logger.info("Extraction complete. Results summary:")
        print("\n" + "="*60)
        print("EXTRACTION RESULTS SUMMARY")
        print("="*60)
        
        agents = [
            ("GPT-4.1 Results", "gpt4_1_icd10"),
            ("GPT-4o-mini Results", "gpt4o_mini_icd10"), 
            ("GPT-4.1-mini Results", "gpt4_1_mini_icd10"),
            ("RAG Results", "rag_icd10"),
            ("BM25 Results", "bm25_icd10")
        ]
        
        for agent_name, key in agents:
            codes = results.get(key, [])
            print(f"\n{agent_name}: {len(codes)} codes")
            for i, code_info in enumerate(codes, 1):  # Show all codes
                print(f"  {i}. {code_info.get('code', 'N/A')}: {code_info.get('description', 'N/A')}")
                # Show confidence score if available (for RAG agent)
                if 'confidence' in code_info:
                    print(f"     Confidence: {code_info.get('confidence', 'N/A')}")
            if len(codes) == 0:
                print("  No codes extracted")
        
        # Performance metrics
        print(f"\n{'='*60}")
        print("PERFORMANCE METRICS")
        print("="*60)
        tokens = results.get("tokens_per_node", {})
        retries = results.get("retries_per_node", {})
        
        for agent_name, token_count in tokens.items():
            retry_count = retries.get(agent_name, 0)
            print(f"{agent_name}: {token_count} tokens, {retry_count} retries")
        
        # Export to CSV
        logger.info(f"Exporting results to: {args.output}")
        flattened_results = system.flatten_results_for_csv(results)
        system.export_to_csv(flattened_results, args.output)
        
        print(f"\n✅ Results saved to: {args.output}")
        logger.info("Process completed successfully!")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
