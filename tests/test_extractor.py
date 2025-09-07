"""
Tests for the ICD-10 extraction system.
"""

import pytest
import os
import sys
from pathlib import Path

# Add src to Python path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from icd10_extractor.utils import load_patient_note, ensure_directory_exists
from icd10_extractor.agents import AgentState


class TestUtils:
    """Test utility functions."""
    
    def test_load_patient_note_default(self):
        """Test loading default patient note."""
        note = load_patient_note()
        assert isinstance(note, str)
        assert len(note) > 0
        assert "COPD" in note
    
    def test_load_patient_note_file(self, tmp_path):
        """Test loading patient note from file."""
        test_file = tmp_path / "test_note.txt"
        test_content = "Test patient note content"
        test_file.write_text(test_content)
        
        note = load_patient_note(str(test_file))
        assert note == test_content
    
    def test_load_patient_note_nonexistent_file(self):
        """Test loading from nonexistent file returns default."""
        note = load_patient_note("nonexistent_file.txt")
        assert isinstance(note, str)
        assert len(note) > 0
    
    def test_ensure_directory_exists(self, tmp_path):
        """Test directory creation."""
        test_dir = tmp_path / "test_directory"
        ensure_directory_exists(str(test_dir))
        assert test_dir.exists()
        assert test_dir.is_dir()


class TestAgentState:
    """Test agent state structure."""
    
    def test_agent_state_structure(self):
        """Test that AgentState has required fields."""
        # This is a TypedDict, so we can't instantiate it directly
        # but we can check the annotations
        annotations = AgentState.__annotations__
        
        required_fields = [
            'debug_info', 'patient_note', 'gpt4_1_icd10', 'gpt4o_mini_icd10',
            'gpt4_1_mini_icd10', 'rag_icd10', 'rag_confidence', 'bm25_icd10',
            'retries_per_node', 'tokens_per_node'
        ]
        
        for field in required_fields:
            assert field in annotations


# Integration test (requires API key)
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not available")
class TestIntegration:
    """Integration tests requiring API access."""
    
    def test_system_initialization(self):
        """Test that the system can be initialized."""
        from icd10_extractor import ICD10ExtractionSystem
        
        # This test requires the data files to be present
        if not os.path.exists("data/icd10_2019.csv"):
            pytest.skip("ICD-10 data file not available")
        
        system = ICD10ExtractionSystem(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            icd10_csv_path="data/icd10_2019.csv"
        )
        
        assert system is not None
        assert system.llm is not None
        assert system.embeddings is not None
