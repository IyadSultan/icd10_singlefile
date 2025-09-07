"""
Agent definitions and state management for the ICD-10 extraction system.
"""

from typing import List, TypedDict, Dict


class AgentState(TypedDict):
    """
    Represents the state of our graph.
    
    Attributes:
        debug_info: Information for debugging at different logging levels.
        patient_note: The patient's medical note.
        gpt4_1_icd10: List of ICD-10 codes and descriptions extracted by gpt4.1.
        gpt4o_mini_icd10: List of ICD-10 codes extracted by gpt4o-mini.
        gpt4_1_mini_icd10: List of ICD-10 codes extracted by gpt4.1-mini.
        rag_icd10: List of ICD-10 codes extracted using RAG.
        rag_confidence: Confidence scores for RAG extractions.
        bm25_icd10: List of ICD-10 codes extracted using BM25.
        retries_per_node: Dictionary tracking retries per agent.
        tokens_per_node: Dictionary tracking token usage per agent.
    """
    debug_info: dict
    patient_note: str
    gpt4_1_icd10: List[dict]
    gpt4o_mini_icd10: List[dict]
    gpt4_1_mini_icd10: List[dict]
    rag_icd10: List[dict]
    rag_confidence: List[float]
    bm25_icd10: List[dict]
    retries_per_node: dict
    tokens_per_node: dict
