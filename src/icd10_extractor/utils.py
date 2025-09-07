"""
Utility functions for the ICD-10 extraction system.
"""

import os
from typing import Optional


def load_patient_note(filepath: Optional[str] = None) -> str:
    """Load patient note from file or return default."""
    if filepath and os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    
    # Default synthetic patient note
    return """Patient is a 68-year-old male presenting with a chief complaint of increasing shortness of breath over the past two weeks. Symptoms are worse with exertion and improve slightly with rest. He reports a history of chronic obstructive pulmonary disease (COPD), diagnosed 10 years ago, managed with inhaled bronchodilators as needed. He denies fever, chills, or cough with sputum production. He has a history of hypertension, controlled with lisinopril. No known allergies. Social history includes a 40 pack-year smoking history, quit 5 years ago. He lives with his wife and is retired. Physical examination reveals a thin male in mild respiratory distress. Vital signs: BP 140/85, HR 98, RR 22, Temp 98.6 F, SpO2 90% on room air. Auscultation of the lungs reveals diminished breath sounds bilaterally with scattered expiratory wheezes. Cardiac exam is regular rate and rhythm with no murmurs, rubs, or gallops. Extremities show no edema or clubbing. Assessment: Acute exacerbation of COPD. Plan: Administer nebulized albuterol and ipratropium. Start oral prednisone 40mg daily for 5 days. Obtain a chest X-ray and arterial blood gas. Continue home inhalers. Follow up in 1 week or sooner if symptoms worsen. Educate patient on symptom management and when to seek urgent care."""


def ensure_directory_exists(directory: str) -> None:
    """Ensure a directory exists, creating it if necessary."""
    os.makedirs(directory, exist_ok=True)
