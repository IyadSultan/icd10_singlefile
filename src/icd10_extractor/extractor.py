"""
Main ICD-10 extraction system using multi-agent approach.
"""

import os
import logging
import time
import json
import ast
import pandas as pd
from typing import Dict, Any

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# LangGraph imports
from langgraph.graph import StateGraph, END

from .agents import AgentState
from .retrievers import RetrieverManager

logger = logging.getLogger(__name__)


class ICD10ExtractionSystem:
    """Main class for the multi-agent ICD-10 extraction system."""
    
    def __init__(self, openai_api_key: str, icd10_csv_path: str = "data/icd10_2019.csv"):
        """
        Initialize the ICD-10 extraction system.
        
        Args:
            openai_api_key: OpenAI API key
            icd10_csv_path: Path to the ICD-10 CSV file
        """
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.embeddings = OpenAIEmbeddings()
        # LLM instances will be created per agent
        self.icd10_csv_path = icd10_csv_path
        
        # Initialize components
        self.retriever_manager = RetrieverManager(icd10_csv_path, self.embeddings)
        self.workflow = None
        self.app = None
        
        # Setup databases and retrievers
        self._setup_retrievers()
        self._setup_agent_parameters()
        self._build_graph()
    
    def _setup_retrievers(self):
        """Setup FAISS and BM25 retrievers."""
        self.retriever_manager.setup_faiss_database()
        self.retriever_manager.setup_bm25_retriever()
    
    def _setup_agent_parameters(self):
        """Setup parameters for all agents."""
        base_system_prompt = """You are an expert medical coder. Your task is to extract all relevant ICD-10 codes and their descriptions from the provided patient note.

Present the extracted information as a list of dictionaries, where each dictionary has 'code' and 'description' keys."""
        
        self.agent_params = {
            "gpt4_1_icd10": {
                "model": "gpt-4.1",  # Actual GPT-4.1 model
                "system_prompt": base_system_prompt,
                "tools": [],
                "temperature": 0.1,
                "role": "ICD-10 Extractor (GPT-4.1)",
                "debug_logging": "info",
            },
            "gpt4_1_mini_icd10": {
                "model": "gpt-4.1-mini",  # Actual GPT-4.1-mini model
                "system_prompt": base_system_prompt,
                "tools": [],
                "temperature": 0.1,
                "role": "ICD-10 Extractor (GPT-4.1-mini)",
                "debug_logging": "info",
            },
            "gpt4omini_icd10": {
                "model": "gpt-4o-mini",  # GPT-4o-mini model
                "system_prompt": base_system_prompt,
                "tools": [],
                "temperature": 0.1,
                "role": "ICD-10 Extractor (GPT-4o-mini)",
                "debug_logging": "info",
            },
            "rag_icd10": {
                "model": "gpt-4o-mini",
                "system_prompt": """You are an expert medical coder assisting with ICD-10 code extraction.
Use the provided context from the ICD-10 database to identify the most relevant codes and descriptions for the patient note.
The note may have multiple problems so you need to present each problem with its context to RAG to retrieve the codes.
Present all medical problems even if you think they are trivial.
Present the extracted information as a list of dictionaries, where each dictionary has 'code' and 'description' keys, and include a confidence score for each extraction based on the relevance of the retrieved information.""",
                "tools": [],
                "temperature": 0.1,
                "role": "ICD-10 Extractor (RAG)",
                "debug_logging": "info",
                "retriever": self.retriever_manager.retriever,
            },
            "bm25_icd10": {
                "model": "gpt-4o-mini",
                "system_prompt": """You are an expert medical coder assisting with ICD-10 code extraction. Use the provided context from the BM25 retriever to identify the most relevant codes and descriptions for the patient note.

Present the extracted information as a list of dictionaries, where each dictionary has 'code' and 'description' keys.""",
                "tools": [],
                "temperature": 0.1,
                "role": "ICD-10 Extractor (BM25)",
                "debug_logging": "info",
                "retriever": self.retriever_manager.bm25_retriever,
            }
        }
    
    def _create_agent_node(self, agent_params: dict):
        """
        Creates a LangGraph node function from agent parameters.
        
        Args:
            agent_params: A dictionary containing agent configuration
            
        Returns:
            A function that acts as a LangGraph node
        """
        model_name = agent_params.get("model", "gpt-4o-mini")
        system_prompt_template = agent_params.get("system_prompt", "")
        tools = agent_params.get("tools", [])
        temperature = agent_params.get("temperature", 0.1)
        role = agent_params.get("role", "Unnamed Agent")
        debug_logging_level = agent_params.get("debug_logging", "info").upper()
        retriever = agent_params.get("retriever", None)
        retriever_k = agent_params.get("retriever_k", 10)
        
        agent_logger = logging.getLogger(role)
        agent_logger.setLevel(getattr(logging, debug_logging_level, logging.INFO))
        
        # Add JSON instruction
        system_prompt_template_json = (
            system_prompt_template + 
            "\n\nReturn ONLY a valid JSON list of dictionaries, like this: "
            "[{{'code': 'ABC.123', 'description': 'Example Description'}}]. "
            "Do not include any other text or formatting."
        )
        
        def agent_node(state: AgentState) -> dict:
            """LangGraph node function for an agent."""
            agent_logger.info(f"---Executing {role}---")
            start_time = time.time()
            current_state = state.copy()
            state_updates = {}
            
            # Update retry count
            retries = current_state.get("retries_per_node", {}).get(role, 0) + 1
            retries_per_node_update = current_state.get("retries_per_node", {}).copy()
            retries_per_node_update[role] = retries
            state_updates["retries_per_node"] = retries_per_node_update
            
            try:
                context_for_formatting = ""
                if retriever:
                    try:
                        retrieved_docs = retriever.invoke(current_state["patient_note"])
                        retrieved_docs = retrieved_docs[:retriever_k]
                        context_for_formatting = "\n".join([doc.page_content for doc in retrieved_docs])
                        agent_logger.debug(f"Retrieved {len(retrieved_docs)} documents for {role}.")
                    except Exception as retrieve_error:
                        agent_logger.warning(f"Could not retrieve docs for {role}: {retrieve_error}")
                        context_for_formatting = "Retrieval failed."
                
                # Construct prompt
                input_variables = ["patient_note"]
                if retriever:
                    input_variables.append("context")
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt_template_json),
                    ("human", "{patient_note}" + ("\nContext: {context}" if retriever else "")),
                ])
                
                # Prepare input
                chain_input = {"patient_note": current_state["patient_note"]}
                if retriever:
                    chain_input["context"] = context_for_formatting
                
                # Create LLM instance for this specific agent
                agent_logger.info(f"Creating LLM with model: {model_name}, temperature: {temperature}")
                agent_llm = ChatOpenAI(model=model_name, temperature=temperature)
                
                # Build and invoke chain
                chain = prompt | agent_llm
                response = chain.invoke(chain_input)
                
                # Estimate token usage
                formatted_prompt = prompt.format(**chain_input)
                input_tokens = len(formatted_prompt.split())
                output_tokens = len(response.content.split())
                total_tokens = input_tokens + output_tokens
                
                tokens_per_node_update = current_state.get("tokens_per_node", {}).copy()
                tokens_per_node_update[role] = tokens_per_node_update.get(role, 0) + total_tokens
                state_updates["tokens_per_node"] = tokens_per_node_update
                
                # Parse response
                try:
                    extracted_data = self._safe_parse_response(response.content)
                    
                    if not isinstance(extracted_data, list):
                        raise ValueError("Expected a JSON list.")
                    for item in extracted_data:
                        if not isinstance(item, dict) or 'code' not in item or 'description' not in item:
                            raise ValueError("Expected list of dictionaries with 'code' and 'description'.")
                    
                    # Update state based on agent role
                    if role == "ICD-10 Extractor (GPT-4.1)":
                        state_updates["gpt4_1_icd10"] = extracted_data
                    elif role == "ICD-10 Extractor (GPT-4.1-mini)":
                        state_updates["gpt4_1_mini_icd10"] = extracted_data
                    elif role == "ICD-10 Extractor (GPT-4o-mini)":
                        state_updates["gpt4o_mini_icd10"] = extracted_data
                    elif role == "ICD-10 Extractor (RAG)":
                        state_updates["rag_icd10"] = extracted_data
                        # Extract confidence scores if available
                        if extracted_data and isinstance(extracted_data[0], dict) and 'confidence' in extracted_data[0]:
                            state_updates["rag_confidence"] = [item.get('confidence', 1.0) for item in extracted_data]
                        else:
                            state_updates["rag_confidence"] = [1.0] * len(extracted_data)
                    elif role == "ICD-10 Extractor (BM25)":
                        state_updates["bm25_icd10"] = extracted_data
                    
                    agent_logger.info(f"Successfully extracted data for {role}.")
                    
                except Exception as parse_error:
                    agent_logger.error(f"Error parsing response for {role}: {parse_error}")
                    state_updates["debug_info"] = {
                        "level": "error", 
                        "message": f"Parsing error in {role}: {parse_error}"
                    }
                    
            except Exception as e:
                agent_logger.error(f"Error executing {role}: {e}")
                state_updates["debug_info"] = {
                    "level": "error", 
                    "message": f"Execution error in {role}: {e}"
                }
            
            end_time = time.time()
            duration = end_time - start_time
            agent_logger.info(f"---Finished {role} in {duration:.2f} seconds---")
            return state_updates
        
        return agent_node
    
    def _safe_parse_response(self, content: str):
        """Parse response handling both JSON and Python dict syntax."""
        content = content.strip()
        
        # Method 1: Try ast.literal_eval (handles Python dict syntax with single quotes)
        try:
            if content.startswith('[') and content.endswith(']'):
                return ast.literal_eval(content)
        except (ValueError, SyntaxError):
            pass
        
        # Method 2: Try direct JSON parsing
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # Method 3: Convert single quotes to double quotes and retry
        try:
            corrected = content.replace("'", '"')
            return json.loads(corrected)
        except json.JSONDecodeError:
            pass
        
        # If all fails, raise an error
        raise ValueError(f"Could not parse response: {content[:100]}...")
    
    def _start_node(self, state: AgentState) -> dict:
        """The starting node of the graph."""
        logger.info("---START NODE---")
        return {
            "patient_note": state.get("patient_note", ""),
            "retries_per_node": state.get("retries_per_node", {}),
            "tokens_per_node": state.get("tokens_per_node", {})
        }
    
    def _build_graph(self):
        """Build and compile the LangGraph workflow."""
        # Initialize the StateGraph
        self.workflow = StateGraph(AgentState)
        
        # Create agent nodes
        gpt4_1_icd10_node = self._create_agent_node(self.agent_params["gpt4_1_icd10"])
        gpt4_1_mini_icd10_node = self._create_agent_node(self.agent_params["gpt4_1_mini_icd10"])
        gpt4omini_icd10_node = self._create_agent_node(self.agent_params["gpt4omini_icd10"])
        rag_icd10_node = self._create_agent_node(self.agent_params["rag_icd10"])
        bm25_icd10_node = self._create_agent_node(self.agent_params["bm25_icd10"])
        
        # Add nodes to workflow
        self.workflow.add_node("start", self._start_node)
        self.workflow.add_node("gpt4_1_icd10", gpt4_1_icd10_node)
        self.workflow.add_node("gpt4_1_mini_icd10", gpt4_1_mini_icd10_node)
        self.workflow.add_node("gpt4omini_icd10", gpt4omini_icd10_node)
        self.workflow.add_node("rag_icd10", rag_icd10_node)
        self.workflow.add_node("bm25_icd10", bm25_icd10_node)
        
        # Set entry point
        self.workflow.set_entry_point("start")
        
        # Define edges
        self.workflow.add_edge("start", "gpt4_1_icd10")
        self.workflow.add_edge("gpt4_1_icd10", "gpt4_1_mini_icd10")
        self.workflow.add_edge("gpt4_1_mini_icd10", "gpt4omini_icd10")
        self.workflow.add_edge("gpt4omini_icd10", "rag_icd10")
        self.workflow.add_edge("rag_icd10", "bm25_icd10")
        self.workflow.add_edge("bm25_icd10", END)
        
        # Compile workflow
        self.app = self.workflow.compile()
        logger.info("LangGraph workflow built and compiled successfully.")
    
    def extract_icd10_codes(self, patient_note: str) -> dict:
        """
        Extract ICD-10 codes from a patient note using all agents.
        
        Args:
            patient_note: The patient note text
            
        Returns:
            Dictionary containing the final state with all extracted codes
        """
        initial_state = {
            "patient_note": patient_note,
            "retries_per_node": {},
            "tokens_per_node": {},
            "debug_info": {},
            "gpt4_1_icd10": [],
            "gpt4o_mini_icd10": [],
            "gpt4_1_mini_icd10": [],
            "rag_icd10": [],
            "rag_confidence": [],
            "bm25_icd10": [],
        }
        
        final_output = self.app.invoke(initial_state)
        return final_output
    
    def flatten_results_for_csv(self, graph_output: dict) -> dict:
        """Flatten graph output for CSV export."""
        flattened_data = {}
        
        # Add performance metrics
        retries = graph_output.get("retries_per_node", {})
        tokens = graph_output.get("tokens_per_node", {})
        for agent_name in set(retries.keys()) | set(tokens.keys()):
            flattened_data[f"{agent_name}_retries"] = retries.get(agent_name, 0)
            flattened_data[f"{agent_name}_tokens"] = tokens.get(agent_name, 0)
        
        # Add extracted codes
        agent_code_fields = {
            "gpt4_1_icd10": "gpt4_1_icd10",
            "gpt4o_mini_icd10": "gpt4o_mini_icd10",
            "gpt4_1_mini_icd10": "gpt4_1_mini_icd10",
            "rag_icd10": "rag_icd10",
            "bm25_icd10": "bm25_icd10",
        }
        
        for agent_key, state_key in agent_code_fields.items():
            codes_list = graph_output.get(state_key, [])
            codes_str = "; ".join([item.get('code', '') for item in codes_list])
            descriptions_str = "; ".join([item.get('description', '') for item in codes_list])
            flattened_data[f"{agent_key}_codes"] = codes_str
            flattened_data[f"{agent_key}_descriptions"] = descriptions_str
        
        # Add RAG confidence scores
        rag_confidence_list = graph_output.get("rag_confidence", [])
        confidence_str = "; ".join(map(str, rag_confidence_list))
        flattened_data["rag_confidence_scores"] = confidence_str
        
        return flattened_data
    
    def export_to_csv(self, data: dict, filename: str):
        """Export results to CSV file."""
        df = pd.DataFrame([data])
        if not os.path.exists(filename):
            df.to_csv(filename, index=False, mode='w')
            logger.info(f"Created and wrote to new CSV file: {filename}")
        else:
            df.to_csv(filename, index=False, mode='a', header=False)
            logger.info(f"Appended data to existing CSV file: {filename}")
