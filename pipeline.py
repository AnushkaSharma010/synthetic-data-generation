from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Optional
from models.schemas import (
    GenerationRequest, 
    GeneratedData,
    Schema,
)
from agents.FieldInferenceAgent import FieldInferenceAgent
from agents.DataGeneratorAgent import DataGeneratorAgent
from agents.OutputFormatterAgent import OutputFormatterAgent
from agents.PreprocessingAgent import PreprocessingAgent

import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class AgentState(TypedDict):
    request: GenerationRequest
    cleaned_scenario: Annotated[Optional[str], "Cleaned input"]
    schema: Annotated[Optional[Schema], "Inferred schema"]
    # validated_schema: Annotated[Optional[ValidatedSchema], "Approved schema"]
    generated_data: Annotated[Optional[list], "Generated data"]
    output: Annotated[Optional[GeneratedData], "Formatted output"]
    error: Annotated[Optional[str], "Error message if any"]

def create_pipeline() -> StateGraph:
    logger.info("Initializing pipeline agents...")
    try:
        agents = {
            "preprocessor": PreprocessingAgent(),
            "field_inferrer": FieldInferenceAgent(),
            # "human_approver": HumanInteractionAgent(),
            "data_generator": DataGeneratorAgent(),
            "output_formatter": OutputFormatterAgent()
        }
    except Exception as e:
        logger.exception("Agent initialization failed")
        raise RuntimeError("Failed to initialize agents") from e

    workflow = StateGraph(AgentState)

    def preprocess(state: AgentState) -> AgentState:
        try:
            logger.info("Step: Preprocessing input scenario...")
            cleaned = agents["preprocessor"].clean(state["request"].scenario)
            logger.debug(f"Cleaned Scenario: {cleaned}")
            return {"cleaned_scenario": cleaned, "error": None}
        except Exception as e:
            logger.exception("Preprocessing failed")
            return {"error": f"Preprocessing error: {e}"}

    def infer_fields(state: AgentState) -> AgentState:
        if state.get("error"):
           return state
        try:
            logger.info("Step: Inferring schema from cleaned scenario...")
            # Pass sample_size from user request here!
            schema = agents["field_inferrer"].infer_schema(
                scenario=state["cleaned_scenario"],
                sample_size=state["request"].sample_size # <-- use num_records
            )
            if schema is None:
                logger.error("Schema inference returned None")
                return {"error": "Field inference failed: returned None"}
            logger.debug(f"Inferred Schema: {schema}")
            return {"schema": schema, "error": None}
        except Exception as e:
            logger.exception("Field inference failed")
            return {"error": f"Field inference error: {e}"}

    def get_approval(state: AgentState) -> AgentState:
        if state.get("error"):
            return state
        try:
            logger.info("Step: Getting human approval for inferred schema...")
            approval = agents["human_approver"].get_approval(state["schema"])
            logger.debug(f"Approval Result: {approval}")
            return {"validated_schema": approval, "error": None}
        except Exception as e:
            logger.exception("Human approval failed")
            return {"error": f"Approval error: {e}"}
    def generate_data(state: AgentState) -> AgentState:
        if state.get("error") or not state.get("validated_schema") or not state["validated_schema"].approved:
            logger.warning("Skipping data generation due to prior error or unapproved schema.")
            return state
        try:
            logger.info("Step: Generating synthetic data (without validation)...")
            schema_def = state["validated_schema"].schema_def
            schema_def.sample_size = state["request"].sample_size  # ensure sample_size is set
            data = agents["data_generator"].generate(schema_def)

            # Skip any Gemini validation here, directly return generated data
            return {"generated_data": data, "error": None}

        except Exception as e:
            logger.exception("Data generation failed")
            return {"error": f"Data generation error: {e}"}


    def format_output(state: AgentState) -> AgentState:
        if state.get("error"):
            return state
        try:
            logger.info("Step: Formatting generated data for output...")
            formatted = agents["output_formatter"].format(
                state["generated_data"],
                state["request"].output_format
            )
            logger.debug(f"Formatted Output: {formatted}")
            return {"output": formatted, "error": None}
        except Exception as e:
            logger.exception("Output formatting failed")
            return {"error": f"Output formatting error: {e}"}

    def handle_error(state: AgentState) -> AgentState:
        if state.get("error"):
            logger.error(f"Pipeline error encountered: {state['error']}")
            return {
                "output": GeneratedData(
                    message=f"Error: {state['error']}",
                    format="error"
                ),
                "error": None
            }
        return state

    # Register nodes
    workflow.add_node("preprocess", preprocess)
    workflow.add_node("infer_fields", infer_fields)
    workflow.add_node("get_approval", get_approval)
    workflow.add_node("generate_data", generate_data)
    workflow.add_node("format_output", format_output)
    workflow.add_node("error_handler", handle_error)

    # Entry point
    workflow.set_entry_point("preprocess")

    # Main path
    workflow.add_edge("preprocess", "infer_fields")
    workflow.add_edge("infer_fields", "get_approval")
    workflow.add_edge("get_approval", "generate_data")
    workflow.add_edge("generate_data", "format_output")
    workflow.add_edge("format_output", END)

    # Conditional error handling
    for node in ["preprocess", "infer_fields", "get_approval", "generate_data", "format_output"]:
        workflow.add_conditional_edges(
            node,
            lambda s, n=node: "error_handler" if s.get("error") else {
                "preprocess": "infer_fields",
                "infer_fields": "get_approval",
                "get_approval": "generate_data" if s.get("validated_schema") and s["validated_schema"].approved else END,
                "generate_data": "format_output",
                "format_output": END
            }[n]
        )

    logger.info("Pipeline graph successfully built and compiled.")
    return workflow.compile()
