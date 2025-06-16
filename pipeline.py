from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Optional
from models.schemas import GenerationRequest, GeneratedData, Schema
from agents.FieldInferenceAgent import FieldInferenceAgent
from agents.DataGeneratorAgent import DataGeneratorAgent
from agents.OutputFormatterAgent import OutputFormatterAgent
from agents.PreprocessingAgent import PreprocessingAgent
from agents.HumanInteractionAgent import ApprovalResult, HumanInteractionAgent

import logging

logger = logging.getLogger("pipeline")
logging.basicConfig(level=logging.INFO)

class AgentState(TypedDict):
    request: GenerationRequest
    cleaned_scenario: Annotated[Optional[str], "Cleaned input"]
    schema: Annotated[Optional[Schema], "Inferred schema"]
    validated_schema: Annotated[Optional[ApprovalResult], "Approval result"]
    generated_data: Annotated[Optional[list], "Generated data"]
    output: Annotated[Optional[GeneratedData], "Formatted output"]
    error: Annotated[Optional[str], "Error message if any"]

def create_pipeline() -> StateGraph:
    logger.info("Initializing pipeline agents...")

    agents = {
        "preprocessor": PreprocessingAgent(),
        "field_inferrer": FieldInferenceAgent(),
        "human_approver": HumanInteractionAgent(),
        "data_generator": DataGeneratorAgent(),
        "output_formatter": OutputFormatterAgent(),
    }

    workflow = StateGraph(AgentState)

    def preprocess(state: AgentState) -> AgentState:
        try:
            logger.info("Step: Preprocessing input scenario...")
            cleaned = agents["preprocessor"].clean(state["request"].scenario)
            return {**state, "cleaned_scenario": cleaned, "error": None}
        except Exception as e:
            logger.exception("Preprocessing failed")
            return {**state, "error": f"Preprocessing error: {e}"}

    def infer_fields(state: AgentState) -> AgentState:
        if state.get("error"):
            return state
        try:
            logger.info("Step: Inferring schema from cleaned scenario...")
            schema_dict = agents["field_inferrer"].infer_schema(
                scenario=state["cleaned_scenario"],
                sample_size=state["request"].sample_size,
            )
            if schema_dict is None:
                return {**state, "error": "Field inference failed: returned None"}

            schema = Schema(**{
                **schema_dict,
                "scenario": state["request"].scenario
            })

            logger.debug(f"Inferred Schema: {schema}")
            return {**state, "schema": schema, "error": None}
        except Exception as e:
            logger.exception("Field inference failed")
            return {**state, "error": f"Field inference error: {e}"}

    async def get_approval(state: AgentState) -> AgentState:
        if state.get("error"):
            return state
        try:
            logger.info("Step: Getting human approval for schema...")
            approval = await agents["human_approver"].get_approval(state["schema"])
            logger.debug(f"Approval Result: {approval}")
            return {**state, "validated_schema": approval, "error": None}
        except Exception as e:
            logger.exception("Human approval failed")
            return {**state, "error": f"Approval error: {e}"}

    def generate_data(state: AgentState) -> AgentState:
        if state.get("error") or not state.get("validated_schema") or not state["validated_schema"].approved:
            logger.warning("Skipping data generation due to prior error or disapproval.")
            return state
        try:
            logger.info("Step: Generating synthetic data...")
            schema_def = state["validated_schema"].schema_def
            logger.info(f"Generating {schema_def.sample_size} records...")  # ✅ Correct size
            data = agents["data_generator"].generate(schema_def)
            return {**state, "generated_data": data, "error": None}
        except Exception as e:
            logger.exception("Data generation failed")
            return {**state, "error": f"Data generation error: {e}"}


    def format_output(state: AgentState) -> AgentState:
        if state.get("error"):
            return state
        try:
            logger.info("Step: Formatting output...")
            formatted = agents["output_formatter"].format(
                state["generated_data"],
                state["request"].output_format
            )
            return {**state, "output": formatted, "error": None}
        except Exception as e:
            logger.exception("Formatting failed")
            return {**state, "error": f"Output formatting error: {e}"}

    def handle_error(state: AgentState) -> AgentState:
        logger.error(f"Pipeline error encountered: {state.get('error')}")
        return {
            "output": GeneratedData(
                data=None,
                file_content=None,
                format="error",
                message=f"Error: {state.get('error')}"
            ),
            "error": None
        }

    # Register nodes
    workflow.add_node("preprocess", preprocess)
    workflow.add_node("infer_fields", infer_fields)
    workflow.add_node("get_approval", get_approval)
    workflow.add_node("generate_data", generate_data)
    workflow.add_node("format_output", format_output)
    workflow.add_node("error_handler", handle_error)

    workflow.set_entry_point("preprocess")

    workflow.add_edge("preprocess", "infer_fields")
    workflow.add_edge("infer_fields", "get_approval")
    workflow.add_edge("get_approval", "generate_data")
    workflow.add_edge("generate_data", "format_output")
    workflow.add_edge("format_output", END)

    # Conditional error routing
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

    logger.info("✅ Pipeline graph successfully compiled.")
    return workflow.compile()

class Pipeline:
    def __init__(self):
        logger.info("🔧 Initializing Pipeline class...")
        self.graph = create_pipeline()

    async def run_full_pipeline(self, request: GenerationRequest) -> GeneratedData:
        """Run the complete end-to-end pipeline"""
        try:
            logger.info("🔁 Running full generation pipeline for scenario: %s", request.scenario[:60])
            result = await self.graph.ainvoke({"request": request})
            if result.get("error"):
                raise ValueError(result["error"])
            return result["output"]
        except Exception as e:
            logger.exception("Full pipeline execution failed")
            raise RuntimeError(f"Pipeline execution error: {str(e)}")
