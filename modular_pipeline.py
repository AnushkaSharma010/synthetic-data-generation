from langgraph.graph import StateGraph, Graph, END
from typing import Any, TypedDict, Annotated, Optional, Dict, List

from pydantic_core import ValidationError
from models.schemas import (
    GenerationRequest, 
    GeneratedData,
    Schema,
    FieldDefinition
)
from agents.PreprocessingAgent import PreprocessingAgent
from agents.FieldInferenceAgent import FieldInferenceAgent
from agents.DataGeneratorAgent import DataGeneratorAgent
from agents.OutputFormatterAgent import OutputFormatterAgent

import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class AgentState(TypedDict):
    request: GenerationRequest
    cleaned_scenario: Annotated[Optional[str], "Cleaned input"]
    schema: Annotated[Optional[Schema], "Inferred schema"]
    generated_data: Annotated[Optional[list], "Generated data"]
    output: Annotated[Optional[GeneratedData], "Formatted output"]
    error: Annotated[Optional[str], "Error message if any"]

class Pipeline:
    def __init__(self):
        logger.info("Initializing pipeline agents...")
        try:
            self.agents = {
                "preprocessor": PreprocessingAgent(),
                "field_inferrer": FieldInferenceAgent(),
                "data_generator": DataGeneratorAgent(),
                "output_formatter": OutputFormatterAgent()
            }
            self.graph = self._build_full_workflow()
        except Exception as e:
            logger.exception("Agent initialization failed")
            raise RuntimeError("Failed to initialize agents") from e

    def _build_full_workflow(self) -> Graph:
        """Build the complete LangGraph workflow for legacy support"""
        workflow = StateGraph(AgentState)

        # Define nodes
        workflow.add_node("preprocess", self._preprocess)
        workflow.add_node("infer_fields", self._infer_fields)
        workflow.add_node("generate_data", self._generate_data)
        workflow.add_node("format_output", self._format_output)
        workflow.add_node("error_handler", self._handle_error)

        # Entry point
        workflow.set_entry_point("preprocess")

        # Main path
        workflow.add_edge("preprocess", "infer_fields")
        workflow.add_edge("infer_fields", "generate_data")
        workflow.add_edge("generate_data", "format_output")
        workflow.add_edge("format_output", END)

        # Conditional error handling
        for node in ["preprocess", "infer_fields", "generate_data", "format_output"]:
            workflow.add_conditional_edges(
                node,
                lambda s, n=node: "error_handler" if s.get("error") else {
                    "preprocess": "infer_fields",
                    "infer_fields": "generate_data",
                    "generate_data": "format_output",
                    "format_output": END
                }[n]
            )

        logger.info("Full workflow graph successfully built.")
        return workflow.compile()

    # Core pipeline steps as standalone methods
    async def preview_schema(self, request: GenerationRequest) -> Schema:
        """Run preprocessing and field inference to preview schema"""
        try:
            cleaned = await self._preprocess({"request": request})
            if cleaned.get("error"):
                raise ValueError(cleaned["error"])
            
            inferred = await self._infer_fields(cleaned)
            if inferred.get("error"):
                raise ValueError(inferred["error"])
            
            return inferred["schema"]
        except Exception as e:
            logger.exception("Schema preview failed")
            raise RuntimeError(f"Schema preview error: {str(e)}")

    async def update_schema(
        self, 
        current_schema: Schema,
        updates: Dict[str, Any]
    ) -> Schema:
        """
        Enrich or update the existing schema with field additions, modifications, deletions,
        and other updates like sample_size.
        
        Args:
            current_schema (Schema): The existing schema object.
            updates (Dict[str, Any]): Dictionary containing updates, e.g.:
                {
                    "fields": [  # list of field update dicts
                        {"name": "age", "type": "number", "nullable": False},
                        {"name": "new_field", "type": "string"},
                        ...
                    ],
                    "deleted_fields": ["old_field1", "old_field2"],
                    "sample_size": 1000
                }
        
        Returns:
            Schema: Updated schema object.
        
        Raises:
            RuntimeError: If validation or update fails.
        """
        try:
            updated_schema = current_schema.model_copy(deep=True)
            
            # Handle field updates and additions
            if 'fields' in updates:
                for field_update in updates['fields']:
                    if 'name' not in field_update:
                        continue  # skip invalid field update
                    
                    # Find existing field by name
                    existing_field = next(
                        (f for f in updated_schema.fields if f.name == field_update['name']),
                        None
                    )
                    
                    if existing_field:
                        # Update attributes of existing field
                        for attr, value in field_update.items():
                            if attr != 'name' and hasattr(existing_field, attr):
                                setattr(existing_field, attr, value)
                    else:
                        # Add new field
                        updated_schema.fields.append(FieldDefinition(**field_update))
            
            # Handle field deletions
            if 'deleted_fields' in updates:
                remaining_fields = [
                    f for f in updated_schema.fields 
                    if f.name not in updates['deleted_fields']
                ]
                if not remaining_fields:
                    raise ValueError("Schema must have at least one field after deletions")
                updated_schema.fields = remaining_fields
            
            # Update sample size if present
            if 'sample_size' in updates:
                updated_schema.sample_size = updates['sample_size']
            
            return updated_schema
        
        except Exception as e:
            logger.exception("Schema update failed")
            raise RuntimeError(f"Schema update error: {str(e)}")
        
    async def generate_data(
        self,
        schema: Schema,
        output_format: str = "json"
    ) -> GeneratedData:
        """Generate data from finalized schema"""
        try:
            generated = await self._generate_data({
                "schema": schema,
                "request": GenerationRequest(
                    scenario=schema.scenario,  # Not needed for generation
                    sample_size=schema.sample_size,
                    output_format=output_format
                )
            })
            
            if generated.get("error"):
                raise ValueError(generated["error"])
            
            formatted = await self._format_output({
                "generated_data": generated["generated_data"],
                "request": GenerationRequest(
                    scenario=schema.scenario, 
                    sample_size=schema.sample_size,
                    output_format=output_format
                )
            })
            
            if formatted.get("error"):
                raise ValueError(formatted["error"])
            
            return formatted["output"]
        except Exception as e:
            logger.exception("Data generation failed")
            raise RuntimeError(f"Data generation error: {str(e)}")

    async def run_full_pipeline(self, request: GenerationRequest) -> GeneratedData:
        """Run the complete end-to-end pipeline (legacy support)"""
        try:
            result =  self.graph.invoke({"request": request})
            if result.get("error"):
                raise ValueError(result["error"])
            return result["output"]
        except Exception as e:
            logger.exception("Full pipeline execution failed")
            raise RuntimeError(f"Pipeline execution error: {str(e)}")

    # Internal step implementations
    async def _preprocess(self, state: AgentState) -> AgentState:
        try:
            logger.info("Step: Preprocessing input scenario...")
            cleaned = self.agents["preprocessor"].clean(state["request"].scenario)
            logger.debug(f"Cleaned Scenario: {cleaned}")
            return {**state,"cleaned_scenario": cleaned, "error": None}
        except Exception as e:
            logger.exception("Preprocessing failed")
            return {**state,"error": f"Preprocessing error: {e}"}

    async def _infer_fields(self, state: AgentState) -> AgentState:
        if state.get("error"):
            return state
        try:
            logger.info("Step: Inferring schema from cleaned scenario...")
            schema_dict = self.agents["field_inferrer"].infer_schema(
                scenario=state["cleaned_scenario"],
                sample_size=state["request"].sample_size
            )
            if schema_dict is None:
                logger.error("Schema inference returned None")
                return {**state,"error": "Field inference failed: returned None"}
            complete_schema = {
            "fields": schema_dict["fields"],
            "sample_size": schema_dict.get("sample_size", state["request"].sample_size),
            "scenario": state["request"].scenario  # Use original scenario
            }
            try:
                schema = Schema(**complete_schema)
                logger.debug(f"Inferred Schema: {schema}")
                return {**state, "schema": schema, "error": None}
            except ValidationError as ve:
                logger.error(f"Schema validation failed: {str(ve)}")
                return {**state, "error": f"Invalid schema format: {str(ve)}"}
                
        except Exception as e:
            logger.exception("Field inference failed")
            return {**state,"error": f"Field inference error: {e}"}

    async def _generate_data(self, state: AgentState) -> AgentState:
        if state.get("error"):
            return state
        try:
            logger.info("Step: Generating synthetic data...")
            data = self.agents["data_generator"].generate(state["schema"])
            return {**state,"generated_data": data, "error": None}
        except Exception as e:
            logger.exception("Data generation failed")
            return {**state,"error": f"Data generation error: {e}"}

    async def _format_output(self, state: AgentState) -> AgentState:
        if state.get("error"):
            return state
        try:
            logger.info("Step: Formatting generated data for output...")
            formatted = self.agents["output_formatter"].format(
                state["generated_data"],
                state["request"].output_format
            )
            logger.debug(f"Formatted Output: {formatted}")
            return {**state,"output": formatted, "error": None}
        except Exception as e:
            logger.exception("Output formatting failed")
            return {**state,"error": f"Output formatting error: {e}"}

    async def _handle_error(self, state: AgentState) -> AgentState:
        if state.get("error"):
            logger.error(f"Pipeline error encountered: {state['error']}")
            return {
                "output": GeneratedData(
                    data=None,
                    file_content=None,
                    format="error",
                    message=f"Error: {state['error']}"
                ),
                "error": None
            }
        return state