import asyncio
import uuid
import logging
from pydantic import BaseModel, ValidationError
from models.schemas import Schema, FieldDefinition, FieldType
from agents.FieldInferenceAgent import FieldInferenceAgent

logger = logging.getLogger(__name__)


class ApprovalResult(BaseModel):
    approved: bool
    schema_def: Schema


class HumanInteractionAgent:
    def __init__(self, timeout_seconds: int = 60):
        self.timeout_seconds = timeout_seconds
        self.enricher = FieldInferenceAgent().preprocessor  # 🔁 Use Gemini for enrichment

    async def get_approval(self, schema: Schema) -> ApprovalResult:
        schema_id = str(uuid.uuid4())
        print(f"\n🔍 [Human Review] Schema for approval (id: {schema_id})\n")
        for field in schema.fields:
            print(f" - {field.name} ({field.type}): {field.description}")

        print("\n✅ Approve schema? Type 'yes' to approve, 'no' to edit schema, or press Enter to simulate timeout:")

        try:
            user_input = await asyncio.wait_for(async_input(), timeout=self.timeout_seconds)
            choice = user_input.strip().lower()

            if not choice:
                print("⏰ Timeout — proceeding with default approval.\n")
                return ApprovalResult(approved=True, schema_def=schema)

            elif choice == "yes":
                return ApprovalResult(approved=True, schema_def=schema)

            elif choice == "no":
                edited_schema = self._edit_schema(schema)
                return ApprovalResult(approved=True, schema_def=edited_schema)

            else:
                print("❌ Invalid input — treating as rejection. Pipeline will end.")
                return ApprovalResult(approved=False, schema_def=schema)

        except asyncio.TimeoutError:
            print("⏰ Timeout — proceeding with default approval.\n")
            return ApprovalResult(approved=True, schema_def=schema)

    def _edit_schema(self, schema: Schema) -> Schema:
        print("\n📝 Enter new field definitions (press Enter without input to finish):")

        updated_fields = schema.fields.copy()

        while True:
            name = input("   ➤ Field name (or press Enter to finish): ").strip()
            if not name:
                break

            ftype = input("   ➤ Field type (string, number, boolean, date, datetime): ").strip().lower()
            desc = input("   ➤ Field description (optional): ").strip()
            constraints = input("   ➤ Field constraints (optional): ").strip() or None

            # 🔍 Auto-enrich if description missing
            if not desc or len(desc) < 20:
                enriched = self.enricher.enrich_field_metadata(schema.scenario, name, ftype)
                desc = enriched.get("description", desc)
                if not constraints:
                    constraints = enriched.get("constraints")

            try:
                updated_fields.append(FieldDefinition(
                    name=name,
                    type=FieldType(ftype),
                    description=desc,
                    constraints=constraints
                ))
            except (ValueError, ValidationError) as e:
                print(f"❌ Invalid field definition: {e}")
                continue

        try:
            new_sample = input(f"\n🔢 New sample size (default: {schema.sample_size}): ").strip()
            sample_size = int(new_sample) if new_sample else schema.sample_size
        except:
            sample_size = schema.sample_size

        return Schema(
            fields=updated_fields,
            sample_size=sample_size,
            scenario=schema.scenario
        )


# 🔄 Awaitable user input helper
async def async_input():
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input)
