import socketio
from models.schemas import Schema
from agents.HumanInteractionAgent import approval_registry, ApprovalResult

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
app = socketio.ASGIApp(sio)

@sio.event
async def submit_approval(sid, data):
    schema_id = data.get("schema_id")
    approved = data.get("approved", False)

    if schema_id not in approval_registry:
        print(f"⚠️ Unknown schema_id: {schema_id}")
        return

    schema = approval_registry[schema_id]["schema"]
    approval_registry[schema_id]["response"] = ApprovalResult(
        approved=approved,
        schema_def=schema
    )
    approval_registry[schema_id]["event"].set()
    print(f"📩 Approval received for schema_id={schema_id}: {approved}")
