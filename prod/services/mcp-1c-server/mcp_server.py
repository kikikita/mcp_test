from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional

from instructions.howto import howto, InstructionNotFoundError, UserArgsValidationError
from instructions.todo import todo
from odata_client import ODataClient

app = FastAPI(title="MCP Server")


class HowToRequest(BaseModel):
    entity: str
    action: str
    user_args: Optional[Dict[str, Any]] = None


class TodoRequest(BaseModel):
    steps: Any


data_client = ODataClient()


@app.post("/howto")
def howto_endpoint(req: HowToRequest):
    try:
        return howto(req.entity, req.action, req.user_args)
    except InstructionNotFoundError:
        raise HTTPException(status_code=404, detail="no_instruction")
    except UserArgsValidationError as exc:
        raise HTTPException(status_code=400, detail=f"validation_error: {exc}")


@app.post("/todo")
async def todo_endpoint(req: TodoRequest):
    return await todo(req.steps, data_client)
