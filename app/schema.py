from typing import List,Optional,Dict

from pydantic import BaseModel, validator


class AppResponse(BaseModel):
    status: str
    response: object

class RAGPayload(BaseModel):
    # LabelStudio project
    project_id: str