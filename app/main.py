from typing import Dict
import os
from fastapi import FastAPI, HTTPException,File, UploadFile
from fastapi.responses import PlainTextResponse
from loguru import logger
import qdrant_client
from typing import List
from app.config import settings
from app.schema import AppResponse,RAGPayload
from src.rag_core.index import IndexData


app = FastAPI(
    title="Chat with me",
    description=("Enterprise scale chatbot app for business users "),
)

err_resp = PlainTextResponse("Unexpected internal error", status_code=500)


@app.get(
    "/get_collections",
    response_model=AppResponse,
    status_code=201,
    responses={
        201: {
            "description": "Collections is successfully listed",
            "model": AppResponse,
        },
        412: {
            "description": (
                "There is something wrong with the qdrant server"
            ),
            "model": AppResponse,
        },
    },
)
@logger.catch(exclude=HTTPException, default=err_resp)
def list_collections() -> AppResponse:
    """ Get existing qdrant collections

    Raises:
        HTTPException: Raise HTTP_412 if qdrant connection is not successful.

    Returns:
        AppResponse: Success message.
    """
    
    try:
        client = qdrant_client.QdrantClient(url=settings.QDRANT_URL)
        logger.info("Client is initialized")

        collections=client.get_collections().model_dump()

    except Exception as exc:
        logger.error(f"Error :: {exc}")
        return AppResponse(status="failure",response=str(exc)) 
    
    return AppResponse(status="success",response=str(collections))



@app.post(
    "/index",
    response_model=AppResponse,
    status_code=201,
    responses={
        201: {
            "description": "Documents are successfully indexed",
            "model": AppResponse,
        },
        412: {
            "description": (
                "There is something wrong with the qdrant server"
            ),
            "model": AppResponse,
        },
    },
)
@logger.catch(exclude=HTTPException, default=err_resp)
def index_documents(files: List[UploadFile] = File(...),collection_name:str=None) -> AppResponse:
    """ Index the uploaded documents

    Raises:
        HTTPException: Raise HTTP_412 if qdrant connection is not successful.

    Returns:
        AppResponse: Success message.
    """
    
    try:
        for file in files:
            try:
                data_path="./data"
                os.makedirs(data_path, exist_ok=True)
                contents = file.file.read()
                with open(f"{data_path}/{file.filename}", 'wb') as f:
                    f.write(contents)
            except Exception:
                return AppResponse(status="failure",response="There was an error uploading the file(s)")
            
            finally:
                file.file.close()

        logger.info(f"Successfuly uploaded {[file.filename for file in files]}")

        index=IndexData(collection_name=collection_name)
        index_ids=index.index_data()

    except Exception as exc:
        logger.error(f"Error :: {exc}")
        return AppResponse(status="failure",response=str(exc)) 
    
    return AppResponse(status="Success",response=index_ids)  