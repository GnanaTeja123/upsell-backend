from fastapi import APIRouter, HTTPException, UploadFile, File
import pandas as pd
import io
import uuid
from ..services import prediction_service

router = APIRouter()

@router.post("/upload-and-process")
async def upload_and_process(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'): raise HTTPException(status_code=400, detail="Invalid file type.")
    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        file_id = str(uuid.uuid4())
        prediction_service.process_and_store_data(df, file_id)
        return {"file_id": file_id, "filename": file.filename}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@router.get("/{file_id}/summary")
def get_summary(file_id: str):
    try: return prediction_service.get_summary_metrics(file_id)
    except Exception as e: raise HTTPException(status_code=404, detail=str(e))

@router.get("/{file_id}/segmentation")
def get_segmentation(file_id: str):
    try: return prediction_service.get_segmentation_data(file_id)
    except Exception as e: raise HTTPException(status_code=404, detail=str(e))

@router.get("/{file_id}/high-potential")
def get_high_potential(file_id: str):
    try: return prediction_service.get_high_potential_customers(file_id)
    except Exception as e: raise HTTPException(status_code=404, detail=str(e))
        
@router.get("/{file_id}/customer-demographics")
def get_customer_demographics(file_id: str):
    try: return prediction_service.get_customer_demographics(file_id)
    except Exception as e: raise HTTPException(status_code=404, detail=str(e))

@router.get("/{file_id}/dataset-distributions")
def get_dataset_distributions(file_id: str):
    try: return prediction_service.get_dataset_distributions(file_id)
    except Exception as e: raise HTTPException(status_code=404, detail=str(e))

@router.get("/model-performance")
def get_model_performance():
    try: return prediction_service.get_model_performance()
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/{file_id}/customers")
def get_customer_analysis(file_id: str):
    try: return prediction_service.get_customer_analysis(file_id)
    except Exception as e: raise HTTPException(status_code=404, detail=str(e))

@router.post("/upload-and-process")
async def upload_and_process(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'): raise HTTPException(status_code=400, detail="Invalid file type.")
    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        file_id = str(uuid.uuid4())
        prediction_service.process_and_store_data(df, file_id)
        return {"file_id": file_id, "filename": file.filename}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@router.get("/{file_id}/customers")
def get_customer_analysis(file_id: str):
    try: return prediction_service.get_customer_analysis(file_id)
    except Exception as e: raise HTTPException(status_code=404, detail=str(e))