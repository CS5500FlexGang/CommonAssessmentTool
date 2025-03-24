"""
Router module for ML model-related endpoints.
Handles HTTP requests for model operations including switching 
models and comparing results.
"""

# Standard library imports
from typing import List, Dict, Any

# Third-party imports
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

# Local imports
from app.auth.router import get_current_user, get_admin_user
from app.models import User
from app.clients.service.model_manager import model_manager
from app.database import get_db
from sqlalchemy.orm import Session


# SCHEMAS for api responses
class ModelInfo(BaseModel):
    """Model data to be used in responses."""
    name: str
    description: str


class ModelListResponse(BaseModel):
    """Response schema for the available models."""
    models: List[str]
    current_model: str


class ModelSwitchRequest(BaseModel):
    """Request body schema for switching model."""
    model_name: str


class ModelSwitchResponse(BaseModel):
    """Response schema after switching model."""
    success: bool
    current_model: str
    message: str


class ModelComparisonResponse(BaseModel):
    """Response schema for model comparison output."""
    model_results: Dict[str, float]
    client_info: Dict[str, Any]


# ROUTER
router = APIRouter(prefix="/ml-models", tags=["ML Models"])


@router.get("/available", response_model=ModelListResponse)
async def get_available_models(current_user: User = Depends(get_current_user)):
    """Retrieve all available ML models and the currently active model."""
    return {
        "models": model_manager.get_available_models(),
        "current_model": model_manager.get_current_model_name()
    }


@router.get("/current", response_model=str)
async def get_current_model(current_user: User = Depends(get_current_user)):
    """Get the name of the currently active ML model."""
    return model_manager.get_current_model_name()


@router.post("/switch", response_model=ModelSwitchResponse)
async def switch_model(
    request: ModelSwitchRequest,
    current_user: User = Depends(get_admin_user)  # Only admin can switch models
):
    """Switch to a specified ML model."""
    success = model_manager.set_model(request.model_name)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{request.model_name}' not found"
        )
    
    return {
        "success": True,
        "current_model": model_manager.get_current_model_name(),
        "message": f"Successfully switched to model: {request.model_name}"
    }


@router.post("/compare/{client_id}", response_model=ModelComparisonResponse)
async def compare_models_for_client(
    client_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Run predictions for the target client using all available models,
    then compare outputs."""

    try:
        from app.clients.service.client_service import client_service
        from app.clients.service.logic import InputDataCleaner, InterventionMatrix
        
        client = client_service.get_client(db, client_id)
        
        # Format client object into string-based dict
        client_data = {
            "age": str(client.age),
            "gender": str(client.gender),
            "work_experience": str(client.work_experience),
            "canada_workex": str(client.canada_workex),
            "dep_num": str(client.dep_num),
            "canada_born": str(int(client.canada_born)),
            "citizen_status": str(int(client.citizen_status)),
            "level_of_schooling": str(client.level_of_schooling),
            "fluent_english": str(int(client.fluent_english)),
            "reading_english_scale": str(client.reading_english_scale),
            "speaking_english_scale": str(client.speaking_english_scale),
            "writing_english_scale": str(client.writing_english_scale),
            "numeracy_scale": str(client.numeracy_scale),
            "computer_scale": str(client.computer_scale),
            "transportation_bool": str(int(client.transportation_bool)),
            "caregiver_bool": str(int(client.caregiver_bool)),
            "housing": str(client.housing),
            "income_source": str(client.income_source),
            "felony_bool": str(int(client.felony_bool)),
            "attending_school": str(int(client.attending_school)),
            "currently_employed": str(int(client.currently_employed)),
            "substance_use": str(int(client.substance_use)),
            "time_unemployed": str(client.time_unemployed),
            "need_mental_health_support_bool": str(int(client.need_mental_health_support_bool))
        }
        
        # Clean and convert client data
        data_cleaner = InputDataCleaner()
        raw_data = data_cleaner.process(client_data)
        
        # Build baseline input row 
        intervention_matrix = InterventionMatrix()
        baseline_row = intervention_matrix.get_baseline_row(raw_data).reshape(1, -1)
        
        # Compare scores by all 4 models
        model_results = model_manager.compare_models(baseline_row)
        
        # Prepare client info for response
        client_info = {
            "client_id": client_id,
            "age": client.age,
            "gender": "Male" if client.gender == 1 else "Female",
            "education_level": client.level_of_schooling,
            "current_model": model_manager.get_current_model_name()
        }
        
        return {
            "model_results": model_results,
            "client_info": client_info
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error comparing models: {str(e)}"
        )