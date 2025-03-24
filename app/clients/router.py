"""
Router module for client-related endpoints.
Handles all HTTP requests for client operations including create, read, update, and delete.
"""

# Standard library imports
from typing import List, Optional

# Third-party imports
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

# Local imports
from app.auth.router import get_current_user, get_admin_user
from app.database import get_db
from app.models import User
from app.clients.service.client_service import client_service
from app.clients.schema import (
    ClientResponse,
    ClientUpdate,
    ClientListResponse,
    ServiceResponse,
    ServiceUpdate
)

# Create router
router = APIRouter(prefix="/clients", tags=["clients"])

# Dependency for DB injection
def get_service_with_db(db: Session = Depends(get_db)):
    """Dependency to inject both service and database session."""
    return client_service, db

@router.get("/", response_model=ClientListResponse)
async def get_clients(
    service_and_db = Depends(get_service_with_db),
    current_user: User = Depends(get_admin_user),  # Required for authentication
    skip: int = Query(default=0, ge=0, description="Number of records to skip"),
    limit: int = Query(default=50, ge=1, le=150, description="Maximum number of records to return")
):
    """Get all clients with pagination."""
    service, db = service_and_db
    return service.get_clients(db, skip, limit)

@router.get("/{client_id}", response_model=ClientResponse)
async def get_client(
    client_id: int,
    service_and_db = Depends(get_service_with_db),
    current_user: User = Depends(get_admin_user)  # Required for authentication
):
    """Get a specific client by ID"""
    service, db = service_and_db
    return service.get_client(db, client_id)

# pylint: disable=too-many-arguments,too-many-locals
@router.get("/search/by-criteria", response_model=List[ClientResponse])
async def get_clients_by_criteria(
    service_and_db = Depends(get_service_with_db),
    current_user: User = Depends(get_admin_user),  # Required for authentication
    # Client demographic parameters
    employment_status: Optional[bool] = None,
    education_level: Optional[int] = Query(None, ge=1, le=14),
    age_min: Optional[int] = Query(None, ge=18),
    gender: Optional[int] = Query(None, ge=1, le=2),
    work_experience: Optional[int] = Query(None, ge=0),
    canada_workex: Optional[int] = Query(None, ge=0),
    dep_num: Optional[int] = Query(None, ge=0),
    canada_born: Optional[bool] = None,
    citizen_status: Optional[bool] = None,
    # Language and skills parameters
    fluent_english: Optional[bool] = None,
    reading_english_scale: Optional[int] = Query(None, ge=0, le=10),
    speaking_english_scale: Optional[int] = Query(None, ge=0, le=10),
    writing_english_scale: Optional[int] = Query(None, ge=0, le=10),
    numeracy_scale: Optional[int] = Query(None, ge=0, le=10),
    computer_scale: Optional[int] = Query(None, ge=0, le=10),
    # Situation parameters
    transportation_bool: Optional[bool] = None,
    caregiver_bool: Optional[bool] = None,
    housing: Optional[int] = Query(None, ge=1, le=10),
    income_source: Optional[int] = Query(None, ge=1, le=11),
    felony_bool: Optional[bool] = None,
    attending_school: Optional[bool] = None,
    substance_use: Optional[bool] = None,
    time_unemployed: Optional[int] = Query(None, ge=0),
    need_mental_health_support_bool: Optional[bool] = None
):
    """Search clients by any combination of criteria"""
    service, db = service_and_db
    return service.get_clients_by_criteria(
        db,
        currently_employed=employment_status,
        level_of_schooling=education_level,
        age=age_min,
        gender=gender,
        work_experience=work_experience,
        canada_workex=canada_workex,
        dep_num=dep_num,
        canada_born=canada_born,
        citizen_status=citizen_status,
        fluent_english=fluent_english,
        reading_english_scale=reading_english_scale,
        speaking_english_scale=speaking_english_scale,
        writing_english_scale=writing_english_scale,
        numeracy_scale=numeracy_scale,
        computer_scale=computer_scale,
        transportation_bool=transportation_bool,
        caregiver_bool=caregiver_bool,
        housing=housing,
        income_source=income_source,
        felony_bool=felony_bool,
        attending_school=attending_school,
        substance_use=substance_use,
        time_unemployed=time_unemployed,
        need_mental_health_support_bool=need_mental_health_support_bool
    )
# pylint: enable=too-many-arguments,too-many-locals

# pylint: disable=too-many-arguments
@router.get("/search/by-services", response_model=List[ClientResponse])
async def get_clients_by_services(
    service_and_db = Depends(get_service_with_db),
    current_user: User = Depends(get_admin_user),  # Required for authentication
    employment_assistance: Optional[bool] = None,
    life_stabilization: Optional[bool] = None,
    retention_services: Optional[bool] = None,
    specialized_services: Optional[bool] = None,
    employment_related_financial_supports: Optional[bool] = None,
    employer_financial_supports: Optional[bool] = None,
    enhanced_referrals: Optional[bool] = None
):
    """Get clients filtered by multiple service statuses"""
    service, db = service_and_db
    return service.get_clients_by_services(
        db,
        employment_assistance=employment_assistance,
        life_stabilization=life_stabilization,
        retention_services=retention_services,
        specialized_services=specialized_services,
        employment_related_financial_supports=employment_related_financial_supports,
        employer_financial_supports=employer_financial_supports,
        enhanced_referrals=enhanced_referrals
    )
# pylint: enable=too-many-arguments

@router.get("/{client_id}/services", response_model=List[ServiceResponse])
async def get_client_services(
    client_id: int,
    service_and_db = Depends(get_service_with_db),
    current_user: User = Depends(get_admin_user)  # Required for authentication
):
    """Get all services and their status for a specific client, including case worker info"""
    service, db = service_and_db
    return service.get_client_services(db, client_id)

@router.get("/search/success-rate", response_model=List[ClientResponse])
async def get_clients_by_success_rate(
    service_and_db = Depends(get_service_with_db),
    min_rate: int = Query(70, ge=0, le=100, description="Minimum success rate percentage"),
    current_user: User = Depends(get_admin_user)  # Required for authentication
):
    """Get clients with success rate above specified threshold"""
    service, db = service_and_db
    return service.get_clients_by_success_rate(db, min_rate)

@router.get("/case-worker/{case_worker_id}", response_model=List[ClientResponse])
async def get_clients_by_case_worker(
    case_worker_id: int,
    service_and_db = Depends(get_service_with_db),
    current_user: User = Depends(get_current_user)  # Required for authentication
):
    """Get clients assigned to a specific case worker"""
    service, db = service_and_db
    return service.get_clients_by_case_worker(db, case_worker_id)

@router.put("/{client_id}", response_model=ClientResponse)
async def update_client(
    client_id: int,
    client_data: ClientUpdate,
    service_and_db = Depends(get_service_with_db),
    current_user: User = Depends(get_admin_user)  # Required for authentication
):
    """Update a client's information"""
    service, db = service_and_db
    return service.update_client(db, client_id, client_data)

@router.put("/{client_id}/services/{user_id}", response_model=ServiceResponse)
async def update_client_services(
    client_id: int,
    user_id: int,
    service_update: ServiceUpdate,
    service_and_db = Depends(get_service_with_db),
    current_user: User = Depends(get_current_user)  # Required for authentication
):
    """Update services for a specific client and case worker"""
    service, db = service_and_db
    return service.update_client_services(db, client_id, user_id, service_update)

@router.post("/{client_id}/case-assignment", response_model=ServiceResponse)
async def create_case_assignment(
    client_id: int,
    service_and_db = Depends(get_service_with_db),
    case_worker_id: int = Query(..., description="Case worker ID to assign"),
    current_user: User = Depends(get_admin_user)  # Required for authentication
):
    """Create a new case assignment for a client with a case worker"""
    service, db = service_and_db
    return service.create_case_assignment(db, client_id, case_worker_id)

@router.delete("/{client_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_client(
    client_id: int,
    service_and_db = Depends(get_service_with_db),
    current_user: User = Depends(get_admin_user)  # Required for authentication
):
    """Delete a client"""
    service, db = service_and_db
    service.delete_client(db, client_id)
    return None
