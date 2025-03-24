from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from typing import Optional, Dict, Any, List, Type
from abc import ABC, abstractmethod

from app.models import Client, ClientCase, User
from app.clients.schema import ClientUpdate, ServiceUpdate

class Validator(ABC):
    """Abstract base class for validators."""

    @abstractmethod
    def validate(self, **kwargs):
        """Validate input data."""
        pass

class ClientValidator(Validator):
    """Validator for client data."""

    def validate(self, **kwargs):
        """Validate client data."""
        education_level = kwargs.get('education_level')
        age_min = kwargs.get('age_min')
        gender = kwargs.get('gender')

        if education_level is not None and not 1 <= education_level <= 14:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Education level must be between 1 and 14"
            )

        if age_min is not None and age_min < 18:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Minimum age must be at least 18"
            )

        if gender is not None and gender not in [1, 2]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Gender must be 1 or 2"
            )

class QueryBuilder(ABC):
    """Abstract base class for query builders."""

    @abstractmethod
    def build_query(self, db: Session, **kwargs):
        """Build a database query."""
        pass

class ClientQueryBuilder(QueryBuilder):
    """Query builder for client data."""

    def __init__(self, validator: Validator):
        self.validator = validator

    def build_query(self, db: Session, **filters):
        """Build a query for client data."""
        self.validator.validate(
            education_level=filters.get("level_of_schooling"),
            age_min=filters.get("age"),
            gender=filters.get("gender")
        )

        query = db.query(Client)
        for attr, value in filters.items():
            if value is not None:
                column = getattr(Client, attr, None)
                if column is not None:
                    if attr == "age":
                        query = query.filter(column >= value)
                    else:
                        query = query.filter(column == value)
        return query

class Repository(ABC):
    """Abstract base class for repositories."""

    @abstractmethod
    def get_by_id(self, db: Session, id: int):
        """Get an entity by ID."""
        pass

    @abstractmethod
    def get_all(self, db: Session, skip: int = 0, limit: int = 50):
        """Get all entities."""
        pass

    @abstractmethod
    def update(self, db: Session, id: int, data: dict):
        """Update an entity."""
        pass

    @abstractmethod
    def delete(self, db: Session, id: int):
        """Delete an entity."""
        pass

class ClientRepository(Repository):
    """Repository for client data."""

    def get_by_id(self, db: Session, id: int):
        """Get a client by ID."""
        client = db.query(Client).filter(Client.id == id).first()
        if not client:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Client with id {id} not found"
            )
        return client

    def get_all(self, db: Session, skip: int = 0, limit: int = 50):
        """Get all clients."""
        if skip < 0:
            raise HTTPException(status_code=400, detail="Skip value cannot be negative")
        if limit < 1:
            raise HTTPException(status_code=400, detail="Limit must be greater than 0")

        clients = db.query(Client).offset(skip).limit(limit).all()
        total = db.query(Client).count()
        return {"clients": clients, "total": total}

    def update(self, db: Session, id: int, data: Dict[str, Any]):
        """Update a client."""
        client = self.get_by_id(db, id)
        for field, value in data.items():
            setattr(client, field, value)
        try:
            db.commit()
            db.refresh(client)
            return client
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Failed to update client: {str(e)}")

    def delete(self, db: Session, id: int):
        """Delete a client."""
        client = self.get_by_id(db, id)
        try:
            db.query(ClientCase).filter(ClientCase.client_id == id).delete()
            db.delete(client)
            db.commit()
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Failed to delete client: {str(e)}")

class ClientCaseRepository:
    """Repository for client case data."""

    def get_by_client_id(self, db: Session, client_id: int):
        """Get client cases by client ID."""
        client_cases = db.query(ClientCase).filter(ClientCase.client_id == client_id).all()
        if not client_cases:
            raise HTTPException(status_code=404, detail=f"No services found for client with id {client_id}")
        return client_cases

    def get_by_client_and_user(self, db: Session, client_id: int, user_id: int):
        """Get client case by client ID and user ID."""
        client_case = db.query(ClientCase).filter(
            ClientCase.client_id == client_id,
            ClientCase.user_id == user_id
        ).first()
        if not client_case:
            raise HTTPException(
                status_code=404,
                detail=f"No case found for client {client_id} with case worker {user_id}. Cannot update services."
            )
        return client_case

    def update(self, db: Session, client_id: int, user_id: int, data: Dict[str, Any]):
        """Update a client case."""
        client_case = self.get_by_client_and_user(db, client_id, user_id)
        for field, value in data.items():
            setattr(client_case, field, value)
        try:
            db.commit()
            db.refresh(client_case)
            return client_case
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Failed to update client services: {str(e)}")

    def create(self, db: Session, client_id: int, user_id: int):
        """Create a new client case."""
        existing_case = db.query(ClientCase).filter(
            ClientCase.client_id == client_id,
            ClientCase.user_id == user_id
        ).first()
        if existing_case:
            raise HTTPException(
                status_code=400,
                detail=f"Client {client_id} already has a case assigned to case worker {user_id}"
            )
        try:
            new_case = ClientCase(
                client_id=client_id,
                user_id=user_id,
                employment_assistance=False,
                life_stabilization=False,
                retention_services=False,
                specialized_services=False,
                employment_related_financial_supports=False,
                employer_financial_supports=False,
                enhanced_referrals=False,
                success_rate=0
            )
            db.add(new_case)
            db.commit()
            db.refresh(new_case)
            return new_case
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Failed to create case assignment: {str(e)}")

class UserRepository:
    """Repository for user data."""
    
    def get_by_id(self, db: Session, id: int):
        """Get a user by ID."""
        user = db.query(User).filter(User.id == id).first()
        if not user:
            raise HTTPException(status_code=404, detail=f"Case worker with id {id} not found")
        return user

class ClientService:
    """Service for client operations."""

    def __init__(
        self,
        client_repository: ClientRepository,
        client_case_repository: ClientCaseRepository,
        user_repository: UserRepository,
        query_builder: ClientQueryBuilder
    ):
        self.client_repository = client_repository
        self.client_case_repository = client_case_repository
        self.user_repository = user_repository
        self.query_builder = query_builder

    def get_client(self, db: Session, client_id: int):
        """Get a client by ID."""
        return self.client_repository.get_by_id(db, client_id)

    def get_clients(self, db: Session, skip: int = 0, limit: int = 50):
        """Get all clients."""
        return self.client_repository.get_all(db, skip, limit)

    def get_clients_by_criteria(self, db: Session, **filters):
        """Get clients by criteria."""
        query = self.query_builder.build_query(db, **filters)
        return query.all()

    def get_clients_by_services(self, db: Session, **service_filters):
        """Get clients by services."""
        query = db.query(Client).join(ClientCase)
        for service_name, status in service_filters.items():
            if status is not None:
                filter_criteria = getattr(ClientCase, service_name) == status
                query = query.filter(filter_criteria)
        return query.all()

    def get_client_services(self, db: Session, client_id: int):
        """Get client services."""
        return self.client_case_repository.get_by_client_id(db, client_id)

    def get_clients_by_success_rate(self, db: Session, min_rate: int = 70):
        """Get clients by success rate."""
        if not (0 <= min_rate <= 100):
            raise HTTPException(status_code=400, detail="Success rate must be between 0 and 100")
        return db.query(Client).join(ClientCase).filter(ClientCase.success_rate >= min_rate).all()

    def get_clients_by_case_worker(self, db: Session, case_worker_id: int):
        """Get clients by case worker ID."""
        self.user_repository.get_by_id(db, case_worker_id)  # Check if case worker exists
        return db.query(Client).join(ClientCase).filter(ClientCase.user_id == case_worker_id).all()

    def update_client(self, db: Session, client_id: int, client_update: ClientUpdate):
        """Update a client."""
        update_data = client_update.dict(exclude_unset=True)
        return self.client_repository.update(db, client_id, update_data)

    def update_client_services(self, db: Session, client_id: int, user_id: int, service_update: ServiceUpdate):
        """Update client services."""
        update_data = service_update.dict(exclude_unset=True)
        return self.client_case_repository.update(db, client_id, user_id, update_data)

    def create_case_assignment(self, db: Session, client_id: int, case_worker_id: int):
        """Create a case assignment."""
        self.client_repository.get_by_id(db, client_id)  # Check if client exists
        self.user_repository.get_by_id(db, case_worker_id)  # Check if case worker exists
        return self.client_case_repository.create(db, client_id, case_worker_id)

    def delete_client(self, db: Session, client_id: int):
        """Delete a client."""
        self.client_repository.delete(db, client_id)

# Create concrete instances
client_validator = ClientValidator()
client_query_builder = ClientQueryBuilder(client_validator)
client_repository = ClientRepository()
client_case_repository = ClientCaseRepository()
user_repository = UserRepository()

# Create service instance
client_service = ClientService(
    client_repository=client_repository,
    client_case_repository=client_case_repository,
    user_repository=user_repository,
    query_builder=client_query_builder
)
