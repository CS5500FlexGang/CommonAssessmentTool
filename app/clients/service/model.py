"""
Model training module for the Common Assessment Tool.
Handles the preparation, training, and saving of the prediction model.
"""

# Standard library imports
import pickle
from abc import ABC, abstractmethod

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


class ModelTrainer(ABC):
    """Abstract base class for model trainers."""
    
    @abstractmethod
    def prepare_data(self):
        """Prepare data for training."""
    
    @abstractmethod
    def train_model(self):
        """Train the model."""


class ModelPersistence(ABC):
    """Abstract base class for model persistence."""
    
    @abstractmethod
    def save_model(self, model, filename):
        """Save model to file."""
    
    @abstractmethod
    def load_model(self, filename):
        """Load model from file."""


class SuccessRateModelTrainer(ModelTrainer):
    """Concrete implementation of model trainer for success rate prediction."""
    
    def __init__(self, data_path='data_commontool.csv'):
        self.data_path = data_path
        self.feature_columns = [
            'age',                    # Client's age
            'gender',                 # Client's gender (bool)
            'work_experience',        # Years of work experience
            'canada_workex',          # Years of work experience in Canada
            'dep_num',                # Number of dependents
            'canada_born',            # Born in Canada
            'citizen_status',         # Citizenship status
            'level_of_schooling',     # Highest level achieved (1-14)
            'fluent_english',         # English fluency scale (1-10)
            'reading_english_scale',  # Reading ability scale (1-10)
            'speaking_english_scale', # Speaking ability scale (1-10)
            'writing_english_scale',  # Writing ability scale (1-10)
            'numeracy_scale',         # Numeracy ability scale (1-10)
            'computer_scale',         # Computer proficiency scale (1-10)
            'transportation_bool',    # Needs transportation support (bool)
            'caregiver_bool',         # Is primary caregiver (bool)
            'housing',                # Housing situation (1-10)
            'income_source',          # Source of income (1-10)
            'felony_bool',            # Has a felony (bool)
            'attending_school',       # Currently a student (bool)
            'currently_employed',     # Currently employed (bool)
            'substance_use',          # Substance use disorder (bool)
            'time_unemployed',        # Years unemployed
            'need_mental_health_support_bool'  # Needs mental health support (bool)
        ]
        self.intervention_columns = [
            'employment_assistance',
            'life_stabilization',
            'retention_services',
            'specialized_services',
            'employment_related_financial_supports',
            'employer_financial_supports',
            'enhanced_referrals'
        ]
    
    def prepare_data(self):
        """Prepare data for training."""
        data = pd.read_csv(self.data_path)
        all_features = self.feature_columns + self.intervention_columns
        features = np.array(data[all_features])
        targets = np.array(data['success_rate'])
        return train_test_split(features, targets, test_size=0.2, random_state=42)
    
    def train_model(self):
        """Train the model."""
        features_train, _, targets_train, _ = self.prepare_data()
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(features_train, targets_train)
        return model


class PickleModelPersistence(ModelPersistence):
    """Concrete implementation of model persistence using pickle."""
    
    def save_model(self, model, filename="model.pkl"):
        """Save the trained model to a file."""
        with open(filename, "wb") as model_file:
            pickle.dump(model, model_file)
    
    def load_model(self, filename="model.pkl"):
        """Load a trained model from a file."""
        with open(filename, "rb") as model_file:
            return pickle.load(model_file)


class ModelFactory:
    """Factory for creating and managing models."""
    
    def __init__(self, trainer: ModelTrainer, persistence: ModelPersistence):
        self.trainer = trainer
        self.persistence = persistence
    
    def create_and_save_model(self, filename="model.pkl"):
        """Create and save a new model."""
        model = self.trainer.train_model()
        self.persistence.save_model(model, filename)
        return model
    
    def load_model(self, filename="model.pkl"):
        """Load an existing model."""
        return self.persistence.load_model(filename)


def main():
    """Main function to train and save the model."""
    print("Starting model training...")
    
    trainer = SuccessRateModelTrainer()
    persistence = PickleModelPersistence()
    factory = ModelFactory(trainer, persistence)
    
    factory.create_and_save_model()
    
    print("Model training completed and saved successfully.")


if __name__ == "__main__":
    main()