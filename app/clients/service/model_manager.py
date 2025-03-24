"""
Model training management module for the Common Assessment Tool.
Handles the preparation, training, and saving of the prediction model of multiple model types, swicth between them and compare the outputs.
"""

# Standard library imports
import pickle
from abc import ABC, abstractmethod
import os

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor


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


class RandomForestTrainer(ModelTrainer):
    """Concrete implementation of Random Forest Regressor model trainer for success rate prediction."""
    
    def __init__(self, data_path=None):
        if data_path is None:
            base_dir = os.path.dirname(__file__)  
            self.data_path = os.path.join(base_dir, "data_commontool.csv")
        else:
            self.data_path = data_path

        # Columns expected from CSV data for training
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
    

class LinearRegressionTrainer(ModelTrainer):
    """Concrete implementation of Linear Regression model trainer for success rate prediction."""

    def __init__(self, data_path=None):
        if data_path is None:
            base_dir = os.path.dirname(__file__)  
            self.data_path = os.path.join(base_dir, "data_commontool.csv")
        else:
            self.data_path = data_path

        # Columns expected from CSV data for training
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
        model = LinearRegression()
        model.fit(features_train, targets_train)
        return model


class SVRTrainer(ModelTrainer):
    """Concrete implementation of Support Vector Regression model trainer for success rate prediction."""
    
    def __init__(self, data_path=None):
        if data_path is None:
            base_dir = os.path.dirname(__file__)  
            self.data_path = os.path.join(base_dir, "data_commontool.csv")
        else:
            self.data_path = data_path

        # Columns expected from CSV data for training
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
        model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        model.fit(features_train, targets_train)
        return model


class GradientBoostingTrainer(ModelTrainer):
    """Concrete implementation of Gradient Boosting Regressor model trainer for success rate prediction."""


    def __init__(self, data_path=None):
        if data_path is None:
            base_dir = os.path.dirname(__file__) 
            self.data_path = os.path.join(base_dir, "data_commontool.csv")
        else:
            self.data_path = data_path

        # Columns expected from CSV data for training
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
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(features_train, targets_train)
        return model


class PickleModelPersistence(ModelPersistence):
    """Concrete implementation of model persistence using pickle."""
    
    def save_model(self, model, filename):
        """Save the trained model to a file."""
        with open(filename, "wb") as model_file:
            pickle.dump(model, model_file)
    
    def load_model(self, filename):
        """Load a trained model from a file."""
        with open(filename, "rb") as model_file:
            return pickle.load(model_file)


class ModelManagerFactory:
    """Factory for creating and managing models switching."""
    
    def __init__(self, persistence: ModelPersistence, models_dir="models"):
        self.persistence = persistence
        self.models_dir = models_dir
        self.trainers = {
            "random_forest": RandomForestTrainer(),
            "linear_regression": LinearRegressionTrainer(),
            "svr": SVRTrainer(),
            "gradient_boosting": GradientBoostingTrainer()
        }
        self.loaded_models = {}
        self.current_model_name = "random_forest"
        os.makedirs(models_dir, exist_ok=True)
        # this can auto initialize all models if it not trained yet
        self._ensure_all_models_exist()
    
    def _ensure_all_models_exist(self):
        """make sure all model files exist"""
        for model_name in self.trainers.keys():
            model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
            if not os.path.exists(model_path):
                print(f"Training new model: {model_name}")
                self.create_and_save_model(model_name)
    
    def create_and_save_model(self, model_name):
        """Create and save the model we choose."""
        if model_name not in self.trainers:
            raise ValueError(f"Unknown model type: {model_name}")
        
        trainer = self.trainers[model_name]
        model = trainer.train_model()
        
        model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
        self.persistence.save_model(model, model_path)
        
        # Cache the model we choose
        self.loaded_models[model_name] = model
        return model
    
    def load_model(self, model_name):
        """Load the existing model we choose."""
        # Check if the model already loaded
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        # Load it from the file
        model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = self.persistence.load_model(model_path)
        self.loaded_models[model_name] = model
        return model
    
    def get_available_models(self):
        """Get list of all available models."""
        return list(self.trainers.keys())
    
    def get_current_model_name(self):
        """Get the name of the current model."""
        return self.current_model_name
    
    def set_model(self, model_name):
        """Change the current model."""
        if model_name not in self.trainers:
            return False
        
        self.current_model_name = model_name
        # Ensure model is loaded
        self.load_model(model_name)
        return True
    
    def predict(self, data):
        """Make prediction using current model."""
        model = self.load_model(self.current_model_name)
        return model.predict(data)
    
    def predict_with_all_models(self, data):
        """Make predictions using all models."""
        results = {}
        for model_name in self.trainers.keys():
            model = self.load_model(model_name)
            results[model_name] = model.predict(data)
        return results
    
    def compare_models(self, data):
        """Compare predictions from all the model--- return average predictions."""
        results = {}
        for model_name in self.trainers.keys():
            model = self.load_model(model_name)
            predictions = model.predict(data)
            results[model_name] = float(np.mean(predictions))
        return results


def create_model_manager():
    """Create and return the model manager instance."""
    persistence = PickleModelPersistence()
    return ModelManagerFactory(persistence)


model_manager = create_model_manager()


if __name__ == "__main__":
    """Test code for the model manager."""
    print("Testing model manager...")
    
    # List available models
    print(f"Available models: {model_manager.get_available_models()}")
    
    # Get current model
    print(f"Current model: {model_manager.get_current_model_name()}")
    
    # Switch model
    model_manager.set_model("linear_regression")
    print(f"Switched to: {model_manager.get_current_model_name()}")
    
    print("Model manager testing completed.")