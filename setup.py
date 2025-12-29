import os
import sys
import pandas as pd
from src.pipelines.utils import load_object


class PredictPipeline:
    def predict(self, features):
        try:
            # 🔥 Absolute project root
            BASE_DIR = os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                )
            )

            model_path = os.path.join(BASE_DIR, "artifacts", "model.pkl")
            preprocessor_path = os.path.join(BASE_DIR, "artifacts", "preprocessor.pkl")

            print("📂 BASE DIR:", BASE_DIR)
            print("📦 MODEL PATH:", model_path)
            print("📦 PREPROCESSOR PATH:", preprocessor_path)

            print("📌 MODEL EXISTS:", os.path.exists(model_path))
            print("📌 PREPROCESSOR EXISTS:", os.path.exists(preprocessor_path))

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            print("\n🔥 PREDICT PIPELINE ERROR 🔥")
            raise


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        return pd.DataFrame(
            {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
        )
