from pydantic import BaseModel


class PredictionItem(BaseModel):
    class_name: str
    score: float


class PredictionResult(BaseModel):
    predicted_class: str
    score: float
    top_predictions: list[PredictionItem]


class PredictionResponse(PredictionResult):
    filename: str | None


class ModelPrediction(PredictionResult):
    model_name: str
    test_accuracy: float
    training_method: str


class ComparisonResponse(BaseModel):
    filename: str | None
    custom_cnn: ModelPrediction
    resnet18: ModelPrediction
