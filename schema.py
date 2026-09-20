from pydantic import BaseModel


class PredictionItem(BaseModel):
    class_name: str
    score: float


class PredictionResponse(BaseModel):
    filename: str | None
    predicted_class: str
    score: float
    top_predictions: list[PredictionItem]
