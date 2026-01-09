#!/usr/bin/env python3
"""
Pydantic models for structured LLaVA validation responses.
Generalized for any object detection task.
"""

from typing import Optional, Literal, List
from pydantic import BaseModel, Field, validator
from datetime import datetime


class LLaVAValidationResult(BaseModel):
    """
    Structured response from LLaVA for object validation.
    The is_target_object field indicates whether the detected object matches the target class.
    """
    is_target_object: bool = Field(
        ...,
        description="Whether the image contains the target object class"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Confidence level from 0.0 to 1.0"
    )
    reasoning: str = Field(
        ..., min_length=10,
        description="Detailed reasoning for the assessment"
    )
    object_type: Optional[str] = Field(
        default=None,
        description="Categorical label for the detected object"
    )
    image_quality: Optional[str] = None
    visual_cues: List[str] = Field(
        default_factory=list,
        description="List of specific visual cues that influenced the decision"
    )
    parsing_error: Optional[str] = Field(
        default=None,
        description="Parsing error message if JSON output failed"
    )
    llava_latency_ms: Optional[int] = Field(
        default=None,
        description="Time taken by LLaVA call in milliseconds"
    )

    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        return v

    @validator('reasoning')
    def validate_reasoning(cls, v):
        if len(v.strip()) < 10:
            raise ValueError("reasoning must be at least 10 characters")
        return v.strip()


class SegmentValidationRecord(BaseModel):
    """
    Complete validation record for a single segment
    """
    detection_id: str = Field(..., description="Unique detection ID from step1")
    segment_path: str = Field(..., description="Path to the segment image")
    original_image: str = Field(..., description="Original satellite image filename")
    bbox_absolute: List[float] = Field(..., description="Absolute bounding box coordinates")
    bbox_normalized: List[float] = Field(..., description="Normalized YOLO coordinates")
    detector_confidence: float = Field(..., description="Original detector confidence score")
    detector_query: str = Field(..., description="Detector text query used")
    llava_result: LLaVAValidationResult = Field(..., description="LLaVA validation response")
    validated: bool = Field(..., description="Whether segment passed validation")
    validation_threshold: float = Field(..., description="Confidence threshold used")
    processed_at: datetime = Field(default_factory=datetime.now, description="Timestamp when processed")
    prompt_version: str = Field(..., description="Version of LLaVA prompt used")

    @validator('validated', always=True)
    def determine_validation(cls, v, values):
        result = values.get('llava_result')
        threshold = values.get('validation_threshold')
        if result is not None and threshold is not None:
            return (result.is_target_object and result.confidence >= threshold)
        return v


class ValidationStatistics(BaseModel):
    """
    Statistics for the validation process
    """
    total_segments: int = Field(..., description="Total segments processed")
    validated_segments: int = Field(..., description="Segments that passed validation")
    rejected_segments: int = Field(..., description="Segments that failed validation")
    validation_rate: float = Field(..., description="Percentage of segments validated")
    average_detector_confidence: float = Field(..., description="Average detector confidence")
    average_llava_confidence: float = Field(..., description="Average LLaVA confidence")
    processing_time_seconds: float = Field(..., description="Total processing time")
    segments_per_second: float = Field(..., description="Processing rate")
    high_confidence_validations: int = Field(..., description="Validations with >0.8 confidence")
    low_confidence_validations: int = Field(..., description="Validations with <0.5 confidence")
    prompt_version: str = Field(..., description="LLaVA prompt version used")
    validation_threshold: float = Field(..., description="Confidence threshold used")
    processed_at: datetime = Field(default_factory=datetime.now, description="Timestamp when stats computed")

    @validator('validation_rate', always=True)
    def calculate_validation_rate(cls, v, values):
        total = values.get('total_segments')
        valid = values.get('validated_segments')
        if total is not None and total > 0:
            return valid / total
        return 0.0

    @validator('segments_per_second', always=True)
    def calculate_processing_rate(cls, v, values):
        time_s = values.get('processing_time_seconds')
        total = values.get('total_segments')
        if time_s and time_s > 0:
            return total / time_s
        return 0.0


class ValidationBatch(BaseModel):
    """
    Results for a batch of validation operations
    """
    batch_id: str = Field(..., description="Unique batch identifier")
    segment_records: List[SegmentValidationRecord] = Field(..., description="Individual segment results")
    statistics: ValidationStatistics = Field(..., description="Batch statistics")
    prompt_file: str = Field(..., description="Path to prompt file used")
    validation_threshold: float = Field(..., description="Confidence threshold used")
    started_at: datetime = Field(default_factory=datetime.now, description="Batch start timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Batch completion timestamp")

    def mark_completed(self):
        self.completed_at = datetime.now()

    def get_validated_segments(self) -> List[SegmentValidationRecord]:
        return [r for r in self.segment_records if r.validated]

    def get_rejected_segments(self) -> List[SegmentValidationRecord]:
        return [r for r in self.segment_records if not r.validated]
