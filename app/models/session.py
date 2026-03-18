from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from app.models.booking import BookingExtraction, FareEstimate


class ConversationState(str, Enum):
    IDENTIFY = "identify"
    ONBOARD_NAME = "onboard_name"
    ONBOARD_EMAIL = "onboard_email"
    ONBOARD_ADDRESS = "onboard_address"
    COLLECTING = "collecting"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    EDITING = "editing"
    POST_BOOKING = "post_booking"


class ConversationSession(BaseModel):
    state: ConversationState = ConversationState.IDENTIFY
    customer_id: int | None = None
    customer_name: str | None = None
    customer_phone: str | None = None
    # Onboarding scratch fields
    onboard_email: str | None = None
    # Booking data
    booking: BookingExtraction = Field(default_factory=BookingExtraction)
    fare: FareEstimate | None = None
    pickup_coords: str | None = None   # "lat,lng" after geocoding
    dropoff_coords: str | None = None  # "lat,lng" after geocoding
    last_booking_id: int | None = None
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def touch(self):
        self.updated_at = datetime.utcnow()

    def reset_booking(self):
        self.booking = BookingExtraction()
        self.fare = None
        self.pickup_coords = None
        self.dropoff_coords = None
        self.state = ConversationState.COLLECTING
