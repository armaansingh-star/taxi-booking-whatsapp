from pydantic import BaseModel


class CustomerLookup(BaseModel):
    customer_id: int
    full_name: str
    primary_phone_no: str
    email: str | None = None
    address: str | None = None
    town: str | None = None
    postal_code: str | None = None
    is_corporate: bool = False


class CustomerCreate(BaseModel):
    full_name: str
    primary_phone_no: str
    email: str | None = None
    address: str | None = None
    town: str | None = None
    postal_code: str | None = None
    is_corporate: bool = False
