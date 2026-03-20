import logging

from app.database import get_pool
from app.models.customer import CustomerCreate, CustomerLookup

log = logging.getLogger(__name__)


def normalize_phone(raw_phone: str) -> str:
    """Strip 'whatsapp:' prefix from Twilio phone numbers."""
    return raw_phone.replace("whatsapp:", "").strip()


async def lookup_by_phone(raw_phone: str) -> CustomerLookup | None:
    """Find a customer by their phone number."""
    phone = normalize_phone(raw_phone)
    pool = get_pool()
    row = await pool.fetchrow(
        """
        SELECT customer_id, full_name, primary_phone_no, email,
               address, town, postal_code, is_corporate
        FROM ingest_db.customers
        WHERE primary_phone_no = $1 OR secondary_phone_no = $1
        """,
        phone,
    )
    if row:
        return CustomerLookup(**dict(row))
    return None


async def create_customer(data: CustomerCreate) -> int:
    """Insert a new customer and return the customer_id."""
    pool = get_pool()
    customer_id = await pool.fetchval(
        """
        INSERT INTO ingest_db.customers
            (full_name, primary_phone_no, email, address, town, postal_code, is_corporate, status)
        VALUES ($1, $2, $3, $4, $5, $6, $7, 'Active')
        RETURNING customer_id
        """,
        data.full_name,
        data.primary_phone_no,
        data.email,
        data.address,
        data.town,
        data.postal_code,
        data.is_corporate,
    )
    log.info("Created customer %s (id=%s)", data.full_name, customer_id)
    return customer_id
