"""The brief's claim_packet.json contract. Extra fields are allowed; dropping one fails validation before writing.
Numbers are Optional because 'empty when unknown' is correct; evidence fields (frame_ref, source+url) are checked."""
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, model_validator

Num = Optional[float]


class Open(BaseModel):
    model_config = ConfigDict(extra="allow")


class Sweep(Open):
    id: str
    captured_at: str
    device: str
    duration_s: Num
    country: str
    currency: str


class Room(Open):
    length_m: Num
    width_m: Num
    height_m: Num
    floor_area_m2: Num
    wall_area_m2: Num
    shelved_wall_area_m2: Num
    scale_method: str
    confidence: float


class Price(Open):
    amount: Num
    source: str
    url: str
    retrieved_at: str

    @model_validator(mode="after")
    def sourced(self):  # a price with no retrievable source is a fabricated price
        if self.amount is not None and not (self.source and self.url and self.retrieved_at):
            raise ValueError("price without source/url/retrieved_at")
        return self


class Replacement(Price):
    converted: bool


class Used(Price):
    condition_assumed: str


class Book(Open):
    id: str
    shelf: str
    position: int
    frame_ref: str
    status: Literal["identified", "unidentified", "needs_appraisal"]
    title: str
    author: str
    edition: str
    isbn: str
    spine_height_cm: Num
    spine_thickness_cm: Num
    id_confidence: float
    replacement_cost: Replacement
    used_value: Used

    @model_validator(mode="after")
    def rules(self):
        if not self.frame_ref.startswith("frames/"):
            raise ValueError("frame_ref must point to a saved frame")
        if self.status == "needs_appraisal" and (self.replacement_cost.amount or self.used_value.amount):
            raise ValueError("appraisal items must not be auto-priced")
        if self.status == "unidentified" and self.title:
            raise ValueError("unidentified book with a title")
        return self


class ItemPrice(Open):
    low: Num
    high: Num
    source: str
    url: str
    retrieved_at: str


class Item(Open):
    id: str
    category: str
    description: str
    brand_model: str
    frame_ref: str
    dimensions_cm: dict[Literal["w", "h", "d"], Num]
    status: Literal["priced", "range", "needs_appraisal"]
    replacement_cost: ItemPrice
    confidence: float

    @model_validator(mode="after")
    def rules(self):
        if self.status == "needs_appraisal" and self.replacement_cost.low is not None:
            raise ValueError("appraisal items must not be auto-priced")
        if self.replacement_cost.low is not None and not (self.replacement_cost.source and self.replacement_cost.url):
            raise ValueError("item price without source/url")
        return self


class Totals(Open):
    book_count: int
    books_identified: int
    books_unidentified: int
    shelf_run_m: Num
    books_replacement_cost: Num
    books_used_value: Num
    items_replacement_cost_low: Num
    items_replacement_cost_high: Num
    excluded_from_totals: int


class ReviewLine(Open):
    ref_id: str
    reason: str


class ClaimPacket(Open):
    sweep: Sweep
    room: Room
    books: list[Book]
    items: list[Item]
    totals: Totals
    review_queue: list[ReviewLine]
