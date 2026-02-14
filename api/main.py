from contextlib import asynccontextmanager
from typing import Optional
import fastapi
from sqlmodel import Field, SQLModel, Session, create_engine, select
from sqlalchemy.orm import Mapped

EPOCH = 1756684800 # [2025-09-07T04:52:31+00:00] in UNIX time
from sqlalchemy.orm import RelationshipProperty

from sqlmodel import SQLModel, Field, Relationship
from fastapi import Depends


# --- Core models ---
class Train(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    train_code: str = Field(index=True, unique=True)
    name: str
    type: str
    length_m: float
    max_speed_kmh: float
    priority: int
    direction: str

    accel_mps2: float | None = None
    decel_mps2: float | None = None

    instances: list["TrainInstance"] = Relationship(back_populates="train")
    schedules: list["Schedule"] = Relationship(back_populates="train")


class TrainInstance(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    train_id: int | None = Field(default=None, foreign_key="train.id", index=True)
    service_epoch: int | None = Field(default=None, description="epoch seconds for the run")
    status: str = Field(default="scheduled")
    consist_id: str | None = None

    train_id: int | None = Field(default=None, foreign_key="train.id", index=True)
    train: Optional["Train"] = Relationship(back_populates="instances")
    actuals: list["ActualSchedule"] = Relationship(back_populates="train_instance")


class Schedule(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    train_id: int | None = Field(default=None, foreign_key="train.id", index=True)

    scheduled_departure_epoch: int
    scheduled_arrival_epoch: int
    layover_seconds: int | None = None

    station_id: int | None = Field(default=None, foreign_key="station.id", index=True)
    station: Optional["Station"] = Relationship()

    train: Optional["Train"] = Relationship(back_populates="schedules")
    actuals: list["ActualSchedule"] = Relationship(back_populates="schedule")


class ActualSchedule(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)

    train_instance_id: int | None = Field(default=None, foreign_key="traininstance.id", index=True)
    schedule_id: int | None = Field(default=None, foreign_key="schedule.id", index=True)

    scheduled_departure: int | None = None
    actual_departure: int | None = None
    scheduled_arrival: int | None = None
    actual_arrival: int | None = None
    layover_seconds: int | None = None
    actual_layover_seconds: int | None = None

    train_instance: Optional["TrainInstance"] = Relationship(back_populates="actuals")
    schedule: Optional["Schedule"] = Relationship(back_populates="actuals")


# --- Topology / infra ---
class Track(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    track_id: str = Field(index=True, unique=True)
    name: str | None = None

    length_m: float | None = None
    has_platform: bool = True
    is_cargo_platform: bool = False
    is_main_line: bool = True
    is_dead_end: bool = False
    is_loop_line: bool = False

    station_id: int | None = Field(default=None, foreign_key="station.id", index=True)
    station: Optional["Station"] = Relationship(back_populates="tracks")


class Station(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    station_code: str = Field(index=True, unique=True)
    name: str

    tracks: list["Track"] = Relationship(back_populates="station")
    block_sections_from: list["BlockSection"] = Relationship(
        back_populates="from_station_rel",
        sa_relationship_kwargs={"primaryjoin": "Station.id==BlockSection.from_station_id"}
    )
    block_sections_to: list["BlockSection"] = Relationship(
        back_populates="to_station_rel",
        sa_relationship_kwargs={"primaryjoin": "Station.id==BlockSection.to_station_id"}
    )


class BlockSection(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    block_id: str = Field(index=True, unique=True)
    from_station_id: int | None = Field(default=None, foreign_key="station.id", index=True)
    to_station_id: int | None = Field(default=None, foreign_key="station.id", index=True)

    length_km: float | None = None
    line_speed_kmh: float | None = None
    bidirectional: bool = True
    electric: bool = True
    signal_num: int | None = None
    signal_aspects: int | None = None

    from_station_rel: Optional["Station"] = Relationship(back_populates="block_sections_from", sa_relationship=RelationshipProperty("Station", foreign_keys="[BlockSection.from_station_id]"))
    to_station_rel: Optional["Station"] = Relationship(back_populates="block_sections_to", sa_relationship=RelationshipProperty("Station", foreign_keys="[BlockSection.to_station_id]"))


class Route(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    route_id: str = Field(index=True, unique=True)
    name: str | None = None
    from_station_id: int | None = Field(default=None, foreign_key="station.id", index=True)
    to_station_id: int | None = Field(default=None, foreign_key="station.id", index=True)

    from_station: Optional["Station"] = Relationship(sa_relationship_kwargs={"primaryjoin": "Route.from_station_id==Station.id"})
    to_station: Optional["Station"] = Relationship(sa_relationship_kwargs={"primaryjoin": "Route.to_station_id==Station.id"})
    route_stations: list["RouteStation"] = Relationship(back_populates="route")


class RouteStation(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    route_id: int | None = Field(default=None, foreign_key="route.id", index=True)
    station_id: int | None = Field(default=None, foreign_key="station.id", index=True)
    ordinal: int

    route: Optional["Route"] = Relationship(back_populates="route_stations")
    station: Optional["Station"] = Relationship()


class TrainLog(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    train_id: int | None = Field(default=None, foreign_key="train.id", index=True)
    timestamp_epoch: int = Field(index=True)
    event: str
    details: str | None = None


sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)

@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    # Startup
    create_db_and_tables()
    yield
    # Shutdown

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    from sqlmodel import Session
    with Session(engine) as session:
        yield session

app = fastapi.FastAPI(lifespan=lifespan)

@app.get("/trains")
def get_trains(db: Session = Depends(get_session)):
    return db.exec(select(Train)).all()

@app.get("/trains/{train_id}")
def get_train(train_id: int, db: Session = Depends(get_session)):
    return db.exec(select(Train).where(Train.id == train_id)).first()

@app.post("/trains")
def create_train(train: Train, db: Session = Depends(get_session)):
    db.add(train)
    db.commit()
    db.refresh(train)
    return train

@app.delete("/trains/{train_id}")
def delete_train(train_id: int, db: Session = Depends(get_session)):
    train = db.exec(select(Train).where(Train.id == train_id)).first()
    if train:
        db.delete(train)
        db.commit()
        return {"message": "Train deleted"}
    return {"message": "Train not found"}

@app.post("/stations/")
def create_station(station: Station, db: Session = Depends(get_session)):
    db.add(station)
    db.commit()
    db.refresh(station)
    return station

@app.delete("/stations/{station_id}")
def delete_station(station_id: int, db: Session = Depends(get_session)):
    station = db.exec(select(Station).where(Station.id == station_id)).first()
    if station:
        db.delete(station)
        db.commit()
        return {"message": "Station deleted"}
    return {"message": "Station not found"}

@app.get("/stations/")
def get_stations(db: Session = Depends(get_session)):
    return db.exec(select(Station)).all()

# @app.post("/trains/{train_id}/schedule")
# def create_train_schedule(train_id: int, schedule: Schedule, db: Session = Depends(get_session)):
#     schedule.train_id = train_id
#     db.add(schedule)
#     db.commit()
#     db.refresh(schedule)
#     return schedule

@app.post("/trains/{train_id}/schedule")
def create_train_schedule(train_id: int, schedule: list[Schedule], db: Session = Depends(get_session)):
    for item in schedule:
        item.train_id = train_id
        db.add(item)
    db.commit()
    return {"message": "Schedule created"}

@app.get("/trains/{train_id}/schedule")
def get_train_schedule(train_id: int):
    pass

@app.get("/train-instances/{train_id}")
def get_train_instances(train_id: int, date: str | None = None):
    pass

@app.post("/whatif/start")
def whatif_start():
    pass

@app.get("/whatif/{scenario_id}/status")
def whatif_status(scenario_id: int):
    pass

@app.get("/routes")
def get_routes():
    pass

@app.get("/routes/{route_id}")
def get_route(route_id: int):
    pass

@app.post("/whatif/create")
def whatif_create():
    pass

@app.get("/whatif/{scenario_id}/results")
def whatif_results(scenario_id: int):
    pass

import uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)