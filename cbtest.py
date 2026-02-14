import datetime
import math
import random
import simpy
import simpy.resources.resource
from simpy.core import EmptySchedule, Infinity as SimpyInfinity

from networks.kkdi_tpj_network import create_alu_tpj_network, create_tpj_kkdi_network
from train_lib.models import DecisionSuggestion, Train, Station, Track, BlockSection
from train_lib.constants import ARRIVAL, DEPARTURE
from train_lib.simulation import Simulation
from etrainlib import ETrainAPISync

sim = Simulation()

network = create_alu_tpj_network(sim)

# lli_network = create_lli_tpj_network(sim)

# [TPJ, KRMG, KRUR, VEL, PDKT, TYM, CTND, KKDI] = network.get_stations()
[TPJ, GOC, SRGM, LLI, PMB, KKPM, SLTH, ALU] = network.get_stations()

scheduling_horizon = 120  # minutes
current_time = int(datetime.datetime.now().timestamp())


etrain = ETrainAPISync()

# 1. Fetch live trains from all boundary stations
boundary_stations = [
    stn for stn in network.stations if getattr(stn, "is_boundary", False)
]
print(f"Boundary Stations: {[stn.stn_code for stn in boundary_stations]}")

live_trains_map = {}
for stn in boundary_stations:
    print(f"Fetching live trains for {stn.stn_code}...")
    try:
        # We need the full station name for the API, but `stn.stn_code` is just the code.
        # Ideally we'd have the name in the Station object.
        # For now, let's use a mapping or just the code if the API accepts it (API usually needs "NAME-CODE").
        # The `get_live_station` in `etrainlib` expects `stn_name` and `stn_code`.
        # API construction: f"/station/{stn_name.replace(' ', '-')}-{stn_code.upper()}/live"
        # If we don't have the name, we might fail or need a lookup.
        # Let's assume a default or lookup.
        # Mapping for known stations in this specific network:
        stn_names = {
            "TPJ": "TIRUCHCHIRAPALI",
            "ALU": "ARIYALUR",
            "KKDI": "KARAIKKUDI JN",
            # Add others if they become boundaries later
        }
        stn_name = stn_names.get(stn.stn_code.upper(), stn.stn_code.upper())

        l_trains = etrain.get_live_station(stn.stn_code.upper(), stn_name)
        for t in l_trains:
            # key by train_no
            if t["train_no"] not in live_trains_map:
                live_trains_map[t["train_no"]] = t
                # Verify we keep the train object that has the most relevant info?
                # The 'exp_arr' depends on the station we fetched it from.
                # We need this to calculate the date.
                # We should probably store which station we fetched it from to use for date calc.
                t["_fetched_from"] = stn.stn_code.upper()
    except Exception as e:
        print(f"Error fetching {stn.stn_code}: {e}")

print(f"Found {len(live_trains_map)} unique trains.")


for train_no, train in live_trains_map.items():
    print(
        f"{train['train_no']} {train['train_name']} - Fetched from {train['_fetched_from']}"
    )
    try:
        schedule = etrain.get_train_schedule(train["train_no"], train["train_name"])
    except Exception as e:
        print(f"Failed to get schedule for {train['train_no']}: {e}")
        continue

    # sim_train instantiation moved to after validation

    # 1. Filter schedule to only include stations in our network
    schedule_in_network = []
    for station in schedule:
        stn_obj = network.get_station_by_code(station["code"].lower())
        if stn_obj:
            schedule_in_network.append((station, stn_obj))

    if not schedule_in_network:
        print(
            f"Train {train['train_no']} does not pass through the simulation network."
        )
        continue

    # 2. Determine start and end stations in the network
    start_schedule_entry, start_stn_obj = schedule_in_network[0]
    end_schedule_entry, end_stn_obj = schedule_in_network[-1]

    print(
        f"Train traverses network from {start_stn_obj.stn_code} to {end_stn_obj.stn_code}"
    )

    # 3. Get all stations in the simulation path
    sim_path_stations = network.get_stations_between(start_stn_obj, end_stn_obj)
    if not sim_path_stations:
        if start_stn_obj == end_stn_obj:
            sim_path_stations = [start_stn_obj]
        else:
            print(
                f"[WARNING] Could not find path between {start_stn_obj.stn_code} and {end_stn_obj.stn_code}. Using schedule only."
            )
            sim_path_stations = [s[1] for s in schedule_in_network]

    if len(sim_path_stations) <= 1:
        print(
            f"Train {train['train_no']} only touches {sim_path_stations[0].stn_code} in the network. Simulating single stop arrival/departure."
        )
    else:
        print(f"Train {train['train_no']} traverses {len(sim_path_stations)} stations.")

    # Calculate the start date of the train
    # We use the station we fetched the data from (train['_fetched_from']) as the reference point
    current_station_code = train["_fetched_from"]

    # Helper to parse "HH:MM"
    def get_time_obj(time_str):
        try:
            return datetime.datetime.strptime(time_str, "%H:%M, %d %b").time()
        except ValueError:
            return None

    # 1. Get Live Event Time
    event_time_str = train["exp_arr"]
    print(f"DEBUG: Event time string for {train['train_no']}: {event_time_str}")
    if event_time_str == "Source":
        event_time_str = train["exp_dept"]

    if event_time_str in ["N/A", None]:
        # Fallback if no time info
        train_scheduled_date = datetime.datetime.now().date()
        print(f"[WARNING] No live time info for {train['train_no']}. Assuming today.")
    else:
        # 2. Determine Event Date at Current Station
        now = datetime.datetime.now()
        event_time = get_time_obj(event_time_str)

        if event_time:
            event_date = now.date()
            event_dt = datetime.datetime.combine(event_date, event_time)

            # Adjust for midnight crossover (heuristic: if diff > 12h, likely separate days)
            diff = (event_dt - now).total_seconds()
            if diff > 12 * 3600:
                # e.g., Now=10:00, Event=23:00 (of yesterday, mostly) -> actually if Now is 01:00 and Event is 23:00, Diff is +22h
                # If Event is 23:00 and Now is 01:00 (next day), we want Event to be yesterday.
                # Wait: Event (01/01 23:00) vs Now (01/01 01:00) -> Diff +22h?
                # No: EventDt (Today 23:00) - Now (Today 01:00) = +22h.
                # Real: Event was yesterday 23:00. So we subtract 1 day.
                event_dt -= datetime.timedelta(days=1)
            elif diff < -12 * 3600:
                # e.g., Now=23:00, Event=01:00 (Today 01:00). Diff = -22h.
                # Real: Event is tomorrow 01:00. So we add 1 day.
                event_dt += datetime.timedelta(days=1)

            # 3. Find Day Offset from Schedule
            current_stn_schedule = next(
                (
                    s
                    for s in schedule
                    if s["code"].lower() == current_station_code.lower()
                ),
                None,
            )

            if current_stn_schedule:
                # 'a_day' / 'd_day' example: "1", "2"
                # Use 'a_day' if arriving, 'd_day' if departing/starting
                day_val = (
                    current_stn_schedule.get("a_day")
                    if current_station_code != schedule[0]["code"]
                    else current_stn_schedule.get("d_day")
                )
                if not day_val:
                    day_val = "1"

                day_offset = int(day_val)

                # 4. Calculate Origin Date
                # Start Date = Event Date - (Day Offset - 1)
                train_scheduled_date = event_dt.date() - datetime.timedelta(
                    days=(day_offset - 1)
                )
                print(
                    f"DEBUG: Calculated Start Date for {train['train_no']}: {train_scheduled_date} (Event: {event_dt}, Day Offset: {day_offset})"
                )
            else:
                train_scheduled_date = now.date()
                print(
                    f"DEBUG: Could not find current station in schedule for {train['train_no']}. Defaulting to today: {train_scheduled_date}"
                )
        else:
            train_scheduled_date = now.date()
            print(
                f"DEBUG: Could not parse event time for {train['train_no']}. Defaulting to today: {train_scheduled_date}"
            )
    # 4. Fetch running status ONCE (already done sort of, but let's be sure to have it)
    running_status = etrain.get_running_status(
        train["train_no"],
        train["train_name"],
        train_scheduled_date,  # Train's original departure date, calculate it from schedule if needed for accuracy
        schedule[0]["code"],  # Train's source station code
    )

    # Helper to parse time
    def parse_time(time_obj):
        if (
            time_obj is None
            or time_obj == "N/A"
            or time_obj == "Source"
            or time_obj == "Destination"
        ):
            return None
        if isinstance(time_obj, (int, float)):
            return int(time_obj)
        if isinstance(time_obj, datetime.datetime):
            return int(time_obj.timestamp())
        try:
            # Try parsing if string? logic from original code seems complicated
            return None
        except:
            return None

    # We need to collect stop data for all stations in sim_path_stations
    stops_data = []  # List of {'stn': stn_obj, 'arr': timestamp, 'dept': timestamp, 'is_stop': bool}

    for stn_obj in sim_path_stations:
        # Check if station is in schedule
        schedule_entry = next(
            (s for s, obj in schedule_in_network if obj == stn_obj), None
        )

        # Check running status
        def match_status(s):
            # 1. Check by stn_code (if available in running status - usually not, only name)
            if s.get("name") and s["name"].lower() == stn_obj.stn_code.lower():
                return True
            # 2. Check by name from schedule (if exists)
            if schedule_entry and s.get("name") and s["name"] == schedule_entry["name"]:
                return True
            # 3. Check by partial match or known mapping? Likely unsafe without explicit mapping.
            return False

        status_entry = next((s for s in running_status if match_status(s)), None)

        arr_time = None
        dept_time = None
        is_scheduled_stop = False

        if status_entry:
            # Use Actual Arrival/Departure if available, else Scheduled
            if (
                status_entry.get("act_arr")
                and status_entry["act_arr"] != "N/A"
                and status_entry["act_arr"] != "Source"
            ):
                arr_time = status_entry["act_arr"]
            elif (
                status_entry.get("tt_arr")
                and status_entry["tt_arr"] != "N/A"
                and status_entry["tt_arr"] != "Source"
            ):
                arr_time = status_entry["tt_arr"]

            if (
                status_entry.get("act_dept")
                and status_entry["act_dept"] != "N/A"
                and status_entry["act_dept"] != "Destination"
            ):
                dept_time = status_entry["act_dept"]
            elif (
                status_entry.get("tt_dept")
                and status_entry["tt_dept"] != "N/A"
                and status_entry["tt_dept"] != "Destination"
            ):
                dept_time = status_entry["tt_dept"]

        if schedule_entry:
            is_scheduled_stop = True  # It's in the main schedule
            # Fallback to schedule times if running status missing
            if arr_time is None and schedule_entry.get("a"):
                try:
                    sch_time_str = schedule_entry["a"]  # "HH:MM"
                    sch_day_str = schedule_entry.get("a_day", "1")
                    sch_time = datetime.datetime.strptime(sch_time_str, "%H:%M").time()
                    sch_day_offset = int(sch_day_str)

                    # Calculate datetime
                    sch_date = train_scheduled_date + datetime.timedelta(
                        days=(sch_day_offset - 1)
                    )
                    sch_dt = datetime.datetime.combine(sch_date, sch_time)
                    arr_time = int(sch_dt.timestamp())
                    print(
                        f"DEBUG: Using schedule fallback for Arr at {stn_obj.stn_code}: {sch_dt}"
                    )
                except Exception as e:
                    print(
                        f"DEBUG: Schedule fallback failed for Arr at {stn_obj.stn_code}: {e}"
                    )

            if dept_time is None and schedule_entry.get("d"):
                try:
                    sch_time_str = schedule_entry["d"]  # "HH:MM"
                    sch_day_str = schedule_entry.get("d_day", "1")
                    sch_time = datetime.datetime.strptime(sch_time_str, "%H:%M").time()
                    sch_day_offset = int(sch_day_str)

                    # Calculate datetime
                    sch_date = train_scheduled_date + datetime.timedelta(
                        days=(sch_day_offset - 1)
                    )
                    sch_dt = datetime.datetime.combine(sch_date, sch_time)
                    dept_time = int(sch_dt.timestamp())
                    print(
                        f"DEBUG: Using schedule fallback for Dept at {stn_obj.stn_code}: {sch_dt}"
                    )
                except Exception as e:
                    print(
                        f"DEBUG: Schedule fallback failed for Dept at {stn_obj.stn_code}: {e}"
                    )

        # Normalize times to timestamps (int)
        if isinstance(arr_time, datetime.datetime):
            arr_time = int(arr_time.timestamp())
        if isinstance(dept_time, datetime.datetime):
            dept_time = int(dept_time.timestamp())

        # Handle Source/Dest logic
        if arr_time == "Source":
            arr_time = None
        if dept_time == "Destination":
            dept_time = None
        if arr_time == "N/A":
            arr_time = None
        if dept_time == "N/A":
            dept_time = None

        # Logic: "only simulate it arriving and departing the station if the train arrives tpj from out of simulation and exits out of simulation."
        # Meaning:
        # - If Current Stn == Train Source: It doesn't "arrive from out of sim". It starts here. So Arr = Dept (Instant spawn).
        # - If Current Stn == Train Dest: It doesn't "exit out of sim" (it stops here). So Dept = Arr (Instant terminate).
        # - Otherwise (Standard Stop/Pass): It arrives from out and exits to out (or next station). Use buffer if missing.

        is_source_station = schedule[0]["code"].lower() == stn_obj.stn_code.lower()
        is_dest_station = schedule[-1]["code"].lower() == stn_obj.stn_code.lower()

        if arr_time is None and isinstance(dept_time, int):
            if is_source_station:
                arr_time = dept_time  # Start exactly at departure (no arrival buffer)
                print(
                    f"DEBUG: {stn_obj.stn_code} is Source. Setting Arr = Dept ({dept_time})"
                )
            else:
                arr_time = dept_time - 300  # 5 mins before (Arriving from outside)

        if dept_time is None and isinstance(arr_time, int):
            if is_dest_station:
                dept_time = (
                    arr_time  # Terminate exactly at arrival (no departure buffer)
                )
                print(
                    f"DEBUG: {stn_obj.stn_code} is Dest. Setting Dept = Arr ({arr_time})"
                )
            else:
                dept_time = arr_time + 300  # 5 mins after (Departing to outside)

        if isinstance(arr_time, int) and isinstance(dept_time, int):
            stops_data.append(
                {
                    "stn": stn_obj,
                    "arr": arr_time,
                    "dept": dept_time,
                    "is_scheduled": is_scheduled_stop,
                }
            )
        else:
            stops_data.append(
                {
                    "stn": stn_obj,
                    "arr": arr_time if isinstance(arr_time, int) else None,
                    "dept": dept_time if isinstance(dept_time, int) else None,
                    "is_scheduled": is_scheduled_stop,
                }
            )

    # 5. Interpolation
    # Find indices of stations with known times
    known_indices = [i for i, data in enumerate(stops_data) if data["arr"] is not None]

    if not known_indices:
        print(
            f"[WARNING] No timing info found for train {train['train_no']}. Skipping."
        )
        continue

    # Interpolate between known points
    for i in range(len(known_indices) - 1):
        start_idx = known_indices[i]
        end_idx = known_indices[i + 1]

        start_data = stops_data[start_idx]
        end_data = stops_data[end_idx]

        total_time_diff = end_data["arr"] - start_data["dept"]
        if total_time_diff < 0:
            print(
                f"[WARNING] Negative time difference between {start_data['stn'].stn_code} and {end_data['stn'].stn_code}. Data might be bad."
            )
            continue

        # Calculate total distance
        # We need distance between stations. Assuming roughly proportional to index difference or count
        # Ideally use network.get_block_sections_between to sum distances

        # Simple interpolation by count (assuming roughly equal distance blocks if no dist info)
        # OR use cumulative distance if available.
        # Let's try to get distance from BlockSection objects

        segment_stations = [s["stn"] for s in stops_data[start_idx : end_idx + 1]]
        # Calculate distances for each step in segment
        segment_dists = [0]
        cumulative_dist = 0
        for k in range(len(segment_stations) - 1):
            u = segment_stations[k]
            v = segment_stations[k + 1]
            # Get block dist
            blocks = network.get_block_sections_between(u, v)
            dist = (
                sum(b.length_km for b in blocks) if blocks else 10
            )  # Default 10km if missing
            cumulative_dist += dist
            segment_dists.append(cumulative_dist)

        total_dist = segment_dists[-1]

        if total_dist == 0:
            total_dist = 1  # Avoid div by zero

        # Interpolate
        for j in range(start_idx + 1, end_idx):
            # Relative position in segment
            rel_idx = j - start_idx
            dist_from_start = segment_dists[rel_idx]

            fraction = dist_from_start / total_dist
            interpolated_travel_time = fraction * total_time_diff

            arrival = start_data["dept"] + interpolated_travel_time
            departure = arrival  # 0 stop time for non-scheduled stops

            stops_data[j]["arr"] = int(arrival)
            stops_data[j]["dept"] = int(departure)
            stops_data[j]["is_interpolated"] = True

    # 6. Horizon Filter
    # Check if the train enters the network within the horizon
    should_add_train = False

    if stops_data:
        # Find first valid time point to determine entry
        first_valid_stop = next((s for s in stops_data if s["arr"] is not None), None)

        if first_valid_stop:
            entry_time = first_valid_stop["arr"]
            horizon_time = current_time + (scheduling_horizon * 60)

            if entry_time > horizon_time:
                print(
                    f"Skipping {train['train_no']} - Enters network at {datetime.datetime.fromtimestamp(entry_time)} (Horizon: {datetime.datetime.fromtimestamp(horizon_time)})"
                )
                continue
            else:
                print(
                    f"Including {train['train_no']} - Enters at {datetime.datetime.fromtimestamp(entry_time)}"
                )
                should_add_train = True
        else:
            print(
                f"Skipping {train['train_no']} - No valid time points after interpolation."
            )
            continue
    else:
        print(f"Skipping {train['train_no']} - No stops generated.")
        continue

    # 7. Add to Simulation
    if should_add_train:
        sim_train = Train(
            sim,
            f"{train['train_no']} - {train['train_name']}",
            [],
            max_speed=110,  # Default higher speed
            priority=1,
            length=300,
            weight=1173,
            initial_delay=0,
            hp=6120,
            accel_mps2=0.5,
            decel_mps2=0.5,
        )

        stops_added_count = 0
        for data in stops_data:
            if data["arr"] is not None:
                arr_min = max(0, data["arr"] - current_time) // 60
                dept_min = max(0, data["dept"] - current_time) // 60

                sim_train.schedule_stop(data["stn"], arr_min, dept_min, 1)
                stops_added_count += 1
                print(
                    f"Scheduled stop at {data['stn'].stn_code} ({'Interp' if data.get('is_interpolated') else 'Real'}) Arr: {arr_min} Dept: {dept_min}"
                )

        if stops_added_count == 0:
            print(
                f"[WARNING] Created train {train['train_no']} but added 0 stops! (arr times were None or < current? check logic)"
            )
            # Might want to remove from sim if 0 stops?
            # But we already created it.

    print(
        f"Train {train['train_no']} processed. Stops added: {stops_added_count if 'stops_added_count' in locals() else 0}"
    )


# Print the full schedule for verification
for train in sim.trains:
    print(f"Train {train.id} schedule:")
    for stop in train.schedule:
        print(
            f"  Station: {stop.station.stn_code}, Arr: {stop.arrival_time} min, Dept: {stop.departure_time} min"
        )


# Existing trains
# train1 = Train(
#     sim,
#     "T1",
#     [],
#     max_speed=80,
#     priority=1,
#     length=300,
#     weight=1173,
#     initial_delay=0,
#     hp=6120,
#     accel_mps2=0.5,
#     decel_mps2=0.5,
# )
# train1.schedule_stop(TPJ, 0, 10, 1)
# train1.schedule_stop(KRMG, 20, 25, 0)
# train1.schedule_stop(KRUR, 40, 45, 1)
# train1.schedule_stop(VEL, 50, 50, 0)
# train1.schedule_stop(PDKT, 70, 75, 0)
# train1.schedule_stop(TYM, 85, 90, 0)
# train1.schedule_stop(CTND, 100, 105, 1)
# train1.schedule_stop(KKDI, 120, 125, 0)

# 0 = 5:30
# 10 = 5:40
# 33 = 6:03
# 35 = 6:05
# 80 = 06:50
# 85 = 06:55

# train2 = Train(
#     sim,
#     "T2",
#     [],
#     max_speed=110,
#     priority=1,
#     length=523,
#     weight=1130,
#     initial_delay=7,
#     hp=6350,
#     accel_mps2=0.9,
#     decel_mps2=0.85,
# )
# train2.start_time(0)
# train2.schedule_stop(KKDI, 0, 10, 0)
# train2.schedule_stop(CTND, 16, 16, 0)
# train2.schedule_stop(TYM, 23, 23, 0)
# train2.schedule_stop(PDKT, 33, 35, 0)
# train2.schedule_stop(VEL, 44, 44, 0)
# train2.schedule_stop(KRUR, 55, 55, 0)
# train2.schedule_stop(KRMG, 68, 68, 0)
# train2.schedule_stop(TPJ, 90, 95, 0)
# train2.set_direction("DOWN")

# train3 = Train(sim, "T3", [], max_speed=110, priority=2, length=268, weight=1170, initial_delay=5, hp=6120, accel_mps2=0.5, decel_mps2=0.5)
# train3.start_time(50)
# train3.schedule_stop(TPJ, 0, 5, 1)
# train3.schedule_stop(KRMG, 14, 15, 1)
# train3.schedule_stop(KRUR, 27, 28, 1) # Run through main
# train3.schedule_stop(VEL, 39, 40, 1)
# train3.schedule_stop(PDKT, 53, 55, 0)
# train3.schedule_stop(TYM, 67, 68, 1)
# train3.schedule_stop(CTND, 75, 76, 0) # Run through main
# train3.schedule_stop(KKDI, 93, 95, 1)
# train3.set_direction("UP")

# train4 = Train(sim, "T4", [], max_speed=120, priority=2, length=320, weight=2580, initial_delay=0, hp=4500, accel_mps2=0.5, decel_mps2=0.5)
# train4.schedule_stop(TPJ, 40, 45, 1)
# train4.schedule_stop(GOC, 60, 65, 1)
# train4.schedule_stop(SRGM, 80, 85, 1)
# train4.schedule_stop(LLI, 95, 100, 1)
# train4.schedule_stop(PMB, 110, 115, 1)
# train4.schedule_stop(KKPM, 130, 135, 1)
# train4.schedule_stop(SLTH, 140, 140, 0)
# train4.schedule_stop(ALU, 150, 155, 0)
# train4.set_direction("UP")

# train5 = Train(env, "T5", [], max_speed=90, priority=3, length=250, weight=1277, initial_delay=0, hp=4500, accel_mps2=0.3, decel_mps2=0.2)
# train5.start_time(10)
# train5.schedule_stop(TPJ, 60, 65, 2)
# train5.schedule_stop(KRMG, 80, 80, 0)
# train5.schedule_stop(KRUR, 90, 90, 0)
# train5.schedule_stop(VEL, 110, 110, 0)
# train5.schedule_stop(PDKT, 150, 150, 0)
# train5.schedule_stop(TYM, 210, 210, 0)
# train5.schedule_stop(CTND, 220, 220, 0)
# train5.schedule_stop(KKDI, 270, 280, 1)
# train5.set_direction("DOWN")

# train6 = Train(env, "T6", [], max_speed=160, priority=1, length=300, weight=4000, initial_delay=0, hp=12000, accel_mps2=0.5, decel_mps2=0.5)
# train6.start_time(20)
# train6.schedule_stop(KKDI, 75, 80, 0)
# train6.schedule_stop(CTND, 100, 100, 0) # Run through main
# train6.schedule_stop(TYM, 120, 120, 0)
# train6.schedule_stop(PDKT, 130, 140, 0)
# train6.schedule_stop(VEL, 150, 150, 0)
# train6.schedule_stop(KRUR, 160, 160, 0)
# train6.schedule_stop(KRMG, 170, 170, 0)
# train6.schedule_stop(TPJ, 185, 190, 1)
# train6.set_direction("UP")

# train7 = Train(env, "T7", [], max_speed=100, priority=2, length=270, weight=3600, initial_delay=0, hp=3125, accel_mps2=0.5, decel_mps2=0.5)
# train7.start_time(50)
# train7.schedule_stop(TPJ, 95, 100, 1)
# train7.schedule_stop(KRMG, 110, 115, 0)
# train7.schedule_stop(KRUR, 120, 125, 1)
# train7.schedule_stop(VEL, 130, 135, 0)
# train7.schedule_stop(PDKT, 150, 155, 0)
# train7.schedule_stop(TYM, 170, 175, 0)
# train7.schedule_stop(CTND, 180, 185, 1)
# train7.schedule_stop(KKDI, 200, 205, 1)
# train7.set_direction("DOWN")

# train8 = Train(env, "T8", [], max_speed=160, priority=0, length=384, weight=430, initial_delay=0, hp=9010, accel_mps2=0.7, decel_mps2=0.7) # Non stop vande bharat express
# train8.schedule_stop(KKDI, 10, 10, 0)
# train8.schedule_stop(CTND, 40, 40, 0)
# train8.schedule_stop(TYM, 60, 60, 0)
# train8.schedule_stop(PDKT, 70, 70, 0)
# train8.schedule_stop(VEL, 90, 90, 0)
# train8.schedule_stop(KRUR, 105, 105, 0)
# train8.schedule_stop(KRMG, 115, 115, 0)
# train8.schedule_stop(TPJ, 220, 220, 0)

# # # === EXTRA TRAINS (T9–T16) ===

# train9 = Train(env, "T9", [], max_speed=80, priority=2, length=300, weight=4100, initial_delay=0, hp=6120, accel_mps2=0.5, decel_mps2=0.5)
# train9.schedule_stop(TPJ, 15, 20, 1)
# train9.schedule_stop(KRMG, 30, 35, 1)
# train9.schedule_stop(KRUR, 50, 55, 1)
# train9.schedule_stop(VEL, 70, 75, 1)
# train9.schedule_stop(PDKT, 85, 90, 0)
# train9.schedule_stop(TYM, 100, 105, 1)
# train9.schedule_stop(CTND, 115, 120, 1)
# train9.schedule_stop(KKDI, 130, 135, 0)

# train10 = Train(env, "T10", [], max_speed=110, priority=1, length=310, weight=4050, initial_delay=0, hp=6120, accel_mps2=0.5, decel_mps2=0.5)
# train10.schedule_stop(KKDI, 20, 25, 1)
# train10.schedule_stop(CTND, 45, 50, 0)
# train10.schedule_stop(TYM, 60, 65, 0)
# train10.schedule_stop(PDKT, 70, 75, 0)
# train10.schedule_stop(VEL, 80, 85, 1)
# train10.schedule_stop(KRUR, 95, 100, 1)
# train10.schedule_stop(KRMG, 110, 115, 1)
# train10.schedule_stop(TPJ, 125, 130, 1)

# train11 = Train(env, "T11", [], max_speed=70, priority=3, length=280, weight=3700, initial_delay=0, hp=3125, accel_mps2=0.5, decel_mps2=0.5)
# train11.schedule_stop(TPJ, 30, 35, 1)
# train11.schedule_stop(KRUR, 60, 60, 0)
# train11.schedule_stop(PDKT, 95, 100, 0)
# train11.schedule_stop(CTND, 120, 120, 0)
# train11.schedule_stop(KKDI, 140, 145, 1)

# train12 = Train(env, "T12", [], max_speed=120, priority=1, length=330, weight=4200, initial_delay=0, hp=6000, accel_mps2=0.5, decel_mps2=0.5)
# train12.schedule_stop(KKDI, 50, 55, 1)
# train12.schedule_stop(CTND, 75, 80, 1)
# train12.schedule_stop(PDKT, 105, 110, 0)
# train12.schedule_stop(KRUR, 140, 145, 1)
# train12.schedule_stop(TPJ, 165, 170, 0)

# train13 = Train(env, "T13", [], max_speed=100, priority=2, length=260, weight=3500, initial_delay=0, hp=3125, accel_mps2=0.5, decel_mps2=0.5)
# train13.schedule_stop(TPJ, 70, 75, 0)
# train13.schedule_stop(KRUR, 100, 100, 0)
# train13.schedule_stop(PDKT, 130, 130, 0)
# train13.schedule_stop(CTND, 155, 160, 1)
# train13.schedule_stop(KKDI, 180, 185, 1)

# train14 = Train(env, "T14", [], max_speed=140, priority=1, length=300, weight=3950, initial_delay=0, hp=4500, accel_mps2=0.5, decel_mps2=0.5)
# train14.schedule_stop(KKDI, 85, 90, 0)
# train14.schedule_stop(CTND, 110, 110, 0)
# train14.schedule_stop(PDKT, 140, 140, 0)
# train14.schedule_stop(KRUR, 165, 165, 0)
# train14.schedule_stop(TPJ, 190, 195, 1)

# train15 = Train(env, "T15", [], max_speed=90, priority=3, length=250, weight=4900, initial_delay=0, hp=4500, accel_mps2=0.5, decel_mps2=0.5)
# train15.schedule_stop(TPJ, 105, 110, 1)
# train15.schedule_stop(KRUR, 135, 135, 0)
# train15.schedule_stop(PDKT, 160, 160, 0)
# train15.schedule_stop(CTND, 185, 185, 0)
# train15.schedule_stop(KKDI, 210, 215, 1)

# train16 = Train(env, "T16", [], max_speed=160, priority=0, length=384, weight=430, initial_delay=0, hp=9010, accel_mps2=0.15, decel_mps2=0.18)  # Another VB express
# train16.schedule_stop(KKDI, 40, 40, 0)
# train16.schedule_stop(CTND, 65, 65, 0)
# train16.schedule_stop(PDKT, 80, 80, 0)
# train16.schedule_stop(KRUR, 115, 115, 0)
# train16.schedule_stop(TPJ, 145, 145, 0)

# exit(0)  # Exit before running the simulation loop for testing purposes

time_elapsed = 0
while True:
    DecisionSuggestion.SUGGESTIONS.clear()
    time_elapsed += 5
    sim.run(until=time_elapsed)  # Run for 300 minutes
    # Process decision suggestions
    for sugg in DecisionSuggestion.SUGGESTIONS:
        print(
            f"Suggestion: {sugg.train_id} at {sugg.station_code} - {sugg.action} for {sugg.hold_time} mins ({sugg.reason})"
        )
        approve = input("Approve? (y/n): ")
        if approve.lower() == "y":
            sugg.approve()
        else:
            sugg.reject("User rejected the suggestion.")
    if sim.peek() == SimpyInfinity:
        print("Simulation ends here..")
        break

trains = Train.TRAINS
train_logs = sum([train.log.entries for train in trains], [])
train_marks = sum([train.log.marks for train in trains], [])

train_logs.sort(key=lambda x: (x[0], x[1]))
train_marks.sort(key=lambda x: (x[0], x[1]))
print("Time\tTrain\tEvent\tStation")
# for train_log in train_logs:
#     print(f"{train_log[0]}\t{train_log[1]}\t{train_log[2]}")

for train_mark in train_marks:
    print(f"{train_mark[0]:.3f}\t{train_mark[1]}\t{train_mark[2]}\t{train_mark[3]}")

import matplotlib.pyplot as plt
import pandas as pd


def plot_timetable(trains: list[Train], stations: list[Station]):
    # Build structured log dataframe
    # logs = []
    # for train in trains:
    #     for t, tid, ev, stn in train.log.marks:
    #         logs.append(dict(time=t, train=tid, event=ev, station=stn))
    # df = pd.DataFrame(logs).sort_values(["time", "train"])

    # Print logs nicely (instead of subplot)
    # print("\n===== Event Logs =====")
    # print(df.to_string(index=False))

    stations: list[str] = [
        stn.stn_code if isinstance(stn, Station) else stn for stn in stations
    ]

    # ---------- create 2x2 layout ----------
    fig, axs = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]

    # ---------- (1) CLEAN GANTT CHART ----------
    ax1.set_title("Train Schedule Gantt Chart (with Station/Block labels)")

    for i, train in enumerate(trains):
        color = plt.cm.tab20.colors[i % 20]
        marks = train.log.marks
        for j in range(len(marks) - 1):
            t0, tid, type0, stn0 = marks[j]
            t1, _, type1, stn1 = marks[j + 1]

            # Station dwell
            if type0 == "ARRIVE" and type1 == "DEPART":
                bar = ax1.barh(
                    i,
                    t1 - t0,
                    left=t0,
                    color=color,
                    alpha=0.6,
                    label=tid if j == 0 else "",
                )
                label = stn0
                mid = (t0 + t1) / 2

            # Block run
            elif type0 == "ENTRY" and type1 == "EXIT":
                bar = ax1.barh(
                    i, t1 - t0, left=t0, color=color, alpha=0.9, hatch="//", label=""
                )
                label = stn0  # block name
                mid = (t0 + t1) / 2
            else:
                continue

            # --- Only add text if it fits inside bar ---
            bar_width = t1 - t0
            renderer = ax1.figure.canvas.get_renderer()
            txt = ax1.text(
                mid,
                i,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if type0 == "ENTRY" else "black",
            )
            bb = txt.get_window_extent(renderer=renderer)
            txt_width = bb.width / ax1.figure.dpi * 72  # approx in data coords

            if txt_width > bar_width * 1.2:  # too wide -> remove
                txt.remove()
    ax1.set_yticks(range(len(trains)))
    ax1.set_yticklabels([t.id for t in trains])
    ax1.set_xlabel("Time (min)")
    ax1.legend()
    ax1.grid(True, axis="x")

    # ---------- (2) DELAY HEATMAP ----------
    ax2.set_title("Delay Heatmap (min)")
    station_list = stations
    matrix = []

    # Filter out short schedules (e.g. 1-stop entries) to avoid ragged array
    trains_for_heatmap = [t for t in trains if len(t.schedule) >= len(stations)]

    for train in trains_for_heatmap:
        scheduled = [sp.arrival_time for sp in train.schedule]
        actual = [sp.actual_arrival_time for sp in train.schedule]
        if len(scheduled) == len(actual):
            delay = (
                list(reversed([a - s for a, s in zip(actual, scheduled)]))
                if train.direction == "DOWN"
                else [a - s for a, s in zip(actual, scheduled)]
            )
        else:
            delay = [None] * len(scheduled)  # incomplete runs
        matrix.append(delay)

    import numpy as np

    matrix = np.array(matrix, dtype=float)  # force float for NaN handling

    print(matrix)

    im = ax2.imshow(matrix, cmap="coolwarm", aspect="auto", interpolation="nearest")
    ax2.set_xticks(range(len(station_list)))
    ax2.set_xticklabels(station_list)
    ax2.set_yticks(range(len(trains_for_heatmap)))
    ax2.set_yticklabels([t.id for t in trains_for_heatmap])

    for i in range(matrix.shape[0]):  # rows = trains
        for j in range(matrix.shape[1]):  # cols = stations
            val = matrix[i, j]
            if not np.isnan(val):  # skip missing
                ax2.text(
                    j,
                    i,
                    f"{int(val)}",
                    ha="center",
                    va="center",
                    color="black" if abs(val) < 5 else "white",
                    fontsize=8,
                )

    plt.colorbar(im, ax=ax2, label="Delay (min)")

    # ---------- (3) TRAIN GRAPH ----------
    ax3.set_title("Train Graph (Time vs Stations/Blocks)")

    # map stations+blocks to Y positions
    ypos = {st: i for i, st in enumerate(stations)}  # add blocks if needed
    print(ypos)
    for i, train in enumerate(trains):
        col = plt.cm.tab20.colors[i % 20]
        marks = list(filter(lambda x: x[2] in [ARRIVAL, DEPARTURE], train.log.marks))
        for j in range(len(marks) - 1):
            t0, tid, type0, stn0 = marks[j]
            t1, _, type1, stn1 = marks[j + 1]
            print(t0, tid, type0, stn0, "|", t1, type1, stn1)
            if type0 == ARRIVAL and type1 == DEPARTURE:
                # dwell (horizontal in your old plot, vertical now!)
                ax3.plot(
                    [t0, t1],
                    [ypos[stn0], ypos[stn0]],
                    linestyle="dashed",
                    color=col,
                    label=f"{tid} - {train.priority}" if j == 0 else "",
                )
            elif type0 == DEPARTURE and type1 == ARRIVAL:
                # run (diagonal from one station to next)
                ax3.plot(
                    [t0, t1],
                    [ypos[stn0], ypos[stn1]],
                    linestyle="solid",
                    marker="o",
                    color=col,
                )

    ax3.set_yticks(range(len(stations)))
    ax3.set_yticklabels(stations)
    ax3.set_xlabel("Time (min)")
    ax3.set_ylabel("Stations / Blocks")
    ax3.grid(True)
    ax3.legend()

    ax4.set_title("Schedule vs Actual Timetable (Overlay)")

    for i, train in enumerate(trains):
        col = plt.cm.tab20.colors[i % 20]

        # Plot scheduled timetable
        for sp in train.schedule:
            arr, dep = sp.arrival_time, sp.departure_time
            ax4.barh(
                i,
                dep - arr,
                left=arr,
                edgecolor=col,
                color=col,
                fill=True,
                hatch="//",
                linewidth=1.5,
                alpha=0.9,
            )

        # Plot actual timetable from logs
        marks = train.log.marks
        for j in range(len(marks) - 1):
            t0, tid, type0, stn0 = marks[j]
            t1, _, type1, stn1 = marks[j + 1]
            if type0 == "ARRIVE" and type1 == "DEPART":
                ax4.barh(i, t1 - t0, left=t0, color=col, alpha=0.6)

    ax4.set_yticks(range(len(trains)))
    ax4.set_yticklabels([t.id for t in trains])
    ax4.set_xlabel("Time (min)")
    ax4.grid(True, axis="x")

    # fig.tight_layout()
    plt.show()


plot_timetable(Train.TRAINS, [stn for stn in network.get_stations()])

# import matplotlib.pyplot as plt

# # ---------- (1) GANTT CHART ----------
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))  # side by side
# # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))  # top & bottom (use this instead if you prefer)

# ax1.set_title("Train Schedule Gantt Chart")

# def split_text(text, max_chars=20):
#     words = text.split()
#     lines, current_line = [], ""
#     for word in words:
#         if len(current_line) + len(word) + 1 <= max_chars:
#             current_line += (" " + word if current_line else word)
#         else:
#             lines.append(current_line)
#             current_line = word
#     if current_line: lines.append(current_line)
#     return "\n".join(lines)

# for i, train in enumerate(trains):
#     color = plt.cm.tab20.colors[i % 20]
#     entries = train.log.entries
#     for idx in range(len(entries) - 1):
#         start = entries[idx][0]
#         end = entries[idx + 1][0]
#         message = entries[idx][2]

#         ax1.barh(i, end - start, left=start, color=color, edgecolor='black', alpha=0.7)
#         ax1.text(start + (end - start) / 2, i, split_text(message, (end-start)/2),
#                  va='center', ha='center', fontsize=8, color='black')

# ax1.set_xlabel("Time")
# ax1.set_ylabel("Train")
# ax1.set_yticks(range(len(trains)))
# ax1.set_yticklabels([train.id for train in trains])
# handles = [plt.Rectangle((0, 0), 1, 1, color=plt.cm.tab20.colors[i % 20]) for i in range(len(trains))]
# ax1.legend(handles, [f"{train.id} - {train.priority}" for train in trains], title="Trains")
# ax1.grid(True, which="both", axis="x")

# # ---------- (2) TRAIN GRAPH ----------
# stations = ["tpj", "pdkt", "kkdi"]
# xpos = {st: i for i, st in enumerate(stations)}

# edge_spans = {}
# for train in trains:
#     spans = []
#     marks = list(filter(lambda x: x[2] in [ARRIVAL, DEPARTURE], train.log.marks))
#     for i in range(len(marks) - 1):
#         t0, tid, mtype0, stn0 = marks[i]
#         t1, _, mtype1, stn1 = marks[i+1]

#         # --- dwell detection ---
#         # if "Entering" in msg0 and "Accepted" in msg1:
#         if mtype0 == ARRIVAL and mtype1 == DEPARTURE:
#             # Train is dwelling inside platform (between accepted & dispatch)
#             spans.append((stn0, stn0, t0, t1, "dwell"))

#         # # --- run detection ---
#         # elif mtype0 == DEPARTURE and mtype1 == ENTRY:
#         #     spans.append((stn0, stn1, t0, t1, "run"))

#         elif mtype0 == DEPARTURE and mtype1 == ARRIVAL:
#             spans.append((stn0, stn1, t0, t1, "run"))

#         # --- terminate detection ---
#         # elif "Accepted" in msg0 and "Terminating" in msg1:
#         #     u = msg0.split()[1]           # current station
#         #     spans.append((u, u, t0, t1, "dwell"))

#     edge_spans[train.id] = spans

# colors = plt.cm.tab20.colors
# color_map = {train.id: colors[i % len(colors)] for i, train in enumerate(trains)}

# for train in trains:
#     tid = train.id
#     col = color_map[tid]
#     for j, (u, v, t0, t1, kind) in enumerate(edge_spans[tid]):
#         xs = [xpos[u], xpos[v]]
#         ys = [t0, t1]
#         style = "solid" if kind == "run" else "dashed"
#         ax2.plot(xs, ys, marker="o", linewidth=2, color=col, linestyle=style,
#                  label=tid if j == 0 else None)
#         # annotate arrival & departure
#         ax2.text(xs[0], ys[0], f"{ys[0]}", fontsize=7, va="bottom", ha="right")
#         ax2.text(xs[1], ys[1], f"{ys[1]}", fontsize=7, va="bottom", ha="left")

# ax2.set_xticks(range(len(stations)))
# ax2.set_xticklabels(stations)
# ax2.set_ylabel("Time (minutes)")
# ax2.set_xlabel("Stations (left to right)")
# ax2.set_title("Train Graph with Runs + Dwells")
# ax2.invert_yaxis()
# ax2.grid(True, which="both", axis="both")
# ax2.legend()

# plt.tight_layout()
# plt.show()
