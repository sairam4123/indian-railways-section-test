from ortools.sat.python import cp_model
from ortools.sat.python import cp_model

def build_train_scheduling_model(trains, stations, blocks,
                                 penalty_hold=10,
                                 penalty_chg_pf=5,
                                 penalty_delay=1):
    model = cp_model.CpModel()

    # ---------------- Horizon ----------------
    horizon = max(sp.departure_time for t in trains for sp in t.schedule) + 60

    arrive, depart, pf, hold, chg_pf, delay = {}, {}, {}, {}, {}, {}
    block_interval = {}

    # ---------------- STATION VARS ----------------
    for t in trains:
        for sp in t.schedule:
            key = (t.id, sp.station.stn_code)

            # Time vars
            arrive[key] = model.NewIntVar(0, horizon, f"arrive_{t.id}_{sp.station.stn_code}")
            depart[key] = model.NewIntVar(0, horizon, f"depart_{t.id}_{sp.station.stn_code}")

            # PF & actions
            pf[key] = model.NewIntVar(0, len(sp.station.tracks)-1,
                                      f"pf_{t.id}_{sp.station.stn_code}")
            hold[key] = model.NewBoolVar(f"hold_{t.id}_{sp.station.stn_code}")
            chg_pf[key] = model.NewBoolVar(f"chgpf_{t.id}_{sp.station.stn_code}")

            # Delay var
            delay[key] = model.NewIntVar(0, horizon, f"delay_{t.id}_{sp.station.stn_code}")

            # Dwell constraint
            model.Add(depart[key] >= arrive[key] + sp.layover_time)

            # HOLD definition: dep ≥ scheduled, hold=1 if strictly later
            model.Add(depart[key] >= sp.departure_time)
            model.Add(depart[key] - sp.departure_time >= 1).OnlyEnforceIf(hold[key])
            model.Add(depart[key] - sp.departure_time <= 0).OnlyEnforceIf(hold[key].Not())

            # PF assignment constraint
            valid_alts = [i for i in range(len(sp.station.tracks)) if i != sp.expected_platform]
            if valid_alts:
                tuples = [(sp.expected_platform, 0)] + [(alt, 1) for alt in valid_alts]
                model.AddAllowedAssignments([pf[key], chg_pf[key]], tuples)
            else:
                model.Add(pf[key] == sp.expected_platform)
                model.Add(chg_pf[key] == 0)

            # Delay definition
            model.Add(delay[key] >= depart[key] - sp.departure_time)
            model.Add(delay[key] >= 0)

    # ---------------- BLOCK CONSTRAINTS ----------------
    for t in trains:
        for i in range(len(t.schedule)-1):
            s1 = t.schedule[i].station
            s2 = t.schedule[i+1].station
            k1 = (t.id, s1.stn_code)
            k2 = (t.id, s2.stn_code)

            block = s1.get_block_to(s2)
            run_time = block._run_minutes(t)
            clearance = block.headway_min

            # Interval for block occupancy
            block_interval[(t.id, block.name)] = model.NewIntervalVar(
                depart[k1], run_time + clearance, arrive[k2],
                f"block_{t.id}_{block.name}"
            )

            # Travel consistency
            model.Add(arrive[k2] >= depart[k1] + run_time)

    for block in blocks:
        intervals = [
            block_interval[(t.id, block.name)]
            for t in trains
            for i in range(len(t.schedule)-1)
            if t.schedule[i].station.get_block_to(t.schedule[i+1].station) == block
        ]
        if intervals:
            model.AddNoOverlap(intervals)

    # ---------------- PLATFORM CONFLICTS ----------------
    for st in stations:
        for t1 in trains:
            for t2 in trains:
                if t1.id >= t2.id: 
                    continue
                sp1 = next((sp for sp in t1.schedule if sp.station == st), None)
                sp2 = next((sp for sp in t2.schedule if sp.station == st), None)
                if not sp1 or not sp2:
                    continue

                k1 = (t1.id, st.stn_code)
                k2 = (t2.id, st.stn_code)

                # If same PF, no overlap
                same_pf = model.NewBoolVar(f"samepf_{t1.id}_{t2.id}_{st.stn_code}")
                model.Add(pf[k1] == pf[k2]).OnlyEnforceIf(same_pf)
                model.Add(pf[k1] != pf[k2]).OnlyEnforceIf(same_pf.Not())

                model.Add(depart[k1] <= arrive[k2]).OnlyEnforceIf(same_pf)
                model.Add(depart[k2] <= arrive[k1]).OnlyEnforceIf(same_pf)

    # ---------------- OBJECTIVE ----------------
    obj_terms = []
    for t in trains:
        for sp in t.schedule:
            k = (t.id, sp.station.stn_code)
            obj_terms.append(delay[k] * penalty_delay)
            obj_terms.append(hold[k] * penalty_hold)
            obj_terms.append(chg_pf[k] * penalty_chg_pf)

    model.Minimize(sum(obj_terms))
    return model, arrive, depart, pf, hold, chg_pf, delay


# ---------------- SOLVER ----------------
# def solve_and_print(trains, stations, blocks):
#     model, arrive, depart, pf, hold, chg_pf, delay = build_train_scheduling_model(
#         trains=trains, stations=stations, blocks=blocks
#     )
#     solver = cp_model.CpSolver()
#     solver.parameters.max_time_in_seconds = 20
#     solver.parameters.num_search_workers = 8
#     solver.parameters.linearization_level = 0

#     status = solver.Solve(model)
#     if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
#         for t in trains:
#             for sp in t.schedule:
#                 k = (t.id, sp.station.stn_code)
#                 print(f"{t.id}@{sp.station.stn_code}: "
#                       f"arr={solver.Value(arrive[k])}, dep={solver.Value(depart[k])}, "
#                       f"pf={solver.Value(pf[k])}, "
#                       f"hold={solver.Value(hold[k])}, chgpf={solver.Value(chg_pf[k])}, "
#                       f"delay={solver.Value(delay[k])}")
#     else:
#         print("No solution found.")


import math
import random
import simpy
import simpy.resources.resource

APPROACH_TIME = 2
DEPARTURE_CLEAR_TIME = 2

ENABLE_RANDOM_DELAYS = True
EARLY_PENALTY = 5
LATE_PENALTY = 10

TRAINS = []

class Station:
    STATIONS = []
    def __init__(self, env: simpy.Environment, stn_code: str, tracks: list['Track']) -> None:
        self.stn_code = stn_code
        self.tracks: list[Track] = tracks
        self.env = env
        self.train_map: dict['Train', tuple['Track', simpy.resources.resource.PriorityRequest]] = {}
        self.connections: dict['Station', list['BlockSection']] = {}

        Station.STATIONS.append(self)


    # def accept(self, train: 'Train'):
    #     track = self.get_first_available_track(train)
    #     if track:
    #         if (req := track.accept(train)):
    #             # Simulate time taken to reach platform
    #             yield self.env.timeout(2) # 2 mins
    #             self.train_map[train] = req
    #             return req
    #     return None

    def accept(self, train: 'Train', sp: 'SchedulePoint'):
        """
        Acquire a suitable platform track as soon as the train arrives.
        We do NOT release here; we keep the request in train_map for live view + later dispatch().
        """
        # choose track: prefer expected platform if valid & fits; else first available that fits
        track = None
        if 0 <= sp.expected_platform < len(self.tracks):
            cand = self.tracks[sp.expected_platform]
            if cand.length >= train.length:
                track = cand

        if track is None or track.resource.count > 0: # Track is not valid or occupied
            track = self.get_first_available_track(train)
            if track is None:
                # all suitable tracks busy -> queue on any suitable track; pick the first that fits
                for t in self.tracks:
                    if t.length >= train.length:
                        track = t
                        break
                if track is None:
                    raise RuntimeError(f"No track long enough for train {train} at {self.stn_code}")


        # Request the platform (PriorityResource). This will queue if busy.
        req = track.resource.request(priority=train.priority)
        yield req  # WAIT here until platform is granted

        # (optional) mark live state
        self.train_map[train] = (track, req)

        train.log.log(f"Accepted {self.stn_code} on track {track.id}")
        # print(f"{self.env.now}: Accepted {self.stn_code} - {train.id} on track {track.id}")
        # simulate approach to the platform starter (don’t release request!)
        if sp.expected_platform != 0: # Not main line    
            yield self.env.timeout(APPROACH_TIME)

        # done: the train is now occupying the platform track.
        return req  # keep request handle in case caller wants it too


    def get_first_available_track(self, train: 'Train'):
        for track in self.tracks:
            if not track.resource.count and track.length >= train.length:
                return track
        return None
    
    # def get_first_available_block(self, train: 'Train', sp: 'SchedulePoint'):
    #     next_sp = train.get_next_schedule_point()
    #     if next_sp is None:
    #         return None
    #     for block in self.connections.get(next_sp.station, []):
    #         if not block.resource.count > 0:
    #             return block
    #     return None
    
    def get_block_to(self, next_station: 'Station') -> 'BlockSection | None':
        lst = self.connections.get(next_station, [])
        # return the first available block
        for block in lst:
            if not block.resource.count > 0:
                return block
        # if no block is available, return the first block
        return lst[0] if lst else None

    def dispatch(self, train: 'Train', sp: 'SchedulePoint'):
        """Release the platform after departure, including a small clear time."""
        entry = self.train_map.get(train)
        if not entry:
            return False
        track, req = entry


        # simulate shunt/starting and clearing the platform starter
        if sp.expected_platform != 0:  # Not main line
            yield self.env.timeout(DEPARTURE_CLEAR_TIME)

        # release platform and update live map
        track.resource.release(req)
        self.train_map.pop(train, None)

        return True

class Track:
    def __init__(self, env: simpy.Environment, id: str, has_platform: bool, length: int) -> None:
        self.resource = simpy.PriorityResource(env, capacity=1)
        self.env = env
        self.has_platform = has_platform
        self.length = length
        self.id = id

        # self.train: 'Train | None' = None

    # def accept(self, train: 'Train'):
    #     if train.length > self.length:
    #         return
    #     self.train = train
    #     return self.resource.request(priority=train.priority)
    

class BlockSection:
    def __init__(self, env: simpy.Environment, 
                 name: str, from_station: Station, to_station: Station, 
                 line_speed: int, length_km: int, headway_min: int, 
                 bidirectional: bool = False, 
                 electric: bool = False) -> None:
        self.name = name
        self.from_station = from_station
        self.to_station = to_station
        self.line_speed = line_speed  # in km/h
        self.length_km = length_km
        self.headway_min = headway_min
        self.bidirectional = bidirectional
        self.env = env
        self.electric = electric

        self.resource = simpy.PriorityResource(env, capacity=1)
        # one-way only
        self.from_station.connections.setdefault(to_station, []).append(self)
        if bidirectional:
            self.to_station.connections.setdefault(from_station, []).append(self)

        self.occupied_train: tuple['Train', simpy.resources.resource.PriorityRequest] | None = None


    def _run_minutes(self, train: 'Train') -> int:
        """
        Compute run time (minutes) over this block/section using a simple
        accel–cruise–decel model with unit-consistent kinematics.

        Assumes:
        - self.line_speed in km/h
        - self.length_km in km
        - train.max_speed in km/h
        - Optional train.accel_mps2, train.decel_mps2 in m/s^2
        """
        # --- Parameters & unit conversions ---
        L = max(0.0, float(self.length_km)) * 1000.0                  # meters
        if L == 0:
            return 0

        v_cap_kmh = max(1.0, min(float(self.line_speed), float(train.max_speed)))
        v_cap = v_cap_kmh * (1000.0 / 3600.0)                         # m/s

        # Use train-provided accel/decel if available; otherwise reasonable defaults
        a = getattr(train, "accel_mps2", 0.9 if getattr(self, "electric", False) else 0.7)
        b = getattr(train, "decel_mps2", 0.9 if getattr(self, "electric", False) else 0.8)
        print(a, b)
        # Guard rails
        a = max(a, 0.1)
        b = max(b, 0.1)

        # --- Distances to accel to v_cap and decel from v_cap ---
        d_accel = 0.5 * v_cap * v_cap / a
        d_decel = 0.5 * v_cap * v_cap / b

        if d_accel + d_decel <= L:
            # Trapezoidal: accel → cruise → decel
            d_cruise = L - (d_accel + d_decel)
            t_accel  = v_cap / a
            t_cruise = d_cruise / v_cap if v_cap > 0 else 0.0
            t_decel  = v_cap / b
            total_s  = t_accel + t_cruise + t_decel
        else:
            # Triangular: cannot reach v_cap; peak speed determined by length
            # Solve L = v^2/(2a) + v^2/(2b) → v_peak = sqrt(2abL/(a+b))
            v_peak = math.sqrt((2.0 * a * b * L) / (a + b))
            t_accel = v_peak / a
            t_decel = v_peak / b
            total_s = t_accel + t_decel

        # Convert to minutes, ensure at least 1 minute for any positive length
        total_min = max(1, math.ceil(total_s / 60.0))

        print("L(m):", L, "v_cap(m/s):", v_cap, "a:", a, "b:", b)
        print("d_accel:", d_accel, "d_decel:", d_decel)
        print("t_total(s):", total_s, "t_total(min):", total_s/60)

        return total_min

    def traverse(self, train: 'Train', force_minutes: int | None = None):
        """Acquire -> run (or force_minutes) -> headway -> release, while holding the lock."""
        req = self.resource.request(priority=train.priority)
        yield req
        self.occupied_train = (train, req)
        try:
            run = self._run_minutes(train) if force_minutes is None else force_minutes
            yield self.env.timeout(run)              # inside block
            yield self.env.timeout(self.headway_min) # clearance
        finally:
            self.resource.release(req)
            self.occupied_train = None

    def enter(self, train: 'Train'):
        req = self.resource.request(priority=train.priority)
        yield req
        self.occupied_train = (train, req)

    def exit(self, train: 'Train'):

        occupied_train, req = self.occupied_train if self.occupied_train else (None, None)

        if occupied_train and req and occupied_train == train:
            self.occupied_train = None
            self.resource.release(req)
        yield self.env.timeout(0)  # just to make this a generator
        return 

class Train:

    TRAINS = []

    def __init__(self, env: simpy.Environment, id: str, schedule: list['SchedulePoint'], max_speed: int, priority: int, length: int, weight: int, initial_delay: int) -> None:
        self.env = env
        self.id = id
        self.schedule = schedule
        self.max_speed = max_speed
        self.priority = priority
        self.length = length
        self.initial_delay = initial_delay + random.randint(0, 15) if random.random() < 0.5 and ENABLE_RANDOM_DELAYS else initial_delay
        self.running_delay = self.initial_delay
        self.schedule_pointer = 0
        self.weight = weight
        self.log = TrainLog(self)
        self.current_block: BlockSection | None = None

        Train.TRAINS.append(self)
        env.process(self.run())

    def run(self):

        current_sp = self.schedule[self.schedule_pointer]
        map_enter_delay = current_sp.arrival_time - self.env.now + self.initial_delay
        if map_enter_delay > 0:
            yield self.env.timeout(map_enter_delay)

        while self.schedule_pointer < len(self.schedule):
            current_sp = self.schedule[self.schedule_pointer]

            # print(f"{self.env.now}: Approaching {current_sp.station.stn_code} - {self.id}")
            # Exit previous block if any (you held it until arriving home)
            # if self.current_block:
            #     # If you used traverse(), you won't need this; keeping for compatibility:
            #     yield from self.current_block.exit(self)
            #     self.current_block = None

            # print(f"{self.env.now}: Entering into {current_sp.station.stn_code} - {self.id}")
            # Arrive and occupy platform (even if early)
            yield from self.move_to_schedule_point(current_sp)

            self.running_delay = self.env.now - current_sp.arrival_time
            # Dwell/layover
            dwell_time = max(current_sp.departure_time - self.env.now, current_sp.layover_time)
            print(dwell_time)
            # Wait for departure time else if running late, wait for layover time
            if dwell_time > 0:
                yield self.env.timeout(dwell_time)

            # Look up the next station & block
            next_sp = self.get_next_schedule_point()
            if next_sp is None:
                # terminal: dispatch to yard or just clear platform
                self.log.log(f"Terminating at {current_sp.station.stn_code}")
                self.log.mark_departure(current_sp.station)
                # print(f"{self.env.now}: Terminating at {current_sp.station.stn_code} - {self.id}")
                # clear the platform
                yield from current_sp.station.dispatch(self, current_sp)
                break

            block = current_sp.station.get_block_to(next_sp.station)
            if block is None:
                raise RuntimeError(f"No block from {current_sp.station.stn_code} to {next_sp.station.stn_code}")

            # IMPORTANT: Reserve the next block while still on the platform.
            # We'll queue here if needed, without fouling the main line.

            # Now we dispatch (release platform) only when the block is granted, i.e., right before we start moving.
            # We'll do it like this: pre-acquire by calling traverse(), but we need to dispatch just before moving.
            # So we split traverse() into acquire+run+release by using a small helper:

            # Acquire the block, but don't start timing until we leave platform:
            # We can emulate this by first acquiring, then dispatch, then run+headway while holding.
            req = block.resource.request(priority=self.priority)
            yield req
            block.occupied_train = (self, req)  # mark for live view

            # Leave the platform now (starter clears)
            # print(f"{self.env.now}: Departing {current_sp.station.stn_code} - {self.id} - {block.name}")
            self.log.log(f"Departing {current_sp.station.stn_code} towards {next_sp.station.stn_code} via {block.name}")
            yield from current_sp.station.dispatch(self, current_sp)
            self.log.mark_departure(current_sp.station)
            self.log.mark_entry_block(block)

            # Run inside the block while holding the lock
            try:
                run = block._run_minutes(self)
                yield self.env.timeout(run)
                self.log.log(f"Exited {block.name} - approaching {next_sp.station.stn_code}")
                yield self.env.timeout(block.headway_min)
            finally:
                self.log.mark_exit_block(block)
                # print(f"{self.env.now}: Approaching {next_sp.station.stn_code} - {self.id}")
                block.resource.release(req)
                block.occupied_train = None
                # yield self.env.timeout(2)  # Allow time for the block to clear

            # Next station’s platform will be taken in the next loop iteration by move_to_schedule_point(next_sp)
            self.current_block = None
            self.schedule_pointer += 1
        
    def move_to_schedule_point(self, sp: 'SchedulePoint'):
        # travel_time = sp.arrival_time - self.env.now
        # if travel_time > 0:
        #     yield self.env.timeout(travel_time)
        self.log.log(f"Entering into {sp.station.stn_code}")
        self.log.mark_arrival(sp.station)
        yield from sp.station.accept(self, sp)
    
    def get_next_schedule_point(self) -> 'SchedulePoint | None':
        return self.schedule[self.schedule_pointer + 1] if self.schedule_pointer + 1 < len(self.schedule) else None

    def schedule_stop(self, station: Station, arrival_time: int, departure_time: int, expected_platform: int = 0, layover_time: int | None = None) -> 'SchedulePoint':
        # Calculate the layover time from arrival and departure, if not given
        layover_time = layover_time if layover_time is not None else max(0, departure_time - arrival_time)
        sp = SchedulePoint(station, arrival_time, departure_time, layover_time, expected_platform)
        self.schedule.append(sp)
        return sp


class SchedulePoint:
    def __init__(self, station: Station, arrival_time: int, departure_time: int, layover_time: int, expected_platform: int) -> None:
        self.station = station
        self.arrival_time = arrival_time
        self.departure_time = departure_time
        self.layover_time = layover_time
        self.expected_platform = expected_platform

ARRIVAL = "ARRIVE"
DEPARTURE = "DEPART"
ENTRY = "ENTRY"
EXIT = "EXIT"

class TrainLog:
    def __init__(self, train: Train) -> None:
        self.train = train
        self.entries = []
        self.marks = []

    def log(self, message: str) -> None:
        self.entries.append((self.train.env.now, self.train.id, message))
    
    def mark_arrival(self, station: Station):
        # self.entries.append((self.train.env.now, self.train.id, f"Entering {station.stn_code}"))
        self.marks.append((self.train.env.now, self.train.id, ARRIVAL, station.stn_code))
    
    def mark_departure(self, station: Station):
        self.marks.append((self.train.env.now, self.train.id, DEPARTURE, station.stn_code))

    def mark_entry_block(self, block: BlockSection):
        self.marks.append((self.train.env.now, self.train.id, ENTRY, block.name))
    
    def mark_exit_block(self, block: BlockSection):
        self.marks.append((self.train.env.now, self.train.id, EXIT, block.name))


env = simpy.Environment()

TPJ_main = Track(env, "TPJ_main", has_platform=True, length=600)
TPJ_loop1 = Track(env, "TPJ_loop1", has_platform=True, length=400)
TPJ_loop2 = Track(env, "TPJ_loop2", has_platform=True, length=400)
TPJ = Station(env, 'tpj', [TPJ_main, TPJ_loop1, TPJ_loop2])

KRUR_main = Track(env, "KRUR_main", has_platform=False, length=600)
KRUR_loop1 = Track(env, "KRUR_loop1", has_platform=True, length=400)
KRUR_loop2 = Track(env, "KRUR_loop2", has_platform=True, length=400)
KRUR = Station(env, 'krur', [KRUR_main, KRUR_loop1, KRUR_loop2])

PDKT_main = Track(env, "PDKT_main", has_platform=True, length=600)
PDKT_loop1 = Track(env, "PDKT_loop1", has_platform=True, length=400)
PDKT_loop2 = Track(env, "PDKT_loop2", has_platform=True, length=400)
PDKT_loop3 = Track(env, "PDKT_loop3", has_platform=True, length=500)
PDKT = Station(env, 'pdkt', [PDKT_main, PDKT_loop1, PDKT_loop2, PDKT_loop3])

CTND_main = Track(env, "CTND_main", has_platform=False, length=600)
CTND_loop1 = Track(env, "CTND_loop1", has_platform=True, length=400)
CTND_loop2 = Track(env, "CTND_loop2", has_platform=True, length=400)
CTND_loop3 = Track(env, "CTND_loop3", has_platform=True, length=400)
CTND = Station(env, 'ctnd', [CTND_main, CTND_loop1, CTND_loop2, CTND_loop3])

KKDI_main = Track(env, "KKDI_main", has_platform=True, length=600)
KKDI_loop1 = Track(env, "KKDI_loop1", has_platform=True, length=400)
KKDI_loop2 = Track(env, "KKDI_loop2", has_platform=True, length=400)
KKDI_loop3 = Track(env, "KKDI_loop3", has_platform=True, length=400)
KKDI_loop4 = Track(env, "KKDI_loop4", has_platform=True, length=400)
KKDI = Station(env, 'kkdi', [KKDI_main, KKDI_loop1, KKDI_loop2, KKDI_loop3, KKDI_loop4])

# PDKT_KKDI = BlockEdge(env, "PDKT_KKDI", PDKT, KKDI, headway_min=2, length_km=60, line_speed=110, bidirectional=False)
# KKDI_PDKT = BlockEdge(env, "KKDI_PDKT", KKDI, PDKT, headway_min=2, length_km=60, line_speed=110, bidirectional=False)

# PDKT_TPJ = BlockEdge(env, "PDKT_TPJ", PDKT, TPJ, headway_min=3, length_km=80, line_speed=100, bidirectional=False)
# TPJ_PDKT = BlockEdge(env, "TPJ_PDKT", TPJ, PDKT, headway_min=3, length_km=80, line_speed=100, bidirectional=False)



# TPJ_KRUR = BlockSection(env, "TPJ_KRUR", TPJ, KRUR, headway_min=3, length_km=27, line_speed=100, bidirectional=False, electric=True)
# KRUR_TPJ = BlockSection(env, "KRUR_TPJ", KRUR, TPJ, headway_min=3, length_km=27, line_speed=100, bidirectional=False, electric=True)

# KRUR_PDKT = BlockSection(env, "KRUR_PDKT", KRUR, PDKT, headway_min=2, length_km=32, line_speed=110, bidirectional=False, electric=True)
# PDKT_KRUR = BlockSection(env, "PDKT_KRUR", PDKT, KRUR, headway_min=2, length_km=32, line_speed=110, bidirectional=False, electric=True)

# PDKT_CTND = BlockSection(env, "PDKT_CTND", PDKT, CTND, headway_min=2, length_km=26, line_speed=110, bidirectional=False, electric=True)
# CTND_PDKT = BlockSection(env, "CTND_PDKT", CTND, PDKT, headway_min=2, length_km=26, line_speed=110, bidirectional=False, electric=True)

# CTND_KKDI = BlockSection(env, "CTND_KKDI", CTND, KKDI, headway_min=2, length_km=13, line_speed=110, bidirectional=False, electric=True)
# KKDI_CTND = BlockSection(env, "KKDI_CTND", KKDI, CTND, headway_min=2, length_km=13, line_speed=110, bidirectional=False, electric=True)

TPJ_KRUR = BlockSection(env, "TPJ_KRUR", TPJ, KRUR, headway_min=3, length_km=27, line_speed=100, bidirectional=True, electric=True)
KRUR_PDKT = BlockSection(env, "KRUR_PDKT", KRUR, PDKT, headway_min=2, length_km=32, line_speed=110, bidirectional=True, electric=True)
PDKT_CTND = BlockSection(env, "PDKT_CTND", PDKT, CTND, headway_min=2, length_km=26, line_speed=110, bidirectional=True, electric=True)
CTND_KKDI = BlockSection(env, "CTND_KKDI", CTND, KKDI, headway_min=3, length_km=13, line_speed=100, bidirectional=True, electric=True)

# Existing trains
train1 = Train(env, "T1", [], max_speed=80, priority=1, length=300, weight=4000, initial_delay=0)
train1.schedule_stop(TPJ, 0, 10, 1)
train1.schedule_stop(KRUR, 40, 45, 1)
train1.schedule_stop(PDKT, 70, 75, 0)
train1.schedule_stop(CTND, 100, 105, 1)
train1.schedule_stop(KKDI, 110, 115, 0)

train2 = Train(env, "T2", [], max_speed=110, priority=2, length=300, weight=4000, initial_delay=0)
train2.schedule_stop(KKDI, 10, 20, 1)
train2.schedule_stop(CTND, 35, 40, 1)
train2.schedule_stop(PDKT, 55, 65, 0)
train2.schedule_stop(KRUR, 70, 75, 1)
train2.schedule_stop(TPJ, 110, 115, 1)

train3 = Train(env, "T3", [], max_speed=60, priority=1, length=280, weight=3800, initial_delay=0)
train3.schedule_stop(TPJ, 20, 25, 1)
train3.schedule_stop(KRUR, 40, 40, 0) # Run through main
train3.schedule_stop(PDKT, 85, 90, 0)
train3.schedule_stop(CTND, 100, 100, 0) # Run through main
train3.schedule_stop(KKDI, 125, 130, 1)

# train4 = Train(env, "T4", [], max_speed=120, priority=2, length=320, weight=4200, initial_delay=0)
# train4.schedule_stop(KKDI, 40, 45, 1)
# train4.schedule_stop(CTND, 60, 65, 1)
# train4.schedule_stop(PDKT, 95, 100, 1)
# train4.schedule_stop(KRUR, 130, 135, 1)
# train4.schedule_stop(TPJ, 150, 155, 0)

# train5 = Train(env, "T5", [], max_speed=90, priority=3, length=250, weight=5005, initial_delay=0)
# train5.schedule_stop(TPJ, 60, 65, 0)
# train5.schedule_stop(KRUR, 90, 90, 0)
# train5.schedule_stop(PDKT, 120, 120, 0)
# train5.schedule_stop(CTND, 140, 140, 0)
# train5.schedule_stop(KKDI, 170, 175, 1)

# train6 = Train(env, "T6", [], max_speed=160, priority=1, length=300, weight=4000, initial_delay=0)
# train6.schedule_stop(KKDI, 75, 80, 0)
# train6.schedule_stop(CTND, 100, 100, 0) # Run through main
# train6.schedule_stop(PDKT, 130, 130, 0) # Non-stop run (run-through on the main-line)
# train6.schedule_stop(KRUR, 160, 160, 0)
# train6.schedule_stop(TPJ, 185, 190, 1)

# train7 = Train(env, "T7", [], max_speed=100, priority=2, length=270, weight=3600, initial_delay=0)
# train7.schedule_stop(TPJ, 95, 100, 1)
# train7.schedule_stop(KRUR, 120, 125, 1)
# train7.schedule_stop(PDKT, 150, 155, 0)
# train7.schedule_stop(CTND, 180, 185, 1)
# train7.schedule_stop(KKDI, 200, 205, 1)

# train8 = Train(env, "T8", [], max_speed=160, priority=0, length=310, weight=2500, initial_delay=0) # Non stop vande bharat express
# train8.schedule_stop(KKDI, 10, 10, 0)
# train8.schedule_stop(CTND, 30, 30, 0)
# train8.schedule_stop(PDKT, 50, 50, 0)
# train8.schedule_stop(KRUR, 70, 70, 0)
# train8.schedule_stop(TPJ, 90, 90, 0)

# # === EXTRA TRAINS (T9–T16) ===

# train9 = Train(env, "T9", [], max_speed=80, priority=2, length=300, weight=4100, initial_delay=0)
# train9.schedule_stop(TPJ, 15, 20, 1)
# train9.schedule_stop(KRUR, 50, 55, 1)
# train9.schedule_stop(PDKT, 85, 90, 0)
# train9.schedule_stop(CTND, 115, 120, 1)
# train9.schedule_stop(KKDI, 130, 135, 0)

# train10 = Train(env, "T10", [], max_speed=110, priority=1, length=310, weight=4050, initial_delay=0)
# train10.schedule_stop(KKDI, 20, 25, 1)
# train10.schedule_stop(CTND, 45, 50, 0)
# train10.schedule_stop(PDKT, 70, 75, 0)
# train10.schedule_stop(KRUR, 95, 100, 1)
# train10.schedule_stop(TPJ, 125, 130, 1)

# train11 = Train(env, "T11", [], max_speed=70, priority=3, length=280, weight=3700, initial_delay=0)
# train11.schedule_stop(TPJ, 30, 35, 1)
# train11.schedule_stop(KRUR, 60, 60, 0)
# train11.schedule_stop(PDKT, 95, 100, 0)
# train11.schedule_stop(CTND, 120, 120, 0)
# train11.schedule_stop(KKDI, 140, 145, 1)

# train12 = Train(env, "T12", [], max_speed=120, priority=1, length=330, weight=4200, initial_delay=0)
# train12.schedule_stop(KKDI, 50, 55, 1)
# train12.schedule_stop(CTND, 75, 80, 1)
# train12.schedule_stop(PDKT, 105, 110, 0)
# train12.schedule_stop(KRUR, 140, 145, 1)
# train12.schedule_stop(TPJ, 165, 170, 0)

# train13 = Train(env, "T13", [], max_speed=100, priority=2, length=260, weight=3500, initial_delay=0)
# train13.schedule_stop(TPJ, 70, 75, 0)
# train13.schedule_stop(KRUR, 100, 100, 0)
# train13.schedule_stop(PDKT, 130, 130, 0)
# train13.schedule_stop(CTND, 155, 160, 1)
# train13.schedule_stop(KKDI, 180, 185, 1)

# train14 = Train(env, "T14", [], max_speed=140, priority=1, length=300, weight=3950, initial_delay=0)
# train14.schedule_stop(KKDI, 85, 90, 0)
# train14.schedule_stop(CTND, 110, 110, 0)
# train14.schedule_stop(PDKT, 140, 140, 0)
# train14.schedule_stop(KRUR, 165, 165, 0)
# train14.schedule_stop(TPJ, 190, 195, 1)

# train15 = Train(env, "T15", [], max_speed=90, priority=3, length=250, weight=4900, initial_delay=0)
# train15.schedule_stop(TPJ, 105, 110, 1)
# train15.schedule_stop(KRUR, 135, 135, 0)
# train15.schedule_stop(PDKT, 160, 160, 0)
# train15.schedule_stop(CTND, 185, 185, 0)
# train15.schedule_stop(KKDI, 210, 215, 1)

# train16 = Train(env, "T16", [], max_speed=160, priority=0, length=310, weight=2500, initial_delay=0)  # Another VB express
# train16.schedule_stop(KKDI, 30, 30, 0)
# train16.schedule_stop(CTND, 55, 55, 0)
# train16.schedule_stop(PDKT, 80, 80, 0)
# train16.schedule_stop(KRUR, 105, 105, 0)
# train16.schedule_stop(TPJ, 125, 125, 0)


# env.run()

# trains = Train.TRAINS
# train_logs = sum([train.log.entries for train in trains], [])
# train_marks = sum([train.log.marks for train in trains], [])

# train_logs.sort(key=lambda x: (x[0], x[1]))
# train_marks.sort(key=lambda x: (x[0], x[1]))
# print("Time\tTrain\tEvent\tStation")
# # for train_log in train_logs:
# #     print(f"{train_log[0]}\t{train_log[1]}\t{train_log[2]}")

# for train_mark in train_marks:
#     print(f"{train_mark[0]}\t{train_mark[1]}\t{train_mark[2]}\t{train_mark[3]}")

# import matplotlib.pyplot as plt
# import pandas as pd


# def plot_timetable(trains: list[Train], stations: list[Station]):
#     # Build structured log dataframe
#     # logs = []
#     # for train in trains:
#     #     for t, tid, ev, stn in train.log.marks:
#     #         logs.append(dict(time=t, train=tid, event=ev, station=stn))
#     # df = pd.DataFrame(logs).sort_values(["time", "train"])

#     # Print logs nicely (instead of subplot)
#     # print("\n===== Event Logs =====")
#     # print(df.to_string(index=False))

#     # ---------- create 2x2 layout ----------
#     fig, axs = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
#     ax1, ax2, ax3, ax4 = axs[0,0], axs[0,1], axs[1,0], axs[1,1]

#     # ---------- (1) CLEAN GANTT CHART ----------
#     ax1.set_title("Train Schedule Gantt Chart (with Station/Block labels)")

#     for i, train in enumerate(trains):
#         color = plt.cm.tab10.colors[i % 10]
#         marks = train.log.marks
#         for j in range(len(marks)-1):
#             t0, tid, type0, stn0 = marks[j]
#             t1, _, type1, stn1 = marks[j+1]

#             # Station dwell
#             if type0 == "ARRIVE" and type1 == "DEPART":
#                 bar = ax1.barh(i, t1 - t0, left=t0, color=color, alpha=0.6, label=tid if j == 0 else "")
#                 label = stn0
#                 mid = (t0 + t1) / 2

#             # Block run
#             elif type0 == "ENTRY" and type1 == "EXIT":
#                 bar = ax1.barh(i, t1 - t0, left=t0, color=color, alpha=0.9, hatch="//", label="")
#                 label = stn0  # block name
#                 mid = (t0 + t1) / 2
#             else:
#                 continue

#             # --- Only add text if it fits inside bar ---
#             bar_width = t1 - t0
#             renderer = ax1.figure.canvas.get_renderer()
#             txt = ax1.text(mid, i, label, ha="center", va="center", fontsize=8, color="white" if type0=="ENTRY" else "black")
#             bb = txt.get_window_extent(renderer=renderer)
#             txt_width = bb.width / ax1.figure.dpi * 72  # approx in data coords

#             if txt_width > bar_width * 1.2:  # too wide → remove
#                 txt.remove()
#     ax1.set_yticks(range(len(trains)))
#     ax1.set_yticklabels([t.id for t in trains])
#     ax1.set_xlabel("Time (min)")
#     ax1.legend()
#     ax1.grid(True, axis="x")

#     # ---------- (2) DELAY HEATMAP ----------
#     ax2.set_title("Delay Heatmap (min)")
#     station_list = stations
#     matrix = []

#     for train in trains:
#         scheduled = [sp.arrival_time for sp in train.schedule]
#         actual = [t for t, tid, ev, stn in train.log.marks if tid==train.id and ev=="ARRIVE"]
#         if len(scheduled) == len(actual):
#             delay = [a-s for a,s in zip(actual, scheduled)]
#         else:
#             delay = [None]*len(scheduled)  # incomplete runs
#         matrix.append(delay)

#     import numpy as np
#     matrix = np.array(matrix)

#     im = ax2.imshow(matrix, cmap="coolwarm", aspect="auto", interpolation="nearest")
#     ax2.set_xticks(range(len(station_list)))
#     ax2.set_xticklabels(station_list)
#     ax2.set_yticks(range(len(trains)))
#     ax2.set_yticklabels([t.id for t in trains])
#     plt.colorbar(im, ax=ax2, label="Delay (min)")


#     # ---------- (3) TRAIN GRAPH ----------
#     ax3.set_title("Train Graph (Runs + Dwells)")
#     xpos = {st:i for i,st in enumerate(stations)}
#     for i, train in enumerate(trains):
#         col = plt.cm.tab10.colors[i % 10]
#         marks = list(filter(lambda x: x[2] in [ARRIVAL, DEPARTURE], train.log.marks))
#         for j in range(len(marks)-1):
#             t0, tid, type0, stn0 = marks[j]
#             t1, _, type1, stn1 = marks[j+1]
#             if type0==ARRIVAL and type1==DEPARTURE:
#                 ax3.plot([xpos[stn0], xpos[stn0]], [t0, t1], linestyle="dashed", color=col, label=f"{tid} - {train.priority}" if j==0 else "")
#             elif type0==DEPARTURE and type1==ARRIVAL:
#                 ax3.plot([xpos[stn0], xpos[stn1]], [t0, t1], linestyle="solid", marker="o", color=col)
#     ax3.set_xticks(range(len(stations)))
#     ax3.set_xticklabels(stations)
#     ax3.set_ylabel("Time (min)")
#     ax3.invert_yaxis()
#     ax3.grid(True)
#     ax3.legend()

#     ax4.set_title("Schedule vs Actual Timetable (Overlay)")

#     for i, train in enumerate(trains):
#         col = plt.cm.tab10.colors[i % 10]

#         # Plot scheduled timetable
#         for sp in train.schedule:
#             arr, dep = sp.arrival_time, sp.departure_time
#             ax4.barh(i, dep-arr, left=arr, edgecolor=col, fill=False, linewidth=1.5, alpha=0.7)

#         # Plot actual timetable from logs
#         marks = train.log.marks
#         for j in range(len(marks)-1):
#             t0, tid, type0, stn0 = marks[j]
#             t1, _, type1, stn1 = marks[j+1]
#             if type0=="ARRIVE" and type1=="DEPART":
#                 ax4.barh(i, t1-t0, left=t0, color=col, alpha=0.6)

#     ax4.set_yticks(range(len(trains)))
#     ax4.set_yticklabels([t.id for t in trains])
#     ax4.set_xlabel("Time (min)")
#     ax4.grid(True, axis="x")

#     # fig.tight_layout()
#     plt.show()

# plot_timetable(Train.TRAINS, [station.stn_code for station in Station.STATIONS])


model, arrive, depart, pf, hold, chg_pf, delay = build_train_scheduling_model(
    trains=Train.TRAINS,
    stations=Station.STATIONS,
    blocks=[b for st in Station.STATIONS for lst in st.connections.values() for b in lst]
)

solver = cp_model.CpSolver()
# solver.parameters.max_time_in_seconds = 30
status = solver.Solve(model)

if status in (cp_model.INFEASIBLE,):
    print("No solution found.")

if status in (cp_model.MODEL_INVALID,):
    print("Model is invalid.")
    # Handle model invalidation (e.g., log, adjust model, etc.)

if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    for t in Train.TRAINS:
        for sp in t.schedule:
            k = (t.id, sp.station.stn_code)
            print(f"{t.id} at {sp.station.stn_code}: "
                  f"arr {solver.Value(arrive[k])}, dep {solver.Value(depart[k])}, "
                  f"pf {solver.Value(pf[k])}, hold={solver.Value(hold[k])}, chgpf={solver.Value(chg_pf[k])}")
