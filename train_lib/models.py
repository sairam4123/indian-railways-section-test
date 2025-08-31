import random

import simpy
import simpy.resources.resource

from train_lib.physics import total_headway, total_runtime

from .constants import *



class Station:
    STATIONS = []

    def __init__(self, env: simpy.Environment, stn_code: str, tracks: list['Track']) -> None:
        self.stn_code = stn_code
        self.tracks: list[Track] = tracks
        self.env = env
        self.train_map: dict['Train', tuple['Track', simpy.resources.resource.PriorityRequest]] = {}
        self.connections: dict['Station', list['BlockSection']] = {}

        Station.STATIONS.append(self)

    def accept(self, train: 'Train', sp: 'SchedulePoint'):
        """
        Acquire a suitable platform track as soon as the train arrives.
        We do NOT release here; we keep the request in train_map for live view + later dispatch().
        """
        # choose track: prefer expected platform if valid & fits; else first available that fits
        track = None
        if 0 <= sp.expected_platform < len(self.tracks):
            cand = self.tracks[sp.expected_platform]
            if cand.length >= train.length_m:
                track = cand

        if track is None or track.resource.count > 0: # Track is not valid or occupied
            track = self.get_first_available_track(train)
            if track is None:
                # all suitable tracks busy -> queue on any suitable track; pick the first that fits
                for t in self.tracks:
                    if t.length >= train.length_m:
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
            if not track.resource.count and track.length >= train.length_m:
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


class BlockSection:
    def __init__(self, env: simpy.Environment, 
                 name: str, from_station: Station, to_station: Station, 
                 line_speed: int, length_km: int,
                 bidirectional: bool = False, 
                 electric: bool = False,
                 signal_num: int = 2,
                 signal_aspects: int = 2) -> None:
        self.name = name
        self.from_station = from_station
        self.to_station = to_station
        self.line_speed = line_speed  # in km/h
        self.length_km = length_km
        self.signal_num = signal_num
        self.bidirectional = bidirectional
        self.env = env
        self.electric = electric
        self.signal_aspects = signal_aspects

        self.resource = simpy.PriorityResource(env, capacity=max(signal_num-1, 1))
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
        - train.accel_mps2, train.decel_mps2 in m/s^2
        """
        total_min = total_runtime(self.length_km, self.line_speed, train.max_speed, train.accel_mps2, train.decel_mps2)

        return total_min
    
    def _headway_mins(self, train: 'Train') -> float:

        """
        Compute the headway time (minutes) for this block/section.

        Assumes:
        - self.line_speed in km/h
        - self.length_km in km
        - train.max_speed in km/h
        - train.accel_mps2, train.decel_mps2 in m/s^2
        """
        total_min = total_headway(train.max_speed, self.line_speed, self.length_km * 1000, train.accel_mps2, train.decel_mps2)
        return total_min

    # def traverse(self, train: 'Train', force_minutes: int | None = None):
    #     """Acquire -> run (or force_minutes) -> headway -> release, while holding the lock."""
    #     req = self.resource.request(priority=train.priority)
    #     yield req
    #     self.occupied_train = (train, req)
    #     try:
    #         run = self._run_minutes(train) if force_minutes is None else force_minutes
    #         yield self.env.timeout(run)              # inside block
    #         yield self.env.timeout(self.headway_min) # clearance
    #     finally:
    #         self.resource.release(req)
    #         self.occupied_train = None

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

    def __init__(self, env: simpy.Environment, id: str, schedule: list['SchedulePoint'], max_speed: int, priority: int, length: int, weight: int, initial_delay: int, 
                 hp: float, accel_mps2: float | None = None, decel_mps2: float | None = None) -> None:
        self.env = env
        self.id = id
        self.schedule = schedule
        self.max_speed = max_speed
        self.priority = priority
        self.length_m = length
        self.initial_delay = initial_delay + random.randint(0, 15) if random.random() < 0.5 and ENABLE_RANDOM_DELAYS else initial_delay
        self.running_delay = self.initial_delay
        self.schedule_pointer = 0
        self.weight = weight
        self.log = TrainLog(self)

        self.hp = hp

        self.accel_mps2 = accel_mps2 if accel_mps2 else (0.9 if hp / weight >= 20 else 0.7 if hp / weight >= 10 else 0.5)  # simple heuristic
        self.decel_mps2 = decel_mps2 if decel_mps2 else (0.9 if hp / weight >= 20 else 0.7 if hp / weight >= 10 else 0.5)  # simple heuristic
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
                headway = block._headway_mins(self)
                print("Headway:", headway, "Block:", block.name, "Train:", self.id, self.max_speed, self.length_m, self.hp, block.signal_aspects)
                yield self.env.timeout(headway)
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

