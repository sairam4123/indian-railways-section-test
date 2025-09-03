from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from train_lib.models import BlockSection, Train, SchedulePoint, Station

def pass_conflict_hold(train: 'Train', prev_block_section: 'BlockSection', current_sp: 'SchedulePoint') -> float:

    if not prev_block_section.occupied_train:
        return 0.0 # we do not need to wait for next train
    
    succeeding_train, _ = prev_block_section.occupied_train
    # search for current station in the succeeding train schedule

    prev_station = prev_block_section.from_station if prev_block_section.from_station != current_sp.station else prev_block_section.to_station
    
    succeeding_train_csp = next((sp for sp in succeeding_train.schedule if sp.station == current_sp.station), None)
    succeeding_train_psp = next((sp for sp in succeeding_train.schedule if sp.station == prev_station), None)

    if not succeeding_train_csp:
        print("[CRITICAL] THIS SHOULD NOT HAPPEN.. ")
        raise Exception(f"Succeeding train does not have current station {current_sp.station.stn_code} in its schedule")

    if not succeeding_train_psp:
        print("[CRITICAL] THIS SHOULD NOT HAPPEN.. ")
        raise Exception(f"Succeeding train does not have previous station {prev_station.stn_code} in its schedule")

    succeeding_train_pred_arrival = succeeding_train_psp.actual_departure_time + prev_block_section._run_minutes(succeeding_train, succeeding_train.should_accelerate(succeeding_train_csp, succeeding_train_csp.dwell_time), succeeding_train.should_decelerate(succeeding_train_csp))

    current_train_csp = next((sp for sp in train.schedule if sp.station == current_sp.station), None)
    current_train_nsp = next((sp for sp in train.schedule if sp.station != current_sp.station and sp.arrival_time > current_sp.arrival_time), None)
    # current_train_psp = next((sp for sp in train.schedule if sp.station == prev_station), None)


    if not current_train_csp:
        print("[CRITICAL] THIS SHOULD NOT HAPPEN.. ")
        raise Exception(f"Current train does not have current station {current_sp.station.stn_code} in its schedule")

    if not current_train_nsp:
        return 0.0 # no next station, train terminates here
    
    next_station = current_train_nsp.station
    next_block_section = next_station.get_block_to(next_station)
    if not next_block_section:
        print("[WARN] No next block section found. Returning 0.0")
        return 0.0 # no next block section, train terminates here

    # if current train is departing earlier than succeeding train's predicted arrival
    if current_sp.departure_time <= succeeding_train_pred_arrival:
        if train.priority < succeeding_train.priority:
            # current train has lower priority than succeeding train, must wait
            wait_time = (succeeding_train_pred_arrival - current_sp.departure_time) + next_block_section._headway_mins(train, True) # since we are stopping at this station
            return wait_time

    if current_sp.departure_time > succeeding_train_pred_arrival:
        return 0.0 # We do not have to do anything here.

    return 0.0


def meet_conflict_hold(train: 'Train', oncoming_block_section: 'BlockSection', current_sp: 'SchedulePoint') -> float:
    if not oncoming_block_section.occupied_train:
        return 0.0 # we do not need to wait for next train

    oncoming_train, _ = oncoming_block_section.occupied_train
    # search for current station in the oncoming train schedule

    oncoming_train_csp = next((sp for sp in oncoming_train.schedule if sp.station == current_sp.station), None)

    if not oncoming_train_csp:
        print("[CRITICAL] THIS SHOULD NOT HAPPEN.. ")
        raise Exception(f"Oncoming train does not have current station {current_sp.station.stn_code} in its schedule")

    oncoming_train_pred_arrival = oncoming_train_csp.actual_departure_time + oncoming_block_section._run_minutes(oncoming_train, oncoming_train.should_accelerate(oncoming_train_csp, oncoming_train_csp.dwell_time), oncoming_train.should_decelerate(oncoming_train_csp))

    current_train_nsp = next((sp for sp in train.schedule if sp.station != current_sp.station and sp.arrival_time > current_sp.arrival_time), None)


    if not current_train_nsp:
        print("[WARN] No next station found. Returning 0.0")
        return 0.0 # no next station, train terminates here

    next_station = current_train_nsp.station
    next_block_section = current_sp.station.get_block_to(next_station)
    if not next_block_section:
        print("[WARN] No next block section found. Returning 0.0")
        return 0.0 # no next block section, train terminates here

    # if current train is departing earlier than oncoming train's predicted arrival
    if current_sp.departure_time <= oncoming_train_pred_arrival:
        if train.priority < oncoming_train.priority:
            # current train has lower priority than oncoming train, must wait
            wait_time = (oncoming_train_pred_arrival - current_sp.departure_time) + next_block_section._headway_mins(train, True) # since we are stopping at this station
            return wait_time

    if current_sp.departure_time > oncoming_train_pred_arrival:
        return 0.0 # We do not have to do anything here.

    return 0.0

def capacity_conflict_hold(train: 'Train', current_sp: 'SchedulePoint'):

    next_scheduled_halt_sp = next((sp for sp in train.schedule if sp.station != current_sp.station and sp.arrival_time > current_sp.actual_arrival_time and sp.layover_time > 0), None)

    next_station = next((sp for sp in train.schedule if sp != current_sp and sp.arrival_time > current_sp.actual_arrival_time), None)

    if not next_station:
        # we don't have a next station, train is terminating
        return 0.0

    if not next_scheduled_halt_sp:
        # We don't have any next scheduled stop, either train is terminating or it is moving out of map (controlled section)
        return 0.0
    
    next_scheduled_halt_index = train.schedule.index(next_scheduled_halt_sp)
    if next_scheduled_halt_index == -1:
        # Same as above.. Should not happen
        print("[CRITICAL] THIS SHOULD NOT HAPPEN.. {}".format(next_scheduled_halt_sp))
        return 0.0

    next_scheduled_halt_station = next_scheduled_halt_sp.station
    # check for available capacity
    track_availability = next(t for t in next_scheduled_halt_station.tracks if t.has_platform and not t.resource.count > 0)
    if track_availability:
        return 0.0 # we have available platform track, no need to wait

    # next_block_section = current_sp.station.get_block_to(next_station)
    earliest_departing_train = find_earliest_departing_train(next_scheduled_halt_station, current_sp.departure_time)
    if earliest_departing_train:
        other_train, edt_sp = earliest_departing_train
        # we have a earliest departing train
        # check if it's a oncoming train with respect to the current station
        is_oncoming_train = train.direction != other_train.direction

        wait_until = edt_sp.departure_time

        if not is_oncoming_train:
            # If it's not an oncoming train, we can use the earliest departing train's schedule point
            return wait_until - current_sp.actual_arrival_time
        else:
            # it is an oncoming train
            # we need to check if the earliest departing train is going to occupy the next block section

   
            # oncoming case → check block conflict
            next_block = current_sp.station.get_block_to(next_station.station)
            other_next_station = next(
                (sp for sp in other_train.schedule 
                if sp.arrival_time >= edt_sp.departure_time), 
                None
            )
            if other_next_station:
                other_next_block = edt_sp.station.get_block_to(other_next_station.station)
                if other_next_block == next_block and other_next_block and next_block:
                    safety_buffer = next_block._headway_mins(train, True)
                    # collision risk: wait until block is cleared
                    return max(0.0, (wait_until + safety_buffer) - current_sp.actual_arrival_time)

            # if not same block, no problem
            return 0.0


    # No available platform track, we need to wait
    # Find the earliest departing train from that station


def find_earliest_departing_train(station: 'Station', before_time: float) -> tuple['Train', 'SchedulePoint'] | None:
    for train in station.train_map.keys():
        for sp in train.schedule:
            if sp.departure_time < before_time and sp.station == station:
                return train, sp
    return None


