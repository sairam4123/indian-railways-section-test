


from ortools.sat.python import cp_model
from ortools.sat.python import cp_model

def build_block_model(trains, blocks,
                      slack=60,     # how much later than scheduled we allow
                      penalty_delay=1,
                      penalty_hold=10):
    model = cp_model.CpModel()

    arrive, depart, delay, hold = {}, {}, {}, {}

    for t in trains:
        for sp in t.schedule:
            k = (t.id, sp.station.stn_code)

            # Local domains instead of [0..horizon]
            arr_lb, arr_ub = sp.arrival_time, sp.arrival_time + slack
            dep_lb, dep_ub = sp.departure_time, sp.departure_time + slack

            arrive[k] = model.NewIntVar(arr_lb, arr_ub, f"arr_{t.id}_{sp.station.stn_code}")
            depart[k] = model.NewIntVar(dep_lb, dep_ub, f"dep_{t.id}_{sp.station.stn_code}")

            delay[k] = model.NewIntVar(0, slack, f"delay_{t.id}_{sp.station.stn_code}")
            hold[k] = model.NewBoolVar(f"hold_{t.id}_{sp.station.stn_code}")

            # Dwell time constraint
            model.Add(depart[k] >= arrive[k] + sp.layover_time)

            # Hold definition
            model.Add(depart[k] >= sp.departure_time)
            model.Add(depart[k] - sp.departure_time >= 1).OnlyEnforceIf(hold[k])
            model.Add(depart[k] - sp.departure_time <= 0).OnlyEnforceIf(hold[k].Not())

            # Delay definition
            model.Add(delay[k] >= depart[k] - sp.departure_time)

            # Solver hints → seed with timetable
            model.AddHint(arrive[k], sp.arrival_time)
            model.AddHint(depart[k], sp.departure_time)

    # --- Block precedence constraints (no IntervalVars/NoOverlap) ---
    for block in blocks:
        # Collect trains that use this block
        users = []
        for t in trains:
            for i in range(len(t.schedule)-1):
                s1, s2 = t.schedule[i].station, t.schedule[i+1].station
                if s1.get_block_to(s2) == block:
                    k1, k2 = (t.id, s1.stn_code), (t.id, s2.stn_code)
                    run = int(block._run_minutes(t))
                    users.append((t.id, k1, k2, run))

        # Pairwise precedence for trains using this block
        for i in range(len(users)):
            tid1, k1a, k1b, run1 = users[i]
            for j in range(i+1, len(users)):
                tid2, k2a, k2b, run2 = users[j]

                # Either train1 goes first or train2 goes first
                before = model.NewBoolVar(f"{block.name}_ord_{tid1}_{tid2}")
                model.Add(arrive[k2a] >= depart[k1a] + run1).OnlyEnforceIf(before)      # t1 before t2
                model.Add(arrive[k1a] >= depart[k2a] + run2).OnlyEnforceIf(before.Not()) # t2 before t1

    # --- Objective: minimize delays + holds ---
    obj_terms = []
    for k in delay:
        obj_terms.append(delay[k] * penalty_delay)
        obj_terms.append(hold[k] * penalty_hold)
    model.Minimize(sum(obj_terms))

    return model, arrive, depart, delay, hold



def assign_platforms_station(station, trains_at_station, arr_times, dep_times, penalty_chg_pf=5):
    """
    Solve a small CP-SAT model for one station.
    Returns dicts { (train_id, stn): platform, chg_pf }.
    """
    model = cp_model.CpModel()
    pf, chg_pf = {}, {}
    obj_terms = []

    for t in trains_at_station:
        sp = next(sp for sp in t.schedule if sp.station == station)
        k = (t.id, station.stn_code)

        # pf var (0..tracks-1)
        pf[k] = model.NewIntVar(0, len(station.tracks)-1, f"pf_{t.id}_{station.stn_code}")
        chg_pf[k] = model.NewBoolVar(f"chgpf_{t.id}_{station.stn_code}")

        is_expected = model.NewBoolVar(f"isexp_{t.id}_{station.stn_code}")
        model.Add(pf[k] == sp.expected_platform).OnlyEnforceIf(is_expected)
        model.Add(pf[k] != sp.expected_platform).OnlyEnforceIf(is_expected.Not())

        model.Add(chg_pf[k] == 0).OnlyEnforceIf(is_expected)
        model.Add(chg_pf[k] == 1).OnlyEnforceIf(is_expected.Not())

        obj_terms.append(chg_pf[k] * penalty_chg_pf)

    # Platform conflict constraints (only inside this station)
    for i, t1 in enumerate(trains_at_station):
        sp1 = next(sp for sp in t1.schedule if sp.station == station)
        k1 = (t1.id, station.stn_code)
        for t2 in trains_at_station[i+1:]:
            sp2 = next(sp for sp in t2.schedule if sp.station == station)
            k2 = (t2.id, station.stn_code)

            same_pf = model.NewBoolVar(f"samepf_{t1.id}_{t2.id}_{station.stn_code}")
            model.Add(pf[k1] == pf[k2]).OnlyEnforceIf(same_pf)
            model.Add(pf[k1] != pf[k2]).OnlyEnforceIf(same_pf.Not())

            # If same PF, enforce non-overlap
            model.Add(dep_times[k1] <= arr_times[k2]).OnlyEnforceIf(same_pf)
            model.Add(dep_times[k2] <= arr_times[k1]).OnlyEnforceIf(same_pf)

    model.Minimize(sum(obj_terms))

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    pf_vals, chg_vals = {}, {}
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for t in trains_at_station:
            k = (t.id, station.stn_code)
            pf_vals[k] = solver.Value(pf[k])
            chg_vals[k] = solver.Value(chg_pf[k])
    else:
        # Fallback: assign everything to 0
        for t in trains_at_station:
            k = (t.id, station.stn_code)
            pf_vals[k] = 0
            chg_vals[k] = 0

    return pf_vals, chg_vals


def corridor_partition(blocks, stations, junction_threshold=3):
    """
    Split the network into corridors based on station degree (connections).
    - junction_threshold: stations with >= this many connections are corridor boundaries.
    
    Returns a list of corridor block-lists.
    """
    # Build station connectivity graph
    adj = {st: [] for st in stations}
    for b in blocks:
        adj[b.from_station].append(b.to_station)
        adj[b.to_station].append(b.from_station) if b.bidirectional else None

    corridors = []
    visited = set()

    for b in blocks:
        if b in visited:
            continue

        # start corridor from this block
        corridor = [b]
        visited.add(b)

        # walk forward until a junction/terminal
        current = b.to_station
        while (len(adj[current]) == 2 and current not in (b.from_station, b.to_station)):
            # pick next block that isn’t visited
            next_blocks = [blk for blk in blocks if blk.from_station == current and blk not in visited]
            if not next_blocks:
                break
            nb = next_blocks[0]
            corridor.append(nb)
            visited.add(nb)
            current = nb.to_station

        corridors.append(corridor)

    return corridors


import simpy

from train_lib.models import Train, Station
from networks.kkdi_tpj_network import create_tpj_kkdi_network


env = simpy.Environment()

[[TPJ, KRUR, PDKT, CTND, KKDI], block_sections, *loop_tracks] = create_tpj_kkdi_network(env)

# Flatten block sections


# Existing trains
# train1 = Train(env, "T1", [], max_speed=80, priority=1, length=300, weight=1173, initial_delay=0, hp=6120, accel_mps2=0.5, decel_mps2=0.5)
# train1.schedule_stop(TPJ, 0, 10, 1)
# train1.schedule_stop(KRUR, 40, 45, 1)
# train1.schedule_stop(PDKT, 70, 75, 0)
# train1.schedule_stop(CTND, 100, 105, 1)
# train1.schedule_stop(KKDI, 110, 115, 0)

train2 = Train(env, "T2", [], max_speed=110, priority=2, length=300, weight=2244, initial_delay=0, hp=6120, accel_mps2=0.5, decel_mps2=0.5)
train2.schedule_stop(KKDI, 10, 20, 1)
train2.schedule_stop(CTND, 35, 40, 1)
train2.schedule_stop(PDKT, 55, 65, 0)
train2.schedule_stop(KRUR, 70, 75, 1)
train2.schedule_stop(TPJ, 110, 115, 1)

train3 = Train(env, "T3", [], max_speed=80, priority=1, length=280, weight=1170, initial_delay=0, hp=6120, accel_mps2=0.5, decel_mps2=0.5)
train3.schedule_stop(TPJ, 20, 25, 1)
train3.schedule_stop(KRUR, 40, 40, 0) # Run through main
train3.schedule_stop(PDKT, 85, 90, 0)
train3.schedule_stop(CTND, 100, 100, 0) # Run through main
train3.schedule_stop(KKDI, 125, 130, 1)

# train4 = Train(env, "T4", [], max_speed=120, priority=2, length=320, weight=2580, initial_delay=0, hp=4500, accel_mps2=0.5, decel_mps2=0.5)
# train4.schedule_stop(KKDI, 40, 45, 1)
# train4.schedule_stop(CTND, 60, 65, 1)
# train4.schedule_stop(PDKT, 95, 100, 1)
# train4.schedule_stop(KRUR, 130, 135, 1)
# train4.schedule_stop(TPJ, 150, 155, 0)

train5 = Train(env, "T5", [], max_speed=90, priority=3, length=250, weight=1277, initial_delay=0, hp=4500, accel_mps2=0.3, decel_mps2=0.2)
train5.schedule_stop(TPJ, 60, 65, 0)
train5.schedule_stop(KRUR, 90, 90, 0)
train5.schedule_stop(PDKT, 120, 120, 0)
train5.schedule_stop(CTND, 140, 140, 0)
train5.schedule_stop(KKDI, 170, 175, 1)

# train6 = Train(env, "T6", [], max_speed=160, priority=1, length=300, weight=4000, initial_delay=0, hp=12000, accel_mps2=0.5, decel_mps2=0.5)
# train6.schedule_stop(KKDI, 75, 80, 0)
# train6.schedule_stop(CTND, 100, 100, 0) # Run through main
# train6.schedule_stop(PDKT, 130, 130, 0) # Non-stop run (run-through on the main-line)
# train6.schedule_stop(KRUR, 160, 160, 0)
# train6.schedule_stop(TPJ, 185, 190, 1)

# train7 = Train(env, "T7", [], max_speed=100, priority=2, length=270, weight=3600, initial_delay=0, hp=3125, accel_mps2=0.5, decel_mps2=0.5)
# train7.schedule_stop(TPJ, 95, 100, 1)
# train7.schedule_stop(KRUR, 120, 125, 1)
# train7.schedule_stop(PDKT, 150, 155, 0)
# train7.schedule_stop(CTND, 180, 185, 1)
# train7.schedule_stop(KKDI, 200, 205, 1)

# train8 = Train(env, "T8", [], max_speed=160, priority=0, length=384, weight=430, initial_delay=0, hp=9010, accel_mps2=0.15, decel_mps2=0.18) # Non stop vande bharat express
# train8.schedule_stop(KKDI, 10, 10, 0)
# train8.schedule_stop(CTND, 30, 30, 0)
# train8.schedule_stop(PDKT, 50, 50, 0)
# train8.schedule_stop(KRUR, 70, 70, 0)
# train8.schedule_stop(TPJ, 90, 90, 0)

# # === EXTRA TRAINS (T9–T16) ===

# train9 = Train(env, "T9", [], max_speed=80, priority=2, length=300, weight=4100, initial_delay=0, hp=6120, accel_mps2=0.5, decel_mps2=0.5)
# train9.schedule_stop(TPJ, 15, 20, 1)
# train9.schedule_stop(KRUR, 50, 55, 1)
# train9.schedule_stop(PDKT, 85, 90, 0)
# train9.schedule_stop(CTND, 115, 120, 1)
# train9.schedule_stop(KKDI, 130, 135, 0)

# train10 = Train(env, "T10", [], max_speed=110, priority=1, length=310, weight=4050, initial_delay=0, hp=6120, accel_mps2=0.5, decel_mps2=0.5)
# train10.schedule_stop(KKDI, 20, 25, 1)
# train10.schedule_stop(CTND, 45, 50, 0)
# train10.schedule_stop(PDKT, 70, 75, 0)
# train10.schedule_stop(KRUR, 95, 100, 1)
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

blocks = [b for lst in block_sections for b in lst]
stations = Station.STATIONS
trains = Train.TRAINS

solutions = {}
for corridor_blocks in corridor_partition(blocks, stations):  # you decide grouping
    model, arrive, depart, delay, hold = build_block_model(trains, corridor_blocks)
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for k in arrive:
            solutions[k] = (solver.Value(arrive[k]), solver.Value(depart[k]), solver.Value(hold[k]), solver.Value(delay[k]))


# Step 1: Run block model
# block_model, arrive, depart, delay = build_block_model(trains, blocks)
# solver = cp_model.CpSolver()
# status = solver.Solve(block_model)


arr_times = {k: solutions[k][0] for k in solutions}
dep_times = {k: solutions[k][1] for k in solutions}
hold_times = {k: solutions[k][2] for k in solutions}
delay_times = {k: solutions[k][3] for k in solutions}

# Step 2: Run platform model with fixed times
# platform_model, pf, chg_pf = build_platform_model(trains, stations, arr_times, dep_times)
# solver2 = cp_model.CpSolver()
# status2 = solver2.Solve(platform_model)


# model, arrive, depart, pf, hold, chg_pf, delay = build_train_scheduling_model(
#     trains=Train.TRAINS,
#     stations=Station.STATIONS,
#     blocks=[b for st in Station.STATIONS for lst in st.connections.values() for b in lst]
# )

# solver = cp_model.CpSolver()
# # solver.parameters.max_time_in_seconds = 30
# status = solver.Solve(model)

if status in (cp_model.INFEASIBLE,):
    print("No solution found.")

if status in (cp_model.MODEL_INVALID,):
    print("Model is invalid.")
    # Handle model invalidation (e.g., log, adjust model, etc.)

if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):

    all_pf, all_chgpf = {}, {}
    for st in stations:
        trains_here = [t for t in Train.TRAINS if any(sp.station == st for sp in t.schedule)]
        if not trains_here:
            continue

        pf_vals, chg_vals = assign_platforms_station(st, trains_here, arr_times, dep_times)
        all_pf.update(pf_vals)
        all_chgpf.update(chg_vals)

    print("\n=== FINAL TIMETABLE ===")
    for t in Train.TRAINS:
        for sp in t.schedule:
            k = (t.id, sp.station.stn_code)
            arr_val = arr_times[k]
            dep_val = dep_times[k]
            hold_val = hold_times[k]
            delay_val = delay_times[k]
            pf_val = all_pf.get(k, "-")
            chg_val = all_chgpf.get(k, "-")
            print(f"{t.id} at {sp.station.stn_code}: arr={arr_val}, dep={dep_val}, hold={hold_val}, delay={delay_val}, pf={pf_val}, chgpf={chg_val}")