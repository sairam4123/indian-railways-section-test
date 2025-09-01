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

            model.AddHint(depart[key], sp.departure_time)
            model.AddHint(arrive[key], sp.arrival_time)


    # ---------------- BLOCK CONSTRAINTS ----------------
    for t in trains:
        for i in range(len(t.schedule)-1):
            s1 = t.schedule[i].station
            s2 = t.schedule[i+1].station
            k1 = (t.id, s1.stn_code)
            k2 = (t.id, s2.stn_code)

            block = s1.get_block_to(s2)
            run_time = block._run_minutes(t)
            clearance = int(block._headway_mins(t))

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

import simpy

from train_lib.models import Train, Station
from networks.kkdi_tpj_network import create_tpj_kkdi_network

env = simpy.Environment()

[[TPJ, KRUR, PDKT, CTND, KKDI], block_sections, *loop_tracks] = create_tpj_kkdi_network(env)

# Existing trains
train1 = Train(env, "T1", [], max_speed=80, priority=1, length=300, weight=1173, initial_delay=0, hp=6120, accel_mps2=0.5, decel_mps2=0.5)
train1.schedule_stop(TPJ, 0, 10, 1)
train1.schedule_stop(KRUR, 40, 45, 1)
train1.schedule_stop(PDKT, 70, 75, 0)
train1.schedule_stop(CTND, 100, 105, 1)
train1.schedule_stop(KKDI, 110, 115, 0)

# train2 = Train(env, "T2", [], max_speed=110, priority=2, length=300, weight=2244, initial_delay=0, hp=6120, accel_mps2=0.5, decel_mps2=0.5)
# train2.schedule_stop(KKDI, 10, 20, 1)
# train2.schedule_stop(CTND, 35, 40, 1)
# train2.schedule_stop(PDKT, 55, 65, 0)
# train2.schedule_stop(KRUR, 70, 75, 1)
# train2.schedule_stop(TPJ, 110, 115, 1)

# train3 = Train(env, "T3", [], max_speed=80, priority=1, length=280, weight=1170, initial_delay=0, hp=6120, accel_mps2=0.5, decel_mps2=0.5)
# train3.schedule_stop(TPJ, 20, 25, 1)
# train3.schedule_stop(KRUR, 40, 40, 0) # Run through main
# train3.schedule_stop(PDKT, 85, 90, 0)
# train3.schedule_stop(CTND, 100, 100, 0) # Run through main
# train3.schedule_stop(KKDI, 125, 130, 1)

train4 = Train(env, "T4", [], max_speed=120, priority=2, length=320, weight=2580, initial_delay=0, hp=4500, accel_mps2=0.5, decel_mps2=0.5)
train4.schedule_stop(KKDI, 40, 45, 1)
train4.schedule_stop(CTND, 60, 65, 1)
train4.schedule_stop(PDKT, 95, 100, 1)
train4.schedule_stop(KRUR, 130, 135, 1)
train4.schedule_stop(TPJ, 150, 155, 0)

# train5 = Train(env, "T5", [], max_speed=90, priority=3, length=250, weight=1277, initial_delay=0, hp=4500, accel_mps2=0.3, decel_mps2=0.2)
# train5.schedule_stop(TPJ, 60, 65, 0)
# train5.schedule_stop(KRUR, 90, 90, 0)
# train5.schedule_stop(PDKT, 120, 120, 0)
# train5.schedule_stop(CTND, 140, 140, 0)
# train5.schedule_stop(KKDI, 170, 175, 1)

# train6 = Train(env, "T6", [], max_speed=160, priority=1, length=300, weight=4000, initial_delay=0, hp=12000, accel_mps2=0.5, decel_mps2=0.5)
# train6.schedule_stop(KKDI, 75, 80, 0)
# train6.schedule_stop(CTND, 100, 100, 0) # Run through main
# train6.schedule_stop(PDKT, 130, 130, 0) # Non-stop run (run-through on the main-line)
# train6.schedule_stop(KRUR, 160, 160, 0)
# train6.schedule_stop(TPJ, 185, 190, 1)

train7 = Train(env, "T7", [], max_speed=100, priority=2, length=270, weight=3600, initial_delay=0, hp=3125, accel_mps2=0.5, decel_mps2=0.5)
train7.schedule_stop(TPJ, 95, 100, 1)
train7.schedule_stop(KRUR, 120, 125, 1)
train7.schedule_stop(PDKT, 150, 155, 0)
train7.schedule_stop(CTND, 180, 185, 1)
train7.schedule_stop(KKDI, 200, 205, 1)

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
