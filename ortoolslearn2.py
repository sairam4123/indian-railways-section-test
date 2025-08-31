from ortools.sat.python import cp_model

model = cp_model.CpModel()

desired_arrival = {"A": 0, "B": 1, "C": 3}
dwell_time = 2

arr, dept, iv, delay = {}, {}, {}, {}

for t, desired in desired_arrival.items():
    arr[t] = model.NewIntVar(0, 20, f"arr_{t}")
    dept[t] = model.NewIntVar(0, 20, f"dept_{t}")
    
    # enforce depart = arrive + dwell
    model.Add(dept[t] == arr[t] + dwell_time)
    
    iv[t] = model.NewIntervalVar(arr[t], dwell_time, dept[t], f"iv_{t}")
    
    # can't arrive earlier than desired
    model.Add(arr[t] >= desired)
    
    # delay = how late it is
    delay[t] = model.NewIntVar(0, 20, f"delay_{t}")
    model.Add(delay[t] == arr[t] - desired)

# one platform only
model.AddNoOverlap(list(iv.values()))

model.Minimize(sum(delay.values()))

solver = cp_model.CpSolver()
status = solver.Solve(model)

if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    for t in desired_arrival:
        print(f"Train {t}: desired {desired_arrival[t]}, scheduled {solver.Value(arr[t])}, delay {solver.Value(delay[t])}")
    print("Total delay:", sum(solver.Value(delay[t]) for t in delay))
