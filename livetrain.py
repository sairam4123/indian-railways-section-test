from etrainlib import ETrainAPISync


etrain = ETrainAPISync()
live_station = etrain.get_live_station("TPJ", "TIRUCHCHIRAPALI")
for train in live_station[:4]:  # Only first 4 trains for demo
    print(
        f"{train['train_no']} {train['train_name']} - {train['exp_arr']} / {train['exp_dept']} - {train['exp_arr_delay']} mins"
    )
    schedule = etrain.get_train_schedule(train["train_no"], train["train_name"])

    coach_pos = etrain.get_coach_positions(train["train_no"], train["train_name"])
    print(f"Coach positions for {train['train_no']} {train['train_name']}:")
    for pos in coach_pos:
        print(f" {coach_pos[pos]} at position {pos}")
    print(f"Schedule for {train['train_no']} {train['train_name']}:")
    for station in schedule:
        print(
            f"  {station['name']} - Arr: {station['a']}, Dept: {station['d']}, {station['dist']} km from source"
        )
