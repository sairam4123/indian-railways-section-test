import datetime
from typing import Literal, TypedDict

type ArrDepTime = datetime.datetime | Literal["N/A", "Source", "Destination"]

# "train_no": train_info[0],
# "train_name": train_info[1],
# "src": train_info[2],
# "dest": train_info[3],
# "tt_arr": train_info[4],
# "tt_dept": train_info[5],
# "tt_pf": train_info[6],
# "tt_halt": train_info[7],
# "exp_arr": train_info[8],
# "exp_arr_delay": train_info[9],
# "exp_dept": train_info[10],
# "exp_dept_delay": train_info[11],


class ETrainStnLiveArrDepTrain(TypedDict):
    train_no: str
    train_name: str
    src: str
    dest: str
    tt_arr: str
    tt_dept: str
    tt_pf: str
    tt_halt: str
    exp_arr: str
    exp_arr_delay: str
    exp_dept: str
    exp_dept_delay: str


#  "index": sta_info[0],
# "code": sta_info[1],
# "name": sta_info[2],
# "dist": dist,
# "pf": pf,
# "a": arr,
# "d": dept,
# "a_day": a_day,
# "d_day": d_day,


class ETrainScheduleStation(TypedDict):
    index: str
    code: str
    name: str
    dist: str
    pf: str
    a: str
    d: str
    a_day: str
    d_day: str


#  stn = {
#                 "index": running_stn_info[0],
#                 "name": running_stn_info[1],
#                 "s_dist": running_stn_info[2],
#                 "pf": pf,
#                 "tt_arr": scheduled_arr,
#                 "act_arr": actual_arr,
#                 "tt_dept": scheduled_dept,
#                 "act_dept": actual_dept,
#             }


class ETrainRunningStatusStation(TypedDict):
    index: str
    name: str
    s_dist: str
    pf: str
    tt_arr: ArrDepTime
    act_arr: ArrDepTime
    tt_dept: ArrDepTime
    act_dept: ArrDepTime


# "train_no": train_info[0],
#                 "train_name": train_info[1],
#                 "src": train_info[2],
#                 "dest": train_info[3],
#                 "tt_arr": train_info[4],
#                 "tt_dept": train_info[5],
#                 "tt_halt": train_info[6],
#                 "running_days": train_info[7:14],
#                 "classes": train_info[14:],


class ETrainAllTrainsStation(TypedDict):
    train_no: str
    train_name: str
    src: str
    dest: str
    tt_arr: str
    tt_dept: str
    tt_halt: str
    running_days: list[str]
    classes: list[str]
