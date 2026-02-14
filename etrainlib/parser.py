import datetime

import bs4
from bs4.element import PageElement
import parsedatetime

from etrainlib.constants import (
    ETrainAllTrainsConfig,
    ETrainArrivalDepartureConfig,
)
from .models import (
    ETrainRunningStatusStation,
    ETrainScheduleStation,
    ETrainAllTrainsStation,
    ETrainStnLiveArrDepTrain,
)


def parse_schedule(dt: str):
    day = dt.split(" ")[2].strip(")")
    if dt.lower().startswith("source"):
        return "Source", day
    elif dt.lower().startswith("dest"):
        return "Destination", day
    _time = dt.split(" ")[0]
    return _time, day


def parse_running_status_arr_dep(sta_entry_arr_dep: list[str]):
    # ArrDate, ArrYear, ArrDelay, DeptDate, DeptYear, DeptDelay

    if len(sta_entry_arr_dep) < 3:
        print(f"[WARNING] Invalid arr/dep data: {sta_entry_arr_dep}")
        return "N/A", "N/A", "N/A", "N/A"

    if sta_entry_arr_dep[0].lower().startswith(("cancel", "diverted")):
        print(f"[WARNING] Train cancelled/diverted. Marking arr/dep as N/A.")
        return "N/A", "N/A", "N/A", "N/A"
    scheduled_arr, actual_arr = None, None
    scheduled_dept, actual_dept = None, None
    print("DEBUG: Parsing running status arr/dep data:", sta_entry_arr_dep)
    arr = sta_entry_arr_dep[0]
    if arr.lower().startswith("source"):
        scheduled_arr, actual_arr = parse_running_status(arr)
    elif arr.lower().startswith(("diverted", "cancel")):
        pass  # FIXME: we handle this later.
    else:
        arr = sta_entry_arr_dep[0:3]
        scheduled_arr, actual_arr = parse_running_status(*arr)
    if isinstance(scheduled_arr, str) and scheduled_arr.lower().startswith("source"):
        dept = sta_entry_arr_dep[1:4]
        scheduled_dept, actual_dept = parse_running_status(*dept)
    else:
        dept = sta_entry_arr_dep[3]
        if dept.lower().startswith("dest"):
            scheduled_dept, actual_dept = parse_running_status(dept)
        elif dept.lower().startswith(("diverted", "cancel")):
            pass  # FIXME: we handle this later,
        else:
            dept = sta_entry_arr_dep[3:6]
            scheduled_dept, actual_dept = parse_running_status(*dept)

    return (
        scheduled_arr or "N/A",
        actual_arr or "N/A",
        scheduled_dept or "N/A",
        actual_dept or "N/A",
    )


def parse_running_status(dt: str, year=None, delay=None):
    if dt.lower().startswith("source") and year is None and delay is None:
        return "Source", "N/A"
    if dt.lower().startswith("dest") and year is None and delay is None:
        return "Destination", "N/A"
    _time = datetime.datetime.strptime(f"{dt}, {year}", "%H:%M, %d %b, %Y")
    if delay:
        if delay == "(RT)":
            return _time, _time
        else:
            return _time, parse_time_delta(delay, _time)
    return None, None


def parse_time_delta(delay: str, src=datetime.datetime.now()):
    cal = parsedatetime.Calendar()

    actual, _ = cal.parseDT(delay, sourceTime=src)
    return actual


class ETrainParser:
    @staticmethod
    def _parse_larrdep_data(
        json_resp, config: ETrainArrivalDepartureConfig
    ) -> list[ETrainStnLiveArrDepTrain]:
        soup = bs4.BeautifulSoup(json_resp["data"], "html.parser")
        parsed_trains = []
        for table_row in soup.find_all("tr"):
            table_row: PageElement
            train_info = table_row.find_all_next("td")
            train_info = [x.get_text().strip() for x in train_info]
            print("DEBUG:", train_info)
            parsed_train_info: ETrainStnLiveArrDepTrain = {
                "train_no": train_info[0],
                "train_name": train_info[1],
                "src": train_info[2],
                "dest": train_info[3],
                "tt_arr": train_info[4],
                "tt_dept": train_info[5],
                "tt_pf": train_info[6],
                "tt_halt": train_info[7],
                "exp_arr": train_info[8],
                "exp_arr_delay": train_info[9],
                "exp_dept": train_info[10],
                "exp_dept_delay": train_info[11],
            }
            if config.exclude_local and "LOCAL" in parsed_train_info["train_name"]:
                continue
            if config.exclude_memu and "MEMU" in parsed_train_info["train_name"]:
                continue
            if (
                config.exclude_fast_emu
                and "FAST EMU" in parsed_train_info["train_name"]
            ):
                continue
            if (
                config.exclude_parcel_services
                and "JPP" in parsed_train_info["train_name"]
            ):
                continue
            parsed_trains.append(parsed_train_info)
        return parsed_trains[: config.limit]

    @staticmethod
    def _parse_train_schedule_info(json_resp) -> list[ETrainScheduleStation]:
        train_soup = bs4.BeautifulSoup(json_resp["data"]["ldata"], "html.parser")
        sublowerdata = train_soup.find(id="sublowerdata")
        # find("table").find_all("tr")
        table = sublowerdata.find("table") if sublowerdata else None
        table_rows = table.find_all("tr") if table else []

        stations = []
        for table_row in table_rows[1:]:
            table_row: PageElement
            sta_info = [
                x.strip()
                for x in table_row.get_text(" ; ").split(" ; ")
                if not str(x).isspace()
            ]
            dist = sta_info[3].split(" ")[0].strip()
            pf = sta_info[4].split(":")[1].strip()
            arr, a_day = parse_schedule(sta_info[7])
            dept, d_day = parse_schedule(sta_info[8])

            station: ETrainScheduleStation = {
                "index": sta_info[0],
                "code": sta_info[1],
                "name": sta_info[2],
                "dist": dist,
                "pf": pf,
                "a": arr,
                "d": dept,
                "a_day": a_day,
                "d_day": d_day,
            }
            stations.append(station)
        return stations

    @staticmethod
    def _parse_coach_position(json_resp) -> dict[str, str]:
        train_soup = bs4.BeautifulSoup(json_resp["data"]["ldata"], "html.parser")
        table_rows = train_soup.find_all(attrs={"class": "rake"})
        positions = {}
        for table_row in table_rows:
            table_row: PageElement
            coach_pos = table_row.find_parent()
            coach_pos = coach_pos.get_text(" ; ").split(" ; ") if coach_pos else []
            if len(coach_pos) < 2:
                print(f"[WARNING] Invalid coach position data: {coach_pos}")
                positions[str(len(positions))] = coach_pos[
                    0
                ]  # Assign a position based on order
                continue
            if not coach_pos:
                print("[WARNING] No coach position data found for the train.")
                continue
            positions[coach_pos[0]] = coach_pos[1]
        return positions

    @staticmethod
    def _parse_running_status_data(json_resp) -> list[ETrainRunningStatusStation]:
        train_soup = bs4.BeautifulSoup(json_resp["data"], "html.parser")
        sublowerdata = train_soup.find(id="sublowerdata")
        train_table = sublowerdata.find("table") if sublowerdata else None
        table = train_table.find_next_sibling("table") if train_table else None
        table_rows = (
            table.find_all("tr", attrs={"class": ["odd", "even"]}) if table else []
        )

        # table_rows = (
        #     train_soup.find(id="sublowerdata")
        #     .find("table")
        #     .find_next_sibling("table")
        #     .find_all("tr", attrs={"class": ["odd", "even"]})
        # )
        if not table_rows:
            print("[WARNING] No running status data found for the train.")
            return []  # No running status data available

        stns = []
        for table_row in table_rows:
            table_row: PageElement
            running_stn_info = [
                x.strip()
                for x in table_row.get_text(" ; ").split(" ; ")
                if not (x.isspace() or x.startswith("+ "))
            ]
            print("DEBUG: Running status station info:", running_stn_info)
            if running_stn_info[2].startswith("*Diverted"):
                print(
                    f"[WARNING] Train diverted at station {running_stn_info[1]}. Marking arr/dep as N/A."
                )
                pf = "N/A"
                scheduled_arr, actual_arr, scheduled_dept, actual_dept = (
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                )
            elif running_stn_info[2].startswith("*Cancelled"):
                print(
                    f"[WARNING] Train cancelled at station {running_stn_info[1]}. Marking arr/dep as N/A."
                )
                pf = "N/A"
                scheduled_arr, actual_arr, scheduled_dept, actual_dept = (
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                )
            else:
                pf = running_stn_info[3].split(":")[1].strip()
                scheduled_arr, actual_arr, scheduled_dept, actual_dept = (
                    parse_running_status_arr_dep(running_stn_info[6:])
                )
            stn: ETrainRunningStatusStation = {
                "index": running_stn_info[0],
                "name": running_stn_info[1],
                "s_dist": running_stn_info[2],
                "pf": pf,
                "tt_arr": scheduled_arr,
                "act_arr": actual_arr,
                "tt_dept": scheduled_dept,
                "act_dept": actual_dept,
            }
            stns.append(stn)
        return stns

    @staticmethod
    def _parse_all_trains_data(
        json_resp, config: ETrainAllTrainsConfig
    ) -> list[ETrainAllTrainsStation]:
        soup = bs4.BeautifulSoup(json_resp["data"]["udata"], "html.parser")
        parsed_trains = []
        trainlist = soup.find(attrs={"class": "trainlist"})
        trows = trainlist.find_all("tr") if trainlist else []
        if not trows:
            print("[WARNING] No train data found in the response.")
            return []  # No train data available

        for table_row in trows:
            table_row: PageElement

            if len(parsed_trains) >= config.limit:
                break

            train_info = table_row.get_text(" ; ").split(" ; ")
            parsed_train_info: ETrainAllTrainsStation = {
                "train_no": train_info[0],
                "train_name": train_info[1],
                "src": train_info[2],
                "dest": train_info[3],
                "tt_arr": train_info[4],
                "tt_dept": train_info[5],
                "tt_halt": train_info[6],
                "running_days": train_info[7:14],
                "classes": train_info[14:],
            }
            if config.weekday != -1:
                if parsed_train_info["running_days"][config.weekday] == "X":
                    continue

            parsed_trains.append(parsed_train_info)
        return parsed_trains
