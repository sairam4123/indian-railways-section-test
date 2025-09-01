from train_lib.models import Station, Track, BlockSection
import simpy

def create_tpj_kkdi_network(env: simpy.Environment):
    
    TPJ_main = Track(env, "TPJ_main", has_platform=True, length=600)
    TPJ_loop1 = Track(env, "TPJ_loop1", has_platform=True, length=400)
    TPJ_loop2 = Track(env, "TPJ_loop2", has_platform=True, length=400)
    TPJ = Station(env, 'tpj', [TPJ_main, TPJ_loop1, TPJ_loop2])


    KRMG_main = Track(env, "KRMG_main", has_platform=False, length=600)
    KRMG_loop1 = Track(env, "KRMG_loop1", has_platform=True, length=400)
    KRMG_loop2 = Track(env, "KRMG_loop2", has_platform=True, length=400)
    KRMG = Station(env, 'krmg', [KRMG_main, KRMG_loop1, KRMG_loop2])

    KRUR_main = Track(env, "KRUR_main", has_platform=False, length=600)
    KRUR_loop1 = Track(env, "KRUR_loop1", has_platform=True, length=400)
    KRUR_loop2 = Track(env, "KRUR_loop2", has_platform=True, length=400)
    KRUR = Station(env, 'krur', [KRUR_main, KRUR_loop1, KRUR_loop2])

    VEL_main = Track(env, "VEL_main", has_platform=False, length=600)
    VEL_loop1 = Track(env, "VEL_loop1", has_platform=True, length=400)
    VEL_loop2 = Track(env, "VEL_loop2", has_platform=True, length=400)
    VEL = Station(env, 'vel', [VEL_main, VEL_loop1, VEL_loop2])

    PDKT_main = Track(env, "PDKT_main", has_platform=True, length=600)
    PDKT_loop1 = Track(env, "PDKT_loop1", has_platform=True, length=400)
    PDKT_loop2 = Track(env, "PDKT_loop2", has_platform=True, length=400)
    PDKT_loop3 = Track(env, "PDKT_loop3", has_platform=True, length=500)
    PDKT = Station(env, 'pdkt', [PDKT_main, PDKT_loop1, PDKT_loop2, PDKT_loop3])

    TYM_main = Track(env, "TYM_main", has_platform=False, length=600)
    TYM_loop1 = Track(env, "TYM_loop1", has_platform=True, length=400)
    TYM_loop2 = Track(env, "TYM_loop2", has_platform=True, length=400)
    TYM = Station(env, 'tym', [TYM_main, TYM_loop1, TYM_loop2])


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


    # TPJ_KRUR = BlockSection(env, "TPJ_KRUR", TPJ, KRUR, length_km=27, line_speed=100, bidirectional=False, electric=True, signal_num=3, signal_aspects=4)
    # KRUR_PDKT = BlockSection(env, "KRUR_PDKT", KRUR, PDKT, length_km=32, line_speed=110, bidirectional=False, electric=True, signal_num=3, signal_aspects=4)
    # PDKT_CTND = BlockSection(env, "PDKT_CTND", PDKT, CTND, length_km=26, line_speed=110, bidirectional=False, electric=True, signal_num=3, signal_aspects=4)
    # CTND_KKDI = BlockSection(env, "CTND_KKDI", CTND, KKDI, length_km=13, line_speed=110, bidirectional=False, electric=True, signal_num=3, signal_aspects=4)


    # KRUR_TPJ = BlockSection(env, "KRUR_TPJ", KRUR, TPJ, length_km=27, line_speed=100, bidirectional=False, electric=True, signal_num=3, signal_aspects=4)
    # PDKT_KRUR = BlockSection(env, "PDKT_KRUR", PDKT, KRUR, length_km=32, line_speed=110, bidirectional=False, electric=True, signal_num=3, signal_aspects=4)
    # CTND_PDKT = BlockSection(env, "CTND_PDKT", CTND, PDKT, length_km=26, line_speed=110, bidirectional=False, electric=True, signal_num=3, signal_aspects=4)
    # KKDI_CTND = BlockSection(env, "KKDI_CTND", KKDI, CTND, length_km=13, line_speed=110, bidirectional=False, electric=True, signal_num=3, signal_aspects=4)

    # KRUR_TPJ_4 = BlockSection(env, "KRUR_TPJ", KRUR, TPJ, length_km=27, line_speed=100, bidirectional=False, electric=True, signal_num=4, signal_aspects=4)
    # PDKT_KRUR_4 = BlockSection(env, "PDKT_KRUR", PDKT, KRUR, length_km=32, line_speed=110, bidirectional=False, electric=True, signal_num=4, signal_aspects=4)
    # CTND_PDKT_4 = BlockSection(env, "CTND_PDKT", CTND, PDKT, length_km=26, line_speed=110, bidirectional=False, electric=True, signal_num=4, signal_aspects=4)
    # KKDI_CTND_4 = BlockSection(env, "KKDI_CTND", KKDI, CTND, length_km=13, line_speed=110, bidirectional=False, electric=True, signal_num=4, signal_aspects=4)


    TPJ_KRMG_3 = BlockSection(env, "TPJ_KRMG", TPJ, KRMG, length_km=13, line_speed=110, bidirectional=True, electric=True, signal_num=2, signal_aspects=4)
    KRMG_KRUR_3 = BlockSection(env, "KRMG_KRUR", KRMG, KRUR, length_km=15, line_speed=110, bidirectional=True, electric=True, signal_num=2, signal_aspects=4)
    KRUR_VEL_3 = BlockSection(env, "KRUR_VEL", KRUR, VEL, length_km=13, line_speed=110, bidirectional=True, electric=True, signal_num=2, signal_aspects=4)
    VEL_PDKT_3 = BlockSection(env, "VEL_PDKT", VEL, PDKT, length_km=11, line_speed=110, bidirectional=True, electric=True, signal_num=2, signal_aspects=4)
    PDKT_TYM_3 = BlockSection(env, "PDKT_TYM", PDKT, TYM, length_km=16, line_speed=110, bidirectional=True, electric=True, signal_num=2, signal_aspects=4)
    TYM_CTND_3 = BlockSection(env, "TYM_CTND", TYM, CTND, length_km=10, line_speed=110, bidirectional=True, electric=True, signal_num=2, signal_aspects=4)
    CTND_KKDI_3 = BlockSection(env, "CTND_KKDI", CTND, KKDI, length_km=11, line_speed=100, bidirectional=True, electric=True, signal_num=2, signal_aspects=4)

    return (
        # Stations
        [TPJ, KRMG, KRUR, VEL, PDKT, TYM, CTND, KKDI],

        # Block Sections
        [
            # [TPJ_KRUR, KRUR_PDKT, PDKT_CTND, CTND_KKDI],
            # [KRUR_TPJ, PDKT_KRUR, CTND_PDKT, KKDI_CTND],
            # [KRUR_TPJ_4, PDKT_KRUR_4, CTND_PDKT_4, KKDI_CTND_4],
            [TPJ_KRMG_3, KRMG_KRUR_3, KRUR_VEL_3, VEL_PDKT_3, PDKT_TYM_3, TYM_CTND_3, CTND_KKDI_3],
        ],


        # Loop lines
        [KKDI_main, KKDI_loop1, KKDI_loop2, KKDI_loop3, KKDI_loop4],
        [CTND_main, CTND_loop1, CTND_loop2, CTND_loop3],
        [TYM_main, TYM_loop1, TYM_loop2],
        [PDKT_main, PDKT_loop1, PDKT_loop2, PDKT_loop3],
        [VEL_main, VEL_loop1, VEL_loop2],
        [KRUR_main, KRUR_loop1, KRUR_loop2],
        [KRMG_main, KRMG_loop1, KRMG_loop2],
        [TPJ_main, TPJ_loop1, TPJ_loop2],
    )

def create_vm_tpj_network(env: simpy.Environment):
    pass