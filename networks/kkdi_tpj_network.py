from train_lib.models import Station, Track, BlockSection
from train_lib.simulation import Simulation

from train_lib.models import Network


def create_tpj_kkdi_network(sim: Simulation):

    TPJ_main = Track(sim, "TPJ_main", has_platform=True, is_mainline=True, length=600)
    TPJ_loop1 = Track(
        sim, "TPJ_loop1", has_platform=True, is_mainline=False, length=400
    )
    TPJ_loop2 = Track(
        sim, "TPJ_loop2", has_platform=True, is_mainline=False, length=400
    )
    TPJ = Station(sim, "tpj", [TPJ_main, TPJ_loop1, TPJ_loop2])

    KRMG_main = Track(
        sim, "KRMG_main", has_platform=False, is_mainline=True, length=600
    )
    KRMG_loop1 = Track(
        sim, "KRMG_loop1", has_platform=True, is_mainline=False, length=400
    )
    KRMG_loop2 = Track(
        sim, "KRMG_loop2", has_platform=True, is_mainline=False, length=400
    )
    KRMG = Station(sim, "krmg", [KRMG_main, KRMG_loop1, KRMG_loop2])

    KRUR_main = Track(
        sim, "KRUR_main", has_platform=False, is_mainline=True, length=600
    )
    KRUR_loop1 = Track(
        sim, "KRUR_loop1", has_platform=True, is_mainline=False, length=400
    )
    KRUR_loop2 = Track(
        sim, "KRUR_loop2", has_platform=True, is_mainline=False, length=400
    )
    KRUR = Station(sim, "krur", [KRUR_main, KRUR_loop1, KRUR_loop2])

    VEL_main = Track(sim, "VEL_main", has_platform=False, is_mainline=True, length=600)
    VEL_loop1 = Track(
        sim, "VEL_loop1", has_platform=True, is_mainline=False, length=400
    )
    VEL_loop2 = Track(
        sim, "VEL_loop2", has_platform=True, is_mainline=False, length=400
    )
    VEL = Station(sim, "vel", [VEL_main, VEL_loop1, VEL_loop2])

    PDKT_main = Track(sim, "PDKT_main", has_platform=True, is_mainline=True, length=600)
    PDKT_loop1 = Track(
        sim, "PDKT_loop1", has_platform=True, is_mainline=False, length=400
    )
    PDKT_loop2 = Track(
        sim, "PDKT_loop2", has_platform=True, is_mainline=False, length=400
    )
    PDKT_loop3 = Track(
        sim, "PDKT_loop3", has_platform=True, is_mainline=False, length=500
    )
    PDKT = Station(sim, "pdkt", [PDKT_main, PDKT_loop1, PDKT_loop2, PDKT_loop3])

    TYM_main = Track(sim, "TYM_main", has_platform=False, is_mainline=True, length=600)
    TYM_loop1 = Track(
        sim, "TYM_loop1", has_platform=True, is_mainline=False, length=400
    )
    TYM_loop2 = Track(
        sim, "TYM_loop2", has_platform=True, is_mainline=False, length=400
    )
    TYM = Station(sim, "tym", [TYM_main, TYM_loop1, TYM_loop2])

    CTND_main = Track(
        sim, "CTND_main", has_platform=False, is_mainline=True, length=600
    )
    CTND_loop1 = Track(
        sim, "CTND_loop1", has_platform=True, is_mainline=False, length=400
    )
    CTND_loop2 = Track(
        sim, "CTND_loop2", has_platform=True, is_mainline=False, length=400
    )
    CTND_loop3 = Track(
        sim, "CTND_loop3", has_platform=True, is_mainline=False, length=400
    )
    CTND = Station(sim, "ctnd", [CTND_main, CTND_loop1, CTND_loop2, CTND_loop3])

    KKDI_main = Track(sim, "KKDI_main", has_platform=True, is_mainline=True, length=600)
    KKDI_loop1 = Track(
        sim, "KKDI_loop1", has_platform=True, is_mainline=False, length=400
    )
    KKDI_loop2 = Track(
        sim, "KKDI_loop2", has_platform=True, is_mainline=False, length=400
    )
    KKDI_loop3 = Track(
        sim, "KKDI_loop3", has_platform=True, is_mainline=False, length=400
    )
    KKDI_loop4 = Track(
        sim, "KKDI_loop4", has_platform=True, is_mainline=False, length=400
    )
    KKDI = Station(
        sim, "kkdi", [KKDI_main, KKDI_loop1, KKDI_loop2, KKDI_loop3, KKDI_loop4]
    )

    TPJ_KRMG_BD = BlockSection(
        sim,
        "TPJ_KRMG",
        TPJ,
        KRMG,
        length_km=13,
        line_speed=110,
        bidirectional=True,
        electric=True,
        signal_num=3,
        signal_aspects=4,
    )
    KRMG_KRUR_BD = BlockSection(
        sim,
        "KRMG_KRUR",
        KRMG,
        KRUR,
        length_km=16,
        line_speed=110,
        bidirectional=True,
        electric=True,
        signal_num=3,
        signal_aspects=4,
    )
    KRUR_VEL_BD = BlockSection(
        sim,
        "KRUR_VEL",
        KRUR,
        VEL,
        length_km=12,
        line_speed=110,
        bidirectional=True,
        electric=True,
        signal_num=3,
        signal_aspects=4,
    )
    VEL_PDKT_BD = BlockSection(
        sim,
        "VEL_PDKT",
        VEL,
        PDKT,
        length_km=12,
        line_speed=110,
        bidirectional=True,
        electric=True,
        signal_num=3,
        signal_aspects=4,
    )
    PDKT_TYM_BD = BlockSection(
        sim,
        "PDKT_TYM",
        PDKT,
        TYM,
        length_km=16,
        line_speed=110,
        bidirectional=True,
        electric=True,
        signal_num=3,
        signal_aspects=4,
    )
    TYM_CTND_BD = BlockSection(
        sim,
        "TYM_CTND",
        TYM,
        CTND,
        length_km=9,
        line_speed=110,
        bidirectional=True,
        electric=True,
        signal_num=3,
        signal_aspects=4,
    )
    CTND_KKDI_BD = BlockSection(
        sim,
        "CTND_KKDI",
        CTND,
        KKDI,
        length_km=11,
        line_speed=110,
        bidirectional=True,
        electric=True,
        signal_num=3,
        signal_aspects=4,
    )

    # TPJ_KRMG_3 = BlockSection(env, "TPJ_KRMG", TPJ, KRMG, length_km=13, line_speed=110, bidirectional=False, electric=True, signal_num=3, signal_aspects=4)
    # KRMG_KRUR_3 = BlockSection(env, "KRMG_KRUR", KRMG, KRUR, length_km=16, line_speed=110, bidirectional=False, electric=True, signal_num=3, signal_aspects=4)
    # KRUR_VEL_3 = BlockSection(env, "KRUR_VEL", KRUR, VEL, length_km=12, line_speed=110, bidirectional=False, electric=True, signal_num=3, signal_aspects=4)
    # VEL_PDKT_3 = BlockSection(env, "VEL_PDKT", VEL, PDKT, length_km=12, line_speed=110, bidirectional=False, electric=True, signal_num=3, signal_aspects=4)
    # PDKT_TYM_3 = BlockSection(env, "PDKT_TYM", PDKT, TYM, length_km=16, line_speed=110, bidirectional=False, electric=True, signal_num=3, signal_aspects=4)
    # TYM_CTND_3 = BlockSection(env, "TYM_CTND", TYM, CTND, length_km=9, line_speed=110, bidirectional=False, electric=True, signal_num=3, signal_aspects=4)
    # CTND_KKDI_3 = BlockSection(env, "CTND_KKDI", CTND, KKDI, length_km=11, line_speed=110, bidirectional=False, electric=True, signal_num=3, signal_aspects=4)

    # KRMG_TPJ_4 = BlockSection(env, "KRMG_TPJ", KRMG, TPJ, length_km=13, line_speed=110, bidirectional=False, electric=True, signal_num=3, signal_aspects=4)
    # KRUR_KRMG_4 = BlockSection(env, "KRUR_KRMG", KRUR, KRMG, length_km=16, line_speed=110, bidirectional=False, electric=True, signal_num=3, signal_aspects=4)
    # VEL_KRUR_4 = BlockSection(env, "VEL_KRUR", VEL, KRUR, length_km=12, line_speed=110, bidirectional=False, electric=True, signal_num=3, signal_aspects=4)
    # PDKT_VEL_4 = BlockSection(env, "PDKT_VEL", PDKT, VEL, length_km=12, line_speed=110, bidirectional=False, electric=True, signal_num=3, signal_aspects=4)
    # TYM_PDKT_4 = BlockSection(env, "TYM_PDKT", TYM, PDKT, length_km=16, line_speed=110, bidirectional=False, electric=True, signal_num=3, signal_aspects=4)
    # CTND_TYM_4 = BlockSection(env, "CTND_TYM", CTND, TYM, length_km=9, line_speed=110, bidirectional=False, electric=True, signal_num=3, signal_aspects=4)
    # KKDI_CTND_4 = BlockSection(env, "KKDI_CTND", KKDI, CTND, length_km=11, line_speed=110, bidirectional=False, electric=True, signal_num=3, signal_aspects=4)

    # return (
    #     # Stations
    #     [TPJ, KRMG, KRUR, VEL, PDKT, TYM, CTND, KKDI],

    #     # Block Sections
    #     [
    #         # [TPJ_KRUR, KRUR_PDKT, PDKT_CTND, CTND_KKDI],
    #         # [KRUR_TPJ, PDKT_KRUR, CTND_PDKT, KKDI_CTND],
    #         # [KRUR_TPJ_4, PDKT_KRUR_4, CTND_PDKT_4, KKDI_CTND_4],
    #         # [TPJ_KRMG_3, KRMG_KRUR_3, KRUR_VEL_3, VEL_PDKT_3, PDKT_TYM_3, TYM_CTND_3, CTND_KKDI_3],
    #         # [KRMG_TPJ_4, KRUR_KRMG_4, VEL_KRUR_4, PDKT_VEL_4, TYM_PDKT_4, CTND_TYM_4, KKDI_CTND_4],
    #         [TPJ_KRMG_BD, KRMG_KRUR_BD, KRUR_VEL_BD, VEL_PDKT_BD, PDKT_TYM_BD, TYM_CTND_BD, CTND_KKDI_BD],
    #     ],

    #     # Loop lines
    #     [KKDI_main, KKDI_loop1, KKDI_loop2, KKDI_loop3, KKDI_loop4],
    #     [CTND_main, CTND_loop1, CTND_loop2, CTND_loop3],
    #     [TYM_main, TYM_loop1, TYM_loop2],
    #     [PDKT_main, PDKT_loop1, PDKT_loop2, PDKT_loop3],
    #     [VEL_main, VEL_loop1, VEL_loop2],
    #     [KRUR_main, KRUR_loop1, KRUR_loop2],
    #     [KRMG_main, KRMG_loop1, KRMG_loop2],
    #     [TPJ_main, TPJ_loop1, TPJ_loop2],
    # )

    return Network(
        sim,
        stations=[TPJ, KRMG, KRUR, VEL, PDKT, TYM, CTND, KKDI],
        block_sections=[
            [
                TPJ_KRMG_BD,
                KRMG_KRUR_BD,
                KRUR_VEL_BD,
                VEL_PDKT_BD,
                PDKT_TYM_BD,
                TYM_CTND_BD,
                CTND_KKDI_BD,
            ],
        ],
        loop_lines={
            KKDI: [KKDI_main, KKDI_loop1, KKDI_loop2, KKDI_loop3, KKDI_loop4],
            CTND: [CTND_main, CTND_loop1, CTND_loop2, CTND_loop3],
            TYM: [TYM_main, TYM_loop1, TYM_loop2],
            PDKT: [PDKT_main, PDKT_loop1, PDKT_loop2, PDKT_loop3],
            VEL: [VEL_main, VEL_loop1, VEL_loop2],
            KRUR: [KRUR_main, KRUR_loop1, KRUR_loop2],
            KRMG: [KRMG_main, KRMG_loop1, KRMG_loop2],
            TPJ: [TPJ_main, TPJ_loop1, TPJ_loop2],
        },
    )


def create_alu_tpj_network(sim: Simulation):
    TPJ_main = Track(sim, "TPJ_main", has_platform=True, is_mainline=True, length=600)
    TPJ_loop1 = Track(
        sim, "TPJ_loop1", has_platform=True, is_mainline=False, length=400
    )
    TPJ_loop2 = Track(
        sim, "TPJ_loop2", has_platform=True, is_mainline=False, length=400
    )
    TPJ = Station(sim, "tpj", [TPJ_main, TPJ_loop1, TPJ_loop2], is_boundary=True)

    GOC_main = Track(sim, "GOC_main", has_platform=True, is_mainline=True, length=600)
    GOC_loop1 = Track(
        sim, "GOC_loop1", has_platform=True, is_mainline=False, length=400
    )
    GOC_loop2 = Track(
        sim, "GOC_loop2", has_platform=True, is_mainline=False, length=400
    )
    GOC_loop3 = Track(
        sim, "GOC_loop3", has_platform=True, is_mainline=False, length=400
    )
    GOC_loop4 = Track(
        sim, "GOC_loop4", has_platform=True, is_mainline=False, length=400
    )
    GOC = Station(sim, "goc", [GOC_main, GOC_loop1, GOC_loop2], is_boundary=True)

    SRGM_main = Track(
        sim, "SRGM_main", has_platform=False, is_mainline=True, length=600
    )
    SRGM_loop1 = Track(
        sim, "SRGM_loop1", has_platform=True, is_mainline=False, length=400
    )
    SRGM_loop2 = Track(
        sim, "SRGM_loop2", has_platform=True, is_mainline=False, length=400
    )
    SRGM_loop3 = Track(
        sim, "SRGM_loop3", has_platform=True, is_mainline=False, length=400
    )
    SRGM = Station(sim, "srgm", [SRGM_main, SRGM_loop1, SRGM_loop2, SRGM_loop3])

    LLI_main = Track(sim, "LLI_main", has_platform=True, is_mainline=True, length=600)
    LLI_loop1 = Track(
        sim, "LLI_loop1", has_platform=True, is_mainline=False, length=400
    )
    LLI_loop2 = Track(
        sim, "LLI_loop2", has_platform=True, is_mainline=False, length=400
    )
    LLI_loop3 = Track(
        sim, "LLI_loop3", has_platform=True, is_mainline=False, length=400
    )
    LLI = Station(sim, "lli", [LLI_main, LLI_loop1, LLI_loop2, LLI_loop3])

    PMB_main = Track(sim, "PMB_main", has_platform=True, is_mainline=True, length=600)
    PMB_loop1 = Track(
        sim, "PMB_loop1", has_platform=True, is_mainline=False, length=400
    )
    PMB_loop2 = Track(
        sim, "PMB_loop2", has_platform=True, is_mainline=False, length=400
    )
    PMB_loop3 = Track(
        sim, "PMB_loop3", has_platform=True, is_mainline=False, length=400
    )
    PMB = Station(sim, "pmb", [PMB_main, PMB_loop1, PMB_loop2, PMB_loop3])

    KKPM_main = Track(sim, "KKPM_main", has_platform=True, is_mainline=True, length=600)
    KKPM_loop1 = Track(
        sim, "KKPM_loop1", has_platform=True, is_mainline=False, length=400
    )
    KKPM_loop2 = Track(
        sim, "KKPM_loop2", has_platform=True, is_mainline=False, length=400
    )
    KKPM_loop3 = Track(
        sim, "KKPM_loop3", has_platform=True, is_mainline=False, length=400
    )
    KKPM = Station(sim, "kkpm", [KKPM_main, KKPM_loop1, KKPM_loop2, KKPM_loop3])

    SLTH_main = Track(
        sim, "SLTH_main", has_platform=False, is_mainline=True, length=600
    )
    SLTH_loop1 = Track(
        sim, "SLTH_loop1", has_platform=True, is_mainline=False, length=400
    )
    SLTH_loop2 = Track(
        sim, "SLTH_loop2", has_platform=True, is_mainline=False, length=400
    )
    SLTH_loop3 = Track(
        sim, "SLTH_loop3", has_platform=True, is_mainline=False, length=400
    )
    SLTH = Station(sim, "slth", [SLTH_main, SLTH_loop1, SLTH_loop2, SLTH_loop3])

    ALU_main = Track(sim, "ALU_main", has_platform=False, is_mainline=True, length=600)
    ALU_loop1 = Track(
        sim, "ALU_loop1", has_platform=True, is_mainline=False, length=400
    )
    ALU_loop2 = Track(
        sim, "ALU_loop2", has_platform=True, is_mainline=False, length=400
    )
    ALU_loop3 = Track(
        sim, "ALU_loop3", has_platform=True, is_mainline=False, length=400
    )
    ALU_loop4 = Track(
        sim, "ALU_loop4", has_platform=True, is_mainline=False, length=400
    )
    ALU = Station(sim, "alu", [ALU_main, ALU_loop1, ALU_loop2, ALU_loop3, ALU_loop4], is_boundary=True)

    TPJ_GOC_1 = BlockSection(
        sim,
        "TPJ_GOC",
        TPJ,
        GOC,
        length_km=4,
        line_speed=50,
        bidirectional=False,
        electric=True,
        signal_num=3,
        signal_aspects=4,
    )
    GOC_SRGM_1 = BlockSection(
        sim,
        "GOC_SRGM",
        GOC,
        SRGM,
        length_km=7,
        line_speed=110,
        bidirectional=False,
        electric=True,
        signal_num=3,
        signal_aspects=4,
    )
    SRGM_LLI_1 = BlockSection(
        sim,
        "SRGM_LLI",
        SRGM,
        LLI,
        length_km=15,
        line_speed=110,
        bidirectional=False,
        electric=True,
        signal_num=3,
        signal_aspects=4,
    )
    LLI_PMB_1 = BlockSection(
        sim,
        "LLI_PMB",
        LLI,
        PMB,
        length_km=14,
        line_speed=110,
        bidirectional=False,
        electric=True,
        signal_num=3,
        signal_aspects=4,
    )
    PMB_KKPM_1 = BlockSection(
        sim,
        "PMB_KKPM",
        PMB,
        KKPM,
        length_km=13,
        line_speed=110,
        bidirectional=False,
        electric=True,
        signal_num=3,
        signal_aspects=4,
    )
    KKPM_SLTH_1 = BlockSection(
        sim,
        "KKPM_SLTH",
        KKPM,
        SLTH,
        length_km=17,
        line_speed=110,
        bidirectional=False,
        electric=True,
        signal_num=3,
        signal_aspects=4,
    )
    SLTH_ALU_1 = BlockSection(
        sim,
        "SLTH_ALU",
        SLTH,
        ALU,
        length_km=7,
        line_speed=110,
        bidirectional=False,
        electric=True,
        signal_num=3,
        signal_aspects=4,
    )

    ALU_SLTH_2 = BlockSection(
        sim,
        "ALU_SLTH",
        ALU,
        SLTH,
        length_km=7,
        line_speed=110,
        bidirectional=False,
        electric=True,
        signal_num=3,
        signal_aspects=4,
    )
    SLTH_KKPM_2 = BlockSection(
        sim,
        "SLTH_KKPM",
        SLTH,
        KKPM,
        length_km=17,
        line_speed=110,
        bidirectional=False,
        electric=True,
        signal_num=3,
        signal_aspects=4,
    )
    KKPM_PMB_2 = BlockSection(
        sim,
        "KKPM_PMB",
        KKPM,
        PMB,
        length_km=13,
        line_speed=110,
        bidirectional=False,
        electric=True,
        signal_num=3,
        signal_aspects=4,
    )
    PMB_LLI_2 = BlockSection(
        sim,
        "PMB_LLI",
        PMB,
        LLI,
        length_km=14,
        line_speed=110,
        bidirectional=False,
        electric=True,
        signal_num=3,
        signal_aspects=4,
    )
    LLI_SRGM_2 = BlockSection(
        sim,
        "LLI_SRGM",
        LLI,
        SRGM,
        length_km=15,
        line_speed=110,
        bidirectional=False,
        electric=True,
        signal_num=3,
        signal_aspects=4,
    )
    SRGM_GOC_2 = BlockSection(
        sim,
        "SRGM_GOC",
        SRGM,
        GOC,
        length_km=7,
        line_speed=110,
        bidirectional=False,
        electric=True,
        signal_num=3,
        signal_aspects=4,
    )
    GOC_TPJ_2 = BlockSection(
        sim,
        "GOC_TPJ",
        GOC,
        TPJ,
        length_km=4,
        line_speed=50,
        bidirectional=False,
        electric=True,
        signal_num=3,
        signal_aspects=4,
    )

    return Network(
        sim,
        stations=[TPJ, GOC, SRGM, LLI, PMB, KKPM, SLTH, ALU],
        block_sections=[
            [
                TPJ_GOC_1,
                GOC_SRGM_1,
                SRGM_LLI_1,
                LLI_PMB_1,
                PMB_KKPM_1,
                KKPM_SLTH_1,
                SLTH_ALU_1,
            ],
            [
                ALU_SLTH_2,
                SLTH_KKPM_2,
                KKPM_PMB_2,
                PMB_LLI_2,
                LLI_SRGM_2,
                SRGM_GOC_2,
                GOC_TPJ_2,
            ],
        ],
        loop_lines={
            TPJ: [TPJ_main, TPJ_loop1, TPJ_loop2],
            GOC: [GOC_main, GOC_loop1, GOC_loop2, GOC_loop3, GOC_loop4],
            SRGM: [SRGM_main, SRGM_loop1, SRGM_loop2, SRGM_loop3],
            LLI: [LLI_main, LLI_loop1, LLI_loop2, LLI_loop3],
            PMB: [PMB_main, PMB_loop1, PMB_loop2, PMB_loop3],
            KKPM: [KKPM_main, KKPM_loop1, KKPM_loop2, KKPM_loop3],
            SLTH: [SLTH_main, SLTH_loop1, SLTH_loop2, SLTH_loop3],
            ALU: [ALU_main, ALU_loop1, ALU_loop2, ALU_loop3, ALU_loop4],
        },
    )
