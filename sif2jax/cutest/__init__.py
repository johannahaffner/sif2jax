from ._bounded_minimisation import (
    bounded_minimisation_problems as bounded_minimisation_problems,
    EXP2B as EXP2B,
    HS1 as HS1,
    HS2 as HS2,
    HS3 as HS3,
    HS4 as HS4,
    HS5 as HS5,
    HS25 as HS25,
    HS38 as HS38,
    HS45 as HS45,
    HS110 as HS110,
)
from ._constrained_minimisation import (
    # ACOPP14 as ACOPP14,  # TODO: needs human review - complex AC OPF formulation
    # AIRPORT as AIRPORT,  # TODO: Human review - constraint values don't match pycutest
    # ALLINITA as ALLINITA,  # TODO: needs human review - L2 group type interpretation
    ALSOTAME as ALSOTAME,
    # ANTWERP as ANTWERP,  # TODO: needs human review - initial value calculation
    AVGASA as AVGASA,
    AVGASB as AVGASB,
    # AVION2 as AVION2,  # TODO: Human review - gradient discrepancies
    BIGGSC4 as BIGGSC4,
    # BOOTH as BOOTH,  # TODO: needs human review - incorrectly classified
    BOXBOD as BOXBOD,
    BT1 as BT1,
    BT2 as BT2,
    BT3 as BT3,
    BT4 as BT4,
    BT5 as BT5,
    BT6 as BT6,
    BT7 as BT7,
    BT8 as BT8,
    BT9 as BT9,
    BT10 as BT10,
    BT11 as BT11,
    BT12 as BT12,
    BT13 as BT13,
    BURKEHAN as BURKEHAN,
    BYRDSPHR as BYRDSPHR,
    CANTILVR as CANTILVR,
    CB2 as CB2,
    CB3 as CB3,
    CHACONN1 as CHACONN1,
    CHACONN2 as CHACONN2,
    CHANDHEQ as CHANDHEQ,
    CLUSTER as CLUSTER,
    CONCON as CONCON,
    constrained_minimisation_problems as constrained_minimisation_problems,
    COOLHANS as COOLHANS,
    # CRESC4 as CRESC4,  # TODO: Human review - complex crescent area formula
    CSFI1 as CSFI1,
    CSFI2 as CSFI2,
    CVXQP1 as CVXQP1,
    # DALLASS as DALLASS,  # TODO: needs complex element implementations
    DANIWOOD as DANIWOOD,
    DECONVC as DECONVC,
    HS6 as HS6,
    HS7 as HS7,
    HS8 as HS8,
    HS9 as HS9,
    HS10 as HS10,
    HS11 as HS11,
    HS12 as HS12,
    HS13 as HS13,
    HS14 as HS14,
    HS15 as HS15,
    HS16 as HS16,
    HS17 as HS17,
    HS18 as HS18,
    HS19 as HS19,
    HS20 as HS20,
    HS21 as HS21,
    HS22 as HS22,
    HS23 as HS23,
    HS24 as HS24,
    HS26 as HS26,
    HS27 as HS27,
    HS28 as HS28,
    HS29 as HS29,
    HS30 as HS30,
    HS31 as HS31,
    HS32 as HS32,
    HS33 as HS33,
    HS34 as HS34,
    HS35 as HS35,
    HS36 as HS36,
    HS37 as HS37,
    HS39 as HS39,
    HS40 as HS40,
    HS41 as HS41,
    HS42 as HS42,
    HS43 as HS43,
    HS44 as HS44,
    HS46 as HS46,
    HS47 as HS47,
    HS48 as HS48,
    HS49 as HS49,
    HS50 as HS50,
    HS51 as HS51,
    HS52 as HS52,
    HS53 as HS53,
    HS54 as HS54,
    HS55 as HS55,
    HS56 as HS56,
    HS57 as HS57,
    # HS59 as HS59,  # TODO: Human review - objective function discrepancy
    HS60 as HS60,
    HS61 as HS61,
    HS62 as HS62,
    HS63 as HS63,
    HS64 as HS64,
    HS65 as HS65,
    HS66 as HS66,
    # HS67 as HS67,  # TODO: Human review - different SIF file version
    HS68 as HS68,
    HS69 as HS69,
    # HS70 as HS70,  # TODO: Human review - test failures
    HS71 as HS71,
    HS72 as HS72,
    HS73 as HS73,
    HS74 as HS74,
    HS75 as HS75,
    HS76 as HS76,
    HS77 as HS77,
    HS78 as HS78,
    HS79 as HS79,
    HS80 as HS80,
    HS81 as HS81,
    HS83 as HS83,
    # HS84 as HS84,  # TODO: Human review - objective value discrepancy
    HS93 as HS93,
    # HS99 as HS99,  # TODO: Needs human review - complex recursive formulation
    HS100 as HS100,
    HS101 as HS101,
    HS102 as HS102,
    HS103 as HS103,
    HS104 as HS104,
    HS105 as HS105,
    HS106 as HS106,
    HS107 as HS107,
    HS108 as HS108,
    HS109 as HS109,
    HS111 as HS111,
    HS112 as HS112,
    HS113 as HS113,
    HS114 as HS114,
    HS116 as HS116,
    HS117 as HS117,
    HS118 as HS118,
    HS119 as HS119,
    LOOTSMA as LOOTSMA,
    LUKVLE1 as LUKVLE1,
    # LUKVLE2 as LUKVLE2,
    LUKVLE3 as LUKVLE3,
    # LUKVLE4 as LUKVLE4,  # Use LUKVLE4C instead
    # LUKVLE4C as LUKVLE4C,
    LUKVLE5 as LUKVLE5,
    LUKVLE6 as LUKVLE6,
    LUKVLE7 as LUKVLE7,
    LUKVLE8 as LUKVLE8,
    LUKVLE9 as LUKVLE9,
    LUKVLE10 as LUKVLE10,
    LUKVLE11 as LUKVLE11,
    # LUKVLE12 as LUKVLE12,  # Has constraint function inconsistencies
    LUKVLE13 as LUKVLE13,
    LUKVLE14 as LUKVLE14,
    LUKVLE15 as LUKVLE15,
    LUKVLE16 as LUKVLE16,
    LUKVLE17 as LUKVLE17,
    LUKVLE18 as LUKVLE18,
    LUKVLI1 as LUKVLI1,
    # LUKVLI2 as LUKVLI2,
    LUKVLI3 as LUKVLI3,
    # LUKVLI4 as LUKVLI4,  # Use LUKVLI4C instead
    # LUKVLI4C as LUKVLI4C,
    LUKVLI5 as LUKVLI5,
    LUKVLI6 as LUKVLI6,
    LUKVLI7 as LUKVLI7,
    LUKVLI8 as LUKVLI8,
    LUKVLI9 as LUKVLI9,
    LUKVLI10 as LUKVLI10,
    LUKVLI11 as LUKVLI11,
    # LUKVLI12 as LUKVLI12,  # Has constraint function inconsistencies
    LUKVLI13 as LUKVLI13,
    LUKVLI14 as LUKVLI14,
    LUKVLI15 as LUKVLI15,
    LUKVLI16 as LUKVLI16,
    LUKVLI17 as LUKVLI17,
    LUKVLI18 as LUKVLI18,
    MAKELA1 as MAKELA1,
    MAKELA2 as MAKELA2,
    MAKELA3 as MAKELA3,
    MAKELA4 as MAKELA4,
    MARATOS as MARATOS,
    PENTAGON as PENTAGON,
    POLAK1 as POLAK1,
    POLAK2 as POLAK2,
    POLAK5 as POLAK5,
    POWELLBS as POWELLBS,
    POWELLSE as POWELLSE,
    POWELLSQ as POWELLSQ,
    SIPOW1 as SIPOW1,
    SIPOW2 as SIPOW2,
    # SIPOW3 as SIPOW3,  # TODO: Human review - constraint formulation issues
    # SIPOW4 as SIPOW4,  # TODO: Human review - constraint formulation issues
    TRUSPYR1 as TRUSPYR1,
    # TRUSPYR2 as TRUSPYR2,  # TODO: Human review - test requested to be removed
    VANDERM1 as VANDERM1,
    VANDERM2 as VANDERM2,
    # VANDERM3 as VANDERM3,  # TODO: Human review - constraints mismatch
    # VANDERM4 as VANDERM4,  # TODO: Human review - constraints mismatch
    ZECEVIC2 as ZECEVIC2,
    ZECEVIC3 as ZECEVIC3,
    ZECEVIC4 as ZECEVIC4,
)
from ._unconstrained_minimisation import (
    AKIVA as AKIVA,
    ALLINITU as ALLINITU,
    ARGLINA as ARGLINA,
    ARGLINB as ARGLINB,
    ARGLINC as ARGLINC,
    ARGTRIGLS as ARGTRIGLS,
    ARWHEAD as ARWHEAD,
    # BA_L1LS as BA_L1LS,  # TODO: BA_L family needs human review - removed from imports
    # BA_L1SPLS as BA_L1SPLS,  # TODO: BA_L family needs human review
    BARD as BARD,
    BDQRTIC as BDQRTIC,
    BEALE as BEALE,
    BIGGS6 as BIGGS6,
    BOX as BOX,
    BOX3 as BOX3,
    # BOXBOD as BOXBOD,  # Moved to constrained formulation
    BOXBODLS as BOXBODLS,
    # BOXPOWER as BOXPOWER,  # TODO: Human review - minor gradient discrepancy
    # BRKMCC as BRKMCC,  # TODO: Human review - significant discrepancies
    # BROWNAL as BROWNAL,  # TODO: Human review - small Hessian discrepancies
    BROWNBS as BROWNBS,
    BROWNDEN as BROWNDEN,
    BROYDN3DLS as BROYDN3DLS,
    BROYDN7D as BROYDN7D,
    # BROYDNBDLS as BROYDNBDLS,  # TODO: Gradient test fails - needs human review
    # BRYBND as BRYBND,  # TODO: Gradient test fails - needs human review
    # CERI651ALS as CERI651ALS,  # TODO: Human review - numerical instability
    # CERI651BLS as CERI651BLS,  # TODO: Human review - numerical instability
    # CERI651CLS as CERI651CLS,  # TODO: Human review - numerical instability
    # CERI651DLS as CERI651DLS,  # TODO: Human review - numerical instability
    # CERI651ELS as CERI651ELS,  # TODO: Human review - numerical instability
    CHAINWOO as CHAINWOO,
    # CHANDHEQ as CHANDHEQ,  # Moved to constrained formulation
    CHNROSNB as CHNROSNB,
    CHNRSNBM as CHNRSNBM,
    # CHWIRUT1 as CHWIRUT1,  # TODO: needs external data file
    CHWIRUT1LS as CHWIRUT1LS,
    # CHWIRUT2 as CHWIRUT2,  # TODO: needs implementation with 54 data points
    CHWIRUT2LS as CHWIRUT2LS,
    CLIFF as CLIFF,
    # CLUSTER as CLUSTER,  # Moved to constrained formulation
    CLUSTERLS as CLUSTERLS,
    COATING as COATING,
    # COOLHANS as COOLHANS,  # Moved to constrained formulation
    COOLHANSLS as COOLHANSLS,
    COSINE as COSINE,
    CRAGGLVY as CRAGGLVY,
    CUBE as CUBE,
    CURLY10 as CURLY10,
    CURLY20 as CURLY20,
    CURLY30 as CURLY30,
    # CYCLOOCFLS as CYCLOOCFLS,  # TODO: Human review - times out with default p=10000
    # DANIWOOD as DANIWOOD,  # Moved to constrained formulation
    DANIWOODLS as DANIWOODLS,
    DENSCHNA as DENSCHNA,
    DENSCHNB as DENSCHNB,
    DENSCHNC as DENSCHNC,
    DENSCHND as DENSCHND,
    DENSCHNE as DENSCHNE,
    DENSCHNF as DENSCHNF,
    DEVGLA1 as DEVGLA1,
    DEVGLA2 as DEVGLA2,
    DIXMAANA1 as DIXMAANA1,
    DIXMAANB as DIXMAANB,
    DIXMAANC as DIXMAANC,
    DIXMAAND as DIXMAAND,
    DIXMAANE1 as DIXMAANE1,
    DIXMAANF as DIXMAANF,
    DIXMAANG as DIXMAANG,
    DIXMAANH as DIXMAANH,
    DIXMAANI1 as DIXMAANI1,
    DIXMAANJ as DIXMAANJ,
    DIXMAANK as DIXMAANK,
    DIXMAANL as DIXMAANL,
    DIXMAANM1 as DIXMAANM1,
    DIXMAANN as DIXMAANN,
    DIXMAANO as DIXMAANO,
    DIXMAANP as DIXMAANP,
    DIXON3DQ as DIXON3DQ,
    DJTL as DJTL,
    DQDRTIC as DQDRTIC,
    DQRTIC as DQRTIC,
    # ECKERLE4LS as ECKERLE4LS,  # TODO: Human review - significant discrepancies
    EDENSCH as EDENSCH,
    EG2 as EG2,
    EGGCRATE as EGGCRATE,
    EIGENALS as EIGENALS,
    EIGENBLS as EIGENBLS,
    EIGENCLS as EIGENCLS,
    ELATVIDU as ELATVIDU,
    ENGVAL1 as ENGVAL1,
    ENGVAL2 as ENGVAL2,
    # ENSOLS as ENSOLS,  # TODO: Human review - significant discrepancies
    ERRINROS as ERRINROS,
    # ERRINRSM as ERRINRSM,  # TODO: Human review - significant discrepancies
    EXP2 as EXP2,
    EXPFIT as EXPFIT,
    # EXTROSNB as EXTROSNB,  # TODO: Human review - objective/gradient discrepancies
    # FBRAIN3LS as FBRAIN3LS,  # TODO: Human review - complex data dependencies
    FLETBV3M as FLETBV3M,
    FLETCBV2 as FLETCBV2,
    FLETCBV3 as FLETCBV3,
    # FLETCHBV as FLETCHBV,  # TODO: Human review - objective/gradient discrepancies
    FLETCHCR as FLETCHCR,
    # FMINSRF2 as FMINSRF2,  # TODO: Human review - starting value/gradient issues
    # FMINSURF as FMINSURF,  # TODO: Human review - starting value/gradient issues
    # FREURONE as FREURONE,  # TODO: Human review - miscategorized (constrained)
    FREUROTH as FREUROTH,
    # GAUSS1LS as GAUSS1LS,  # TODO: Human review - issues reported by user
    # GAUSS2LS as GAUSS2LS,  # TODO: Human review - issues reported by user
    # GAUSS3LS as GAUSS3LS,  # TODO: Human review - issues reported by user
    GAUSSIAN as GAUSSIAN,
    # GBRAINLS as GBRAINLS,  # TODO: Human review - complex data dependencies
    GENHUMPS as GENHUMPS,
    GENROSE as GENROSE,
    GROWTHLS as GROWTHLS,
    # GULF as GULF,  # TODO: Human review - issues reported by user
    HAHN1LS as HAHN1LS,
    HAIRY as HAIRY,
    # HATFLDD as HATFLDD,  # TODO: HATFLD family needs human review - discrepancies
    # HATFLDE as HATFLDE,  # TODO: HATFLD family needs human review - discrepancies
    # HATFLDFL as HATFLDFL,  # TODO: HATFLD family needs human review - discrepancies
    # HATFLDFLS as HATFLDFLS,  # TODO: HATFLD family needs human review - discrepancies
    # HATFLDGLS as HATFLDGLS,  # TODO: HATFLD family needs human review - discrepancies
    # HEART6LS as HEART6LS,  # TODO: Human review - significant discrepancies
    # HEART8LS as HEART8LS,  # TODO: Human review - significant discrepancies
    HELIX as HELIX,
    # HIELOW as HIELOW,  # TODO: Human review - significant discrepancies
    HILBERTA as HILBERTA,
    HILBERTB as HILBERTB,
    # HIMMELBB as HIMMELBB,  # TODO: Human review - Hessian discrepancies
    HIMMELBCLS as HIMMELBCLS,
    # HIMMELBF as HIMMELBF,  # TODO: Human review - Hessian discrepancies
    HIMMELBG as HIMMELBG,
    HIMMELBH as HIMMELBH,
    HUMPS as HUMPS,
    INDEF as INDEF,
    INDEFM as INDEFM,
    INTEQNELS as INTEQNELS,
    JENSMP as JENSMP,
    JUDGE as JUDGE,
    KIRBY2LS as KIRBY2LS,
    KOWOSB as KOWOSB,
    # KSSLS as KSSLS,  # TODO: Human review - significant obj/grad discrepancies
    LANCZOS1LS as LANCZOS1LS,
    LANCZOS2LS as LANCZOS2LS,
    LIARWHD as LIARWHD,
    LOGHAIRY as LOGHAIRY,
    LSC1LS as LSC1LS,
    LSC2LS as LSC2LS,
    # MANCINO as MANCINO,  # TODO: Human review - significant discrepancies in all
    # MEXHAT as MEXHAT,  # TODO: Human review - complex scaling issues
    # NONDIA as NONDIA,  # TODO: Human review - SCALE factor issue
    NONCVXU2 as NONCVXU2,
    NONCVXUN as NONCVXUN,
    NONDQUAR as NONDQUAR,
    NONMSQRT as NONMSQRT,
    # PENALTY1 as PENALTY1,  # TODO: Human review - minor numerical precision issues
    # PENALTY2 as PENALTY2,  # TODO: Human review - SCALE factor issue
    POWER as POWER,
    # POWELLSG as POWELLSG,  # TODO: Human review - objective off by factor of 4.15
    ROSENBR as ROSENBR,
    TENFOLDTRLS as TENFOLDTRLS,
    # TOINTGOR as TOINTGOR,  # TODO: Human review - runtime test fails
    TOINTGSS as TOINTGSS,
    # TOINTPSP as TOINTPSP,  # TODO: Human review - gradient test fails
    TRIGON1 as TRIGON1,
    # TRIGON2 as TRIGON2,  # TODO: Human review - Hessian test fails
    unconstrained_minimisation_problems as unconstrained_minimisation_problems,
    WAYSEA1 as WAYSEA1,
    WAYSEA2 as WAYSEA2,
)


problems_dict = {
    # "ACOPP14": ACOPP14(),  # TODO: needs human review - complex AC OPF formulation
    # "AIRPORT": AIRPORT(),  # TODO: Human review - constraints don't match pycutest
    # "ALLINITA": ALLINITA(),  # TODO: needs human review
    "ALSOTAME": ALSOTAME(),
    # "ANTWERP": ANTWERP(),  # TODO: needs human review
    "BIGGSC4": BIGGSC4(),
    # "BOOTH": BOOTH(),  # TODO: needs human review - incorrectly classified
    "BURKEHAN": BURKEHAN(),
    "BYRDSPHR": BYRDSPHR(),
    "CANTILVR": CANTILVR(),
    "CB2": CB2(),
    "CB3": CB3(),
    "CHACONN1": CHACONN1(),
    "CHACONN2": CHACONN2(),
    "HS1": HS1(),
    "HS2": HS2(),
    "HS3": HS3(),
    "HS4": HS4(),
    "HS5": HS5(),
    "HS6": HS6(),
    "HS7": HS7(),
    "HS8": HS8(),
    "HS9": HS9(),
    "HS10": HS10(),
    "HS11": HS11(),
    "HS12": HS12(),
    "HS13": HS13(),
    "HS14": HS14(),
    "HS15": HS15(),
    "HS16": HS16(),
    "HS17": HS17(),
    "HS18": HS18(),
    "HS19": HS19(),
    "HS20": HS20(),
    "HS21": HS21(),
    "HS22": HS22(),
    "HS23": HS23(),
    "HS24": HS24(),
    "HS25": HS25(),
    "HS26": HS26(),
    "HS27": HS27(),
    "HS28": HS28(),
    "HS29": HS29(),
    "HS30": HS30(),
    "HS31": HS31(),
    "HS32": HS32(),
    "HS33": HS33(),
    "HS34": HS34(),
    "HS35": HS35(),
    "HS36": HS36(),
    "HS37": HS37(),
    "HS38": HS38(),
    "HS39": HS39(),
    "HS40": HS40(),
    "HS41": HS41(),
    "HS42": HS42(),
    "HS43": HS43(),
    "HS44": HS44(),
    "HS45": HS45(),
    "HS46": HS46(),
    "HS47": HS47(),
    "HS48": HS48(),
    "HS49": HS49(),
    "HS50": HS50(),
    "HS51": HS51(),
    "HS52": HS52(),
    "HS53": HS53(),
    "HS54": HS54(),
    "HS55": HS55(),
    "HS56": HS56(),
    "HS57": HS57(),
    # "HS59": HS59(),  # TODO: Human review - objective function discrepancy
    "HS60": HS60(),
    "HS61": HS61(),
    "HS62": HS62(),
    "HS63": HS63(),
    "HS64": HS64(),
    "HS65": HS65(),
    "HS66": HS66(),
    # "HS67": HS67(),  # TODO: Human review - different SIF file version
    "HS68": HS68(),
    "HS69": HS69(),
    "HS71": HS71(),
    "HS72": HS72(),
    "HS73": HS73(),
    "HS74": HS74(),
    "HS75": HS75(),
    "HS76": HS76(),
    "HS77": HS77(),
    "HS78": HS78(),
    "HS79": HS79(),
    "HS80": HS80(),
    "HS81": HS81(),
    "HS83": HS83(),
    "HS93": HS93(),
    # "HS99": HS99(),  # TODO: Needs human review - complex recursive formulation
    "HS100": HS100(),
    "HS101": HS101(),
    "HS102": HS102(),
    "HS103": HS103(),
    "HS104": HS104(),
    "HS105": HS105(),
    "HS106": HS106(),
    "HS107": HS107(),
    "HS108": HS108(),
    "HS109": HS109(),
    "HS110": HS110(),
    "HS111": HS111(),
    "HS112": HS112(),
    "HS113": HS113(),
    "HS114": HS114(),
    "HS116": HS116(),
    "HS117": HS117(),
    "HS118": HS118(),
    "HS119": HS119(),
    "LOOTSMA": LOOTSMA(),
    "MARATOS": MARATOS(),
    "PENTAGON": PENTAGON(),
    "POLAK1": POLAK1(),
    "POLAK2": POLAK2(),
    "POLAK5": POLAK5(),
    "SIPOW1": SIPOW1(),
    "SIPOW2": SIPOW2(),
    # "SIPOW3": SIPOW3(),  # TODO: Human review - constraint formulation issues
    # "SIPOW4": SIPOW4(),  # TODO: Human review - constraint formulation issues
    "VANDERM1": VANDERM1(),
    "VANDERM2": VANDERM2(),
    # "VANDERM3": VANDERM3(),  # TODO: Human review - constraints mismatch
    # "VANDERM4": VANDERM4(),  # TODO: Human review - constraints mismatch
    "MAKELA1": MAKELA1(),
    "MAKELA2": MAKELA2(),
    "MAKELA3": MAKELA3(),
    "MAKELA4": MAKELA4(),
    # "HS70": HS70(),  # TODO: Human review - test failures
    # "HS84": HS84(),  # TODO: Human review - objective value discrepancy
    "ZECEVIC2": ZECEVIC2(),
    "ZECEVIC3": ZECEVIC3(),
    "ZECEVIC4": ZECEVIC4(),
    "TRUSPYR1": TRUSPYR1(),
    # "TRUSPYR2": TRUSPYR2(),  # TODO: Human review - test requested to be removed
    "BT1": BT1(),
    "BT2": BT2(),
    "BT3": BT3(),
    "BT4": BT4(),
    "BT5": BT5(),
    "BT6": BT6(),
    "BT7": BT7(),
    "BT8": BT8(),
    "BT9": BT9(),
    "BT10": BT10(),
    "BT11": BT11(),
    "BT12": BT12(),
    "BT13": BT13(),
    "LUKVLE1": LUKVLE1(),
    # "LUKVLE2": LUKVLE2(),
    "LUKVLE3": LUKVLE3(),
    # "LUKVLE4": LUKVLE4(),  # Use LUKVLE4C instead
    # "LUKVLE4C": LUKVLE4C(),
    "LUKVLE5": LUKVLE5(),
    "LUKVLE6": LUKVLE6(),
    "LUKVLE7": LUKVLE7(),
    "LUKVLE8": LUKVLE8(),
    "LUKVLE9": LUKVLE9(),
    "LUKVLE10": LUKVLE10(),
    "LUKVLE11": LUKVLE11(),
    # "LUKVLE12": LUKVLE12(),  # Has constraint function inconsistencies
    "LUKVLE13": LUKVLE13(),
    "LUKVLE14": LUKVLE14(),
    "LUKVLE15": LUKVLE15(),
    "LUKVLE16": LUKVLE16(),
    "LUKVLE17": LUKVLE17(),
    "LUKVLE18": LUKVLE18(),
    "LUKVLI1": LUKVLI1(),
    # "LUKVLI2": LUKVLI2(),
    "LUKVLI3": LUKVLI3(),
    # "LUKVLI4": LUKVLI4(),  # Use LUKVLI4C instead
    # "LUKVLI4C": LUKVLI4C(),
    "LUKVLI5": LUKVLI5(),
    "LUKVLI6": LUKVLI6(),
    "LUKVLI7": LUKVLI7(),
    "LUKVLI8": LUKVLI8(),
    "LUKVLI9": LUKVLI9(),
    "LUKVLI10": LUKVLI10(),
    "LUKVLI11": LUKVLI11(),
    # "LUKVLI12": LUKVLI12(),  # Has constraint function inconsistencies
    "LUKVLI13": LUKVLI13(),
    "LUKVLI14": LUKVLI14(),
    "LUKVLI15": LUKVLI15(),
    "LUKVLI16": LUKVLI16(),
    "LUKVLI17": LUKVLI17(),
    "LUKVLI18": LUKVLI18(),
    "AKIVA": AKIVA(),
    "ALLINITU": ALLINITU(),
    "ARGLINA": ARGLINA(),
    "ARGLINB": ARGLINB(),
    "ARGLINC": ARGLINC(),
    "ARGTRIGLS": ARGTRIGLS(),
    "ARWHEAD": ARWHEAD(),
    "AVGASA": AVGASA(),
    "AVGASB": AVGASB(),
    # "AVION2": AVION2(),  # TODO: Human review - gradient discrepancies
    # "BA_L1LS": BA_L1LS(),  # TODO: BA_L family needs to be split into files
    # "BA_L1SPLS": BA_L1SPLS(),  # TODO: BA_L family needs human review
    "BARD": BARD(),
    "BDQRTIC": BDQRTIC(),
    "BEALE": BEALE(),
    "BIGGS6": BIGGS6(),
    "BOX": BOX(),
    "BOX3": BOX3(),
    "BOXBOD": BOXBOD(),
    "BOXBODLS": BOXBODLS(),
    # "BOXPOWER": BOXPOWER(),  # TODO: Human review - minor gradient discrepancy
    # "BRKMCC": BRKMCC(),  # TODO: Human review - significant discrepancies
    # "BROWNAL": BROWNAL(),  # TODO: Human review - small Hessian discrepancies
    "BROWNBS": BROWNBS(),
    "BROWNDEN": BROWNDEN(),
    "BROYDN3DLS": BROYDN3DLS(),
    "BROYDN7D": BROYDN7D(),
    # "BROYDNBDLS": BROYDNBDLS(),  # TODO: Gradient test fails - needs human review
    # "BRYBND": BRYBND(),  # TODO: Gradient test fails - needs human review
    # "CERI651ALS": CERI651ALS(),  # TODO: Human review - numerical instability
    # "CERI651BLS": CERI651BLS(),  # TODO: Human review - numerical instability
    # "CERI651CLS": CERI651CLS(),  # TODO: Human review - numerical instability
    # "CERI651DLS": CERI651DLS(),  # TODO: Human review - numerical instability
    # "CERI651ELS": CERI651ELS(),  # TODO: Human review - numerical instability
    "CHAINWOO": CHAINWOO(),
    "CHANDHEQ": CHANDHEQ(),
    "CHNROSNB": CHNROSNB(),
    "CHNRSNBM": CHNRSNBM(),
    "CHWIRUT1LS": CHWIRUT1LS(),
    "CHWIRUT2LS": CHWIRUT2LS(),
    "CLIFF": CLIFF(),
    "CLUSTER": CLUSTER(),
    "CLUSTERLS": CLUSTERLS(),
    "COATING": COATING(),
    "CONCON": CONCON(),
    "COOLHANS": COOLHANS(),
    "COOLHANSLS": COOLHANSLS(),
    "COSINE": COSINE(),
    "CRAGGLVY": CRAGGLVY(),
    # "CRESC4": CRESC4(),  # TODO: Human review - complex crescent area formula
    "CSFI1": CSFI1(),
    "CSFI2": CSFI2(),
    "CUBE": CUBE(),
    "CURLY10": CURLY10(),
    "CURLY20": CURLY20(),
    "CURLY30": CURLY30(),
    "CVXQP1": CVXQP1(),
    # "CYCLOOCFLS": CYCLOOCFLS(),  # TODO: Human review - times out with default p=10000
    # "DALLASS": DALLASS(),  # TODO: needs complex element implementations
    "DANIWOOD": DANIWOOD(),
    "DANIWOODLS": DANIWOODLS(),
    "DECONVC": DECONVC(),
    "DENSCHNA": DENSCHNA(),
    "DENSCHNB": DENSCHNB(),
    "DENSCHNC": DENSCHNC(),
    "DENSCHND": DENSCHND(),
    "DENSCHNE": DENSCHNE(),
    "DENSCHNF": DENSCHNF(),
    "DEVGLA1": DEVGLA1(),
    "DEVGLA2": DEVGLA2(),
    "DIXMAANA1": DIXMAANA1(),
    "DIXMAANB": DIXMAANB(),
    "DIXMAANC": DIXMAANC(),
    "DIXMAAND": DIXMAAND(),
    "DIXMAANE1": DIXMAANE1(),
    "DIXMAANF": DIXMAANF(),
    "DIXMAANG": DIXMAANG(),
    "DIXMAANH": DIXMAANH(),
    "DIXMAANI1": DIXMAANI1(),
    "DIXMAANJ": DIXMAANJ(),
    "DIXMAANK": DIXMAANK(),
    "DIXMAANL": DIXMAANL(),
    "DIXMAANM1": DIXMAANM1(),
    "DIXMAANN": DIXMAANN(),
    "DIXMAANO": DIXMAANO(),
    "DIXMAANP": DIXMAANP(),
    "DIXON3DQ": DIXON3DQ(),
    "DJTL": DJTL(),
    "DQDRTIC": DQDRTIC(),
    "DQRTIC": DQRTIC(),
    # "ECKERLE4LS": ECKERLE4LS(),  # TODO: Human review - significant discrepancies
    "EDENSCH": EDENSCH(),
    "EG2": EG2(),
    "EGGCRATE": EGGCRATE(),
    "EIGENALS": EIGENALS(),
    "EIGENBLS": EIGENBLS(),
    "EIGENCLS": EIGENCLS(),
    "ELATVIDU": ELATVIDU(),
    "ENGVAL1": ENGVAL1(),
    "ENGVAL2": ENGVAL2(),
    # "ENSOLS": ENSOLS(),  # TODO: Human review - significant discrepancies
    "ERRINROS": ERRINROS(),
    # "ERRINRSM": ERRINRSM(),  # TODO: Human review - significant discrepancies
    "EXP2": EXP2(),
    "EXP2B": EXP2B(),
    "EXPFIT": EXPFIT(),
    # "EXTROSNB": EXTROSNB(),  # TODO: Human review - objective/gradient discrepancies
    # "FBRAIN3LS": FBRAIN3LS(),  # TODO: Human review - complex data dependencies
    # "FLETCHBV": FLETCHBV(),  # TODO: Human review - objective/gradient discrepancies
    "FLETBV3M": FLETBV3M(),
    "FLETCBV2": FLETCBV2(),
    "FLETCHCR": FLETCHCR(),
    "FLETCBV3": FLETCBV3(),
    # "FMINSRF2": FMINSRF2(),  # TODO: Human review - starting value/gradient issues
    # "FMINSURF": FMINSURF(),  # TODO: Human review - starting value/gradient issues
    # "FREURONE": FREURONE(),  # TODO: Human review - miscategorized (constrained)
    "FREUROTH": FREUROTH(),
    # "GAUSS1LS": GAUSS1LS(),  # TODO: Human review - issues reported by user
    # "GAUSS2LS": GAUSS2LS(),  # TODO: Human review - issues reported by user
    # "GAUSS3LS": GAUSS3LS(),  # TODO: Human review - issues reported by user
    "GAUSSIAN": GAUSSIAN(),
    # "GBRAINLS": GBRAINLS(),  # TODO: Human review - complex data dependencies
    "GENHUMPS": GENHUMPS(),
    "GENROSE": GENROSE(),
    "GROWTHLS": GROWTHLS(),
    # "GULF": GULF(),  # TODO: Human review - issues reported by user
    "HAHN1LS": HAHN1LS(),
    "HAIRY": HAIRY(),
    # "HATFLDD": HATFLDD(),  # TODO: HATFLD family needs human review - discrepancies
    # "HATFLDE": HATFLDE(),  # TODO: HATFLD family needs human review - discrepancies
    # "HATFLDFL": HATFLDFL(),  # TODO: HATFLD family needs human review - discrepancies
    # "HATFLDFLS": HATFLDFLS(),  # TODO: HATFLD family needs human review
    # "HATFLDGLS": HATFLDGLS(),  # TODO: HATFLD family needs human review
    # "HEART6LS": HEART6LS(),  # TODO: Human review - significant discrepancies
    # "HEART8LS": HEART8LS(),  # TODO: Human review - significant discrepancies
    "HELIX": HELIX(),
    # "HIELOW": HIELOW(),  # TODO: Human review - significant discrepancies
    "HILBERTA": HILBERTA(),
    "HILBERTB": HILBERTB(),
    # "HIMMELBB": HIMMELBB(),  # TODO: Human review - Hessian discrepancies
    "HIMMELBCLS": HIMMELBCLS(),
    # "HIMMELBF": HIMMELBF(),  # TODO: Human review - Hessian discrepancies
    "HIMMELBG": HIMMELBG(),
    "HIMMELBH": HIMMELBH(),
    "HUMPS": HUMPS(),
    "INDEF": INDEF(),
    "INDEFM": INDEFM(),
    "INTEQNELS": INTEQNELS(),
    "JENSMP": JENSMP(),
    "JUDGE": JUDGE(),
    "KIRBY2LS": KIRBY2LS(),
    "KOWOSB": KOWOSB(),
    # "KSSLS": KSSLS(),  # TODO: Human review - significant obj/grad discrepancies
    "LANCZOS1LS": LANCZOS1LS(),
    "LANCZOS2LS": LANCZOS2LS(),
    "LIARWHD": LIARWHD(),
    "LOGHAIRY": LOGHAIRY(),
    "LSC1LS": LSC1LS(),
    "LSC2LS": LSC2LS(),
    # "MANCINO": MANCINO(),  # TODO: Human review - significant discrepancies in all
    # "MEXHAT": MEXHAT(),  # TODO: Human review - complex scaling issues
    # "NONDIA": NONDIA(),  # TODO: Human review - SCALE factor issue
    "NONCVXU2": NONCVXU2(),
    "NONCVXUN": NONCVXUN(),
    "NONDQUAR": NONDQUAR(),
    "NONMSQRT": NONMSQRT(),
    # "PENALTY1": PENALTY1(),  # TODO: Human review - minor numerical precision issues
    # "PENALTY2": PENALTY2(),  # TODO: Human review - SCALE factor issue
    "POWER": POWER(),
    # "POWELLSG": POWELLSG(),  # TODO: Human review - objective off by factor of 4.15
    "ROSENBR": ROSENBR(),
    "TENFOLDTRLS": TENFOLDTRLS(),
    "POWELLBS": POWELLBS(),
    "POWELLSE": POWELLSE(),
    "POWELLSQ": POWELLSQ(),
    "WAYSEA1": WAYSEA1(),
    "WAYSEA2": WAYSEA2(),
    "TRIGON1": TRIGON1(),
    # "TRIGON2": TRIGON2(),  # TODO: Human review - Hessian test fails
    # "TOINTGOR": TOINTGOR(),  # TODO: Human review - runtime test fails
    "TOINTGSS": TOINTGSS(),
    # "TOINTPSP": TOINTPSP(),  # TODO: Human review - gradient test fails
}


def get_problem(name: str):
    return problems_dict.get(name, None)  # TODO: try except with nicer error message


problems = (
    unconstrained_minimisation_problems
    + bounded_minimisation_problems
    + constrained_minimisation_problems
)
