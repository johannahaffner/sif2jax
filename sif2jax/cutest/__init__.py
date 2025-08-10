from ._bounded_minimisation import (
    BDEXP as BDEXP,
    BIGGS3 as BIGGS3,
    BIGGS5 as BIGGS5,
    bounded_minimisation_problems as bounded_minimisation_problems,
    BOX2 as BOX2,
    BRANIN as BRANIN,
    CAMEL6 as CAMEL6,
    CYCLOOCTLS as CYCLOOCTLS,
    DEGDIAG as DEGDIAG,
    DEGTRID as DEGTRID,
    DEGTRID2 as DEGTRID2,
    DGOSPEC as DGOSPEC,
    DIAGIQB as DIAGIQB,
    DIAGIQE as DIAGIQE,
    DIAGIQT as DIAGIQT,
    DIAGNQB as DIAGNQB,
    DIAGNQE as DIAGNQE,
    DIAGNQT as DIAGNQT,
    DIAGPQB as DIAGPQB,
    DIAGPQE as DIAGPQE,
    DIAGPQT as DIAGPQT,
    EXP2B as EXP2B,
    EXPLIN as EXPLIN,
    EXPLIN2 as EXPLIN2,
    EXPQUAD as EXPQUAD,
    HART6 as HART6,
    HATFLDA as HATFLDA,
    HATFLDB as HATFLDB,
    HATFLDC as HATFLDC,
    # HIMMELP1 as HIMMELP1,  # TODO: Human review needed - OBNL element issues
    HS1 as HS1,
    HS2 as HS2,
    HS3 as HS3,
    HS4 as HS4,
    HS5 as HS5,
    HS25 as HS25,
    HS38 as HS38,
    HS45 as HS45,
    HS110 as HS110,
    LEVYMONT as LEVYMONT,
    LEVYMONT5 as LEVYMONT5,
    LEVYMONT6 as LEVYMONT6,
    LEVYMONT7 as LEVYMONT7,
    LEVYMONT8 as LEVYMONT8,
    LEVYMONT9 as LEVYMONT9,
    LEVYMONT10 as LEVYMONT10,
    LOGROS as LOGROS,
    PALMER1 as PALMER1,
    PALMER1A as PALMER1A,
    # PALMER1B as PALMER1B,  # TODO: Fix Hessian issues
    # PALMER1E as PALMER1E,  # TODO: Fix Hessian issues
    PALMER2 as PALMER2,
    PALMER2A as PALMER2A,
    PALMER2B as PALMER2B,
    PALMER2E as PALMER2E,
    PALMER3 as PALMER3,
    PALMER3A as PALMER3A,
    PALMER3B as PALMER3B,
    PALMER3E as PALMER3E,
    PALMER4 as PALMER4,
    # PALMER4A as PALMER4A,  # TODO: Fix Hessian issues
    PALMER4B as PALMER4B,
    PALMER4E as PALMER4E,
    # PALMER5A as PALMER5A,  # TODO: Fix Chebyshev polynomial calculation
    PALMER5B as PALMER5B,
    PALMER6A as PALMER6A,
    PALMER6E as PALMER6E,
    # PALMER7A as PALMER7A,  # TODO: Fix Hessian issues
    PALMER7E as PALMER7E,
    PALMER8A as PALMER8A,
    PALMER8E as PALMER8E,
    PFIT1LS as PFIT1LS,
    PFIT2LS as PFIT2LS,
    PFIT3LS as PFIT3LS,
    PFIT4LS as PFIT4LS,
    PRICE4B as PRICE4B,
    QUDLIN as QUDLIN,
    TRIGON1B as TRIGON1B,
)

# TRIGON2B as TRIGON2B,  # TODO: Human review - tiny Hessian discrepancies
from ._constrained_minimisation import (
    # ACOPP14 as ACOPP14,  # TODO: needs human review - complex AC OPF formulation
    # AIRPORT as AIRPORT,  # TODO: Human review - constraint values don't match pycutest
    # ALLINITA as ALLINITA,  # TODO: needs human review - L2 group type interpretation
    # ALLINITC as ALLINITC,  # TODO: Human review - dimension mismatch with pycutest
    ALSOTAME as ALSOTAME,
    # ANTWERP as ANTWERP,  # TODO: needs human review - initial value calculation
    # AUG2D as AUG2D,  # TODO: needs human review - edge variable structure
    AVGASA as AVGASA,
    AVGASB as AVGASB,
    # AVION2 as AVION2,  # TODO: Human review - gradient discrepancies
    BIGGSC4 as BIGGSC4,
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
    # CLNLBEAM as CLNLBEAM,  # TODO: Dimension mismatch in constraints
    # CONCON as CONCON,  # TODO: Removed - automatic derivative mismatches
    constrained_minimisation_problems as constrained_minimisation_problems,
    COSHFUN as COSHFUN,
    # CRESC4 as CRESC4,  # TODO: Human review - complex crescent area formula
    CSFI1 as CSFI1,
    CSFI2 as CSFI2,
    DALLASS as DALLASS,
    DECONVC as DECONVC,
    # DEGTRIDL as DEGTRIDL,  # TODO: Human review - causes segfault
    DTOC1L as DTOC1L,
    DTOC1NA as DTOC1NA,
    DTOC1NB as DTOC1NB,
    DTOC1NC as DTOC1NC,
    DTOC1ND as DTOC1ND,
    DTOC2 as DTOC2,
    # DTOC3 as DTOC3,  # Human review needed
    DTOC4 as DTOC4,
    DTOC5 as DTOC5,
    DTOC6 as DTOC6,
    ELATTAR as ELATTAR,
    # EXPFITA as EXPFITA,  # TODO: Human review - fundamental formulation differences
    # EXPFITB as EXPFITB,  # TODO: Human review - fundamental formulation differences
    # EXPFITC as EXPFITC,  # TODO: Human review - fundamental formulation differences
    GIGOMEZ2 as GIGOMEZ2,
    HAGER1 as HAGER1,
    HAGER2 as HAGER2,
    # HAGER3 as HAGER3,  # TODO: HAGER3 needs human review - marked for future import
    HAGER4 as HAGER4,
    HIMMELBC as HIMMELBC,
    HIMMELBD as HIMMELBD,
    HIMMELBE as HIMMELBE,
    # HIMMELP2 as HIMMELP2,  # TODO: Human review needed - OBNL element issues
    # HIMMELP3 as HIMMELP3,  # TODO: Human review needed - OBNL element issues
    # HIMMELP4 as HIMMELP4,  # TODO: Human review needed - OBNL element issues
    # HIMMELP5 as HIMMELP5,  # TODO: Human review needed - OBNL element issues
    # HIMMELP6 as HIMMELP6,  # TODO: Human review needed - OBNL element issues
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
    # HS74 as HS74,  # TODO: Human review - constraint Jacobian discrepancies
    # HS75 as HS75,  # TODO: Human review - same issues as HS74
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
    # HS109 as HS109,  # TODO: Human review needed - sign convention issues
    HS111 as HS111,
    HS112 as HS112,
    HS113 as HS113,
    HS114 as HS114,
    HS116 as HS116,
    HS117 as HS117,
    # HS118 as HS118,  # TODO: Human review - constraint Jacobian ordering mismatch
    HS119 as HS119,
    HYDROELL as HYDROELL,
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
    # LUKVLE9 as LUKVLE9,  # TODO: Human review needed - Jacobian issues
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
    # LUKVLI9 as LUKVLI9,  # TODO: Human review needed - Jacobian issues
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
    MSS1 as MSS1,
    MSS2 as MSS2,
    MSS3 as MSS3,
    ODFITS as ODFITS,
    # ORTHRDM2 as ORTHRDM2,  # TODO: Human review - singular Jacobian issues
    # ORTHRDS2 as ORTHRDS2,  # TODO: Human review - singular Jacobian issues
    ORTHRDS2C as ORTHRDS2C,
    # ORTHREGA as ORTHREGA,  # TODO: Human review - formulation differences
    ORTHREGB as ORTHREGB,
    ORTHREGC as ORTHREGC,
    ORTHREGD as ORTHREGD,
    ORTHREGE as ORTHREGE,
    ORTHREGF as ORTHREGF,
    ORTHRGDM as ORTHRGDM,
    ORTHRGDS as ORTHRGDS,
    PENTAGON as PENTAGON,
    POLAK1 as POLAK1,
    POLAK2 as POLAK2,
    POLAK3 as POLAK3,
    POLAK4 as POLAK4,
    POLAK5 as POLAK5,
    POLAK6 as POLAK6,
    # POLYGON as POLYGON,  # TODO: Human review - fixed variable conventions
    SIMPLLPA as SIMPLLPA,
    SIMPLLPB as SIMPLLPB,
    SIPOW1 as SIPOW1,
    SIPOW2 as SIPOW2,
    TRO3X3 as TRO3X3,
    TRO4X4 as TRO4X4,
    TRO5X5 as TRO5X5,
    TRO6X2 as TRO6X2,
    TRO11X3 as TRO11X3,
    TRO21X5 as TRO21X5,
    TRO41X9 as TRO41X9,
    # SPIN2OP as SPIN2OP,  # TODO: Human review - constraint test failures
    # SPINOP as SPINOP,  # TODO: Human review - auxiliary variable constraint issues
    # STEENBRB as STEENBRB,  # TODO: Human review - gradient test failing
    # SIPOW3 as SIPOW3,  # TODO: Human review - constraint formulation issues
    # SIPOW4 as SIPOW4,  # TODO: Human review - constraint formulation issues
    # TENBARS4 as TENBARS4,  # TODO: Human review - pycutest Jacobian inconsistency
    # TRUSPYR1 as TRUSPYR1,  # TODO: Human review - complex constraint scaling issues
    # TRUSPYR2 as TRUSPYR2,  # TODO: Human review - test requested to be removed
    # VANDERM3 as VANDERM3,  # TODO: Human review - constraints mismatch
    # VANDERM4 as VANDERM4,  # TODO: Human review - constraints mismatch
    # TODO: TWIR problems need human review - complex trilinear constraint formulation
    # TWIRISM1 as TWIRISM1,
    # TWIRIMD1 as TWIRIMD1,
    # TWIRIBG1 as TWIRIBG1,
    ZECEVIC2 as ZECEVIC2,
    ZECEVIC3 as ZECEVIC3,
    ZECEVIC4 as ZECEVIC4,
)
from ._nonlinear_equations import (
    AIRCRFTA as AIRCRFTA,
    ARGAUSS as ARGAUSS,
    ARGLALE as ARGLALE,
    ARGLBLE as ARGLBLE,
    ARGLCLE as ARGLCLE,
    ARGTRIG as ARGTRIG,
    ARTIF as ARTIF,
    # TODO: Human review needed - constraint dimension mismatch
    # ARWHDNE as ARWHDNE,
    BARDNE as BARDNE,
    BDVALUES as BDVALUES,
    # BDQRTICNE as BDQRTICNE,  # TODO: Human review needed
    BEALENE as BEALENE,
    BENNETT5 as BENNETT5,
    BIGGS6NE as BIGGS6NE,
    BOOTH as BOOTH,
    BOX3NE as BOX3NE,
    BOXBOD as BOXBOD,
    BRATU2DT as BRATU2DT,
    # BROWNALE as BROWNALE,  # TODO: Human review needed - Jacobian precision issues
    BROWNBSNE as BROWNBSNE,
    BROWNDENE as BROWNDENE,
    # CERI651A as CERI651A,  # TODO: Human review - numerical overflow at ones vector
    # CERI651B as CERI651B,  # TODO: Human review - numerical overflow at ones vector
    # CERI651C as CERI651C,  # TODO: Human review - numerical overflow at ones vector
    # CHAINWOONE as CHAINWOONE,  # TODO: Human review - constraint values mismatch
    CHANDHEQ as CHANDHEQ,
    # CHANNEL as CHANNEL,  # TODO: Human review needed
    CHEBYQADNE as CHEBYQADNE,
    CLUSTER as CLUSTER,
    # COATINGNE as COATINGNE,  # TODO: Human review - formulation differences
    COOLHANS as COOLHANS,
    # CHNRSBNE as CHNRSBNE,  # TODO: Human review needed
    # CHNRSNBMNE as CHNRSNBMNE,  # TODO: Human review needed
    # CUBENE as CUBENE,  # TODO: Human review - constraint and Jacobian mismatch
    CYCLIC3 as CYCLIC3,
    CYCLOOCF as CYCLOOCF,
    CYCLOOCT as CYCLOOCT,
    DANIWOOD as DANIWOOD,
    DECONVBNE as DECONVBNE,
    DECONVNE as DECONVNE,
    DENSCHNBNE as DENSCHNBNE,
    DENSCHNCNE as DENSCHNCNE,
    DENSCHNDNE as DENSCHNDNE,
    DENSCHNENE as DENSCHNENE,
    DENSCHNFNE as DENSCHNFNE,
    DEVGLA1NE as DEVGLA1NE,
    DEVGLA2NE as DEVGLA2NE,
    # BROYDN3D as BROYDN3D,  # TODO: Human review needed - constraint values mismatch
    # BROYDNBD as BROYDNBD,  # TODO: Human review needed - systematic differences
    # BRYBNDNE as BRYBNDNE,  # TODO: Human review needed - constraint values mismatch
    DRCAVTY1 as DRCAVTY1,
    DRCAVTY2 as DRCAVTY2,
    DRCAVTY3 as DRCAVTY3,
    EGGCRATENE as EGGCRATENE,
    # EIGENAU as EIGENAU,  # TODO: Human review - non-standard initial values
    ELATVIDUNE as ELATVIDUNE,
    ENGVAL2NE as ENGVAL2NE,
    ERRINROSNE as ERRINROSNE,
    ERRINRSMNE as ERRINRSMNE,
    EXP2NE as EXP2NE,
    EXPFITNE as EXPFITNE,
    EXTROSNBNE as EXTROSNBNE,
    # FLOSP2HH as FLOSP2HH,  # TODO: Human review needed - NQR constraint handling
    # FLOSP2HL as FLOSP2HL,  # TODO: Human review needed - NQR constraint handling
    # FLOSP2HM as FLOSP2HM,  # TODO: Human review needed - NQR constraint handling
    # FLOSP2TH as FLOSP2TH,  # TODO: Human review needed - NQR constraint handling
    # FLOSP2TL as FLOSP2TL,  # TODO: Human review needed - NQR constraint handling
    # FLOSP2TM as FLOSP2TM,  # TODO: Human review needed - NQR constraint handling
    FREURONE as FREURONE,
    GENROSEBNE as GENROSEBNE,
    GENROSENE as GENROSENE,
    GOTTFR as GOTTFR,
    GULFNE as GULFNE,
    HATFLDANE as HATFLDANE,
    HATFLDBNE as HATFLDBNE,
    HATFLDCNE as HATFLDCNE,
    HATFLDDNE as HATFLDDNE,
    # HATFLDENE as HATFLDENE,  # TODO: Human review - sign convention issues
    HATFLDF as HATFLDF,
    HATFLDFLNE as HATFLDFLNE,
    HATFLDG as HATFLDG,
    HELIXNE as HELIXNE,
    HIMMELBA as HIMMELBA,
    HIMMELBFNE as HIMMELBFNE,
    HS1NE as HS1NE,
    HS2NE as HS2NE,
    HS25NE as HS25NE,
    # HYDCAR6 as HYDCAR6,  # TODO: Human review needed
    HYPCIR as HYPCIR,
    INTEQNE as INTEQNE,
    JENSMPNE as JENSMPNE,
    JUDGENE as JUDGENE,
    KOEBHELBNE as KOEBHELBNE,
    KOWOSBNE as KOWOSBNE,
    LEVYMONE as LEVYMONE,
    LEVYMONE5 as LEVYMONE5,
    LEVYMONE6 as LEVYMONE6,
    LEVYMONE7 as LEVYMONE7,
    LEVYMONE8 as LEVYMONE8,
    LEVYMONE9 as LEVYMONE9,
    LEVYMONE10 as LEVYMONE10,
    LIARWHDNE as LIARWHDNE,
    # LINVERSENE as LINVERSENE,  # TODO: Human review - incomplete implementation
    LUKSAN11 as LUKSAN11,
    LUKSAN12 as LUKSAN12,
    LUKSAN13 as LUKSAN13,
    LUKSAN14 as LUKSAN14,
    LUKSAN15 as LUKSAN15,
    LUKSAN16 as LUKSAN16,
    LUKSAN17 as LUKSAN17,
    LUKSAN21 as LUKSAN21,
    LUKSAN22 as LUKSAN22,
    MANCINONE as MANCINONE,
    MEYER3NE as MEYER3NE,
    MGH09 as MGH09,
    MGH10 as MGH10,
    MGH10S as MGH10S,
    MGH17 as MGH17,
    MGH17S as MGH17S,
    MISRA1D as MISRA1D,
    # MODBEALENE as MODBEALENE,  # TODO: Human review - constraint ordering issues
    # MOREBVNE as MOREBVNE,  # TODO: Human review - SIF file bug on line 64
    MSQRTA as MSQRTA,
    MSQRTB as MSQRTB,
    # MUONSINE as MUONSINE,  # TODO: Human review - hardcoded data values
    NONDIANE as NONDIANE,
    nonlinear_equations_problems as nonlinear_equations_problems,
    # NONMSQRTNE as NONMSQRTNE,  # TODO: Human review - element structure
    NONSCOMPNE as NONSCOMPNE,
    OSCIGRNE as OSCIGRNE,
    OSCIPANE as OSCIPANE,
    PALMER1ANE as PALMER1ANE,
    PALMER1BNE as PALMER1BNE,
    PALMER1ENE as PALMER1ENE,
    PALMER1NE as PALMER1NE,
    PALMER2ANE as PALMER2ANE,
    PALMER2BNE as PALMER2BNE,
    PALMER2ENE as PALMER2ENE,
    PALMER2NE as PALMER2NE,
    PALMER3ANE as PALMER3ANE,
    PALMER3BNE as PALMER3BNE,
    PALMER3ENE as PALMER3ENE,
    PALMER3NE as PALMER3NE,
    PALMER4ANE as PALMER4ANE,
    PALMER4BNE as PALMER4BNE,
    PALMER4ENE as PALMER4ENE,
    PALMER4NE as PALMER4NE,
    # PALMER5ANE as PALMER5ANE,  # TODO: Fix Chebyshev polynomial calculation
    PALMER5BNE as PALMER5BNE,
    # PALMER5ENE as PALMER5ENE,  # TODO: Human review - numerical precision
    PALMER6ANE as PALMER6ANE,
    PALMER6ENE as PALMER6ENE,
    PALMER7ANE as PALMER7ANE,
    PALMER7ENE as PALMER7ENE,
    PALMER8ANE as PALMER8ANE,
    PALMER8ENE as PALMER8ENE,
    PFIT1 as PFIT1,
    PFIT2 as PFIT2,
    PFIT3 as PFIT3,
    PFIT4 as PFIT4,
    POWELLBS as POWELLBS,
    POWELLSE as POWELLSE,
    POWELLSQ as POWELLSQ,
    POWERSUMNE as POWERSUMNE,
    # RES as RES,  # TODO: Human review needed - mixed constraint types
    SANTA as SANTA,
    SINVALNE as SINVALNE,
    SPIN as SPIN,
    # SPIN2 as SPIN2,  # TODO: Human review - constraint test failures
    # SSBRYBNDNE as SSBRYBNDNE,  # TODO: Human review needed - complex element structure
    TENFOLDTR as TENFOLDTR,
    TRIGON1NE as TRIGON1NE,
    # TRIGON2NE as TRIGON2NE,  # TODO: Human review - Jacobian tolerance 1.26e-05
    VANDANIUMS as VANDANIUMS,
    VARDIMNE as VARDIMNE,
    VESUVIA as VESUVIA,
    VESUVIO as VESUVIO,
    VESUVIOU as VESUVIOU,
    VIBRBEAMNE as VIBRBEAMNE,
    YATP1CNE as YATP1CNE,
    YATP1NE as YATP1NE,
)

# YATP2CNE as YATP2CNE,  # TODO: Human review - constraint ordering mismatch
# YATP2SQ as YATP2SQ,  # TODO: Human review - constraint ordering mismatch
# VANDERM1 as VANDERM1,  # TODO: Human review - mixed constraint types
# VANDERM2 as VANDERM2,  # TODO: Human review - mixed constraint types
from ._quadratic_problems import (
    bounded_quadratic_problems as bounded_quadratic_problems,
    constrained_quadratic_problems as constrained_quadratic_problems,
    # CHENHARK as CHENHARK,  # TODO: Human review needed - see file
    CVXBQP1 as CVXBQP1,
    CVXQP1 as CVXQP1,
    CVXQP2 as CVXQP2,
    CVXQP3 as CVXQP3,
    DUAL1 as DUAL1,
    DUAL2 as DUAL2,
    DUAL3 as DUAL3,
    DUAL4 as DUAL4,
    DUALC1 as DUALC1,
    DUALC2 as DUALC2,
    DUALC5 as DUALC5,
    DUALC8 as DUALC8,
    GOULDQP1 as GOULDQP1,
    GOULDQP2 as GOULDQP2,
    GOULDQP3 as GOULDQP3,
    HATFLDH as HATFLDH,
    HS44NEW as HS44NEW,
    NCVXBQP1 as NCVXBQP1,
    NCVXBQP2 as NCVXBQP2,
    NCVXBQP3 as NCVXBQP3,
    NCVXQP1 as NCVXQP1,
    NCVXQP2 as NCVXQP2,
    NCVXQP3 as NCVXQP3,
    NCVXQP4 as NCVXQP4,
    NCVXQP5 as NCVXQP5,
    NCVXQP6 as NCVXQP6,
    NCVXQP7 as NCVXQP7,
    NCVXQP8 as NCVXQP8,
    NCVXQP9 as NCVXQP9,
    QPBAND as QPBAND,
    QPNBAND as QPNBAND,
    # QPNBLEND as QPNBLEND,  # TODO: Human review - complex constraint matrix
    # QPNBOEI1 as QPNBOEI1,  # TODO: Human review - Boeing routing constraints
    # QPNBOEI2 as QPNBOEI2,  # TODO: Human review - Boeing routing constraints
    # QPNSTAIR as QPNSTAIR,  # TODO: Human review - complex constraint dimensions
    quadratic_problems as quadratic_problems,
    TAME as TAME,
    # TORSIOND as TORSIOND,  # TODO: Human review - objective mismatch with pycutest
    YAO as YAO,
)

# VANDERM3 as VANDERM3,  # TODO: Human review needed - originally had issues
# VANDERM4 as VANDERM4,  # TODO: Human review needed - originally had issues
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
    BENNETT5LS as BENNETT5LS,
    BIGGS6 as BIGGS6,
    BOX as BOX,
    BOX3 as BOX3,
    BOXBODLS as BOXBODLS,
    # BOXPOWER as BOXPOWER,  # TODO: Human review - minor gradient discrepancy
    # BRATU1D as BRATU1D,  # TODO: Human review needed - see file
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
    CHNROSNB as CHNROSNB,
    CHNRSNBM as CHNRSNBM,
    # CHWIRUT1 as CHWIRUT1,  # TODO: needs external data file
    CHWIRUT1LS as CHWIRUT1LS,
    # CHWIRUT2 as CHWIRUT2,  # TODO: needs implementation with 54 data points
    CHWIRUT2LS as CHWIRUT2LS,
    CLIFF as CLIFF,
    CLUSTERLS as CLUSTERLS,
    COATING as COATING,
    COOLHANSLS as COOLHANSLS,
    COSINE as COSINE,
    CRAGGLVY as CRAGGLVY,
    CUBE as CUBE,
    CURLY10 as CURLY10,
    CURLY20 as CURLY20,
    CURLY30 as CURLY30,
    CYCLIC3LS as CYCLIC3LS,
    CYCLOOCFLS as CYCLOOCFLS,
    DANIWOODLS as DANIWOODLS,
    DENSCHNA as DENSCHNA,
    DENSCHNB as DENSCHNB,
    DENSCHNC as DENSCHNC,
    DENSCHND as DENSCHND,
    DENSCHNE as DENSCHNE,
    DENSCHNF as DENSCHNF,
    DEVGLA1 as DEVGLA1,
    DEVGLA2 as DEVGLA2,
    # DIAMON3DLS as DIAMON3DLS,  # TODO: Human review needed - see file
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
    DRCAV1LQ as DRCAV1LQ,
    DRCAV2LQ as DRCAV2LQ,
    DRCAV3LQ as DRCAV3LQ,
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
    HATFLDD as HATFLDD,
    HATFLDE as HATFLDE,
    HATFLDFL as HATFLDFL,
    HATFLDFLS as HATFLDFLS,
    # HATFLDGLS as HATFLDGLS,  # TODO: Known gradient/Hessian discrepancies
    # HEART6LS as HEART6LS,  # TODO: Human review - significant discrepancies
    # HEART8LS as HEART8LS,  # TODO: Human review - significant discrepancies
    HELIX as HELIX,
    # HIELOW as HIELOW,  # TODO: Human review - significant discrepancies
    HILBERTA as HILBERTA,
    HILBERTB as HILBERTB,
    # HIMMELBB as HIMMELBB,  # TODO: needs human review - Hessian issues
    HIMMELBCLS as HIMMELBCLS,
    # HIMMELBF as HIMMELBF,  # TODO: needs human review - Hessian issues
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
    LUKSAN11LS as LUKSAN11LS,
    LUKSAN12LS as LUKSAN12LS,
    LUKSAN13LS as LUKSAN13LS,
    LUKSAN14LS as LUKSAN14LS,
    LUKSAN15LS as LUKSAN15LS,
    LUKSAN16LS as LUKSAN16LS,
    LUKSAN17LS as LUKSAN17LS,
    LUKSAN21LS as LUKSAN21LS,
    # LUKSAN22LS as LUKSAN22LS,  # TODO: Human review needed - gradient issues
    # MANCINO as MANCINO,  # TODO: Human review - significant discrepancies in all
    MARATOSB as MARATOSB,
    MEXHAT as MEXHAT,
    MGH09LS as MGH09LS,
    MGH10LS as MGH10LS,
    MGH10SLS as MGH10SLS,
    MGH17LS as MGH17LS,
    MGH17SLS as MGH17SLS,
    # MOREBV as MOREBV,  # TODO: Human review - minor gradient precision differences
    # MODBEALE as MODBEALE,  # TODO: Human review - SCALE interpretation issue
    # NONDIA as NONDIA,  # TODO: Human review - SCALE factor issue
    NONCVXU2 as NONCVXU2,
    NONCVXUN as NONCVXUN,
    NONDQUAR as NONDQUAR,
    NONMSQRT as NONMSQRT,
    OSBORNEA as OSBORNEA,
    # OSBORNEB as OSBORNEB,  # TODO: Human review - objective discrepancy
    PALMER1C as PALMER1C,
    PALMER1D as PALMER1D,
    PALMER2C as PALMER2C,
    PALMER3C as PALMER3C,
    PALMER4C as PALMER4C,
    PALMER5C as PALMER5C,
    PALMER5D as PALMER5D,
    PALMER6C as PALMER6C,
    PALMER7C as PALMER7C,
    PALMER8C as PALMER8C,
    # PENALTY1 as PENALTY1,  # TODO: Human review - minor numerical precision issues
    # PENALTY2 as PENALTY2,  # TODO: Human review - SCALE factor issue
    PENALTY3 as PENALTY3,
    POWER as POWER,
    POWERSUM as POWERSUM,
    # POWELLSG as POWELLSG,  # TODO: Human review - objective off by factor of 4.15
    PRICE3 as PRICE3,
    PRICE4 as PRICE4,
    QUARTC as QUARTC,
    ROSENBR as ROSENBR,
    ROSZMAN1LS as ROSZMAN1LS,
    S308 as S308,
    SISSER as SISSER,
    SPIN2LS as SPIN2LS,
    # SPINLS as SPINLS,  # TODO: Human review - gradient/Hessian issues
    # SPMSRTLS as SPMSRTLS,  # TODO: Human review - complex matrix multiplication
    TENFOLDTRLS as TENFOLDTRLS,
    # TOINTGOR as TOINTGOR,  # TODO: Human review - runtime test fails
    TOINTGSS as TOINTGSS,
    # TOINTPSP as TOINTPSP,  # TODO: Human review - gradient test fails
    # TQUARTIC as TQUARTIC,  # TODO: Human review - objective calculation incorrect
    TRIGON1 as TRIGON1,
    # TRIGON2 as TRIGON2,  # TODO: Human review - Hessian test fails
    unconstrained_minimisation_problems as unconstrained_minimisation_problems,
    VANDANMSLS as VANDANMSLS,
    VARDIM as VARDIM,
    # VAREIGVL as VAREIGVL,  # TODO: Human review - matrix computation discrepancy
    VESUVIALS as VESUVIALS,
    VESUVIOLS as VESUVIOLS,
    VESUVIOULS as VESUVIOULS,
    VIBRBEAM as VIBRBEAM,
    # WATSON as WATSON,  # TODO: Human review - Hessian computation issues
    WAYSEA1 as WAYSEA1,
    WAYSEA2 as WAYSEA2,
    WOODS as WOODS,
    YATP1CLS as YATP1CLS,
    YATP1LS as YATP1LS,
    YATP2CLS as YATP2CLS,
    # YATP2LS as YATP2LS,  # TODO: Human review - Hessian test failures
    ZANGWIL2 as ZANGWIL2,
)


problems_dict = {
    # "ACOPP14": ACOPP14(),  # TODO: needs human review - complex AC OPF formulation
    # "AIRPORT": AIRPORT(),  # TODO: Human review - constraints don't match pycutest
    # "ALLINITA": ALLINITA(),  # TODO: needs human review
    # "ALLINITC": ALLINITC(),  # Human review needed - dimension mismatch
    "ALSOTAME": ALSOTAME(),
    "TRO3X3": TRO3X3(),
    "TRO4X4": TRO4X4(),
    "TRO5X5": TRO5X5(),
    "TRO6X2": TRO6X2(),
    "TRO11X3": TRO11X3(),
    "TRO21X5": TRO21X5(),
    "TRO41X9": TRO41X9(),
    # "ANTWERP": ANTWERP(),  # TODO: needs human review
    "BIGGSC4": BIGGSC4(),
    "BOOTH": BOOTH(),
    "BURKEHAN": BURKEHAN(),
    "BYRDSPHR": BYRDSPHR(),
    "CANTILVR": CANTILVR(),
    "CB2": CB2(),
    "CB3": CB3(),
    "CHACONN1": CHACONN1(),
    "CHACONN2": CHACONN2(),
    # "CLNLBEAM": CLNLBEAM(),  # TODO: Dimension mismatch in constraints
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
    # "HS74": HS74(),  # TODO: Human review - constraint Jacobian issues
    # "HS75": HS75(),  # TODO: Human review - same issues as HS74
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
    # "HS109": HS109(),  # TODO: Human review needed - sign convention issues
    "HS110": HS110(),
    "LEVYMONT": LEVYMONT(),
    "LEVYMONT5": LEVYMONT5(),
    "LEVYMONT6": LEVYMONT6(),
    "LEVYMONT7": LEVYMONT7(),
    "LEVYMONT8": LEVYMONT8(),
    "LEVYMONT9": LEVYMONT9(),
    "LEVYMONT10": LEVYMONT10(),
    "LOGROS": LOGROS(),
    "HS111": HS111(),
    "HS112": HS112(),
    "HS113": HS113(),
    "HS114": HS114(),
    "HS116": HS116(),
    "HS117": HS117(),
    # "HS118": HS118(),  # TODO: Human review - constraint Jacobian ordering mismatch
    "HS119": HS119(),
    "HIMMELBC": HIMMELBC(),
    "HIMMELBD": HIMMELBD(),
    "HIMMELBE": HIMMELBE(),
    "LOOTSMA": LOOTSMA(),
    "MARATOS": MARATOS(),
    "MSS1": MSS1(),
    "MSS2": MSS2(),
    "MSS3": MSS3(),
    "ODFITS": ODFITS(),
    # "ORTHRDM2": ORTHRDM2(),  # TODO: Human review - gradient issues
    # "ORTHRDS2": ORTHRDS2(),  # TODO: Human review - gradient issues
    "ORTHRDS2C": ORTHRDS2C(),
    # "ORTHREGA": ORTHREGA(),  # TODO: Human review - complex formulation differences
    "ORTHREGB": ORTHREGB(),
    "ORTHREGC": ORTHREGC(),
    "ORTHREGD": ORTHREGD(),
    "ORTHREGE": ORTHREGE(),
    "ORTHREGF": ORTHREGF(),
    "ORTHRGDM": ORTHRGDM(),
    "ORTHRGDS": ORTHRGDS(),
    "PENTAGON": PENTAGON(),
    "POLAK1": POLAK1(),
    "POLAK2": POLAK2(),
    "POLAK3": POLAK3(),
    "POLAK4": POLAK4(),
    "POLAK5": POLAK5(),
    "POLAK6": POLAK6(),
    # "POLYGON": POLYGON(),  # TODO: Human review - fixed variable conventions
    "SIMPLLPA": SIMPLLPA(),
    "SIMPLLPB": SIMPLLPB(),
    "SIPOW1": SIPOW1(),
    "SIPOW2": SIPOW2(),
    # "SPINOP": SPINOP(),  # TODO: Human review - auxiliary variable constraint issues
    # "SPIN2OP": SPIN2OP(),  # TODO: Human review - constraint test failures
    # "SIPOW3": SIPOW3(),  # TODO: Human review - constraint formulation issues
    # "SIPOW4": SIPOW4(),  # TODO: Human review - constraint formulation issues
    # "VANDERM1": VANDERM1(),  # TODO: Human review - mixed constraint types
    # "VANDERM2": VANDERM2(),  # TODO: Human review - mixed constraint types
    # "VANDERM3": VANDERM3(),  # TODO: Human review - constraints mismatch
    # "VANDERM4": VANDERM4(),  # TODO: Human review - constraints mismatch
    "MAKELA1": MAKELA1(),
    "MAKELA2": MAKELA2(),
    "MAKELA3": MAKELA3(),
    "MAKELA4": MAKELA4(),
    # "HS70": HS70(),  # TODO: Human review - test failures
    # "HS84": HS84(),  # TODO: Human review - objective value discrepancy
    # TODO: TWIR problems need human review - complex trilinear constraint formulation
    # "TWIRISM1": TWIRISM1(),
    # "TWIRIMD1": TWIRIMD1(),
    # "TWIRIBG1": TWIRIBG1(),
    "ZECEVIC2": ZECEVIC2(),
    "ZECEVIC3": ZECEVIC3(),
    "ZECEVIC4": ZECEVIC4(),
    # "TRUSPYR1": TRUSPYR1(),  # TODO: Human review - complex constraint scaling issues
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
    # "LUKVLE9": LUKVLE9(),  # TODO: Human review needed - Jacobian issues
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
    # "LUKVLI9": LUKVLI9(),  # TODO: Human review needed - Jacobian issues
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
    # "AUG2D": AUG2D(),  # TODO: needs human review - edge variable structure
    "AVGASA": AVGASA(),
    "AVGASB": AVGASB(),
    # "DEGTRIDL": DEGTRIDL(),  # TODO: Human review - causes segfault
    # "AVION2": AVION2(),  # TODO: Human review - gradient discrepancies
    # "BA_L1LS": BA_L1LS(),  # TODO: BA_L family needs to be split into files
    # "BA_L1SPLS": BA_L1SPLS(),  # TODO: BA_L family needs human review
    "BARD": BARD(),
    "BDEXP": BDEXP(),
    "BDQRTIC": BDQRTIC(),
    "BEALE": BEALE(),
    "BENNETT5LS": BENNETT5LS(),
    "BIGGS3": BIGGS3(),
    "BIGGS5": BIGGS5(),
    "BIGGS6": BIGGS6(),
    "BOX": BOX(),
    "BOX2": BOX2(),
    "BOX3": BOX3(),
    "BOXBOD": BOXBOD(),
    "BOXBODLS": BOXBODLS(),
    # "BOXPOWER": BOXPOWER(),  # TODO: Human review - minor gradient discrepancy
    "BRANIN": BRANIN(),
    # "BRATU1D": BRATU1D(),  # TODO: Human review needed - see file
    # "BRKMCC": BRKMCC(),  # TODO: Human review - significant discrepancies
    "CAMEL6": CAMEL6(),
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
    # "CONCON": CONCON(),  # TODO: Removed - automatic derivative mismatches
    "COSHFUN": COSHFUN(),
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
    "CVXBQP1": CVXBQP1(),
    "CVXQP1": CVXQP1(),
    "CVXQP2": CVXQP2(),
    "CVXQP3": CVXQP3(),
    "CYCLOOCFLS": CYCLOOCFLS(),
    "DALLASS": DALLASS(),
    "DANIWOOD": DANIWOOD(),
    "DANIWOODLS": DANIWOODLS(),
    "DEGTRID": DEGTRID(),
    "DECONVC": DECONVC(),
    "DTOC1L": DTOC1L(),
    "DTOC1NA": DTOC1NA(),
    "DTOC1NB": DTOC1NB(),
    "DTOC1NC": DTOC1NC(),
    "DTOC1ND": DTOC1ND(),
    "DTOC2": DTOC2(),
    # "DTOC3": DTOC3(),  # Human review needed
    "DTOC4": DTOC4(),
    "DTOC5": DTOC5(),
    "DTOC6": DTOC6(),
    "DENSCHNA": DENSCHNA(),
    "DENSCHNB": DENSCHNB(),
    "DENSCHNC": DENSCHNC(),
    "DENSCHND": DENSCHND(),
    "DENSCHNE": DENSCHNE(),
    "DENSCHNF": DENSCHNF(),
    "DEVGLA1": DEVGLA1(),
    # "DIAMON3DLS": DIAMON3DLS(),  # TODO: Human review needed - see file
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
    "DRCAV1LQ": DRCAV1LQ(),
    "DRCAV2LQ": DRCAV2LQ(),
    "DRCAV3LQ": DRCAV3LQ(),
    "DRCAVTY1": DRCAVTY1(),
    "DRCAVTY2": DRCAVTY2(),
    "DRCAVTY3": DRCAVTY3(),
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
    "DGOSPEC": DGOSPEC(),
    "EXPLIN": EXPLIN(),
    "EXPLIN2": EXPLIN2(),
    "EXPQUAD": EXPQUAD(),
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
    "GIGOMEZ2": GIGOMEZ2(),
    "HAGER1": HAGER1(),
    "HAGER2": HAGER2(),
    # "HAGER3": HAGER3(),  # TODO: HAGER3 needs human review - marked for future import
    "HAGER4": HAGER4(),
    "GENROSE": GENROSE(),
    "GROWTHLS": GROWTHLS(),
    # "GULF": GULF(),  # TODO: Human review - issues reported by user
    "HAHN1LS": HAHN1LS(),
    "HAIRY": HAIRY(),
    "HART6": HART6(),
    "HATFLDA": HATFLDA(),
    "HATFLDB": HATFLDB(),
    "HATFLDC": HATFLDC(),
    "HATFLDD": HATFLDD(),
    "HATFLDE": HATFLDE(),
    "HATFLDFL": HATFLDFL(),
    "HATFLDFLS": HATFLDFLS(),
    # "HATFLDGLS": HATFLDGLS(),  # TODO: Known gradient/Hessian discrepancies
    # "HEART6LS": HEART6LS(),  # TODO: Human review - significant discrepancies
    # "HEART8LS": HEART8LS(),  # TODO: Human review - significant discrepancies
    "HELIX": HELIX(),
    # "HIELOW": HIELOW(),  # TODO: Human review - significant discrepancies
    "HILBERTA": HILBERTA(),
    "HILBERTB": HILBERTB(),
    # "HIMMELBB": HIMMELBB(),  # TODO: needs human review - Hessian issues
    "HIMMELBCLS": HIMMELBCLS(),
    # "HIMMELBF": HIMMELBF(),  # TODO: needs human review - Hessian issues
    "HIMMELBG": HIMMELBG(),
    "HIMMELBH": HIMMELBH(),
    # "HIMMELP1": HIMMELP1(),  # TODO: Human review needed - OBNL element issues
    # "HIMMELP2": HIMMELP2(),  # TODO: Human review needed - OBNL element issues
    # "HIMMELP3": HIMMELP3(),  # TODO: Human review needed - OBNL element issues
    # "HIMMELP4": HIMMELP4(),  # TODO: Human review needed - OBNL element issues
    # "HIMMELP5": HIMMELP5(),  # TODO: Human review needed - OBNL element issues
    # "HIMMELP6": HIMMELP6(),  # TODO: Human review needed - OBNL element issues
    "HUMPS": HUMPS(),
    "HYDROELL": HYDROELL(),
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
    # "MODBEALE": MODBEALE(),  # TODO: Human review - SCALE interpretation issue
    "MGH09LS": MGH09LS(),
    "MGH10LS": MGH10LS(),
    "MGH10SLS": MGH10SLS(),
    "MGH17LS": MGH17LS(),
    "MGH17SLS": MGH17SLS(),
    "MARATOSB": MARATOSB(),
    "MEXHAT": MEXHAT(),
    # "MOREBV": MOREBV(),  # TODO: Human review - minor gradient precision differences
    "NCVXBQP1": NCVXBQP1(),
    "NCVXBQP2": NCVXBQP2(),
    "NCVXBQP3": NCVXBQP3(),
    "NCVXQP1": NCVXQP1(),
    "NCVXQP2": NCVXQP2(),
    "NCVXQP3": NCVXQP3(),
    "NCVXQP4": NCVXQP4(),
    "NCVXQP5": NCVXQP5(),
    "NCVXQP6": NCVXQP6(),
    "NCVXQP7": NCVXQP7(),
    "NCVXQP8": NCVXQP8(),
    "NCVXQP9": NCVXQP9(),
    # "NONDIA": NONDIA(),  # TODO: Human review - SCALE factor issue
    "NONCVXU2": NONCVXU2(),
    "NONCVXUN": NONCVXUN(),
    "NONDQUAR": NONDQUAR(),
    "NONMSQRT": NONMSQRT(),
    "OSBORNEA": OSBORNEA(),
    # "OSBORNEB": OSBORNEB(),  # TODO: Human review - objective discrepancy
    "PALMER1C": PALMER1C(),
    "PALMER1D": PALMER1D(),
    "PALMER2C": PALMER2C(),
    "PALMER3C": PALMER3C(),
    "PALMER4C": PALMER4C(),
    "PALMER5C": PALMER5C(),
    "PALMER5D": PALMER5D(),
    "PALMER6C": PALMER6C(),
    "PALMER7C": PALMER7C(),
    "PALMER8C": PALMER8C(),
    # "PALMER4A": PALMER4A(),  # TODO: Fix Hessian issues
    "PALMER4E": PALMER4E(),
    # "PALMER5A": PALMER5A(),  # TODO: Fix Chebyshev polynomial calculation
    "PALMER5B": PALMER5B(),
    "PALMER6A": PALMER6A(),
    "PALMER6E": PALMER6E(),
    # "PALMER7A": PALMER7A(),  # TODO: Fix Hessian issues
    "PALMER7E": PALMER7E(),
    "PALMER8A": PALMER8A(),
    "PALMER8E": PALMER8E(),
    "PALMER2A": PALMER2A(),
    "PALMER2B": PALMER2B(),
    "PALMER2E": PALMER2E(),
    # "PENALTY1": PENALTY1(),  # TODO: Human review - minor numerical precision issues
    # "PENALTY2": PENALTY2(),  # TODO: Human review - SCALE factor issue
    "PENALTY3": PENALTY3(),
    "POWER": POWER(),
    "POWERSUM": POWERSUM(),
    # "POWELLSG": POWELLSG(),  # TODO: Human review - objective off by factor of 4.15
    "PRICE3": PRICE3(),
    "PRICE4": PRICE4(),
    "QUARTC": QUARTC(),
    "ROSENBR": ROSENBR(),
    "DIAGIQB": DIAGIQB(),
    "DIAGIQE": DIAGIQE(),
    "DIAGIQT": DIAGIQT(),
    "DIAGNQB": DIAGNQB(),
    "DIAGNQE": DIAGNQE(),
    "DIAGNQT": DIAGNQT(),
    "DIAGPQB": DIAGPQB(),
    "DIAGPQE": DIAGPQE(),
    "DIAGPQT": DIAGPQT(),
    "ROSZMAN1LS": ROSZMAN1LS(),
    "S308": S308(),
    "SISSER": SISSER(),
    # "SPINLS": SPINLS(),  # TODO: Human review - gradient/Hessian issues
    "SPIN2LS": SPIN2LS(),
    # "SPMSRTLS": SPMSRTLS(),  # TODO: Human review - complex matrix multiplication
    # "TENBARS4": TENBARS4(),  # TODO: Human review - pycutest Jacobian inconsistency
    "10FOLDTRLS": TENFOLDTRLS(),
    "POWELLBS": POWELLBS(),
    "POWELLSE": POWELLSE(),
    "POWELLSQ": POWELLSQ(),
    "PRICE4B": PRICE4B(),
    # "WATSON": WATSON(),  # TODO: Human review - Hessian computation issues
    "WAYSEA1": WAYSEA1(),
    "WAYSEA2": WAYSEA2(),
    "WOODS": WOODS(),
    "YATP1CLS": YATP1CLS(),
    "YATP1CNE": YATP1CNE(),
    "YATP1LS": YATP1LS(),
    "YATP1NE": YATP1NE(),
    "YATP2CLS": YATP2CLS(),
    # "YATP2CNE": YATP2CNE(),  # TODO: Human review - constraint ordering mismatch
    # "YATP2LS": YATP2LS(),  # TODO: Human review - Hessian test failures
    # "YATP2SQ": YATP2SQ(),  # TODO: Human review - constraint ordering mismatch
    "ZANGWIL2": ZANGWIL2(),
    "TRIGON1": TRIGON1(),
    "TRIGON1B": TRIGON1B(),
    "TRIGON1NE": TRIGON1NE(),
    # "TRIGON2": TRIGON2(),  # TODO: Human review - Hessian test fails
    # "TRIGON2B": TRIGON2B(),  # TODO: Human review - tiny Hessian discrepancies
    # "TRIGON2NE": TRIGON2NE(),  # TODO: Human review - Jacobian tolerance 1.26e-05
    "VANDANIUMS": VANDANIUMS(),
    "VARDIMNE": VARDIMNE(),
    "VESUVIA": VESUVIA(),
    "VESUVIO": VESUVIO(),
    "VESUVIOU": VESUVIOU(),
    "VIBRBEAMNE": VIBRBEAMNE(),
    # "TOINTGOR": TOINTGOR(),  # TODO: Human review - runtime test fails
    "TOINTGSS": TOINTGSS(),
    # "TORSIOND": TORSIOND(),  # TODO: Human review - objective mismatch
    # "TQUARTIC": TQUARTIC(),  # TODO: Human review - objective calculation incorrect
    "YAO": YAO(),
    "QPBAND": QPBAND(),
    "QPNBAND": QPNBAND(),
    # "QPNBLEND": QPNBLEND(),  # TODO: Human review - complex constraint matrix
    # "QPNBOEI1": QPNBOEI1(),  # TODO: Human review - Boeing routing constraints
    # "QPNBOEI2": QPNBOEI2(),  # TODO: Human review - Boeing routing constraints
    # "QPNSTAIR": QPNSTAIR(),  # TODO: Human review - complex constraint dimensions
    # "CHENHARK": CHENHARK(),  # TODO: Human review needed - see file
    "DEGDIAG": DEGDIAG(),
    "DUAL1": DUAL1(),
    "DUAL2": DUAL2(),
    "DUAL3": DUAL3(),
    "DUAL4": DUAL4(),
    "DUALC1": DUALC1(),
    "DUALC2": DUALC2(),
    "DUALC5": DUALC5(),
    "DUALC8": DUALC8(),
    "GOULDQP1": GOULDQP1(),
    "GOULDQP2": GOULDQP2(),
    "GOULDQP3": GOULDQP3(),
    "QUDLIN": QUDLIN(),
    "TAME": TAME(),
    "HATFLDH": HATFLDH(),
    "HS44NEW": HS44NEW(),
    "VANDANMSLS": VANDANMSLS(),
    "VARDIM": VARDIM(),
    # "VAREIGVL": VAREIGVL(),  # TODO: Human review - matrix computation discrepancy
    "VESUVIALS": VESUVIALS(),
    "VIBRBEAM": VIBRBEAM(),
    "VESUVIOLS": VESUVIOLS(),
    "VESUVIOULS": VESUVIOULS(),
    # "TOINTPSP": TOINTPSP(),  # TODO: Human review - gradient test fails
    "AIRCRFTA": AIRCRFTA(),
    "ARGAUSS": ARGAUSS(),
    "ARGLALE": ARGLALE(),
    "ARGLBLE": ARGLBLE(),
    "ARGLCLE": ARGLCLE(),
    "ARGTRIG": ARGTRIG(),
    "ARTIF": ARTIF(),
    # TODO: Human review needed - constraint dimension mismatch
    # "ARWHDNE": ARWHDNE(),
    "BARDNE": BARDNE(),
    "BDVALUES": BDVALUES(),
    # "BDQRTICNE": BDQRTICNE(),  # TODO: Human review needed
    "BEALENE": BEALENE(),
    "BENNETT5": BENNETT5(),
    "BIGGS6NE": BIGGS6NE(),
    "BOX3NE": BOX3NE(),
    # "BROWNALE": BROWNALE(),  # TODO: Human review needed - Jacobian precision issues
    "BROWNBSNE": BROWNBSNE(),
    "BROWNDENE": BROWNDENE(),
    "BRATU2DT": BRATU2DT(),
    "LEVYMONE9": LEVYMONE9(),
    # "BROYDN3D": BROYDN3D(),  # TODO: Human review needed - constraint values mismatch
    # "BROYDNBD": BROYDNBD(),  # TODO: Human review needed - systematic differences
    # "BRYBNDNE": BRYBNDNE(),  # TODO: Human review needed - constraint values mismatch
    "HYPCIR": HYPCIR(),
    "MSQRTA": MSQRTA(),
    "MSQRTB": MSQRTB(),
    # "CERI651A": CERI651A(),  # TODO: Human review - numerical overflow at ones vector
    # "CERI651B": CERI651B(),  # TODO: Human review - numerical overflow at ones vector
    # "CERI651C": CERI651C(),  # TODO: Human review - numerical overflow at ones vector
    # "CHAINWOONE": CHAINWOONE(),  # TODO: Human review - constraint values mismatch
    # "CHANNEL": CHANNEL(),  # TODO: Human review needed
    "CHEBYQADNE": CHEBYQADNE(),
    # "CHNRSBNE": CHNRSBNE(),  # TODO: Human review needed
    # "CHNRSNBMNE": CHNRSNBMNE(),  # TODO: Human review needed
    # "COATINGNE": COATINGNE(),  # TODO: Human review - formulation differences
    # "CUBENE": CUBENE(),  # TODO: Human review - constraint and Jacobian mismatch
    "CYCLIC3": CYCLIC3(),
    "CYCLIC3LS": CYCLIC3LS(),
    "CYCLOOCF": CYCLOOCF(),
    "CYCLOOCT": CYCLOOCT(),
    "CYCLOOCTLS": CYCLOOCTLS(),
    "DEGTRID2": DEGTRID2(),
    "DENSCHNBNE": DENSCHNBNE(),
    "DENSCHNCNE": DENSCHNCNE(),
    "DENSCHNDNE": DENSCHNDNE(),
    "DENSCHNENE": DENSCHNENE(),
    "DENSCHNFNE": DENSCHNFNE(),
    "DECONVBNE": DECONVBNE(),
    "DECONVNE": DECONVNE(),
    "DEVGLA1NE": DEVGLA1NE(),
    "DEVGLA2NE": DEVGLA2NE(),
    "EGGCRATENE": EGGCRATENE(),
    # "EIGENAU": EIGENAU(),  # TODO: Human review - non-standard initial values
    "ELATTAR": ELATTAR(),
    "ELATVIDUNE": ELATVIDUNE(),
    "ENGVAL2NE": ENGVAL2NE(),
    "ERRINRSMNE": ERRINRSMNE(),
    "ERRINROSNE": ERRINROSNE(),
    "EXP2NE": EXP2NE(),
    # "EXPFITA": EXPFITA(),  # TODO: Human review - fundamental formulation differences
    # "EXPFITB": EXPFITB(),  # TODO: Human review - fundamental formulation differences
    # "EXPFITC": EXPFITC(),  # TODO: Human review - fundamental formulation differences
    "EXPFITNE": EXPFITNE(),
    "EXTROSNBNE": EXTROSNBNE(),
    # "FLOSP2HH": FLOSP2HH(),  # TODO: Human review needed - NQR constraint handling
    # "FLOSP2HL": FLOSP2HL(),  # TODO: Human review needed - NQR constraint handling
    # "FLOSP2HM": FLOSP2HM(),  # TODO: Human review needed - NQR constraint handling
    # "FLOSP2TH": FLOSP2TH(),  # TODO: Human review needed - NQR constraint handling
    # "FLOSP2TL": FLOSP2TL(),  # TODO: Human review needed - NQR constraint handling
    # "FLOSP2TM": FLOSP2TM(),  # TODO: Human review needed - NQR constraint handling
    "FREURONE": FREURONE(),
    "GENROSEBNE": GENROSEBNE(),
    "GENROSENE": GENROSENE(),
    "GOTTFR": GOTTFR(),
    "GULFNE": GULFNE(),
    "HATFLDANE": HATFLDANE(),
    "HATFLDBNE": HATFLDBNE(),
    "HATFLDCNE": HATFLDCNE(),
    "HATFLDDNE": HATFLDDNE(),
    # "HATFLDENE": HATFLDENE(),  # TODO: Human review - sign convention issues
    "HATFLDF": HATFLDF(),
    "HATFLDFLNE": HATFLDFLNE(),
    "HATFLDG": HATFLDG(),
    "HELIXNE": HELIXNE(),
    "HIMMELBA": HIMMELBA(),
    "HIMMELBFNE": HIMMELBFNE(),
    "HS1NE": HS1NE(),
    "HS2NE": HS2NE(),
    "HS25NE": HS25NE(),
    # "HYDCAR6": HYDCAR6(),  # TODO: Human review needed
    "INTEQNE": INTEQNE(),
    "JENSMPNE": JENSMPNE(),
    "JUDGENE": JUDGENE(),
    "KOEBHELBNE": KOEBHELBNE(),
    "KOWOSBNE": KOWOSBNE(),
    "LEVYMONE": LEVYMONE(),
    "LEVYMONE5": LEVYMONE5(),
    "LEVYMONE6": LEVYMONE6(),
    "LEVYMONE7": LEVYMONE7(),
    "LEVYMONE8": LEVYMONE8(),
    "LEVYMONE10": LEVYMONE10(),
    "LIARWHDNE": LIARWHDNE(),
    # "LINVERSENE": LINVERSENE(),  # TODO: Human review - incomplete implementation
    "LUKSAN11": LUKSAN11(),
    "LUKSAN12": LUKSAN12(),
    "LUKSAN13": LUKSAN13(),
    "LUKSAN14": LUKSAN14(),
    "LUKSAN15": LUKSAN15(),
    "LUKSAN16": LUKSAN16(),
    "LUKSAN17": LUKSAN17(),
    "LUKSAN21": LUKSAN21(),
    "LUKSAN22": LUKSAN22(),
    "LUKSAN11LS": LUKSAN11LS(),
    "LUKSAN12LS": LUKSAN12LS(),
    "LUKSAN13LS": LUKSAN13LS(),
    "LUKSAN14LS": LUKSAN14LS(),
    "LUKSAN15LS": LUKSAN15LS(),
    "LUKSAN16LS": LUKSAN16LS(),
    "LUKSAN17LS": LUKSAN17LS(),
    "LUKSAN21LS": LUKSAN21LS(),
    # "LUKSAN22LS": LUKSAN22LS(),  # TODO: Human review needed - gradient issues
    "MANCINONE": MANCINONE(),
    "MEYER3NE": MEYER3NE(),
    "MGH09": MGH09(),
    "MGH10": MGH10(),
    "MGH10S": MGH10S(),
    "MGH17": MGH17(),
    "MGH17S": MGH17S(),
    "MISRA1D": MISRA1D(),
    # "MODBEALENE": MODBEALENE(),  # TODO: Human review - constraint ordering issues
    # "MOREBVNE": MOREBVNE(),  # TODO: Human review - SIF file bug on line 64
    # "MUONSINE": MUONSINE(),  # TODO: Human review - hardcoded data values
    "NONDIANE": NONDIANE(),
    # "NONMSQRTNE": NONMSQRTNE(),  # TODO: Human review - element structure
    "NONSCOMPNE": NONSCOMPNE(),
    "OSCIGRNE": OSCIGRNE(),
    "OSCIPANE": OSCIPANE(),
    "PALMER1ANE": PALMER1ANE(),
    "PALMER1BNE": PALMER1BNE(),
    "PALMER1ENE": PALMER1ENE(),
    "PALMER1NE": PALMER1NE(),
    "PALMER2ANE": PALMER2ANE(),
    "PALMER2BNE": PALMER2BNE(),
    "PALMER2ENE": PALMER2ENE(),
    "PALMER2NE": PALMER2NE(),
    "PALMER1": PALMER1(),
    "PALMER1A": PALMER1A(),
    # "PALMER1B": PALMER1B(),  # TODO: Fix Hessian issues
    # "PALMER1E": PALMER1E(),  # TODO: Fix Hessian issues
    "PALMER2": PALMER2(),
    "PALMER3": PALMER3(),
    "PALMER3A": PALMER3A(),
    "PALMER3B": PALMER3B(),
    "PALMER3E": PALMER3E(),
    "PALMER4": PALMER4(),
    "PALMER4B": PALMER4B(),
    "PALMER3ANE": PALMER3ANE(),
    "PALMER3BNE": PALMER3BNE(),
    "PALMER3ENE": PALMER3ENE(),
    "PALMER3NE": PALMER3NE(),
    "PALMER4ANE": PALMER4ANE(),
    "PALMER4BNE": PALMER4BNE(),
    "PALMER4ENE": PALMER4ENE(),
    "PALMER4NE": PALMER4NE(),
    # "PALMER5ANE": PALMER5ANE(),  # TODO: Fix Chebyshev polynomial calculation
    "PALMER5BNE": PALMER5BNE(),
    # "PALMER5ENE": PALMER5ENE(),  # TODO: Human review - numerical precision
    "PALMER6ANE": PALMER6ANE(),
    "PALMER6ENE": PALMER6ENE(),
    "PALMER7ANE": PALMER7ANE(),
    "PALMER7ENE": PALMER7ENE(),
    "PALMER8ANE": PALMER8ANE(),
    "PALMER8ENE": PALMER8ENE(),
    "PFIT1": PFIT1(),
    "PFIT2": PFIT2(),
    "PFIT3": PFIT3(),
    "PFIT4": PFIT4(),
    "PFIT1LS": PFIT1LS(),
    "PFIT2LS": PFIT2LS(),
    "PFIT3LS": PFIT3LS(),
    "PFIT4LS": PFIT4LS(),
    "POWERSUMNE": POWERSUMNE(),
    # "RES": RES(),  # TODO: Human review needed - mixed constraint types
    "SANTA": SANTA(),
    "SINVALNE": SINVALNE(),
    "SPIN": SPIN(),
    # "SPIN2": SPIN2(),  # TODO: Human review - constraint test failures
    # "SSBRYBNDNE": SSBRYBNDNE(),  # TODO: Human review - complex element structure
    # "STEENBRB": STEENBRB(),  # TODO: Human review - gradient test failing
    "10FOLDTR": TENFOLDTR(),
}


def get_problem(name: str):
    return problems_dict.get(name, None)  # TODO: try except with nicer error message


# Gather problems into categories
# Note: While mathematically bounds are a type of constraint, we keep bounded problems
# separate from constrained problems in our API for cleaner problem categorization
# and to maintain clear inheritance hierarchies.
#
# Bounded problems include:
# - bounded minimisation problems (bounds only)
# - bounded quadratic problems (quadratic objective with bounds only)
#
# Constrained problems include:
# - constrained minimisation problems (equality/inequality constraints, may have bounds)
# - constrained quadratic problems (quadratic objective with constraints)

# Add bounded quadratic problems to bounded minimisation problems
bounded_minimisation_problems += bounded_quadratic_problems

# Add constrained quadratic problems to constrained minimisation problems
constrained_minimisation_problems += constrained_quadratic_problems

# Combine all problem categories
problems = (
    unconstrained_minimisation_problems
    + bounded_minimisation_problems
    + constrained_minimisation_problems
    + nonlinear_equations_problems
)
