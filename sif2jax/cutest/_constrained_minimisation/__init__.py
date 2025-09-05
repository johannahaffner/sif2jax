# TODO: ACOPP14 needs human review - complex AC OPF formulation
# from .acopp14 import ACOPP14 as ACOPP14
# TODO: AIRPORT needs human review - constraint values don't match pycutest
# from .airport import AIRPORT as AIRPORT

# TODO: ALLINITA needs human review - L2 group type interpretation issues
# from .allinita import ALLINITA as ALLINITA
# TODO: ALLINITC needs human review - dimension mismatch with pycutest (fixed variables)
# from .allinitc import ALLINITC as ALLINITC
from .aljazzaf import ALJAZZAF as ALJAZZAF
from .alsotame import ALSOTAME as ALSOTAME

# TODO: ANTWERP needs human review - initial value calculation issues
# from .antwerp import ANTWERP as ANTWERP
# TODO: AUG2D needs human review - edge variable structure
# from .aug2d import AUG2D as AUG2D
from .avgasa import AVGASA as AVGASA
from .avgasb import AVGASB as AVGASB

# from .avion2 import AVION2 as AVION2  # TODO: Human review - gradient discrepancies
# from .bdry2 import BDRY2 as BDRY2  # TODO: Human review - objective values don't match
# from .bigbank import BIGBANK as BIGBANK  # TODO: Human review - objective mismatch
from .biggsc4 import BIGGSC4 as BIGGSC4
from .bt1 import BT1 as BT1
from .bt2 import BT2 as BT2
from .bt3 import BT3 as BT3
from .bt4 import BT4 as BT4
from .bt5 import BT5 as BT5
from .bt6 import BT6 as BT6
from .bt7 import BT7 as BT7
from .bt8 import BT8 as BT8
from .bt9 import BT9 as BT9
from .bt10 import BT10 as BT10
from .bt11 import BT11 as BT11
from .bt12 import BT12 as BT12
from .bt13 import BT13 as BT13
from .burkehan import BURKEHAN as BURKEHAN
from .byrdsphr import BYRDSPHR as BYRDSPHR
from .cantilvr import CANTILVR as CANTILVR
from .cb2 import CB2 as CB2
from .cb3 import CB3 as CB3
from .chaconn1 import CHACONN1 as CHACONN1
from .chaconn2 import CHACONN2 as CHACONN2

# from .chardis1 import CHARDIS1 as CHARDIS1  # TODO: Human review needed
# from .chardis12 import CHARDIS12 as CHARDIS12  # TODO: Human review needed
# from .cresc4 import CRESC4 as CRESC4  # TODO: Human review - complex crescent area
# TODO: CLNLBEAM needs fixing - dimension mismatch in constraints
# from .clnlbeam import CLNLBEAM as CLNLBEAM
from .cleuven2 import CLEUVEN2 as CLEUVEN2
from .cleuven3 import CLEUVEN3 as CLEUVEN3
from .cleuven4 import CLEUVEN4 as CLEUVEN4
from .cleuven5 import CLEUVEN5 as CLEUVEN5
from .cleuven6 import CLEUVEN6 as CLEUVEN6
from .cleuven7 import CLEUVEN7 as CLEUVEN7

# TODO: CONCON - removed from API due to automatic derivative mismatches with pycutest
# from .concon import CONCON as CONCON
from .coshfun import COSHFUN as COSHFUN
from .csfi1 import CSFI1 as CSFI1
from .csfi2 import CSFI2 as CSFI2
from .dallass import DALLASS as DALLASS
from .deconvc import DECONVC as DECONVC
from .degenlpa import DEGENLPA as DEGENLPA
from .degenlpb import DEGENLPB as DEGENLPB
from .degenqp import DEGENQP as DEGENQP
from .degenqpc import DEGENQPC as DEGENQPC
from .dembo7 import DEMBO7 as DEMBO7
from .demymalo import DEMYMALO as DEMYMALO
from .dipigri import DIPIGRI as DIPIGRI

# from .dittert import DITTERT as DITTERT  # TODO: Human review needed
# TODO: DEGTRIDL needs human review - causes segfault despite correct implementation
# from .degtridl import DEGTRIDL as DEGTRIDL
from .dtoc1l import DTOC1L as DTOC1L
from .dtoc1na import DTOC1NA as DTOC1NA
from .dtoc1nb import DTOC1NB as DTOC1NB
from .dtoc1nc import DTOC1NC as DTOC1NC
from .dtoc1nd import DTOC1ND as DTOC1ND
from .dtoc2 import DTOC2 as DTOC2

# TODO: DTOC3 needs human review - X(1498) and X(1499) treated as 0 by pycutest
# from .dtoc3 import DTOC3 as DTOC3
from .dtoc4 import DTOC4 as DTOC4

# TODO: Human review needed - same constraint issues as EIGENA
# from .eigenaco import EIGENACO as EIGENACO
from .dtoc5 import DTOC5 as DTOC5
from .dtoc6 import DTOC6 as DTOC6
from .elattar import ELATTAR as ELATTAR
from .elec import ELEC as ELEC
from .expfita import EXPFITA as EXPFITA
from .expfitb import EXPFITB as EXPFITB
from .expfitc import EXPFITC as EXPFITC

# TODO: FCCU needs human review - objective value discrepancies
# from .fccu import FCCU as FCCU
# TODO: FEEDLOC needs human review - constraint dimension mismatch
# from .feedloc import FEEDLOC as FEEDLOC
from .fletcher import FLETCHER as FLETCHER
from .flt import FLT as FLT
from .gigomez2 import GIGOMEZ2 as GIGOMEZ2
from .hadamard import HADAMARD as HADAMARD

# TODO: Human review needed - parameter/dimension mismatch with pycutest
# JAX implementation uses correct SIF parameters (N=10 → 21 variables)
# But pycutest expects 5001 variables, suggesting different problem variant
# Issue persists even with drop_fixed_variables=False
from .hager1 import HAGER1 as HAGER1
from .hager2 import HAGER2 as HAGER2

# TODO: HAGER3 needs human review - marked for future import
# from .hager3 import HAGER3 as HAGER3
from .hager4 import HAGER4 as HAGER4

# TODO: Human review needed - complex SIF structure
from .haifal import HAIFAL as HAIFAL
from .haifam import (
    HAIFAM as HAIFAM,
)
from .haifas import HAIFAS as HAIFAS
from .himmelbc import HIMMELBC as HIMMELBC
from .himmelbd import HIMMELBD as HIMMELBD
from .himmelbe import HIMMELBE as HIMMELBE

# from .himmelp2 import HIMMELP2 as HIMMELP2  # TODO: Human review - OBNL issues
# from .himmelp3 import HIMMELP3 as HIMMELP3  # TODO: Human review - OBNL issues
# from .himmelp4 import HIMMELP4 as HIMMELP4  # TODO: Human review - OBNL issues
# from .himmelp5 import HIMMELP5 as HIMMELP5  # TODO: Human review - OBNL issues
# from .himmelp6 import HIMMELP6 as HIMMELP6  # TODO: Human review - OBNL issues
from .hs6 import HS6 as HS6
from .hs7 import HS7 as HS7
from .hs8 import HS8 as HS8
from .hs9 import HS9 as HS9
from .hs10 import HS10 as HS10
from .hs11 import HS11 as HS11
from .hs12 import HS12 as HS12
from .hs13 import HS13 as HS13
from .hs14 import HS14 as HS14
from .hs15 import HS15 as HS15
from .hs16 import HS16 as HS16
from .hs17 import HS17 as HS17
from .hs18 import HS18 as HS18
from .hs19 import HS19 as HS19
from .hs20 import HS20 as HS20
from .hs21 import HS21 as HS21
from .hs22 import HS22 as HS22
from .hs23 import HS23 as HS23
from .hs24 import HS24 as HS24
from .hs26 import HS26 as HS26
from .hs27 import HS27 as HS27
from .hs28 import HS28 as HS28
from .hs29 import HS29 as HS29
from .hs30 import HS30 as HS30
from .hs31 import HS31 as HS31
from .hs32 import HS32 as HS32
from .hs33 import HS33 as HS33
from .hs34 import HS34 as HS34
from .hs35 import HS35 as HS35
from .hs36 import HS36 as HS36
from .hs37 import HS37 as HS37
from .hs39 import HS39 as HS39
from .hs40 import HS40 as HS40
from .hs41 import HS41 as HS41
from .hs42 import HS42 as HS42
from .hs43 import HS43 as HS43
from .hs44 import HS44 as HS44
from .hs46 import HS46 as HS46
from .hs47 import HS47 as HS47
from .hs48 import HS48 as HS48
from .hs49 import HS49 as HS49
from .hs50 import HS50 as HS50
from .hs51 import HS51 as HS51
from .hs52 import HS52 as HS52
from .hs53 import HS53 as HS53
from .hs54 import HS54 as HS54
from .hs55 import HS55 as HS55
from .hs56 import HS56 as HS56
from .hs57 import HS57 as HS57

# from .hs59 import HS59 as HS59  # TODO: Human review - objective function discrepancy
from .hs60 import HS60 as HS60
from .hs61 import HS61 as HS61
from .hs62 import HS62 as HS62
from .hs63 import HS63 as HS63
from .hs64 import HS64 as HS64
from .hs65 import HS65 as HS65
from .hs66 import HS66 as HS66

# from .hs67 import HS67 as HS67  # TODO: Human review - several discrepancies
from .hs68 import HS68 as HS68
from .hs69 import HS69 as HS69

# TODO: HS70 needs human review - test failures
# from .hs70 import HS70 as HS70
from .hs71 import HS71 as HS71
from .hs72 import HS72 as HS72
from .hs73 import HS73 as HS73

# TODO: HS74 needs human review - constraint Jacobian values differ by large factors
# from .hs74 import HS74 as HS74
# TODO: HS75 needs human review - same issues as HS74
# from .hs75 import HS75 as HS75
from .hs76 import HS76 as HS76
from .hs77 import HS77 as HS77
from .hs78 import HS78 as HS78
from .hs79 import HS79 as HS79
from .hs80 import HS80 as HS80
from .hs81 import HS81 as HS81
from .hs83 import HS83 as HS83

# TODO: HS84 needs human review - objective value discrepancy (~2%)
# from .hs84 import HS84 as HS84
from .hs93 import HS93 as HS93

# from .hs99 import HS99 as HS99  # TODO: Needs human review - complex recursion
from .hs100 import HS100 as HS100
from .hs101 import HS101 as HS101
from .hs102 import HS102 as HS102
from .hs103 import HS103 as HS103
from .hs104 import HS104 as HS104
from .hs105 import HS105 as HS105
from .hs106 import HS106 as HS106
from .hs107 import HS107 as HS107
from .hs108 import HS108 as HS108

# from .hs109 import HS109 as HS109  # TODO: Human review - sign convention issues
from .hs111 import HS111 as HS111
from .hs112 import HS112 as HS112
from .hs113 import HS113 as HS113
from .hs114 import HS114 as HS114
from .hs116 import HS116 as HS116
from .hs117 import HS117 as HS117

# TODO: HS118 needs human review - constraint Jacobian ordering mismatch
# from .hs118 import HS118 as HS118
from .hs119 import HS119 as HS119
from .hydroell import HYDROELL as HYDROELL

# TODO: JANNSON3 needs human review - Jacobian tests hang due to computational cost
# from .jannson3 import JANNSON3 as JANNSON3
from .jannson4 import JANNSON4 as JANNSON4

# TODO: KISSING needs human review - runtime issue (5.37x, slightly over 5x threshold)
# from .kissing import KISSING as KISSING
# from .kissing2 import KISSING2 as KISSING2  # TODO: Human review needed
# TODO: Human review needed - KIWCRESC constraint values differ by 2.0 from pycutest
# from .kiwcresc import KIWCRESC as KIWCRESC
# TODO: Human review needed - KSIP needs vectorization, dtype promotion errors with JAX
# from .ksip import KSIP as KSIP
# TODO: Human review needed - marked for human review
# from .leuven1 import LEUVEN1 as LEUVEN1
from .liswet1 import LISWET1 as LISWET1
from .liswet2 import LISWET2 as LISWET2
from .liswet3 import LISWET3 as LISWET3
from .liswet4 import LISWET4 as LISWET4
from .liswet5 import LISWET5 as LISWET5
from .liswet6 import LISWET6 as LISWET6
from .liswet7 import LISWET7 as LISWET7
from .liswet8 import LISWET8 as LISWET8
from .liswet9 import LISWET9 as LISWET9
from .liswet10 import LISWET10 as LISWET10
from .liswet11 import LISWET11 as LISWET11
from .liswet12 import LISWET12 as LISWET12
from .lootsma import LOOTSMA as LOOTSMA
from .lukvle1 import LUKVLE1 as LUKVLE1

# from .lukvle2 import LUKVLE2 as LUKVLE2
from .lukvle3 import LUKVLE3 as LUKVLE3

# from .lukvle4 import LUKVLE4 as LUKVLE4  # Use LUKVLE4C instead
# from .lukvle4c import LUKVLE4C as LUKVLE4C
from .lukvle5 import LUKVLE5 as LUKVLE5
from .lukvle6 import LUKVLE6 as LUKVLE6
from .lukvle7 import LUKVLE7 as LUKVLE7
from .lukvle8 import LUKVLE8 as LUKVLE8

# from .lukvle9 import LUKVLE9 as LUKVLE9  # TODO: Human review needed - Jacobian issues
from .lukvle10 import LUKVLE10 as LUKVLE10
from .lukvle11 import LUKVLE11 as LUKVLE11

# from .lukvle12 import LUKVLE12 as LUKVLE12  # Has constraint function inconsistencies
from .lukvle13 import LUKVLE13 as LUKVLE13
from .lukvle14 import LUKVLE14 as LUKVLE14
from .lukvle15 import LUKVLE15 as LUKVLE15
from .lukvle16 import LUKVLE16 as LUKVLE16
from .lukvle17 import LUKVLE17 as LUKVLE17
from .lukvle18 import LUKVLE18 as LUKVLE18
from .lukvli1 import LUKVLI1 as LUKVLI1

# from .lukvli2 import LUKVLI2 as LUKVLI2
from .lukvli3 import LUKVLI3 as LUKVLI3

# from .lukvli4 import LUKVLI4 as LUKVLI4  # Use LUKVLI4C instead
# from .lukvli4c import LUKVLI4C as LUKVLI4C
from .lukvli5 import LUKVLI5 as LUKVLI5
from .lukvli6 import LUKVLI6 as LUKVLI6
from .lukvli7 import LUKVLI7 as LUKVLI7
from .lukvli8 import LUKVLI8 as LUKVLI8

# from .lukvli9 import LUKVLI9 as LUKVLI9  # TODO: Human review needed - Jacobian issues
from .lukvli10 import LUKVLI10 as LUKVLI10
from .lukvli11 import LUKVLI11 as LUKVLI11

# from .lukvli12 import LUKVLI12 as LUKVLI12  # Has constraint function inconsistencies
from .lukvli13 import LUKVLI13 as LUKVLI13
from .lukvli14 import LUKVLI14 as LUKVLI14
from .lukvli15 import LUKVLI15 as LUKVLI15
from .lukvli16 import LUKVLI16 as LUKVLI16
from .lukvli17 import LUKVLI17 as LUKVLI17
from .lukvli18 import LUKVLI18 as LUKVLI18
from .makela1 import MAKELA1 as MAKELA1
from .makela2 import MAKELA2 as MAKELA2
from .makela3 import MAKELA3 as MAKELA3
from .makela4 import MAKELA4 as MAKELA4
from .maratos import MARATOS as MARATOS
from .mss1 import MSS1 as MSS1
from .mss2 import MSS2 as MSS2
from .mss3 import MSS3 as MSS3
from .odfits import ODFITS as ODFITS
from .oet1 import OET1 as OET1
from .oet2 import OET2 as OET2
from .oet3 import OET3 as OET3
from .oet4 import OET4 as OET4
from .oet5 import OET5 as OET5
from .oet6 import OET6 as OET6
from .oet7 import OET7 as OET7
from .optcdeg2 import OPTCDEG2 as OPTCDEG2
from .optcdeg3 import OPTCDEG3 as OPTCDEG3
from .optcntrl import OPTCNTRL as OPTCNTRL
from .optctrl3 import OPTCTRL3 as OPTCTRL3
from .optctrl6 import OPTCTRL6 as OPTCTRL6
from .optmass import OPTMASS as OPTMASS
from .optprloc import OPTPRLOC as OPTPRLOC

# from .orthrdm2 import ORTHRDM2 as ORTHRDM2  # TODO: Human review - singular Jacobian
# from .orthrds2 import ORTHRDS2 as ORTHRDS2  # TODO: Human review - singular Jacobian
from .orthrds2c import ORTHRDS2C as ORTHRDS2C

# from .orthrega import ORTHREGA as ORTHREGA  # TODO: Human review - formulation diffs
from .orthregb import ORTHREGB as ORTHREGB
from .orthregc import ORTHREGC as ORTHREGC
from .orthregd import ORTHREGD as ORTHREGD
from .orthrege import ORTHREGE as ORTHREGE
from .orthregf import ORTHREGF as ORTHREGF
from .orthrgdm import ORTHRGDM as ORTHRGDM
from .orthrgds import ORTHRGDS as ORTHRGDS
from .pentagon import PENTAGON as PENTAGON
from .polak1 import POLAK1 as POLAK1
from .polak2 import POLAK2 as POLAK2
from .polak3 import POLAK3 as POLAK3
from .polak4 import POLAK4 as POLAK4
from .polak5 import POLAK5 as POLAK5
from .polak6 import POLAK6 as POLAK6

# from .polygon import POLYGON as POLYGON  # TODO: Human review - sign conventions
from .reading1 import READING1 as READING1
from .reading2 import READING2 as READING2
from .reading3 import READING3 as READING3
from .reading4 import READING4 as READING4
from .reading5 import READING5 as READING5

# from .reading6 import READING6 as READING6  # TODO: Human review needed
# Note: READING7 and READING8 exist but are not implemented due to a CUTEst bug:
# the starting point is the solution too
from .reading9 import READING9 as READING9

# from .rdw2d51f import RDW2D51F as RDW2D51F  # TODO: Human review needed
from .rosepetal import ROSEPETAL as ROSEPETAL
from .rosepetal2 import ROSEPETAL2 as ROSEPETAL2
from .simpllpa import SIMPLLPA as SIMPLLPA
from .simpllpb import SIMPLLPB as SIMPLLPB

# from .sinrosnb import SINROSNB as SINROSNB  # TODO: Human review - scaling issues
from .sipow1 import SIPOW1 as SIPOW1
from .sipow2 import SIPOW2 as SIPOW2

# TODO: Human review needed - constraint test failures (vectorized but sign issues)
# from .spin2op import SPIN2OP as SPIN2OP
# TODO: Human review needed - constraint issues with auxiliary variables
# from .spinop import SPINOP as SPINOP
# TODO: STEENBRB needs human review - gradient test failing
# from .steenbrb import STEENBRB as STEENBRB
# TODO: SIPOW3 needs human review - constraint formulation issues
# from .sipow3 import SIPOW3 as SIPOW3
# TODO: SIPOW4 needs human review - constraint formulation issues
# from .sipow4 import SIPOW4 as SIPOW4
# TODO: TAX13322 needs human review - complex objective structure, off by ~80x
# from .tax13322 import TAX13322 as TAX13322
from .tenbars1 import TENBARS1 as TENBARS1
from .tenbars2 import TENBARS2 as TENBARS2
from .tenbars3 import TENBARS3 as TENBARS3
from .tenbars4 import TENBARS4 as TENBARS4
from .trainf import TRAINF as TRAINF

# TODO: Human review needed - marked for human review
# from .trainh import TRAINH as TRAINH
from .tro3x3 import TRO3X3 as TRO3X3
from .tro4x4 import TRO4X4 as TRO4X4
from .tro5x5 import TRO5X5 as TRO5X5
from .tro6x2 import TRO6X2 as TRO6X2
from .tro11x3 import TRO11X3 as TRO11X3
from .tro21x5 import TRO21X5 as TRO21X5
from .tro41x9 import TRO41X9 as TRO41X9

# TODO: TRUSPYR1 needs human review - complex constraint scaling issues
# from .truspyr1 import TRUSPYR1 as TRUSPYR1
# TODO: TRUSPYR2 needs human review - test requested to be removed
# from .truspyr2 import TRUSPYR2 as TRUSPYR2
# from .vanderm3 import VANDERM3 as VANDERM3  # TODO: Human review needed
# from .vanderm4 import VANDERM4 as VANDERM4  # TODO: Human review needed
# TODO: TWIR problems need human review - complex trilinear constraint formulation
# from .twirism1 import TWIRISM1 as TWIRISM1
# from .twirimd1 import TWIRIMD1 as TWIRIMD1
# from .twiribg1 import TWIRIBG1 as TWIRIBG1
from .zecevic2 import ZECEVIC2 as ZECEVIC2
from .zecevic3 import ZECEVIC3 as ZECEVIC3
from .zecevic4 import ZECEVIC4 as ZECEVIC4


constrained_minimisation_problems = (
    # ACOPP14(),  # TODO: needs human review - complex AC OPF formulation
    # AIRPORT(),  # TODO: Human review - constraint values don't match pycutest
    # ALLINITA(),  # TODO: needs human review - L2 group type interpretation
    # ALLINITC(),  # Human review needed - dimension mismatch
    ALJAZZAF(),
    ALSOTAME(),
    TRO3X3(),
    TRO4X4(),
    TRO5X5(),
    TRO6X2(),
    TRO11X3(),
    TRO21X5(),
    TRO41X9(),
    TRAINF(),
    # TRAINH(),  # TODO: Human review needed
    # ANTWERP(),  # TODO: needs human review - initial value calculation
    AVGASA(),
    AVGASB(),
    # DEGTRIDL(),  # TODO: Human review - causes segfault despite correct implementation
    # AVION2(),  # TODO: Human review - gradient discrepancies
    # BDRY2(),  # TODO: Human review - objective values don't match pycutest
    # BIGBANK(),  # TODO: Human review - objective values don't match pycutest
    BIGGSC4(),
    BURKEHAN(),
    BYRDSPHR(),
    CANTILVR(),
    CB2(),
    CB3(),
    CHACONN1(),
    CHACONN2(),
    # CHARDIS1(),  # TODO: Human review needed
    # CHARDIS12(),  # TODO: Human review needed
    # CLNLBEAM(),  # TODO: Dimension mismatch in constraints
    CLEUVEN2(),
    CLEUVEN3(),
    CLEUVEN4(),
    CLEUVEN5(),
    CLEUVEN6(),
    CLEUVEN7(),
    # CONCON(),  # TODO: Removed - automatic derivative mismatches
    COSHFUN(),
    # CRESC4(),  # TODO: Human review - complex crescent area formula
    CSFI1(),
    CSFI2(),
    DALLASS(),
    DECONVC(),
    DEGENLPA(),
    DEGENLPB(),
    DEGENQP(),
    DEGENQPC(),
    DEMBO7(),
    DEMYMALO(),
    DIPIGRI(),
    # DITTERT(),  # TODO: Human review needed
    DTOC1L(),
    DTOC1NA(),
    DTOC1NB(),
    DTOC1NC(),
    DTOC1ND(),
    DTOC2(),
    # DTOC3(),  # Human review needed
    DTOC4(),
    DTOC5(),
    DTOC6(),
    # EIGENACO(),  # TODO: Human review needed - same constraint issues as EIGENA
    ELATTAR(),
    ELEC(),
    # EXPFITA(),  # TODO: Human review - fundamental formulation differences
    # EXPFITB(),  # TODO: Human review - fundamental formulation differences
    # EXPFITC(),  # TODO: Human review - fundamental formulation differences
    # FCCU(),  # TODO: FCCU needs human review - objective value discrepancies
    # FEEDLOC(),  # TODO: FEEDLOC needs human review - constraint dimension mismatch
    FLETCHER(),
    FLT(),
    GIGOMEZ2(),
    HADAMARD(),
    HAGER1(),
    HAGER2(),
    # HAGER3(),  # TODO: Human review needed - marked for future import
    HAGER4(),
    HAIFAS(),
    HAIFAM(),  # TODO: Human review needed - complex SIF structure
    HAIFAL(),
    HS6(),
    HS7(),
    HS8(),
    HS9(),
    HS10(),
    HS11(),
    HS12(),
    HS13(),
    HS14(),
    HS15(),
    HS16(),
    HS17(),
    HS18(),
    HS19(),
    HS20(),
    HS21(),
    HS22(),
    HS23(),
    HS24(),
    HS26(),
    HS27(),
    HS28(),
    HS29(),
    HS30(),
    HS31(),
    HS32(),
    HS33(),
    HS34(),
    HS35(),
    HS36(),
    HS37(),
    HS39(),
    HS40(),
    HS41(),
    HS42(),
    HS43(),
    HS44(),
    HS46(),
    HS47(),
    HS48(),
    HS49(),
    HS50(),
    HS51(),
    HS52(),
    HS53(),
    HS54(),
    HS55(),
    HS56(),
    HS57(),
    # HS59(),  # TODO: Human review - objective function discrepancy
    HS60(),
    HS61(),
    HS62(),
    HS63(),
    HS64(),
    HS65(),
    HS66(),
    # HS67(),  # TODO: Human review - several discrepancies
    HS68(),
    HS69(),
    HS71(),
    HS72(),
    HS73(),
    # HS74(),  # Human review needed - constraint Jacobian issues
    # HS75(),  # Human review needed - same issues as HS74
    HS76(),
    HS77(),
    HS78(),
    HS79(),
    HS80(),
    HS81(),
    HS83(),
    HS93(),
    # HS99(),  # TODO: Needs human review - complex recursive formulation
    HS100(),
    HS101(),
    HS102(),
    HS103(),
    HS104(),
    HS105(),
    HS106(),
    HS107(),
    HS108(),
    # HS109(),  # TODO: Human review needed - sign convention issues
    HS111(),
    HS112(),
    HS113(),
    HS114(),
    HS116(),
    HS117(),
    # HS118(),  # TODO: Human review - constraint Jacobian ordering mismatch
    HS119(),
    HYDROELL(),
    # JANNSON3(),  # TODO: Human review - Jacobian tests hang due to computational cost
    JANNSON4(),
    # KIWCRESC(),  # TODO: Human review - constraint values differ by 2.0 from pycutest
    # KISSING(),  # TODO: Human review - runtime issue (5.37x)
    # KISSING2(),  # TODO: Human review needed
    # KSIP(),  # TODO: Human review needed - needs vectorization, dtype promotion errors
    # LEUVEN1(),  # TODO: Human review needed
    LISWET1(),
    LISWET2(),
    LISWET3(),
    LISWET4(),
    LISWET5(),
    LISWET6(),
    LISWET7(),
    LISWET8(),
    LISWET9(),
    LISWET10(),
    LISWET11(),
    LISWET12(),
    LOOTSMA(),
    HIMMELBC(),
    HIMMELBD(),
    HIMMELBE(),
    # HIMMELP2(),  # TODO: Human review needed - OBNL element issues
    # HIMMELP3(),  # TODO: Human review needed - OBNL element issues
    # HIMMELP4(),  # TODO: Human review needed - OBNL element issues
    # HIMMELP5(),  # TODO: Human review needed - OBNL element issues
    # HIMMELP6(),  # TODO: Human review needed - OBNL element issues
    HYDROELL(),
    LOOTSMA(),
    MARATOS(),
    MSS1(),
    MSS2(),
    MSS3(),
    ODFITS(),
    OET1(),
    OET2(),
    OET3(),
    OET4(),
    OET5(),
    OET6(),
    OET7(),
    OPTCDEG2(),
    OPTCDEG3(),
    OPTCNTRL(),
    OPTCTRL3(),
    OPTCTRL6(),
    OPTMASS(),
    OPTPRLOC(),
    # ORTHRDM2(),  # TODO: Human review - singular Jacobian issues
    # ORTHRDS2(),  # TODO: Human review - singular Jacobian issues
    ORTHRDS2C(),
    # ORTHREGA(),  # TODO: Human review - fundamental formulation differences
    ORTHREGB(),
    ORTHREGC(),
    ORTHREGD(),
    ORTHREGE(),
    ORTHREGF(),
    ORTHRGDM(),
    ORTHRGDS(),
    PENTAGON(),
    POLAK1(),
    POLAK2(),
    POLAK3(),
    POLAK4(),
    POLAK5(),
    POLAK6(),
    # POLYGON(),  # TODO: Human review - constraint sign convention differences
    READING1(),
    READING2(),
    READING3(),
    READING4(),
    READING5(),
    # READING6(),  # TODO: Human review needed
    # Note: READING7 and READING8 exist but are not implemented due to a CUTEst bug:
    # the starting point is the solution too
    READING9(),
    # RDW2D51F(),  # TODO: Human review needed
    ROSEPETAL(),
    ROSEPETAL2(),
    SIMPLLPA(),
    SIMPLLPB(),
    # SINROSNB(),  # TODO: Human review - objective scaling issues
    SIPOW1(),
    SIPOW2(),
    # TAX13322(),  # TODO: Human review - complex objective structure
    TENBARS1(),
    TENBARS2(),
    TENBARS3(),
    TENBARS4(),
    # SPINOP(),  # TODO: Human review - constraint issues with auxiliary variables
    # SPIN2OP(),  # TODO: Human review - constraint test failures
    # STEENBRB(),  # TODO: Human review - gradient test failing
    # SIPOW3(),  # TODO: Human review - constraint formulation issues
    # SIPOW4(),  # TODO: Human review - constraint formulation issues
    # VANDERM1(),  # Moved to nonlinear equations (NOR2 classification)
    # VANDERM2(),  # Moved to nonlinear equations (NOR2 classification)
    # VANDERM3(),  # TODO: Human review - constraint values don't match
    # VANDERM4(),  # TODO: Human review - constraint values don't match
    MAKELA1(),
    MAKELA2(),
    MAKELA3(),
    MAKELA4(),
    # HS70(),  # TODO: Human review - test failures
    # HS84(),  # TODO: Human review - objective value discrepancy
    # TODO: TWIR problems need human review - complex trilinear constraint formulation
    # TWIRISM1(),
    # TWIRIMD1(),
    # TWIRIBG1(),
    ZECEVIC2(),
    ZECEVIC3(),
    ZECEVIC4(),
    # TRUSPYR1(),  # TODO: Human review - complex constraint scaling issues
    # TRUSPYR2(),  # TODO: Human review - test requested to be removed
    BT1(),
    BT2(),
    BT3(),
    BT4(),
    BT5(),
    BT6(),
    BT7(),
    BT8(),
    BT9(),
    BT10(),
    BT11(),
    BT12(),
    BT13(),
    LUKVLE1(),
    # LUKVLE2(),
    LUKVLE3(),
    # LUKVLE4(),  # Has factor ~2380 error due to SIF bug, use LUKVLE4C instead
    # LUKVLE4C(),  # Has 3% discrepancy with pycutest
    LUKVLE5(),
    LUKVLE6(),
    LUKVLE7(),
    LUKVLE8(),
    # LUKVLE9(),  # TODO: Human review needed - Jacobian issues
    LUKVLE10(),
    LUKVLE11(),
    # LUKVLE12(),  # Has constraint function inconsistencies
    LUKVLE13(),
    LUKVLE14(),
    LUKVLE15(),
    LUKVLE16(),
    LUKVLE17(),
    LUKVLE18(),
    LUKVLI1(),
    # LUKVLI2(),
    LUKVLI3(),
    # LUKVLI4(),  # Has factor ~2380 error due to SIF bug, use LUKVLI4C instead
    # LUKVLI4C(),  # Has 3% discrepancy with pycutest
    LUKVLI5(),
    LUKVLI6(),
    LUKVLI7(),
    LUKVLI8(),
    # LUKVLI9(),  # TODO: Human review needed - Jacobian issues
    LUKVLI10(),
    LUKVLI11(),
    # LUKVLI12(),  # Has constraint function inconsistencies
    LUKVLI13(),
    LUKVLI14(),
    LUKVLI15(),
    LUKVLI16(),
    LUKVLI17(),
    LUKVLI18(),
)
