from .akiva import AKIVA as AKIVA
from .allinitu import ALLINITU as ALLINITU
from .arglina import ARGLINA as ARGLINA
from .arglinb import ARGLINB as ARGLINB
from .arglinc import ARGLINC as ARGLINC
from .argtrigls import ARGTRIGLS as ARGTRIGLS
from .arwhead import ARWHEAD as ARWHEAD

# TODO: BA_L family needs human review
# from .ba_l1ls import BA_L1LS as BA_L1LS  # TODO: BA_L family needs human review
# from .ba_l1spls import BA_L1SPLS as BA_L1SPLS  # TODO: Human review - Hessian fails
from .bard import BARD as BARD
from .bdqrtic import BDQRTIC as BDQRTIC
from .beale import BEALE as BEALE
from .biggs6 import BIGGS6 as BIGGS6
from .box import BOX as BOX
from .box3 import BOX3 as BOX3
from .boxbodls import BOXBODLS as BOXBODLS

# TODO: BOXPOWER needs human review - minor gradient discrepancy at last element
# from .boxpower import BOXPOWER as BOXPOWER
# TODO: BRKMCC needs human review - significant discrepancies
# from .brkmcc import BRKMCC as BRKMCC
# TODO: BROWNAL needs human review - small Hessian discrepancies
# from .brownal import BROWNAL as BROWNAL
from .brownbs import BROWNBS as BROWNBS
from .brownden import BROWNDEN as BROWNDEN
from .broydn3dls import BROYDN3DLS as BROYDN3DLS
from .broydn7d import BROYDN7D as BROYDN7D

# TODO: BROYDNBDLS and BRYBND require human review - gradient tests fail
# from .broydnbdls import BROYDNBDLS as BROYDNBDLS
# from .brybnd import BRYBND as BRYBND
# TODO: CERI problems need human review - numerical instability in erfc/exp
# from .ceri import (
#     CERI651ALS as CERI651ALS,
#     CERI651BLS as CERI651BLS,
#     CERI651CLS as CERI651CLS,
#     CERI651DLS as CERI651DLS,
#     CERI651ELS as CERI651ELS,
# )
from .chainwoo import CHAINWOO as CHAINWOO
from .chnrosnb import CHNROSNB as CHNROSNB
from .chnrsnbm import CHNRSNBM as CHNRSNBM

# TODO: CHWIRUT1 needs external data file with 214 data points
# from .chwirut1 import CHWIRUT1 as CHWIRUT1
from .chwirut1ls import CHWIRUT1LS as CHWIRUT1LS

# TODO: CHWIRUT2 needs implementation with 54 data points
# from .chwirut2 import CHWIRUT2 as CHWIRUT2
from .chwirut2ls import CHWIRUT2LS as CHWIRUT2LS
from .cliff import CLIFF as CLIFF
from .clusterls import CLUSTERLS as CLUSTERLS
from .coating import COATING as COATING
from .coolhansls import COOLHANSLS as COOLHANSLS
from .cosine import COSINE as COSINE
from .cragglvy import CRAGGLVY as CRAGGLVY
from .cube import CUBE as CUBE
from .curly10 import CURLY10 as CURLY10
from .curly20 import CURLY20 as CURLY20
from .curly30 import CURLY30 as CURLY30

# TODO: CYCLOOCFLS needs optimization - times out with default p=10000 (30k vars)
# from .cycloocfls import CYCLOOCFLS as CYCLOOCFLS
from .daniwoodls import DANIWOODLS as DANIWOODLS
from .denschna import DENSCHNA as DENSCHNA
from .denschnb import DENSCHNB as DENSCHNB
from .denschnc import DENSCHNC as DENSCHNC
from .denschnd import DENSCHND as DENSCHND
from .denschne import DENSCHNE as DENSCHNE
from .denschnf import DENSCHNF as DENSCHNF
from .devgla1 import DEVGLA1 as DEVGLA1
from .devgla2 import DEVGLA2 as DEVGLA2
from .dixmaana1 import DIXMAANA1 as DIXMAANA1
from .dixmaanb import DIXMAANB as DIXMAANB
from .dixmaanc import DIXMAANC as DIXMAANC
from .dixmaand import DIXMAAND as DIXMAAND
from .dixmaane1 import DIXMAANE1 as DIXMAANE1
from .dixmaanf import DIXMAANF as DIXMAANF
from .dixmaang import DIXMAANG as DIXMAANG
from .dixmaanh import DIXMAANH as DIXMAANH
from .dixmaani1 import DIXMAANI1 as DIXMAANI1
from .dixmaanj import DIXMAANJ as DIXMAANJ
from .dixmaank import DIXMAANK as DIXMAANK
from .dixmaanl import DIXMAANL as DIXMAANL
from .dixmaanm1 import DIXMAANM1 as DIXMAANM1
from .dixmaann import DIXMAANN as DIXMAANN
from .dixmaano import DIXMAANO as DIXMAANO
from .dixmaanp import DIXMAANP as DIXMAANP
from .dixon3dq import DIXON3DQ as DIXON3DQ
from .djtl import DJTL as DJTL
from .dqdrtic import DQDRTIC as DQDRTIC
from .dqrtic import DQRTIC as DQRTIC

# TODO: ECKERLE4LS needs human review - significant discrepancies
# from .eckerle4ls import ECKERLE4LS as ECKERLE4LS
from .edensch import EDENSCH as EDENSCH
from .eg1 import EG1 as EG1
from .eg2 import EG2 as EG2
from .eggcrate import EGGCRATE as EGGCRATE
from .eigenals import EIGENALS as EIGENALS
from .eigenbls import EIGENBLS as EIGENBLS
from .eigencls import EIGENCLS as EIGENCLS
from .elatvidu import ELATVIDU as ELATVIDU
from .engval1 import ENGVAL1 as ENGVAL1
from .engval2 import ENGVAL2 as ENGVAL2

# TODO: ENSOLS needs human review - significant discrepancies
# from .ensols import ENSOLS as ENSOLS
from .errinros import ERRINROS as ERRINROS

# TODO: ERRINRSM needs human review - significant discrepancies
# from .errinros import ERRINRSM as ERRINRSM
from .exp_scipy import EXP2 as EXP2
from .expfit import EXPFIT as EXPFIT

# TODO: EXTROSNB needs human review - objective/gradient discrepancies
# from .extrosnb import EXTROSNB as EXTROSNB
# TODO: FBRAIN3LS needs human review - complex data dependencies
# from .fbrain3ls import FBRAIN3LS as FBRAIN3LS
from .fletbv3m import FLETBV3M as FLETBV3M
from .fletcbv2 import FLETCBV2 as FLETCBV2
from .fletcbv3 import FLETCBV3 as FLETCBV3
from .fletchcr import FLETCHCR as FLETCHCR

# TODO: Human review - objective/gradient discrepancies
# from .fletchbv import FLETCHBV as FLETCHBV
# TODO: FMINSURF and FMINSRF2 have bugs - starting value/gradient discrepancies
# from .fminsurf import FMINSRF2 as FMINSRF2, FMINSURF as FMINSURF
# TODO: FREURONE needs human review - miscategorized (should be constrained)
# from .freuroth import FREURONE as FREURONE
from .freuroth import FREUROTH as FREUROTH

# TODO: GAUSS family needs human review - issues reported by user
# from .gauss import GAUSS1LS as GAUSS1LS, GAUSS2LS as GAUSS2LS, GAUSS3LS as GAUSS3LS
from .gaussian import GAUSSIAN as GAUSSIAN

# TODO: GBRAINLS needs human review - complex data dependencies
# from .gbrainls import GBRAINLS as GBRAINLS
from .genhumps import GENHUMPS as GENHUMPS
from .genrose import GENROSE as GENROSE
from .growthls import GROWTHLS as GROWTHLS

# TODO: GULF needs human review - issues reported by user
# from .gulf import GULF as GULF
from .hahn1ls import HAHN1LS as HAHN1LS
from .hairy import HAIRY as HAIRY

# TODO: HATFLD family needs human review - discrepancies in HATFLDGLS
from .hatfldd import HATFLDD as HATFLDD
from .hatflde import HATFLDE as HATFLDE
from .hatfldfl import HATFLDFL as HATFLDFL
from .hatfldfls import HATFLDFLS as HATFLDFLS
from .hatfldgls import HATFLDGLS as HATFLDGLS

# TODO: HEART problems need human review - significant discrepancies
# from .heart import HEART6LS as HEART6LS, HEART8LS as HEART8LS
from .helix import HELIX as HELIX

# TODO: HIELOW needs human review - significant discrepancies
# from .hielow import HIELOW as HIELOW
from .hilberta import HILBERTA as HILBERTA
from .hilbertb import HILBERTB as HILBERTB
from .himmelbcls import HIMMELBCLS as HIMMELBCLS
from .himmelbg import HIMMELBG as HIMMELBG
from .himmelbh import HIMMELBH as HIMMELBH

# TODO: Human review - Hessian discrepancies
# from .himmelbb import HIMMELBB as HIMMELBB
# from .himmelbf import HIMMELBF as HIMMELBF
from .humps import HUMPS as HUMPS
from .indef import INDEF as INDEF
from .indefm import INDEFM as INDEFM
from .inteqnels import INTEQNELS as INTEQNELS
from .jensmp import JENSMP as JENSMP
from .judge import JUDGE as JUDGE
from .kirby import KIRBY2LS as KIRBY2LS
from .kowosb import KOWOSB as KOWOSB

# TODO: KSSLS needs human review - significant objective/gradient discrepancies
# from .kssls import KSSLS as KSSLS
from .lanczos1ls import LANCZOS1LS as LANCZOS1LS
from .lanczos2ls import LANCZOS2LS as LANCZOS2LS
from .liarwhd import LIARWHD as LIARWHD
from .loghairy import LOGHAIRY as LOGHAIRY
from .lsc1ls import LSC1LS as LSC1LS
from .lsc2ls import LSC2LS as LSC2LS

# TODO: MANCINO needs human review - significant discrepancies in all values
# from .mancino import MANCINO as MANCINO
# from .mexhat import MEXHAT as MEXHAT  # TODO: Human review - complex scaling issues
# from .nondia import NONDIA as NONDIA  # TODO: Human review - SCALE factor issue
from .noncvxu2 import NONCVXU2 as NONCVXU2
from .noncvxun import NONCVXUN as NONCVXUN
from .nondquar import NONDQUAR as NONDQUAR
from .nonmsqrt import NONMSQRT as NONMSQRT

# TODO: Human review - minor numerical precision issues
# from .penalty1 import PENALTY1 as PENALTY1
# TODO: Human review - SCALE factor issue
# from .penalty2 import PENALTY2 as PENALTY2
from .power import POWER as POWER

# TODO: Human review - objective off by factor of 4.15
# from .powellsg import POWELLSG as POWELLSG
from .rosenbr import ROSENBR as ROSENBR
from .tenfolds import TENFOLDTRLS as TENFOLDTRLS

# TODO: TOINTGOR needs human review - runtime test fails (~8x slower than threshold)
# from .tointgor import TOINTGOR as TOINTGOR
from .tointgss import TOINTGSS as TOINTGSS

# TODO: TOINTPSP needs human review - gradient test fails with small differences
# from .tointpsp import TOINTPSP as TOINTPSP
from .trigon1 import TRIGON1 as TRIGON1

# TODO: TRIGON2 needs human review - Hessian test fails
# from .trigon2 import TRIGON2 as TRIGON2
from .waysea1 import WAYSEA1 as WAYSEA1
from .waysea2 import WAYSEA2 as WAYSEA2


unconstrained_minimisation_problems = (
    AKIVA(),
    ALLINITU(),
    ARGLINA(),
    ARGLINB(),
    ARGLINC(),
    ARGTRIGLS(),
    ARWHEAD(),
    # BA_L1LS(),  # TODO: BA_L family needs human review
    # BA_L1SPLS(),  # TODO: Human review - Hessian test fails
    BARD(),
    BDQRTIC(),
    BEALE(),
    BIGGS6(),
    BOX(),
    BOX3(),
    BOXBODLS(),
    # BOXPOWER(),  # TODO: Human review - minor gradient discrepancy at last element
    # BRKMCC(),  # TODO: Human review - significant discrepancies
    # BROWNAL(),  # TODO: Human review - small Hessian discrepancies
    BROWNBS(),
    BROWNDEN(),
    BROYDN3DLS(),
    BROYDN7D(),
    # BROYDNBDLS(),  # TODO: Gradient test fails - needs human review
    # BRYBND(),  # TODO: Gradient test fails - needs human review
    # CERI651ALS(),  # TODO: Human review - numerical instability in erfc/exp
    # CERI651BLS(),  # TODO: Human review - numerical instability in erfc/exp
    # CERI651CLS(),  # TODO: Human review - numerical instability in erfc/exp
    # CERI651DLS(),  # TODO: Human review - numerical instability in erfc/exp
    # CERI651ELS(),  # TODO: Human review - numerical instability in erfc/exp
    CHAINWOO(),
    CHNROSNB(),
    CHNRSNBM(),
    CHWIRUT1LS(),
    CHWIRUT2LS(),
    CLIFF(),
    CLUSTERLS(),
    COATING(),
    COOLHANSLS(),
    COSINE(),
    CRAGGLVY(),
    CUBE(),
    CURLY10(),
    CURLY20(),
    CURLY30(),
    # CYCLOOCFLS(),  # TODO: Human review - times out with default p=10000 (30k vars)
    DANIWOODLS(),
    DENSCHNA(),
    DENSCHNB(),
    DENSCHNC(),
    DENSCHND(),
    DENSCHNE(),
    DENSCHNF(),
    DEVGLA1(),
    DEVGLA2(),
    DIXMAANA1(),
    DIXMAANB(),
    DIXMAANC(),
    DIXMAAND(),
    DIXMAANE1(),
    DIXMAANF(),
    DIXMAANG(),
    DIXMAANH(),
    DIXMAANI1(),
    DIXMAANJ(),
    DIXMAANK(),
    DIXMAANL(),
    DIXMAANM1(),
    DIXMAANN(),
    DIXMAANO(),
    DIXMAANP(),
    DIXON3DQ(),
    DJTL(),
    DQDRTIC(),
    DQRTIC(),
    # ECKERLE4LS(),  # TODO: Human review - significant discrepancies
    EDENSCH(),
    EG2(),
    EGGCRATE(),
    EIGENALS(),
    EIGENBLS(),
    EIGENCLS(),
    ELATVIDU(),
    ENGVAL1(),
    ENGVAL2(),
    # ENSOLS(),  # TODO: Human review - significant discrepancies
    ERRINROS(),
    # ERRINRSM(),  # TODO: Human review - significant discrepancies
    EXP2(),
    EXPFIT(),
    # EXTROSNB(),  # TODO: Human review - objective/gradient discrepancies
    # FBRAIN3LS(),  # TODO: Human review - complex data dependencies
    # FLETCH family problems
    # FLETCHBV(),  # TODO: Human review - objective/gradient discrepancies
    FLETBV3M(),
    FLETCBV2(),
    FLETCHCR(),
    # Not varying the scale term in the FLETCBV3 problem
    FLETCBV3(),
    #    FMINSURF(),  # TODO: has a bug
    #    FMINSRF2(),  # TODO: has a bug
    FREUROTH(),
    # FREURONE(),  # TODO: Human review - miscategorized (should be constrained)
    # GAUSS1LS(),  # TODO: Human review - issues reported by user
    # GAUSS2LS(),  # TODO: Human review - issues reported by user
    # GAUSS3LS(),  # TODO: Human review - issues reported by user
    GAUSSIAN(),
    # GBRAINLS(),  # TODO: Human review - complex data dependencies
    GENHUMPS(),
    GENROSE(),
    GROWTHLS(),
    # GULF(),  # TODO: Human review - issues reported by user
    HAHN1LS(),
    # HAHN1LS(y0_id=1),  # Non-default starting point - we only test pycutest defaults
    HAIRY(),
    # HATFLDD(),  # TODO: HATFLD family needs human review - discrepancies
    # HATFLDE(),  # TODO: HATFLD family needs human review - discrepancies
    # HATFLDFL(),  # TODO: HATFLD family needs human review - discrepancies
    # HATFLDFLS(),  # TODO: HATFLD family needs human review - discrepancies
    # HATFLDGLS(),  # TODO: HATFLD family needs human review - discrepancies
    # HEART6LS(),  # TODO: Human review - significant discrepancies
    # HEART8LS(),  # TODO: Human review - significant discrepancies
    HELIX(),
    # HIELOW(),  # TODO: Human review - significant discrepancies
    HILBERTA(),
    HILBERTB(),
    # HIMMELBB(),  # TODO: Human review - Hessian discrepancies
    HIMMELBCLS(),
    # HIMMELBF(),  # TODO: Human review - Hessian discrepancies
    HIMMELBG(),
    HIMMELBH(),
    HUMPS(),
    INDEF(),
    INDEFM(),
    INTEQNELS(),
    JENSMP(),
    JUDGE(),
    KIRBY2LS(),
    KOWOSB(),
    # KSSLS(),  # TODO: Human review - significant objective/gradient discrepancies
    LANCZOS1LS(),
    LANCZOS2LS(),
    LIARWHD(),
    LOGHAIRY(),
    # LR(),
    LSC1LS(),
    LSC2LS(),
    # MANCINO(),  # TODO: Human review - significant discrepancies in all values
    # MEXHAT(),  # TODO: Human review - complex scaling issues
    # NONDIA(),  # TODO: Human review - SCALE factor issue
    NONCVXU2(),
    NONCVXUN(),
    NONDQUAR(),
    NONMSQRT(),
    # PENALTY1(),  # TODO: Human review - minor numerical precision issues
    # PENALTY2(),  # TODO: Human review - SCALE factor issue
    POWER(),
    # POWELLSG(),  # TODO: Human review - objective off by factor of 4.15
    ROSENBR(),
    TENFOLDTRLS(),
    WAYSEA1(),
    WAYSEA2(),
    TRIGON1(),
    # TRIGON2(),  # TODO: Human review - Hessian test fails
    # TOINTGOR(),  # TODO: Human review - runtime test fails
    TOINTGSS(),
    # TOINTPSP(),  # TODO: Human review - gradient test fails
)
