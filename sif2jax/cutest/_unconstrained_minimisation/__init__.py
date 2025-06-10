from .akiva import AKIVA as AKIVA
from .allinitu import ALLINITU as ALLINITU
from .arglin import ARGLINA as ARGLINA, ARGLINB as ARGLINB, ARGLINC as ARGLINC
from .argtrigls import ARGTRIGLS as ARGTRIGLS
from .arwhead import ARWHEAD as ARWHEAD

# TODO: BA_L family needs to be split into separate files and human review
# from .bal import (
#     BA_L1LS as BA_L1LS,
#     BA_L1SPLS as BA_L1SPLS,
# )
from .bard import BARD as BARD
from .bdqrtic import BDQRTIC as BDQRTIC
from .beale import BEALE as BEALE
from .biggs import BIGGS6 as BIGGS6
from .box import BOX as BOX, BOX3 as BOX3, BOXBODLS as BOXBODLS, BOXPOWER as BOXPOWER

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
from .chwirut import CHWIRUT1LS as CHWIRUT1LS, CHWIRUT2LS as CHWIRUT2LS
from .cliff import CLIFF as CLIFF
from .clusterls import CLUSTERLS as CLUSTERLS
from .coating import COATING as COATING
from .coolhansls import COOLHANSLS as COOLHANSLS
from .cosine import COSINE as COSINE
from .cragglvy import CRAGGLVY as CRAGGLVY
from .cube import CUBE as CUBE
from .curly import CURLY10 as CURLY10, CURLY20 as CURLY20, CURLY30 as CURLY30
from .cycloocfls import CYCLOOCFLS as CYCLOOCFLS
from .daniwoodls import DANIWOODLS as DANIWOODLS
from .denschn import (
    DENSCHNA as DENSCHNA,
    DENSCHNB as DENSCHNB,
    DENSCHNC as DENSCHNC,
    DENSCHND as DENSCHND,
    DENSCHNE as DENSCHNE,
    DENSCHNF as DENSCHNF,
)
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
from .eg import EG2 as EG2
from .eggcrate import EGGCRATE as EGGCRATE
from .eigenals import EIGENALS as EIGENALS
from .eigenbls import EIGENBLS as EIGENBLS
from .eigencls import EIGENCLS as EIGENCLS
from .elatvidu import ELATVIDU as ELATVIDU
from .engval import ENGVAL1 as ENGVAL1, ENGVAL2 as ENGVAL2

# TODO: ENSOLS needs human review - significant discrepancies
# from .ensols import ENSOLS as ENSOLS
from .errinros import ERRINROS as ERRINROS

# TODO: ERRINRSM needs human review - significant discrepancies
# from .errinros import ERRINRSM as ERRINRSM
from .exp_scipy import EXP2 as EXP2
from .expfit import EXPFIT as EXPFIT
from .extrosnb import EXTROSNB as EXTROSNB

# TODO: FBRAIN3LS needs human review - complex data dependencies
# from .fbrain3ls import FBRAIN3LS as FBRAIN3LS
from .fletch import (
    FLETBV3M as FLETBV3M,
    FLETCBV2 as FLETCBV2,
    FLETCBV3 as FLETCBV3,
    FLETCHBV as FLETCHBV,
    FLETCHCR as FLETCHCR,
)
from .fminsurf import FMINSRF2 as FMINSRF2, FMINSURF as FMINSURF

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
from .hatfld import (
    HATFLDD as HATFLDD,
    HATFLDE as HATFLDE,
    HATFLDFL as HATFLDFL,
    HATFLDFLS as HATFLDFLS,
    HATFLDGLS as HATFLDGLS,
)

# TODO: HEART problems need human review - significant discrepancies
# from .heart import HEART6LS as HEART6LS, HEART8LS as HEART8LS
from .helix import HELIX as HELIX

# TODO: HIELOW needs human review - significant discrepancies
# from .hielow import HIELOW as HIELOW
from .hilberta import HILBERTA as HILBERTA
from .hilbertb import HILBERTB as HILBERTB
from .himmelblau import (
    HIMMELBB as HIMMELBB,
    HIMMELBCLS as HIMMELBCLS,
    HIMMELBF as HIMMELBF,
    HIMMELBG as HIMMELBG,
    HIMMELBH as HIMMELBH,
)
from .humps import HUMPS as HUMPS
from .indef import INDEF as INDEF
from .inteqnels import INTEQNELS as INTEQNELS
from .jensmp import JENSMP as JENSMP
from .judge import JUDGE as JUDGE
from .kirby import KIRBY2LS as KIRBY2LS
from .kowosb import KOWOSB as KOWOSB

# TODO: KSSLS needs human review - significant objective/gradient discrepancies
# from .kssls import KSSLS as KSSLS
from .lanczos import LANCZOS1LS as LANCZOS1LS, LANCZOS2LS as LANCZOS2LS
from .liarwhd import LIARWHD as LIARWHD
from .lsc import LSC1LS as LSC1LS, LSC2LS as LSC2LS

# TODO: MANCINO needs human review - significant discrepancies in all values
# from .mancino import MANCINO as MANCINO
from .rosenbr import ROSENBR as ROSENBR
from .tenfolds import TENFOLDTRLS as TENFOLDTRLS


unconstrained_minimisation_problems = (
    AKIVA(),
    ALLINITU(),
    ARGLINA(),
    ARGLINB(),
    ARGLINC(),
    ARGTRIGLS(),
    ARWHEAD(),
    # BA_L1LS(),  # TODO: BA_L family needs to be split into separate files
    # BA_L1SPLS(),  # TODO: BA_L family needs to be split into separate files
    BARD(),
    BDQRTIC(),
    BEALE(),
    BIGGS6(),
    BOX(),
    BOX3(),
    BOXBODLS(),
    BOXPOWER(),
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
    CYCLOOCFLS(),
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
    EXTROSNB(),
    # FBRAIN3LS(),  # TODO: Human review - complex data dependencies
    # FLETCH family problems
    FLETCHBV(),
    FLETBV3M(),
    FLETCBV2(),
    FLETCHCR(),
    # Not varying the scale term in the FLETCBV3 problem
    FLETCBV3(n=5000, extra_term=1),
    FLETCBV3(n=5000, extra_term=0),
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
    HATFLDD(),
    HATFLDE(),
    HATFLDFL(),
    HATFLDFLS(),
    HATFLDGLS(),
    # HEART6LS(),  # TODO: Human review - significant discrepancies
    # HEART8LS(),  # TODO: Human review - significant discrepancies
    HELIX(),
    # HIELOW(),  # TODO: Human review - significant discrepancies
    HILBERTA(),
    HILBERTB(),
    HIMMELBB(),
    HIMMELBCLS(),
    HIMMELBF(),
    HIMMELBG(),
    HIMMELBH(),
    HUMPS(),
    INDEF(),
    INTEQNELS(),
    JENSMP(),
    JUDGE(),
    KIRBY2LS(),
    KOWOSB(),
    # KSSLS(),  # TODO: Human review - significant objective/gradient discrepancies
    LANCZOS1LS(),
    LANCZOS2LS(),
    LIARWHD(),
    # LR(),
    LSC1LS(),
    LSC2LS(),
    # MANCINO(),  # TODO: Human review - significant discrepancies in all values
    ROSENBR(),
    TENFOLDTRLS(),
)
