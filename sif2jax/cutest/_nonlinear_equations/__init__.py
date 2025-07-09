from .argauss import ARGAUSS as ARGAUSS
from .argtrig import ARGTRIG as ARGTRIG
from .artif import ARTIF as ARTIF

# TODO: Human review needed - constraint dimension mismatch
# from .arwhdne import ARWHDNE as ARWHDNE
from .bardne import BARDNE as BARDNE

# from .bdqrticne import BDQRTICNE as BDQRTICNE  # TODO: Human review needed
from .bealene import BEALENE as BEALENE
from .bennett5 import BENNETT5 as BENNETT5
from .biggs6ne import BIGGS6NE as BIGGS6NE
from .booth import BOOTH as BOOTH
from .box3ne import BOX3NE as BOX3NE
from .brownale import BROWNALE as BROWNALE
from .brownbsne import BROWNBSNE as BROWNBSNE
from .browndene import BROWNDENE as BROWNDENE
from .broydnbd import BROYDNBD as BROYDNBD
from .brybndne import BRYBNDNE as BRYBNDNE
from .ceri651a import CERI651A as CERI651A
from .ceri651b import CERI651B as CERI651B

# from .errinrosne import ERRINROSNE as ERRINROSNE  # TODO: Human review needed
from .ceri651c import CERI651C as CERI651C
from .chainwoone import CHAINWOONE as CHAINWOONE

# from .channel import CHANNEL as CHANNEL  # TODO: Human review needed
from .chebyqadne import CHEBYQADNE as CHEBYQADNE

# from .chnrsbne import CHNRSBNE as CHNRSBNE  # TODO: Human review needed
# from .chnrsnbmne import CHNRSNBMNE as CHNRSNBMNE  # TODO: Human review needed
from .cubene import CUBENE as CUBENE
from .cyclic3 import CYCLIC3 as CYCLIC3
from .denschnbne import DENSCHNBNE as DENSCHNBNE
from .engval2ne import ENGVAL2NE as ENGVAL2NE
from .hatfldbne import HATFLDBNE as HATFLDBNE
from .hatfldflne import HATFLDFLNE as HATFLDFLNE
from .mgh09 import MGH09 as MGH09
from .misra1d import MISRA1D as MISRA1D
from .nonmsqrtne import NONMSQRTNE as NONMSQRTNE
from .palmer1bne import PALMER1BNE as PALMER1BNE
from .palmer5ene import PALMER5ENE as PALMER5ENE
from .palmer7ane import PALMER7ANE as PALMER7ANE
from .powersumne import POWERSUMNE as POWERSUMNE
from .sinvalne import SINVALNE as SINVALNE
from .ssbrybndne import SSBRYBNDNE as SSBRYBNDNE
from .tenfoldtr import TENFOLDTR as TENFOLDTR


nonlinear_equations_problems = (
    ARGAUSS(),
    ARGTRIG(),
    ARTIF(),
    # TODO: Human review needed - constraint dimension mismatch
    # ARWHDNE(),
    BARDNE(),
    # BDQRTICNE(),  # TODO: Human review needed
    BEALENE(),
    BENNETT5(),
    BIGGS6NE(),
    BOOTH(),
    BOX3NE(),
    BROWNALE(),
    BROWNBSNE(),
    BROWNDENE(),
    BROYDNBD(),
    BRYBNDNE(),
    CERI651A(),
    CERI651B(),
    CERI651C(),
    CHAINWOONE(),
    # CHANNEL(),  # TODO: Human review needed
    CHEBYQADNE(),
    # CHNRSBNE(),  # TODO: Human review needed
    # CHNRSNBMNE(),  # TODO: Human review needed
    CUBENE(),
    CYCLIC3(),
    DENSCHNBNE(),
    ENGVAL2NE(),
    # ERRINROSNE(),  # TODO: Human review needed
    HATFLDBNE(),
    HATFLDFLNE(),
    MGH09(),
    MISRA1D(),
    NONMSQRTNE(),
    PALMER1BNE(),
    PALMER5ENE(),
    PALMER7ANE(),
    POWERSUMNE(),
    SINVALNE(),
    SSBRYBNDNE(),
    TENFOLDTR(),
)
