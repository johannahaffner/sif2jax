from .bdexp import BDEXP as BDEXP
from .branin import BRANIN as BRANIN
from .camel6 import CAMEL6 as CAMEL6
from .dgospec import DGOSPEC as DGOSPEC
from .exp2b import EXP2B as EXP2B
from .explin import EXPLIN as EXPLIN
from .explin2 import EXPLIN2 as EXPLIN2
from .expquad import EXPQUAD as EXPQUAD
from .hart6 import HART6 as HART6
from .hatflda import HATFLDA as HATFLDA
from .hatfldb import HATFLDB as HATFLDB
from .hatfldc import HATFLDC as HATFLDC
from .hs1 import HS1 as HS1
from .hs2 import HS2 as HS2
from .hs3 import HS3 as HS3
from .hs4 import HS4 as HS4
from .hs5 import HS5 as HS5
from .hs25 import HS25 as HS25
from .hs38 import HS38 as HS38
from .hs45 import HS45 as HS45
from .hs110 import HS110 as HS110

# from .himmelp1 import HIMMELP1 as HIMMELP1  # TODO: Human review - OBNL issues
from .logros import LOGROS as LOGROS
from .palmer1 import PALMER1 as PALMER1
from .palmer1a import PALMER1A as PALMER1A

# from .palmer1b import PALMER1B as PALMER1B  # TODO: Fix Hessian issues
# from .palmer1e import PALMER1E as PALMER1E  # TODO: Fix Hessian issues
from .palmer2 import PALMER2 as PALMER2
from .palmer2a import PALMER2A as PALMER2A
from .palmer2b import PALMER2B as PALMER2B
from .palmer2e import PALMER2E as PALMER2E
from .palmer3 import PALMER3 as PALMER3
from .palmer3a import PALMER3A as PALMER3A
from .palmer3b import PALMER3B as PALMER3B
from .palmer3e import PALMER3E as PALMER3E
from .palmer4 import PALMER4 as PALMER4

# from .palmer4a import PALMER4A as PALMER4A  # TODO: Fix Hessian issues
from .palmer4b import PALMER4B as PALMER4B
from .palmer4e import PALMER4E as PALMER4E

# TODO: Fix Chebyshev polynomial calculation
# from .palmer5a import PALMER5A as PALMER5A
from .palmer5b import PALMER5B as PALMER5B

# TODO: Fix Chebyshev polynomial calculation
# from .palmer5e import PALMER5E as PALMER5E
from .palmer6a import PALMER6A as PALMER6A
from .palmer6e import PALMER6E as PALMER6E

# from .palmer7a import PALMER7A as PALMER7A  # TODO: Fix Hessian issues
from .palmer7e import PALMER7E as PALMER7E
from .palmer8a import PALMER8A as PALMER8A
from .palmer8e import PALMER8E as PALMER8E
from .price4b import PRICE4B as PRICE4B


bounded_minimisation_problems = (
    BDEXP(),
    BRANIN(),
    CAMEL6(),
    DGOSPEC(),
    EXP2B(),
    EXPLIN(),
    EXPLIN2(),
    EXPQUAD(),
    HART6(),
    HATFLDA(),
    HATFLDB(),
    HATFLDC(),
    # HIMMELP1(),  # TODO: Human review needed - OBNL element issues
    HS1(),
    HS2(),
    HS3(),
    HS4(),
    HS5(),
    HS25(),
    HS38(),
    HS45(),
    HS110(),
    # LEVYMONT(),  # TODO: Fix SCALE interpretation
    LOGROS(),
    PALMER1(),
    PALMER1A(),
    # PALMER1B(),  # TODO: Fix Hessian issues
    # PALMER1E(),  # TODO: Fix Hessian issues
    PALMER2(),
    PALMER2A(),
    PALMER2B(),
    PALMER2E(),
    PALMER3(),
    PALMER3A(),
    PALMER3B(),
    PALMER3E(),
    PALMER4(),
    # PALMER4A(),  # TODO: Fix Hessian issues
    PALMER4B(),
    PALMER4E(),
    # PALMER5A(),  # TODO: Fix Chebyshev polynomial calculation
    PALMER5B(),
    PALMER6A(),
    PALMER6E(),
    # PALMER7A(),  # TODO: Fix Hessian issues
    PALMER7E(),
    PALMER8A(),
    PALMER8E(),
    PRICE4B(),
)
