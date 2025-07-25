from .branin import BRANIN as BRANIN
from .camel6 import CAMEL6 as CAMEL6
from .dgospec import DGOSPEC as DGOSPEC
from .exp2b import EXP2B as EXP2B
from .explin import EXPLIN as EXPLIN
from .explin2 import EXPLIN2 as EXPLIN2
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
from .palmer1 import PALMER1 as PALMER1
from .palmer2 import PALMER2 as PALMER2
from .palmer3b import PALMER3B as PALMER3B
from .palmer4e import PALMER4E as PALMER4E
from .palmer3b import PALMER3B as PALMER3B
from .price4b import PRICE4B as PRICE4B


bounded_minimisation_problems = (
    BRANIN(),
    CAMEL6(),
    DGOSPEC(),
    EXP2B(),
    EXPLIN(),
    EXPLIN2(),
    HATFLDC(),
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
    PALMER1(),
    PALMER2(),
    PALMER3B(),
    PALMER4E(),
    PALMER3B(),
    PRICE4B(),
)
