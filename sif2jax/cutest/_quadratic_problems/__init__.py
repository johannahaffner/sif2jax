from .biggsc4 import BIGGSC4 as BIGGSC4
from .chenhark import CHENHARK as CHENHARK
from .degdiag import DEGDIAG as DEGDIAG
from .hatfldh import HATFLDH as HATFLDH
from .hs44new import HS44NEW as HS44NEW
from .hs76 import HS76 as HS76
from .qpband import QPBAND as QPBAND
from .qudlin import QUDLIN as QUDLIN
from .tame import TAME as TAME
from .torsiond import TORSIOND as TORSIOND
from .yao import YAO as YAO


quadratic_problems = (
    BIGGSC4(),
    CHENHARK(),
    DEGDIAG(),
    HATFLDH(),
    HS44NEW(),
    HS76(),
    QPBAND(),
    QUDLIN(),
    TAME(),
    TORSIOND(),
    YAO(),
)
