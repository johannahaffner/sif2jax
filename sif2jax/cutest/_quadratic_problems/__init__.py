from .biggsc4 import BIGGSC4 as BIGGSC4

# from .chenhark import CHENHARK as CHENHARK  # TODO: Human review needed - see file
from .hatfldh import HATFLDH as HATFLDH
from .hs44new import HS44NEW as HS44NEW
from .hs76 import HS76 as HS76
from .qpband import QPBAND as QPBAND
from .tame import TAME as TAME

# from .torsiond import TORSIOND as TORSIOND  # TODO: Human review needed - see file
from .yao import YAO as YAO


quadratic_problems = (
    BIGGSC4(),
    # CHENHARK(),  # TODO: Human review needed - see file
    HATFLDH(),
    HS44NEW(),
    HS76(),
    QPBAND(),
    TAME(),
    # TORSIOND(),  # TODO: Human review needed - objective mismatch with pycutest
    YAO(),
)
