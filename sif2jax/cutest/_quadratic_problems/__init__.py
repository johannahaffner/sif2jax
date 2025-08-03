from .biggsc4 import BIGGSC4 as BIGGSC4

# from .chenhark import CHENHARK as CHENHARK  # TODO: Human review needed - see file
from .cvxbqp1 import CVXBQP1 as CVXBQP1
from .cvxqp1 import CVXQP1 as CVXQP1
from .cvxqp2 import CVXQP2 as CVXQP2
from .cvxqp3 import CVXQP3 as CVXQP3
from .hatfldh import HATFLDH as HATFLDH
from .hs44new import HS44NEW as HS44NEW
from .hs76 import HS76 as HS76
from .qpband import QPBAND as QPBAND
from .tame import TAME as TAME

# from .torsiond import TORSIOND as TORSIOND  # TODO: Human review needed - see file
from .yao import YAO as YAO


# Bounded quadratic problems (only bound constraints)
bounded_quadratic_problems = (
    # CHENHARK(),  # TODO: Human review needed - see file
    CVXBQP1(),
    # TORSIOND(),  # TODO: Human review needed - objective mismatch with pycutest
)


# Constrained quadratic problems (equality and/or inequality constraints)
constrained_quadratic_problems = (
    BIGGSC4(),
    CVXQP1(),
    CVXQP2(),
    CVXQP3(),
    HATFLDH(),
    HS44NEW(),
    HS76(),
    QPBAND(),
    TAME(),
    YAO(),
)

# All quadratic problems
quadratic_problems = bounded_quadratic_problems + constrained_quadratic_problems
