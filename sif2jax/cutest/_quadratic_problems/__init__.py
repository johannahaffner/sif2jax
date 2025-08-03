from .biggsc4 import BIGGSC4 as BIGGSC4

# from .chenhark import CHENHARK as CHENHARK  # TODO: Human review needed - see file
from .cvxbqp1 import CVXBQP1 as CVXBQP1
from .cvxqp1 import CVXQP1 as CVXQP1
from .cvxqp2 import CVXQP2 as CVXQP2
from .cvxqp3 import CVXQP3 as CVXQP3
from .hatfldh import HATFLDH as HATFLDH
from .hs44new import HS44NEW as HS44NEW
from .hs76 import HS76 as HS76
from .ncvxbqp1 import NCVXBQP1 as NCVXBQP1
from .ncvxbqp2 import NCVXBQP2 as NCVXBQP2
from .ncvxbqp3 import NCVXBQP3 as NCVXBQP3
from .ncvxqp1 import NCVXQP1 as NCVXQP1
from .ncvxqp2 import NCVXQP2 as NCVXQP2
from .ncvxqp3 import NCVXQP3 as NCVXQP3
from .ncvxqp4 import NCVXQP4 as NCVXQP4
from .ncvxqp5 import NCVXQP5 as NCVXQP5
from .ncvxqp6 import NCVXQP6 as NCVXQP6
from .ncvxqp7 import NCVXQP7 as NCVXQP7
from .ncvxqp8 import NCVXQP8 as NCVXQP8
from .ncvxqp9 import NCVXQP9 as NCVXQP9
from .qpband import QPBAND as QPBAND
from .tame import TAME as TAME

# from .torsiond import TORSIOND as TORSIOND  # TODO: Human review needed - see file
from .yao import YAO as YAO


# Bounded quadratic problems (only bound constraints)
bounded_quadratic_problems = (
    # CHENHARK(),  # TODO: Human review needed - see file
    CVXBQP1(),
    NCVXBQP1(),
    NCVXBQP2(),
    NCVXBQP3(),
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
    NCVXQP1(),
    NCVXQP2(),
    NCVXQP3(),
    NCVXQP4(),
    NCVXQP5(),
    NCVXQP6(),
    NCVXQP7(),
    NCVXQP8(),
    NCVXQP9(),
    QPBAND(),
    TAME(),
    YAO(),
)

# All quadratic problems
quadratic_problems = bounded_quadratic_problems + constrained_quadratic_problems
