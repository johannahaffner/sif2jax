from .biggsc4 import BIGGSC4 as BIGGSC4

# from .chenhark import CHENHARK as CHENHARK  # TODO: Human review needed - see file
from .cvxbqp1 import CVXBQP1 as CVXBQP1
from .cvxqp1 import CVXQP1 as CVXQP1
from .cvxqp2 import CVXQP2 as CVXQP2
from .cvxqp3 import CVXQP3 as CVXQP3
from .dual1 import DUAL1 as DUAL1
from .dual2 import DUAL2 as DUAL2
from .dual3 import DUAL3 as DUAL3
from .dual4 import DUAL4 as DUAL4
from .dualc1 import DUALC1 as DUALC1
from .dualc2 import DUALC2 as DUALC2
from .dualc5 import DUALC5 as DUALC5
from .dualc8 import DUALC8 as DUALC8
from .gouldqp1 import GOULDQP1 as GOULDQP1
from .gouldqp2 import GOULDQP2 as GOULDQP2
from .gouldqp3 import GOULDQP3 as GOULDQP3
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
from .qpnband import QPNBAND as QPNBAND

# from .qpnblend import QPNBLEND as QPNBLEND  # TODO: Human review - constraint matrix
# from .qpnboei1 import QPNBOEI1 as QPNBOEI1  # TODO: Human review - Boeing constraints
# from .qpnboei2 import QPNBOEI2 as QPNBOEI2  # TODO: Human review - Boeing constraints
# from .qpnstair import QPNSTAIR as QPNSTAIR  # TODO: Human review - constraint dims
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
    DUAL1(),
    DUAL2(),
    DUAL3(),
    DUAL4(),
    DUALC1(),
    DUALC2(),
    DUALC5(),
    DUALC8(),
    GOULDQP1(),
    GOULDQP2(),
    GOULDQP3(),
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
    QPNBAND(),
    # QPNBLEND(),  # TODO: Human review - complex constraint matrix
    # QPNBOEI1(),  # TODO: Human review - Boeing routing constraints
    # QPNBOEI2(),  # TODO: Human review - Boeing routing constraints
    # QPNSTAIR(),  # TODO: Human review - complex constraint dimensions
    TAME(),
    YAO(),
)

# All quadratic problems
quadratic_problems = bounded_quadratic_problems + constrained_quadratic_problems
