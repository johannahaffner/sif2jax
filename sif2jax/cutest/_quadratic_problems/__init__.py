from .biggsc4 import BIGGSC4 as BIGGSC4

# from .chenhark import CHENHARK as CHENHARK  # TODO: Human review needed - see file
from .cmpc1 import CMPC1 as CMPC1
from .cmpc2 import CMPC2 as CMPC2
from .cmpc3 import CMPC3 as CMPC3
from .cmpc4 import CMPC4 as CMPC4
from .cmpc5 import CMPC5 as CMPC5
from .cmpc6 import CMPC6 as CMPC6

# from .cmpc7 import CMPC7 as CMPC7  # TODO: Human review - runtime 8.78x vs 5x
from .cmpc8 import CMPC8 as CMPC8

# from .cmpc9 import CMPC9 as CMPC9  # TODO: Human review - runtime 5.81x vs 5x
from .cmpc10 import CMPC10 as CMPC10

# from .cmpc11 import CMPC11 as CMPC11  # TODO: Human review - runtime 5.28x vs 5x
from .cmpc12 import CMPC12 as CMPC12

# from .cmpc13 import CMPC13 as CMPC13  # TODO: Human review - runtime 6.02x vs 5x
# from .cmpc14 import CMPC14 as CMPC14  # TODO: Human review - runtime 5.51x vs 5x
from .cmpc15 import CMPC15 as CMPC15

# from .cmpc16 import CMPC16 as CMPC16  # TODO: Human review - runtime 8.41x vs 5x
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

# TODO: Human review needed - same constraint issues as EIGENA
# from .eigena2 import EIGENA2 as EIGENA2
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
from .table1 import TABLE1 as TABLE1
from .table3 import TABLE3 as TABLE3
from .table6 import TABLE6 as TABLE6
from .table7 import TABLE7 as TABLE7
from .table8 import TABLE8 as TABLE8
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
    CMPC1(),
    CMPC2(),
    CMPC3(),
    CMPC4(),
    CMPC5(),
    CMPC6(),
    # CMPC7(),  # TODO: Human review - runtime 8.78x vs 5x threshold
    CMPC8(),
    # CMPC9(),  # TODO: Human review - runtime 5.81x vs 5x threshold
    CMPC10(),
    # CMPC11(),  # TODO: Human review - runtime 5.28x vs 5x threshold
    CMPC12(),
    # CMPC13(),  # TODO: Human review - runtime 6.02x vs 5x threshold
    # CMPC14(),  # TODO: Human review - runtime 5.51x vs 5x threshold
    CMPC15(),
    # CMPC16(),  # TODO: Human review - runtime 8.41x vs 5x threshold
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
    # EIGENA2(),  # TODO: Human review needed - same constraint issues as EIGENA
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
    TABLE1(),
    TABLE3(),
    TABLE6(),
    TABLE7(),
    TABLE8(),
    TAME(),
    YAO(),
)

# All quadratic problems
quadratic_problems = bounded_quadratic_problems + constrained_quadratic_problems
