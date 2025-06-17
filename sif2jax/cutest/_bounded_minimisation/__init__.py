from .exp2b import EXP2B as EXP2B
from .hs1 import HS1 as HS1
from .hs2 import HS2 as HS2
from .hs3 import HS3 as HS3
from .hs4 import HS4 as HS4
from .hs5 import HS5 as HS5
from .hs25 import HS25 as HS25
from .hs38 import HS38 as HS38
from .hs45 import HS45 as HS45
from .hs110 import HS110 as HS110


bounded_minimisation_problems = (
    EXP2B(),
    HS1(),
    HS2(),
    HS3(),
    HS4(),
    HS5(),
    HS25(),
    HS38(),
    HS45(),
    HS110(),
)
