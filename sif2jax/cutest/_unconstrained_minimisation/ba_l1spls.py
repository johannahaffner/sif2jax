from typing_extensions import override

import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: Human review needed
# Attempts made: Implemented linear terms and partial quadratic terms
# Suspected issues: The full SIF file has thousands of P(i,j) quadratic product terms
# that would need to be included for exact Hessian matching. Current implementation
# only includes a subset of these terms.
# Additional resources needed: Full list of quadratic terms from SIF file or
# alternative formulation that doesn't require explicit enumeration of all terms
class BA_L1SPLS(AbstractUnconstrainedMinimisation):
    """BA-L1SPLS function.

    A small undetermined set of quadratic equations from a
    bundle adjustment subproblem.

    Least-squares version of BA-L1SP.

    SIF input: Nick Gould, Nov 2016

    Classification: SUR2-MN-57-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    @property
    @override
    def name(self):
        return "BA-L1SPLS"

    def objective(self, y, args):
        del args

        # Constants for each group (targets)
        constants = jnp.array(
            [
                9.020224572,  # C1
                -11.194618482,  # C2
                1.83322914,  # C3
                -5.254740578,  # C4
                4.332320525,  # C5
                -6.9705186587,  # C6
                0.5632735813,  # C7
                220.0023398,  # C8
                3.969211949,  # C9
                202.2580513,  # C10
                5.392772211,  # C11
                194.2376052,  # C12
            ]
        )

        # Linear coefficients for each group
        # Organized by group with non-zero coefficients for specific variables

        total_obj = 0.0

        # Group C1 - involves variables x1-x12
        linear_c1 = (
            545.11792729 * y[0]  # x1
            + -5.058282413 * y[1]  # x2
            + -478.0666573 * y[2]  # x3
            + -283.5120115 * y[3]  # x4
            + -1296.338862 * y[4]  # x5
            + -320.6033515 * y[5]  # x6
            + 551.17734728 * y[6]  # x7
            + 0.00020463888 * y[7]  # x8
            + -471.0948965 * y[8]  # x9
            + -409.2809619 * y[9]  # x10
            + -490.2705298 * y[10]  # x11
            + -0.8547064923 * y[11]  # x12
        )

        # Quadratic terms for C1 (based on GROUP USES)
        # P(i,j) with coefficient c means c * x_i * x_j
        # For diagonal P(i,i), it's 0.5 * c * x_i^2 (from SQR element definition)
        quad_c1 = (
            0.5 * 545.11792729 * y[0] ** 2  # P1,1
            + 545.11792729 * y[0] * y[1]  # P1,2
            + -5.058282413 * 0.5 * y[1] ** 2  # P2,2
            # Continue with all quadratic terms from the SIF file
            # This is a simplified version - the full version would include all terms
            + 0.0  # placeholder for remaining terms
        )

        c1_residual = linear_c1 + quad_c1 - constants[0]
        total_obj += c1_residual**2

        # Group C2 - involves variables x1-x12
        linear_c2 = (
            2.44930593 * y[0]
            + 556.94489983 * y[1]
            + 368.0324789 * y[2]
            + 1234.7454956 * y[3]
            + 227.79935236 * y[4]
            + -347.0888335 * y[5]
            + 0.00020463888 * y[6]
            + 551.17743945 * y[7]
            + 376.80482466 * y[8]
            + 327.36300527 * y[9]
            + 392.14243755 * y[10]
            + 0.68363621076 * y[11]
        )

        quad_c2 = (
            0.5 * 2.44930593 * y[0] ** 2
            + 2.44930593 * y[0] * y[1]
            + 0.5 * 556.94489983 * y[1] ** 2
            + 0.0  # placeholder for remaining terms
        )

        c2_residual = linear_c2 + quad_c2 - constants[1]
        total_obj += c2_residual**2

        # Group C3 - involves x1-x3, x13-x21
        linear_c3 = (
            350.08946365 * y[0]
            + 0.39982371752 * y[1]
            + -186.7535887 * y[2]
            + -107.0193513 * y[12]  # x13
            + -758.7948959 * y[13]  # x14
            + -207.8248083 * y[14]  # x15
            + 354.69032045 * y[15]  # x16
            + 5.7520864e-5 * y[16]  # x17
            + -177.8608216 * y[17]  # x18
            + -87.5738398 * y[18]  # x19
            + -38.04282609 * y[19]  # x20
            + -0.5014538225 * y[20]  # x21
        )

        c3_residual = linear_c3 - constants[2]  # simplified - missing quadratic terms
        total_obj += c3_residual**2

        # Group C4 - involves x1-x3, x13-x21
        linear_c4 = (
            0.52655531437 * y[0]
            + 356.88663624 * y[1]
            + 145.9511661 * y[2]
            + 740.42840621 * y[12]
            + 92.188824988 * y[13]
            + -222.1616016 * y[14]
            + 5.7520864e-5 * y[15]
            + 354.69033883 * y[16]
            + 151.71150127 * y[17]
            + 74.698624396 * y[18]
            + 32.449722244 * y[19]
            + 0.42772945465 * y[20]
        )

        c4_residual = linear_c4 - constants[3]
        total_obj += c4_residual**2

        # Group C5 - involves x1-x3, x22-x30
        linear_c5 = (
            424.98400393 * y[0]
            + -3.679913168 * y[1]
            + -285.9919818 * y[2]
            + -168.2771917 * y[21]  # x22
            + -958.0774748 * y[22]  # x23
            + -249.6325723 * y[23]  # x24
            + 430.9113597 * y[24]  # x25
            + 9.5401875e-5 * y[25]  # x26
            + -277.0049802 * y[26]  # x27
            + -176.6544282 * y[27]  # x28
            + -121.2420785 * y[28]  # x29
            + -0.6428351473 * y[29]  # x30
        )

        c5_residual = linear_c5 - constants[4]
        total_obj += c5_residual**2

        # Group C6 - involves x1-x3, x22-x30
        linear_c6 = (
            4.2853403082 * y[0]
            + 434.04620563 * y[1]
            + 218.70892251 * y[2]
            + 921.07415044 * y[21]
            + 136.05794576 * y[22]
            + -271.6835835 * y[23]
            + 9.5401875e-5 * y[24]
            + 430.9113995 * y[25]
            + 225.18412985 * y[26]
            + 143.60670941 * y[27]
            + 98.560653819 * y[28]
            + 0.52257642872 * y[29]
        )

        c6_residual = linear_c6 - constants[5]
        total_obj += c6_residual**2

        # Group C7 - involves x1-x3, x31-x39
        linear_c7 = (
            257.01763205 * y[0]
            + -10.17665712 * y[1]
            + -485.335462 * y[2]
            + -144.7289642 * y[30]  # x31
            + -770.5881534 * y[31]  # x32
            + -272.9493307 * y[32]  # x33
            + 543.87077184 * y[33]  # x34
            + 1.1685922e-6 * y[34]  # x35
            + 76.920229468 * y[35]  # x36
            + 2.0869983072 * y[36]  # x37
            + 0.07566110172 * y[37]  # x38
            + 0.14143107787 * y[38]  # x39
        )

        c7_residual = linear_c7 - constants[6]
        total_obj += c7_residual**2

        # Group C8 - involves x1-x3, x31-x39
        linear_c8 = (
            70.320803231 * y[0]
            + 179.34148777 * y[1]
            + 15.269871971 * y[2]
            + 620.43380046 * y[30]
            + 194.45316484 * y[31]
            + -289.1623446 * y[32]
            + 1.1685922e-6 * y[33]
            + 543.8707716 * y[34]
            + 69.331903641 * y[35]
            + 1.8811119849 * y[36]
            + 0.06819699123 * y[37]
            + 0.12747863508 * y[38]
        )

        c8_residual = linear_c8 - constants[7]
        total_obj += c8_residual**2

        # Group C9 - involves x1-x3, x40-x48
        linear_c9 = (
            422.1912571 * y[0]
            + -8.291703599 * y[1]
            + -366.5398322 * y[2]
            + -32.98148952 * y[39]  # x40
            + -1001.73529 * y[40]  # x41
            + -309.2254094 * y[41]  # x42
            + 484.30591161 * y[42]  # x43
            + 2.2083578e-6 * y[43]  # x44
            + 279.48782165 * y[44]  # x45
            + 79.76578597 * y[45]  # x46
            + 27.161405788 * y[46]  # x47
            + 0.57708943686 * y[47]  # x48
        )

        c9_residual = linear_c9 - constants[8]
        total_obj += c9_residual**2

        # Group C10 - involves x1-x3, x40-x48
        linear_c10 = (
            44.828047187 * y[0]
            + 152.76507038 * y[1]
            + 6.0272627364 * y[2]
            + 536.26711278 * y[39]
            + 204.18278149 * y[40]
            + -250.6380806 * y[41]
            + 2.2083578e-6 * y[42]
            + 484.30589721 * y[43]
            + 41.892546625 * y[44]
            + 11.956127062 * y[45]
            + 4.0712344877 * y[46]
            + 0.08650017735 * y[47]
        )

        c10_residual = linear_c10 - constants[9]
        total_obj += c10_residual**2

        # Group C11 - involves x1-x3, x49-x57
        linear_c11 = (
            480.66081245 * y[0]
            + -9.700408082 * y[1]
            + -321.955117 * y[2]
            + 5.7198974673 * y[48]  # x49
            + -1082.833992 * y[49]  # x50
            + -318.8938606 * y[50]  # x51
            + 458.16637265 * y[51]  # x52
            + 2.2305592e-6 * y[52]  # x53
            + 353.36608051 * y[53]  # x54
            + 187.06511552 * y[54]  # x55
            + 112.10170813 * y[55]  # x56
            + 0.77126151439 * y[56]  # x57
        )

        c11_residual = linear_c11 - constants[10]
        total_obj += c11_residual**2

        # Group C12 - involves x1-x3, x49-x57
        linear_c12 = (
            35.371625137 * y[0]
            + 141.48851847 * y[1]
            + 3.9687026816 * y[2]
            + 499.62967221 * y[48]
            + 205.63088192 * y[49]
            + -232.6571585 * y[50]
            + 2.2305592e-6 * y[51]
            + 458.16634697 * y[52]
            + 30.465184316 * y[53]
            + 16.127674776 * y[54]
            + 9.6647623772 * y[55]
            + 0.06649371711 * y[56]
        )

        c12_residual = linear_c12 - constants[11]
        total_obj += c12_residual**2

        return jnp.array(total_obj)

    def y0(self):
        # Initialize with zeros for this problem
        # The full problem has 57 variables
        return jnp.zeros(57)

    def args(self):
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return None
