# Categorized Constrained Minimisation Problems by Constraint Type

This document categorizes the constrained minimisation problems in sif2jax based on their constraint types as defined in the SIF files.

## 1. Only Equality Constraints (79 problems)
Examples:
- **BT1**: 2 equality constraints
- **BT3**: 2 equality constraints  
- **HS119**: 8 equality constraints

Full list: ALSOTAME, BOXBOD, BT1, BT2, BT3, BT4, BT5, BT6, BT7, BT8, BT9, BT10, BT11, BT12, BT13, BYRDSPHR, CANTILVR, CB2, CB3, CHANDHEQ, CONCON, COOLHANS, DECONVC, HS6, HS7, HS8, HS9, HS13, HS26, HS27, HS28, HS39, HS40, HS41, HS42, HS43, HS44, HS46, HS47, HS48, HS49, HS50, HS51, HS52, HS53, HS54, HS55, HS56, HS57, HS61, HS77, HS78, HS79, HS111, HS119, LOOTSMA, LUKVLI1, LUKVLI3, LUKVLI5, LUKVLI6, LUKVLI7, LUKVLI8, LUKVLI9, LUKVLI10, LUKVLI11, LUKVLI13, LUKVLI14, LUKVLI15, LUKVLI16, LUKVLI17, LUKVLI18, MARATOS, POWELLBS, POWELLSE, POWELLSQ, VANDERM1, VANDERM2

## 2. Only Inequality Constraints of G Type (35 problems)
Examples:
- **HS10**: 1 inequality constraint (G type)
- **HS11**: 1 inequality constraint (G type)
- **HS21**: 1 inequality constraint (G type)

Full list: AVGASA, AVGASB, HS10, HS11, HS12, HS15, HS16, HS17, HS18, HS19, HS20, HS21, HS22, HS23, HS24, HS29, HS30, HS31, HS33, HS34, HS35, HS36, HS37, HS62, HS63, HS65, HS66, HS68, HS69, HS72, HS80, HS81, HS100, HS113, MAKELA1

## 3. Inequality Constraints of G Type and Equality Constraints (8 problems)
Examples:
- **HS14**: 1 equality, 2 inequality (G type)
- **HS32**: 1 equality, 4 inequality (G type)
- **HS73**: 2 equality, 4 inequality (G type)

Full list: HS14, HS32, HS71, HS73, HS74, HS75, HS114, HS117

## 4. Only Inequality Constraints of L Type (5 problems)
Examples:
- **HS64**: 1 inequality constraint (L type)
- **ZECEVIC2**: 1 inequality constraint (L type)
- **ZECEVIC3**: 1 inequality constraint (L type)

Full list: BURKEHAN, HS64, ZECEVIC2, ZECEVIC3, ZECEVIC4

## 5. Inequality Constraints of L Type and Equality Constraints (0 problems)
*No problems found in this category*

## 6. Mixed Inequality Constraints of L and G Types (3 problems)
Examples:
- **HS76**: 4 inequality constraints (mixed L and G)
- **HS93**: 2 inequality constraints (1 L, 1 G)
- **HS108**: 13 inequality constraints (mixed L and G)

Full list: HS76, HS93, HS108

## 7. Mixed Inequality Constraints of L and G Type and Equality Constraints (2 problems)
Examples:
- **CSFI1**: 1 equality, multiple inequality (mixed L and G)
- **CSFI2**: 1 equality, multiple inequality (mixed L and G)

Full list: CSFI1, CSFI2

## 8. Ranged Inequality Constraints Only (1 problem)
Examples:
- **HS83**: 3 ranged constraints

Full list: HS83

## 9. Ranged Inequality Constraints and Equality Constraints (0 problems)
*No problems found in this category*

## 10. Mixed Inequality Constraints of L, G, and R Type (2 problems)
Examples:
- **HS104**: Mixed constraint types including ranged
- **HS116**: 14 inequality constraints (13 G type, 1 ranged)

Full list: HS104, HS116

## Notes

- Ranged constraints are double-sided inequalities of the form `a <= f(x) <= b`
- In SIF files, these are specified using the RANGES section
- L type: Less than constraints `f(x) <= b`
- G type: Greater than constraints `f(x) >= b`
- E type: Equality constraints `f(x) = b`

Some categories have no problems because:
- Most problems with L-type constraints also have G-type constraints
- Ranged constraints are relatively rare in the test set
- The combination of ranged constraints with equality constraints doesn't appear in the active problem set