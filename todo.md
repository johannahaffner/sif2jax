# Work packages for Claude

This file outlines tasks for Claude Code to complete. 

## Assess families of problems to implement

Problems from the same family often show synergies in implementation - patterns may show up more than once, and this can help when troubleshooting. 

@Claude: From this list, prioritise families of problems to implement according to these criteria: 

- problem complexity: lower is better 
- subset already implemented: yes is better (e.g.: we have working implementations of problems from the POWELL series -> then we can probably leverage what we learned from these to implement the rest)
- problem has extensive dataset: work with no or small problem data
- size of problem family: larger is better 

Give either one or zero points according to the criteria above. Then annotate the families below with their priority score: 

- Four points: priority 1
- Three points: priority 2
- Up to two points: priority 3

(Priority) NAME
- (1) SPIN* - Score: 4 (simple problems, 2 already exist (SPINLS/SPINOP), no data, family size: 6)
- (2) MISRA* - Score: 3 (moderate complexity, MISRA1D implemented, small data, family size: 6+)
- (2) GAUSS* - Score: 3 (moderate complexity, several implemented, small data, family size: 6)
- (2) HEART* - Score: 3 (moderate complexity, HEART6LS/HEART8LS implemented, small data, family size: 4)
- (2) MGH* - Score: 3 (moderate complexity, MGH09/MGH10LS implemented, no data, family size: 5+)
- (2) LANCZOS* - Score: 3 (moderate complexity, no subset, small data, family size: 3)
- (2) SYNTHE* - Score: 3 (synthetic test problems, no subset, no data, family size: 3)
- (2) TABLE* - Score: 3 (simple table lookup problems, no subset, small data, family size: 6)
- (2) WALL* - Score: 3 (simple wall problems, no subset, no data, family size: 4)
- (3) PALMER* - Score: 2 (complex, many already implemented, large data, family size: 60+)
- (3) POWELL* - Score: 2 (complex, POWELLSG/POWELLBS/etc implemented, no data, family size: 10+)
- (3) OSBORNE* - Score: 2 (complex, OSBORNEA/B implemented, moderate data, family size: 3)
- (3) TORSION* - Score: 2 (complex mechanics problems, no subset, no data, family size: 12)
- (3) ZAMB2* - Score: 2 (complex problems, no subset, no data, family size: 5)
- (3) WAYSEA* - Score: 2 (complex problems, no subset, no data, family size: 4)
- (3) VESUV* - Score: 2 (complex problems, no subset, no data, family size: 6)
- (3) TRO* - Score: 2 (trust region problems, no subset, no data, family size: 11)
- (3) TENBARS* - Score: 2 (structural problems, no subset, no data, family size: 4)
- (3) TAX* - Score: 2 (tax calculation problems, no subset, no data, family size: 6)
- (3) STEE* - Score: 2 (STEENBRB exists but flagged, no data, family size: 8+)
- (3) SIPOW* - Score: 2 (SIPOW1/2 implemented, no data, family size: 4)
- (3) SEMI* - Score: 2 (semiconductor problems, no subset, no data, family size: 4)
- (3) READING* - Score: 2 (reading problems, no subset, no data, family size: 9)
- (3) RDW2D* - Score: 2 (2D problems, no subset, no data, family size: 5)
- (3) PRIMAL* - Score: 2 (primal formulations, no subset, no data, family size: 8)
- (3) PORTFL* - Score: 2 (portfolio problems, no subset, no data, family size: 6)
- (3) OPTC* - Score: 2 (optimal control, no subset, no data, family size: 6)
- (3) OET* - Score: 2 (test problems, no subset, no data, family size: 7)
- (3) OBST* - Score: 2 (obstacle problems, no subset, no data, family size: 5)
- (3) NET* - Score: 2 (network problems, no subset, no data, family size: 4)
- (3) MSS* - Score: 2 (mass-spring systems, no subset, no data, family size: 3)
- (3) MPC* - Score: 2 (MPC problems, no subset, no data, family size: 16)
- (3) METHAN* - Score: 2 (methane problems, no subset, moderate data, family size: 4)
- (3) LUKVLE* - Score: 2 (Luksan problems, no subset, no data, family size: 8)
- (3) LUKVLI* - Score: 2 (Luksan problems, no subset, no data, family size: 8)
- (3) LISWET* - Score: 2 (Liswet problems, no subset, no data, family size: 12)
- (3) LEUVEN* - Score: 2 (Leuven problems, no subset, no data, family size: 7)
- (3) HYDC* - Score: 1 (complex hydrocarbon, no subset, large data, family size: 2)
- (3) HS* - Score: 1 (complex Hock-Schittkowski, many exist, no data, family size: 40+)
- (3) HIMMEL* - Score: 1 (complex, some exist, no data, family size: 12)
- (3) HIER1* - Score: 1 (hierarchical problems, no subset, large data, family size: 7)
- (3) GRIDNET* - Score: 1 (grid network problems, no subset, large structure, family size: 9)
- (3) GMNCASE* - Score: 1 (complex cases, no subset, large data, family size: 4)
- (3) FLOS* - Score: 1 (flow shop problems, no subset, large data, family size: 6)
- (3) EIG* - Score: 1 (eigenvalue problems, no subset, large structure, family size: 15)
- (3) DUAL* - Score: 1 (dual formulations, no subset, large structure, family size: 8)
- (3) DRCAV* - Score: 1 (driven cavity problems, no subset, large data, family size: 6)
- (3) DMN* - Score: 0 (complex problems, no subset, large data, family size: 12)
- (3) DIAG* - Score: 0 (diagonal problems, no subset, large structure, family size: 9)
- (3) DEG* - Score: 1 (degenerate problems, DEGDIAG exists, large structure, family size: 10)
- (3) DALLAS* - Score: 2 (Dallas problems, no subset, no data, family size: 3)
- (3) CYCL* - Score: 1 (cyclic problems, CYCLIC3 exists, no data, family size: 8)
- (3) CMPC* - Score: 0 (complex MPC, no subset, large data, family size: 16)
- (3) CLEUVEN* - Score: 0 (complex Leuven, no subset, large data, family size: 7)
- (3) CHARD* - Score: 2 (character display, no subset, no data, family size: 4)
- (3) BROY* - Score: 1 (Broyden problems, BROYDN3D etc exist, large structure, family size: 6)
- (3) BRATU* - Score: 1 (Bratu problems, BRATU2DT exists, large structure, family size: 3)
- (3) BLOCK* - Score: 2 (block QP problems, no subset, no data, family size: 5)
- (3) AUG* - Score: 1 (augmented problems, no subset, large structure, family size: 8)
- (3) ALLIN* - Score: 2 (all-in problems, no subset, no data, family size: 5)
- (3) ACOPP* - Score: 0 (complex ACOPP, no subset, large data, family size: 10)
- (3) A5* - Score: 0 (complex A5 problems, no subset, large data, family size: 23)
- (3) A2* - Score: 0 (complex A2 problems, no subset, large data, family size: 15)
- (3) A0* - Score: 0 (complex A0 problems, no subset, large data, family size: 16)

## Research strategies to implement data-heavy problems (Johanna)

@Claude: please add problems that are data-heavy to this section. I will research the best ways we can include them. 

### Already identified:
- MNIST* (MNISTS0, MNISTS0LS, MNISTS5, MNISTS5LS) - MNIST dataset-based problems
- FBRAIN* (FBRAIN, FBRAIN2, FBRAIN2LS, FBRAIN2NE, FBRAIN3, FBRAIN3LS, FBRAINLS, FBRAINNE) - Brain imaging data
- DIAMON* (DIAMON2D, DIAMON2DLS, DIAMON3D, DIAMON3DLS) - Diamond data problems
- CERI* (CERI651A-E and LS variants) - CERI data problems
- BRAIN* (BRAINPC0-9, GBRAIN, GBRAINLS) - Brain data problems

### Additional data-heavy problems found:
- PALMER* family - All PALMER problems use experimental data points (35 data points each)
- OSBORNE* (OSBORNE1, OSBORNE2, OSBORNEB) - Osborne's regression data
- MGH* (MGH09LS, MGH10, MGH10S, MGH10SLS, MGH17, MGH17LS, MGH17S, MGH17SLS) - More, Garbow & Hillstrom test data
- MISRA* (MISRA1A-C and LS variants) - NIST/MISRA regression datasets
- GAUSS* (GAUSS1-3 and LS variants, GAUSSELM) - Gaussian fit data
- HEART* (HEART6, HEART6LS, HEART8, HEART8LS) - Heart rate data
- LANCZOS* (LANCZOS1-3, LANCZOS3LS) - Lanczos data
- TABLE* (TABLE1, TABLE3, TABLE6, TABLE7, TABLE8) - Tabulated data
- RAT* (RAT42, RAT42LS, RAT43, RAT43LS) - Rational function data
- THURBER* (THURBER, THURBERLS) - Thurber semiconductor data
- ROSZMAN1 - Quantum defects data
- ECKERLE4* (ECKERLE4, ECKERLE4LS) - Circular interference data
- CHWIRUT* (CHWIRUT1, CHWIRUT2) - Ultrasonic reference block data
- KIRBY2 - Scanning electron microscope data
- HAHN1 - Thermal expansion data
- BENNETT5 - Bennett's data (already implemented)
- STRATEC - Strategic planning data