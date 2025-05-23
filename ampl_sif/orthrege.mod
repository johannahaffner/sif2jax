#
#**************************
# SET UP THE INITIAL DATA *
#**************************
#   Problem :
#   *********
#   An orthogonal regression problem,
#   The problem is to fit (orthogonally) an elliptic helix to a
#   set of points in 3D space. This set of points is generated by
#   perturbing a first set lying exactly on a predifined helix
#   centered at the origin.
#   Source:
#   M. Gulliksson,
#   "Algorithms for nonlinear Least-squares with Applications to
#   Orthogonal Regression",
#   UMINF-178.90, University of Umea, Sweden, 1990.
#   SIF input: Ph. Toint, June 1990.
#   classification QOR2-AY-V-V
#   Number of data points
#   (number of variables = 3 NPTS + 6 )
#   True helix parameters (centered at the origin)
#   Perturbation parameters
#   Constants
#   Computed parameters
#   Construct the data points
#   Parameters of the helix
#   Projections of the data points onto the helix
#   Solution
	param npts := 10;
	param tp4 := 1.7;
	param tp5 := 0.8;
	param tp6 := 2.0;
	param pseed := 237.1531;
	param psize := 0.2;
	param pi := 3.1415926535;
	param rnpts := 10.0;
	param icr0 := 1.0 / (10.0);
	param incr := (1.0 / (10.0)) * (2.0 * (3.1415926535));
	param im1 := -1 + (10);
	param rim1 := 9.0;
	param theta := (9.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)));
	param st := sin((9.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535))));
	param ct := cos((9.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535))));
	param r1 := (1.7) * (cos((9.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))));
	param r2 := (0.8) * (sin((9.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))));
	param r3 := (2.0) * ((9.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535))));
	param xseed := ((9.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))) * (237.1531);
	param sseed := cos(((9.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))) * 
	(237.1531));
	param perm1 := (0.2) * (cos(((9.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))) 
	* (237.1531)));
	param pert := 1.0 + ((0.2) * (cos(((9.0) * ((1.0 / (10.0)) * (2.0 * 
	(3.1415926535)))) * (237.1531))));
	param xd1 := ((1.7) * (cos((0.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))))) 
	* (1.0 + ((0.2) * (cos(((0.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))) * 
	(237.1531)))));
	param yd1 := ((0.8) * (sin((0.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))))) 
	* (1.0 + ((0.2) * (cos(((0.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))) * 
	(237.1531)))));
	param zd1 := ((2.0) * ((0.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535))))) * 
	(1.0 + ((0.2) * (cos(((0.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))) * 
	(237.1531)))));
	param xd2 := ((1.7) * (cos((1.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))))) 
	* (1.0 + ((0.2) * (cos(((1.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))) * 
	(237.1531)))));
	param yd2 := ((0.8) * (sin((1.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))))) 
	* (1.0 + ((0.2) * (cos(((1.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))) * 
	(237.1531)))));
	param zd2 := ((2.0) * ((1.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535))))) * 
	(1.0 + ((0.2) * (cos(((1.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))) * 
	(237.1531)))));
	param xd3 := ((1.7) * (cos((2.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))))) 
	* (1.0 + ((0.2) * (cos(((2.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))) * 
	(237.1531)))));
	param yd3 := ((0.8) * (sin((2.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))))) 
	* (1.0 + ((0.2) * (cos(((2.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))) * 
	(237.1531)))));
	param zd3 := ((2.0) * ((2.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535))))) * 
	(1.0 + ((0.2) * (cos(((2.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))) * 
	(237.1531)))));
	param xd4 := ((1.7) * (cos((3.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))))) 
	* (1.0 + ((0.2) * (cos(((3.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))) * 
	(237.1531)))));
	param yd4 := ((0.8) * (sin((3.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))))) 
	* (1.0 + ((0.2) * (cos(((3.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))) * 
	(237.1531)))));
	param zd4 := ((2.0) * ((3.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535))))) * 
	(1.0 + ((0.2) * (cos(((3.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))) * 
	(237.1531)))));
	param xd5 := ((1.7) * (cos((4.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))))) 
	* (1.0 + ((0.2) * (cos(((4.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))) * 
	(237.1531)))));
	param yd5 := ((0.8) * (sin((4.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))))) 
	* (1.0 + ((0.2) * (cos(((4.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))) * 
	(237.1531)))));
	param zd5 := ((2.0) * ((4.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535))))) * 
	(1.0 + ((0.2) * (cos(((4.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))) * 
	(237.1531)))));
	param xd6 := ((1.7) * (cos((5.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))))) 
	* (1.0 + ((0.2) * (cos(((5.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))) * 
	(237.1531)))));
	param yd6 := ((0.8) * (sin((5.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))))) 
	* (1.0 + ((0.2) * (cos(((5.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))) * 
	(237.1531)))));
	param zd6 := ((2.0) * ((5.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535))))) * 
	(1.0 + ((0.2) * (cos(((5.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))) * 
	(237.1531)))));
	param xd7 := ((1.7) * (cos((6.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))))) 
	* (1.0 + ((0.2) * (cos(((6.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))) * 
	(237.1531)))));
	param yd7 := ((0.8) * (sin((6.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))))) 
	* (1.0 + ((0.2) * (cos(((6.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))) * 
	(237.1531)))));
	param zd7 := ((2.0) * ((6.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535))))) * 
	(1.0 + ((0.2) * (cos(((6.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))) * 
	(237.1531)))));
	param xd8 := ((1.7) * (cos((7.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))))) 
	* (1.0 + ((0.2) * (cos(((7.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))) * 
	(237.1531)))));
	param yd8 := ((0.8) * (sin((7.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))))) 
	* (1.0 + ((0.2) * (cos(((7.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))) * 
	(237.1531)))));
	param zd8 := ((2.0) * ((7.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535))))) * 
	(1.0 + ((0.2) * (cos(((7.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))) * 
	(237.1531)))));
	param xd9 := ((1.7) * (cos((8.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))))) 
	* (1.0 + ((0.2) * (cos(((8.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))) * 
	(237.1531)))));
	param yd9 := ((0.8) * (sin((8.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))))) 
	* (1.0 + ((0.2) * (cos(((8.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))) * 
	(237.1531)))));
	param zd9 := ((2.0) * ((8.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535))))) * 
	(1.0 + ((0.2) * (cos(((8.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))) * 
	(237.1531)))));
	param xd10 := ((1.7) * (cos((9.0) * ((1.0 / (10.0)) * (2.0 * 
	(3.1415926535)))))) * (1.0 + ((0.2) * (cos(((9.0) * ((1.0 / (10.0)) * (2.0 * 
	(3.1415926535)))) * (237.1531)))));
	param yd10 := ((0.8) * (sin((9.0) * ((1.0 / (10.0)) * (2.0 * 
	(3.1415926535)))))) * (1.0 + ((0.2) * (cos(((9.0) * ((1.0 / (10.0)) * (2.0 * 
	(3.1415926535)))) * (237.1531)))));
	param zd10 := ((2.0) * ((9.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535))))) * 
	(1.0 + ((0.2) * (cos(((9.0) * ((1.0 / (10.0)) * (2.0 * (3.1415926535)))) * 
	(237.1531)))));

	var p1 := 1.0;
	var p2 := 0;
	var p3 := 1.0;
	var p4 := 1.0;
	var p5 := 0;
	var p6 >= 0.0010 ,  := 0.25;
	var x1 := 2.04;
	var y1 := 0;
	var z1 := 0;
	var x2 := 1.3158481289749517;
	var y2 := 0.44989158873434404;
	var z2 := 1.202289530540075;
	var x3 := 0.43008898645023036;
	var y3 := 0.62290719668541;
	var z3 := 2.0576281634294435;
	var x4 := -0.5892382455447966;
	var y4 := 0.8534065167622089;
	var z4 := 4.2285430952720615;
	var x5 := -1.5523083447536603;
	var y5 := 0.5307378960995349;
	var z5 := 5.673372268706862;
	var x6 := -1.3985753920069166;
	var y6 := 5.909729205593858e-11;
	var z6 := 5.1691225610527205;
	var x7 := -1.303813937390017;
	var y7 := -0.44577707007309214;
	var z7 := 7.147763387322456;
	var x8 := -0.6302880692347891;
	var y8 := -0.9128598651915989;
	var z8 := 10.55396634163662;
	var x9 := 0.5072520157785263;
	var y9 := -0.7346640841993881;
	var z9 := 9.707163559929125;
	var x10 := 1.1210098714058645;
	var y10 := -0.3832759276080039;
	var z10 := 9.218393538502124;

minimize obj:
	(x1 - xd1)*(x1 - xd1) + (y1)*(y1) + (z1)*(z1) + (x2 - xd2)*(x2 
	- xd2) + (y2 - yd2)*(y2 - yd2) + 
	(z2 - zd2)*(z2 - zd2) + (x3 - 
	xd3)*(x3 - xd3) + (y3 - yd3)*(y3 - 
	yd3) + (z3 - zd3)*(z3 - zd3) + (x4 + 
	xd4)*(x4 + xd4) + (y4 - yd4)*(y4 - 
	yd4) + (z4 - zd4)*(z4 - zd4) + (x5 
	+ xd5)*(x5 + xd5) + (y5 - yd5)*(y5 
	- yd5) + (z5 - zd5)*(z5 - zd5) + (x6 
	+ xd6)*(x6 + xd6) + (y6 - 
	yd6)*(y6 - yd6) + (z6 - 
	zd6)*(z6 - zd6) + (x7 + xd7)*(x7 + 
	xd7) + (y7 + yd7)*(y7 + yd7) + 
	(z7 - zd7)*(z7 - zd7) + (x8 + 
	xd8)*(x8 + xd8) + (y8 + yd8)*(y8 + 
	yd8) + (z8 - zd8)*(z8 - zd8) + (x9 - 
	xd9)*(x9 - xd9) + (y9 + yd9)*(y9 + 
	yd9) + (z9 - zd9)*(z9 - zd9) + (x10 
	- xd10)*(x10 - xd10) + (y10 + 
	yd10)*(y10 + yd10) + (z10 - zd10)*(z10 
	- zd10);

subject to a1:
	-((p4)*cos(((z1-p3)/p6))) + x1 - p1 = 0;
subject to b1:
	-((p5)*cos(((z1-p3)/p6))) + y1 - p2 = 0;
subject to a2:
	-((p4)*cos(((z2-p3)/p6))) + x2 - p1 = 0;
subject to b2:
	-((p5)*cos(((z2-p3)/p6))) + y2 - p2 = 0;
subject to a3:
	-((p4)*cos(((z3-p3)/p6))) + x3 - p1 = 0;
subject to b3:
	-((p5)*cos(((z3-p3)/p6))) + y3 - p2 = 0;
subject to a4:
	-((p4)*cos(((z4-p3)/p6))) + x4 - p1 = 0;
subject to b4:
	-((p5)*cos(((z4-p3)/p6))) + y4 - p2 = 0;
subject to a5:
	-((p4)*cos(((z5-p3)/p6))) + x5 - p1 = 0;
subject to b5:
	-((p5)*cos(((z5-p3)/p6))) + y5 - p2 = 0;
subject to a6:
	-((p4)*cos(((z6-p3)/p6))) + x6 - p1 = 0;
subject to b6:
	-((p5)*cos(((z6-p3)/p6))) + y6 - p2 = 0;
subject to a7:
	-((p4)*cos(((z7-p3)/p6))) + x7 - p1 = 0;
subject to b7:
	-((p5)*cos(((z7-p3)/p6))) + y7 - p2 = 0;
subject to a8:
	-((p4)*cos(((z8-p3)/p6))) + x8 - p1 = 0;
subject to b8:
	-((p5)*cos(((z8-p3)/p6))) + y8 - p2 = 0;
subject to a9:
	-((p4)*cos(((z9-p3)/p6))) + x9 - p1 = 0;
subject to b9:
	-((p5)*cos(((z9-p3)/p6))) + y9 - p2 = 0;
subject to a10:
	-((p4)*cos(((z10-p3)/p6))) + x10 - p1 = 0;
subject to b10:
	-((p5)*cos(((z10-p3)/p6))) + y10 - p2 = 0;

solve;
	display p1;
	display p2;
	display p3;
	display p4;
	display p5;
	display p6;
	display x1;
	display y1;
	display z1;
	display x2;
	display y2;
	display z2;
	display x3;
	display y3;
	display z3;
	display x4;
	display y4;
	display z4;
	display x5;
	display y5;
	display z5;
	display x6;
	display y6;
	display z6;
	display x7;
	display y7;
	display z7;
	display x8;
	display y8;
	display z8;
	display x9;
	display y9;
	display z9;
	display x10;
	display y10;
	display z10;
display obj;
