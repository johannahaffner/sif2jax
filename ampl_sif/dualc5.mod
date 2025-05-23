#
#**************************
# SET UP THE INITIAL DATA *
#**************************
#   Problem :
#   *********
#   A dual quadratic program from Antonio Frangioni (frangio@DI.UniPi.IT)
#   This is the dual of PRIMALC5.SIF
#   SIF input: Irv Lustig and Nick Gould, June 1996.
#   classification QLR2-MN-8-278
#   Solution
	param n := 8;
	param ip1 := 1 + (8);

	var x1 >= 0.0 ,  <= 1.0, :=0;
	var x2 >= 0.0 ,  <= 1.0, :=0;
	var x3 >= 0.0 ,  <= 1.0, :=0;
	var x4 >= 0.0 ,  <= 1.0, :=0;
	var x5 >= 0.0 ,  <= 1.0, :=0;
	var x6 >= 0.0 ,  <= 1.0, :=0;
	var x7 >= 0.0 ,  <= 1.0, :=0;
	var x8 >= 0.0 ,  <= 1.0, :=0;

minimize obj:
	13053.0*0.5 * x1 * x1 + 2524.0*x1 * x2 + 869.0*x1 * x3 + 2342.0*x1 * x4 - 
	4967.0*x1 * x5 + 2742.0*x1 * x6 + 12580.0*x1 * x7 - 1574.0*x1 * x8 + 9564.0*0.5 
	* x2 * x2 + 4394.0*x2 * x3 + 16968.0*x2 * x4 - 6110.0*x2 * x5 + 1727.0*x2 * x6 
	+ 5191.0*x2 * x7 - 1101.0*x2 * x8 + 11069.0*0.5 * x3 * x3 + 15583.0*x3 * x4 - 
	5984.0*x3 * x5 + 1344.0*x3 * x6 + 392.0*x3 * x7 + 4540.0*x3 * x8 + 54824.0*0.5 
	* x4 * x4 - 10447.0*x4 * x5 - 3459.0*x4 * x6 + 7868.0*x4 * x7 - 6775.0*x4 * x8 
	+ 7219.0*0.5 * x5 * x5 - 4378.0*x5 * x6 - 7789.0*x5 * x7 - 1994.0*x5 * x8 + 
	6429.0*0.5 * x6 * x6 + 6769.0*x6 * x7 + 4433.0*x6 * x8 + 44683.0*0.5 * x7 * x7 
	- 19349.0*x7 * x8 + 22577.0*0.5 * x8 * x8 + 546.88509078*x1 + 122.12095223*x3 + 
	616.11938925*x4 + 586.52942284*x5 + 586.52942284*x6 + 953.05364036*x7 + 
	1585.0340121*x8;

subject to c1:
	x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 = 1;
subject to c2:
	0 <= 64.0*x1 + 64.0*x2 + 64.0*x3 + 64.0*x4 + 64.0*x5 + 64.0*x6 + 64.0*x7 + 
	46.0*x8;
subject to c3:
	0 <= 218.0*x1 + 218.0*x2 + 218.0*x3 + 218.0*x4 + 218.0*x5 + 218.0*x6 + 218.0*x7 
	+ 218.0*x8;
subject to c4:
	0 <= 412.0*x1 + 412.0*x2 + 412.0*x3 + 412.0*x4 + 412.0*x5 + 412.0*x6 + 412.0*x7 
	+ 412.0*x8;
subject to c5:
	0 <= 281.0*x1 + 244.0*x2 + 254.0*x3 + 215.0*x4 + 281.0*x5 + 281.0*x6 + 254.0*x7 
	+ 289.0*x8;
subject to c6:
	0 <= 187.0*x1 + 181.0*x2 + 181.0*x3 + 181.0*x4 + 160.0*x5 + 187.0*x6 + 166.0*x7 
	+ 193.0*x8;
subject to c7:
	0 <= 330.0*x1 + 338.0*x2 + 338.0*x3 + 337.0*x4 + 338.0*x5 + 338.0*x6 + 338.0*x7 
	+ 338.0*x8;
subject to c8:
	0 <= 172.0*x1 + 191.0*x2 + 191.0*x3 + 191.0*x4 + 191.0*x5 + 179.0*x6 + 153.0*x7 
	+ 191.0*x8;
subject to c9:
	0 <= 255.0*x1 + 255.0*x2 + 255.0*x3 + 255.0*x4 + 255.0*x5 + 255.0*x6 + 255.0*x7 
	+ 255.0*x8;
subject to c10:
	0 <= 423.0*x1 + 423.0*x2 + 454.0*x3 + 454.0*x4 + 423.0*x5 + 423.0*x6 + 384.0*x7 
	+ 461.0*x8;
subject to c11:
	0 <= 455.0*x1 + 460.0*x2 + 460.0*x3 + 460.0*x4 + 460.0*x5 + 455.0*x6 + 455.0*x7 
	+ 455.0*x8;
subject to c12:
	0 <= 304.0*x1 + 304.0*x2 + 304.0*x3 + 304.0*x4 + 325.0*x5 + 304.0*x6 + 325.0*x7 
	+ 297.0*x8;
subject to c13:
	0 <= 180.0*x1 + 180.0*x2 + 180.0*x3 + 180.0*x4 + 180.0*x5 + 180.0*x6 + 180.0*x7 
	+ 180.0*x8;
subject to c14:
	0 <= 491.0*x1 + 483.0*x2 + 483.0*x3 + 484.0*x4 + 483.0*x5 + 491.0*x6 + 491.0*x7 
	+ 483.0*x8;
subject to c15:
	0 <= 121.0*x1 + 121.0*x2 + 121.0*x3 + 121.0*x4 + 121.0*x5 + 121.0*x6 + 121.0*x7 
	+ 121.0*x8;
subject to c16:
	0 <= 500.0*x1 + 500.0*x2 + 500.0*x3 + 495.0*x4 + 500.0*x5 + 500.0*x6 + 463.0*x7 
	+ 518.0*x8;
subject to c17:
	0 <= 249.0*x1 + 249.0*x2 + 249.0*x3 + 249.0*x4 + 249.0*x5 + 249.0*x6 + 249.0*x7 
	+ 249.0*x8;
subject to c18:
	0 <= 163.0*x1 + 196.0*x2 + 196.0*x3 + 159.0*x4 + 201.0*x5 + 200.0*x6 + 123.0*x7 
	+ 211.0*x8;
subject to c19:
	0 <= 176.0*x1 + 176.0*x2 + 176.0*x3 + 176.0*x4 + 176.0*x5 + 176.0*x6 + 176.0*x7 
	+ 176.0*x8;
subject to c20:
	0 <= 426.0*x1 + 426.0*x2 + 426.0*x3 + 426.0*x4 + 421.0*x5 + 426.0*x6 + 485.0*x7 
	+ 411.0*x8;
subject to c21:
	0 <= 315.0*x1 + 315.0*x2 + 315.0*x3 + 315.0*x4 + 315.0*x5 + 315.0*x6 + 315.0*x7 
	+ 315.0*x8;
subject to c22:
	0 <= 35.0*x1 + 48.0*x2 + 48.0*x3 + 51.0*x4 + 54.0*x5 + 54.0*x6 + 54.0*x7 + 
	41.0*x8;
subject to c23:
	0 <= 301.0*x1 + 301.0*x2 + 301.0*x3 + 301.0*x4 + 301.0*x5 + 301.0*x6 + 301.0*x7 
	+ 301.0*x8;
subject to c24:
	0 <= 143.0*x1 + 143.0*x2 + 143.0*x3 + 143.0*x4 + 143.0*x5 + 143.0*x6 + 143.0*x7 
	+ 143.0*x8;
subject to c25:
	0 <= 402.0*x1 + 402.0*x2 + 402.0*x3 + 402.0*x4 + 402.0*x5 + 402.0*x6 + 402.0*x7 
	+ 402.0*x8;
subject to c26:
	0 <= 162.0*x1 + 162.0*x2 + 162.0*x3 + 157.0*x4 + 162.0*x5 + 162.0*x6 + 160.0*x7 
	+ 162.0*x8;
subject to c27:
	0 <= 429.0*x1 + 429.0*x2 + 429.0*x3 + 434.0*x4 + 429.0*x5 + 437.0*x6 + 437.0*x7 
	+ 429.0*x8;
subject to c28:
	0 <= 219.0*x1 + 219.0*x2 + 219.0*x3 + 219.0*x4 + 219.0*x5 + 216.0*x6 + 219.0*x7 
	+ 216.0*x8;
subject to c29:
	0 <= 401.0*x1 + 403.0*x2 + 403.0*x3 + 403.0*x4 + 403.0*x5 + 403.0*x6 + 403.0*x7 
	+ 403.0*x8;
subject to c30:
	0 <= 10.0*x1 + 5.0*x2 + 5.0*x3 + 5.0*x4 + 5.0*x5 + 10.0*x6 + 10.0*x7 + 2.0*x8;
subject to c31:
	0 <= 356.0*x1 + 393.0*x2 + 393.0*x3 + 393.0*x4 + 393.0*x5 + 393.0*x6 + 393.0*x7 
	+ 393.0*x8;
subject to c32:
	0 <= 526.0*x1 + 531.0*x2 + 531.0*x3 + 531.0*x4 + 531.0*x5 + 526.0*x6 + 467.0*x7 
	+ 526.0*x8;
subject to c33:
	0 <= 511.0*x1 + 486.0*x2 + 476.0*x3 + 476.0*x4 + 452.0*x5 + 491.0*x6 + 523.0*x7 
	+ 490.0*x8;
subject to c34:
	0 <= 211.0*x1 + 211.0*x2 + 211.0*x3 + 199.0*x4 + 211.0*x5 + 211.0*x6 + 211.0*x7 
	+ 211.0*x8;
subject to c35:
	0 <= 304.0*x1 + 304.0*x2 + 302.0*x3 + 302.0*x4 + 273.0*x5 + 302.0*x6 + 299.0*x7 
	+ 302.0*x8;
subject to c36:
	0 <= 495.0*x1 + 496.0*x2 + 496.0*x3 + 508.0*x4 + 502.0*x5 + 484.0*x6 + 495.0*x7 
	+ 489.0*x8;
subject to c37:
	0 <= 328.0*x1 + 328.0*x2 + 328.0*x3 + 328.0*x4 + 328.0*x5 + 328.0*x6 + 328.0*x7 
	+ 328.0*x8;
subject to c38:
	0 <= 218.0*x1 + 218.0*x2 + 218.0*x3 + 218.0*x4 + 218.0*x5 + 203.0*x6 + 218.0*x7 
	+ 218.0*x8;
subject to c39:
	0 <= 285.0*x1 + 285.0*x2 + 285.0*x3 + 285.0*x4 + 264.0*x5 + 285.0*x6 + 264.0*x7 
	+ 303.0*x8;
subject to c40:
	0 <= 261.0*x1 + 261.0*x2 + 261.0*x3 + 261.0*x4 + 261.0*x5 + 261.0*x6 + 261.0*x7 
	+ 261.0*x8;
subject to c41:
	0 <= 304.0*x1 + 369.0*x2 + 331.0*x3 + 358.0*x4 + 311.0*x5 + 331.0*x6 + 294.0*x7 
	+ 331.0*x8;
subject to c42:
	0 <= 158.0*x1 + 158.0*x2 + 158.0*x3 + 158.0*x4 + 200.0*x5 + 167.0*x6 + 158.0*x7 
	+ 165.0*x8;
subject to c43:
	0 <= 283.0*x1 + 283.0*x2 + 283.0*x3 + 283.0*x4 + 283.0*x5 + 283.0*x6 + 283.0*x7 
	+ 283.0*x8;
subject to c44:
	0 <= 167.0*x1 + 167.0*x2 + 167.0*x3 + 167.0*x4 + 167.0*x5 + 167.0*x6 + 167.0*x7 
	+ 167.0*x8;
subject to c45:
	0 <= 572.0*x1 + 572.0*x2 + 572.0*x3 + 572.0*x4 + 572.0*x5 + 572.0*x6 + 572.0*x7 
	+ 572.0*x8;
subject to c46:
	0 <= 162.0*x1 + 188.0*x2 + 105.0*x3 + 167.0*x4 + 126.0*x5 + 162.0*x6 + 162.0*x7 
	+ 162.0*x8;
subject to c47:
	0 <= 370.0*x1 + 375.0*x2 + 375.0*x3 + 375.0*x4 + 406.0*x5 + 370.0*x6 + 370.0*x7 
	+ 406.0*x8;
subject to c48:
	0 <= 116.0*x1 + 116.0*x2 + 116.0*x3 + 116.0*x4 + 116.0*x5 + 116.0*x6 + 116.0*x7 
	+ 116.0*x8;
subject to c49:
	0 <= 215.0*x1 + 215.0*x2 + 215.0*x3 + 215.0*x4 + 215.0*x5 + 215.0*x6 + 215.0*x7 
	+ 215.0*x8;
subject to c50:
	0 <= 538.0*x1 + 538.0*x2 + 538.0*x3 + 538.0*x4 + 538.0*x5 + 538.0*x6 + 538.0*x7 
	+ 538.0*x8;
subject to c51:
	0 <= 478.0*x1 + 478.0*x2 + 478.0*x3 + 478.0*x4 + 478.0*x5 + 478.0*x6 + 478.0*x7 
	+ 486.0*x8;
subject to c52:
	0 <= 256.0*x1 + 246.0*x2 + 246.0*x3 + 256.0*x4 + 256.0*x5 + 246.0*x6 + 246.0*x7 
	+ 246.0*x8;
subject to c53:
	0 <= 458.0*x1 + 459.0*x2 + 459.0*x3 + 459.0*x4 + 459.0*x5 + 459.0*x6 + 457.0*x7 
	+ 459.0*x8;
subject to c54:
	0 <= 370.0*x1 + 370.0*x2 + 370.0*x3 + 370.0*x4 + 370.0*x5 + 370.0*x6 + 370.0*x7 
	+ 370.0*x8;
subject to c55:
	0 <= 134.0*x1 + 134.0*x2 + 134.0*x3 + 134.0*x4 + 134.0*x5 + 134.0*x6 + 134.0*x7 
	+ 98.0*x8;
subject to c56:
	0 <= 236.0*x1 + 236.0*x2 + 236.0*x3 + 236.0*x4 + 236.0*x5 + 227.0*x6 + 236.0*x7 
	+ 227.0*x8;
subject to c57:
	0 <= 423.0*x1 + 431.0*x2 + 431.0*x3 + 431.0*x4 + 431.0*x5 + 423.0*x6 + 423.0*x7 
	+ 431.0*x8;
subject to c58:
	0 <= 60.0*x1 + 51.0*x2 + 51.0*x3 + 51.0*x4 + 51.0*x5 + 59.0*x6 + 61.0*x7 + 
	43.0*x8;
subject to c59:
	0 <= 484.0*x1 + 484.0*x2 + 484.0*x3 + 484.0*x4 + 484.0*x5 + 484.0*x6 + 484.0*x7 
	+ 484.0*x8;
subject to c60:
	0 <= 316.0*x1 + 316.0*x2 + 316.0*x3 + 333.0*x4 + 316.0*x5 + 333.0*x6 + 316.0*x7 
	+ 333.0*x8;
subject to c61:
	0 <= 107.0*x1 + 105.0*x2 + 105.0*x3 + 105.0*x4 + 105.0*x5 + 105.0*x6 + 105.0*x7 
	+ 105.0*x8;
subject to c62:
	0 <= 274.0*x1 + 274.0*x2 + 274.0*x3 + 274.0*x4 + 269.0*x5 + 274.0*x6 + 274.0*x7 
	+ 259.0*x8;
subject to c63:
	0 <= 79.0*x1 + 79.0*x2 + 81.0*x3 + 110.0*x4 + 101.0*x5 + 81.0*x6 + 118.0*x7 + 
	91.0*x8;
subject to c64:
	0 <= 416.0*x1 + 424.0*x2 + 424.0*x3 + 387.0*x4 + 429.0*x5 + 424.0*x6 + 387.0*x7 
	+ 429.0*x8;
subject to c65:
	0 <= 379.0*x1 + 366.0*x2 + 376.0*x3 + 384.0*x4 + 366.0*x5 + 378.0*x6 + 381.0*x7 
	+ 378.0*x8;
subject to c66:
	0 <= 356.0*x1 + 366.0*x2 + 366.0*x3 + 356.0*x4 + 356.0*x5 + 357.0*x6 + 366.0*x7 
	+ 357.0*x8;
subject to c67:
	0 <= 539.0*x1 + 539.0*x2 + 539.0*x3 + 539.0*x4 + 539.0*x5 + 539.0*x6 + 539.0*x7 
	+ 539.0*x8;
subject to c68:
	0 <= 263.0*x1 + 265.0*x2 + 265.0*x3 + 265.0*x4 + 265.0*x5 + 265.0*x6 + 265.0*x7 
	+ 265.0*x8;
subject to c69:
	0 <= 457.0*x1 + 457.0*x2 + 457.0*x3 + 457.0*x4 + 457.0*x5 + 469.0*x6 + 457.0*x7 
	+ 469.0*x8;
subject to c70:
	0 <= 121.0*x1 + 121.0*x2 + 121.0*x3 + 121.0*x4 + 121.0*x5 + 121.0*x6 + 121.0*x7 
	+ 121.0*x8;
subject to c71:
	0 <= 417.0*x1 + 417.0*x2 + 417.0*x3 + 417.0*x4 + 417.0*x5 + 417.0*x6 + 417.0*x7 
	+ 417.0*x8;
subject to c72:
	0 <= 383.0*x1 + 383.0*x2 + 371.0*x3 + 354.0*x4 + 389.0*x5 + 371.0*x6 + 371.0*x7 
	+ 371.0*x8;
subject to c73:
	0 <= 502.0*x1 + 502.0*x2 + 502.0*x3 + 502.0*x4 + 502.0*x5 + 502.0*x6 + 502.0*x7 
	+ 495.0*x8;
subject to c74:
	0 <= 373.0*x1 + 373.0*x2 + 373.0*x3 + 373.0*x4 + 373.0*x5 + 373.0*x6 + 373.0*x7 
	+ 373.0*x8;
subject to c75:
	0 <= 503.0*x1 + 493.0*x2 + 493.0*x3 + 493.0*x4 + 493.0*x5 + 493.0*x6 + 493.0*x7 
	+ 493.0*x8;
subject to c76:
	0 <= 281.0*x1 + 281.0*x2 + 281.0*x3 + 281.0*x4 + 281.0*x5 + 281.0*x6 + 281.0*x7 
	+ 281.0*x8;
subject to c77:
	0 <= 504.0*x1 + 504.0*x2 + 504.0*x3 + 504.0*x4 + 504.0*x5 + 504.0*x6 + 504.0*x7 
	+ 504.0*x8;
subject to c78:
	0 <= 193.0*x1 + 193.0*x2 + 193.0*x3 + 205.0*x4 + 193.0*x5 + 193.0*x6 + 193.0*x7 
	+ 193.0*x8;
subject to c79:
	0 <= 94.0*x1 + 94.0*x2 + 94.0*x3 + 83.0*x4 + 83.0*x5 + 75.0*x6 + 75.0*x7 + 
	94.0*x8;
subject to c80:
	0 <= 345.0*x1 + 345.0*x2 + 376.0*x3 + 376.0*x4 + 345.0*x5 + 345.0*x6 + 345.0*x7 
	+ 383.0*x8;
subject to c81:
	0 <= 437.0*x1 + 437.0*x2 + 437.0*x3 + 437.0*x4 + 437.0*x5 + 437.0*x6 + 437.0*x7 
	+ 431.0*x8;
subject to c82:
	0 <= 168.0*x1 + 168.0*x2 + 168.0*x3 + 168.0*x4 + 168.0*x5 + 177.0*x6 + 168.0*x7 
	+ 139.0*x8;
subject to c83:
	0 <= 237.0*x1 + 237.0*x2 + 206.0*x3 + 206.0*x4 + 237.0*x5 + 237.0*x6 + 237.0*x7 
	+ 237.0*x8;
subject to c84:
	0 <= 241.0*x1 + 233.0*x2 + 233.0*x3 + 233.0*x4 + 251.0*x5 + 229.0*x6 + 241.0*x7 
	+ 221.0*x8;
subject to c85:
	0 <= 396.0*x1 + 396.0*x2 + 406.0*x3 + 406.0*x4 + 406.0*x5 + 406.0*x6 + 392.0*x7 
	+ 392.0*x8;
subject to c86:
	0 <= 135.0*x1 + 135.0*x2 + 135.0*x3 + 135.0*x4 + 135.0*x5 + 135.0*x6 + 135.0*x7 
	+ 135.0*x8;
subject to c87:
	0 <= 323.0*x1 + 324.0*x2 + 324.0*x3 + 324.0*x4 + 312.0*x5 + 312.0*x6 + 323.0*x7 
	+ 317.0*x8;
subject to c88:
	0 <= 351.0*x1 + 351.0*x2 + 351.0*x3 + 351.0*x4 + 351.0*x5 + 351.0*x6 + 351.0*x7 
	+ 351.0*x8;
subject to c89:
	0 <= 488.0*x1 + 488.0*x2 + 488.0*x3 + 488.0*x4 + 488.0*x5 + 488.0*x6 + 488.0*x7 
	+ 488.0*x8;
subject to c90:
	0 <= 153.0*x1 + 154.0*x2 + 144.0*x3 + 145.0*x4 + 144.0*x5 + 144.0*x6 + 158.0*x7 
	+ 157.0*x8;
subject to c91:
	0 <= 288.0*x1 + 288.0*x2 + 278.0*x3 + 278.0*x4 + 278.0*x5 + 278.0*x6 + 288.0*x7 
	+ 288.0*x8;
subject to c92:
	0 <= 126.0*x1 + 126.0*x2 + 157.0*x3 + 157.0*x4 + 126.0*x5 + 126.0*x6 + 91.0*x7 
	+ 164.0*x8;
subject to c93:
	0 <= 180.0*x1 + 180.0*x2 + 180.0*x3 + 180.0*x4 + 180.0*x5 + 180.0*x6 + 180.0*x7 
	+ 186.0*x8;
subject to c94:
	0 <= 349.0*x1 + 349.0*x2 + 349.0*x3 + 349.0*x4 + 349.0*x5 + 349.0*x6 + 349.0*x7 
	+ 349.0*x8;
subject to c95:
	0 <= 287.0*x1 + 287.0*x2 + 287.0*x3 + 287.0*x4 + 287.0*x5 + 287.0*x6 + 287.0*x7 
	+ 281.0*x8;
subject to c96:
	0 <= 487.0*x1 + 487.0*x2 + 487.0*x3 + 487.0*x4 + 487.0*x5 + 487.0*x6 + 487.0*x7 
	+ 476.0*x8;
subject to c97:
	0 <= 38.0*x1 + 38.0*x2 + 38.0*x3 + 38.0*x4 + 38.0*x5 + 38.0*x6 + 38.0*x7 + 
	82.0*x8;
subject to c98:
	0 <= 175.0*x1 + 175.0*x2 + 175.0*x3 + 175.0*x4 + 151.0*x5 + 175.0*x6 + 175.0*x7 
	+ 163.0*x8;
subject to c99:
	0 <= 416.0*x1 + 416.0*x2 + 416.0*x3 + 416.0*x4 + 416.0*x5 + 416.0*x6 + 416.0*x7 
	+ 416.0*x8;
subject to c100:
	0 <= 70.0*x1 + 70.0*x2 + 70.0*x3 + 70.0*x4 + 70.0*x5 + 70.0*x6 + 68.0*x7 + 
	70.0*x8;
subject to c101:
	0 <= 253.0*x1 + 253.0*x2 + 253.0*x3 + 282.0*x4 + 253.0*x5 + 253.0*x6 + 255.0*x7 
	+ 271.0*x8;
subject to c102:
	0 <= 248.0*x1 + 248.0*x2 + 248.0*x3 + 248.0*x4 + 224.0*x5 + 248.0*x6 + 224.0*x7 
	+ 248.0*x8;
subject to c103:
	0 <= 187.0*x1 + 181.0*x2 + 181.0*x3 + 181.0*x4 + 181.0*x5 + 187.0*x6 + 152.0*x7 
	+ 181.0*x8;
subject to c104:
	0 <= 443.0*x1 + 443.0*x2 + 443.0*x3 + 443.0*x4 + 446.0*x5 + 443.0*x6 + 446.0*x7 
	+ 456.0*x8;
subject to c105:
	0 <= 218.0*x1 + 232.0*x2 + 232.0*x3 + 232.0*x4 + 232.0*x5 + 218.0*x6 + 218.0*x7 
	+ 219.0*x8;
subject to c106:
	0 <= 467.0*x1 + 467.0*x2 + 467.0*x3 + 467.0*x4 + 467.0*x5 + 467.0*x6 + 467.0*x7 
	+ 467.0*x8;
subject to c107:
	0 <= 204.0*x1 + 204.0*x2 + 214.0*x3 + 185.0*x4 + 204.0*x5 + 214.0*x6 + 214.0*x7 
	+ 214.0*x8;
subject to c108:
	0 <= 296.0*x1 + 296.0*x2 + 327.0*x3 + 327.0*x4 + 296.0*x5 + 296.0*x6 + 296.0*x7 
	+ 334.0*x8;
subject to c109:
	0 <= 379.0*x1 + 371.0*x2 + 371.0*x3 + 371.0*x4 + 371.0*x5 + 379.0*x6 + 379.0*x7 
	+ 371.0*x8;
subject to c110:
	0 <= 312.0*x1 + 307.0*x2 + 312.0*x3 + 312.0*x4 + 312.0*x5 + 312.0*x6 + 312.0*x7 
	+ 318.0*x8;
subject to c111:
	0 <= 210.0*x1 + 210.0*x2 + 210.0*x3 + 210.0*x4 + 179.0*x5 + 210.0*x6 + 207.0*x7 
	+ 210.0*x8;
subject to c112:
	0 <= 176.0*x1 + 176.0*x2 + 176.0*x3 + 176.0*x4 + 176.0*x5 + 176.0*x6 + 176.0*x7 
	+ 176.0*x8;
subject to c113:
	0 <= 486.0*x1 + 486.0*x2 + 486.0*x3 + 486.0*x4 + 486.0*x5 + 486.0*x6 + 435.0*x7 
	+ 486.0*x8;
subject to c114:
	0 <= 389.0*x1 + 394.0*x2 + 394.0*x3 + 394.0*x4 + 389.0*x5 + 389.0*x6 + 389.0*x7 
	+ 394.0*x8;
subject to c115:
	0 <= 269.0*x1 + 269.0*x2 + 269.0*x3 + 269.0*x4 + 269.0*x5 + 269.0*x6 + 320.0*x7 
	+ 269.0*x8;
subject to c116:
	0 <= 515.0*x1 + 515.0*x2 + 515.0*x3 + 515.0*x4 + 515.0*x5 + 515.0*x6 + 515.0*x7 
	+ 515.0*x8;
subject to c117:
	0 <= 499.0*x1 + 499.0*x2 + 499.0*x3 + 499.0*x4 + 499.0*x5 + 499.0*x6 + 499.0*x7 
	+ 499.0*x8;
subject to c118:
	0 <= 301.0*x1 + 301.0*x2 + 301.0*x3 + 301.0*x4 + 301.0*x5 + 301.0*x6 + 301.0*x7 
	+ 301.0*x8;
subject to c119:
	0 <= 227.0*x1 + 227.0*x2 + 227.0*x3 + 227.0*x4 + 227.0*x5 + 227.0*x6 + 227.0*x7 
	+ 229.0*x8;
subject to c120:
	0 <= 454.0*x1 + 459.0*x2 + 454.0*x3 + 454.0*x4 + 454.0*x5 + 454.0*x6 + 454.0*x7 
	+ 448.0*x8;
subject to c121:
	0 <= 146.0*x1 + 141.0*x2 + 141.0*x3 + 141.0*x4 + 146.0*x5 + 146.0*x6 + 146.0*x7 
	+ 141.0*x8;
subject to c122:
	0 <= 269.0*x1 + 269.0*x2 + 269.0*x3 + 269.0*x4 + 269.0*x5 + 269.0*x6 + 269.0*x7 
	+ 269.0*x8;
subject to c123:
	0 <= 389.0*x1 + 381.0*x2 + 381.0*x3 + 389.0*x4 + 381.0*x5 + 362.0*x6 + 362.0*x7 
	+ 420.0*x8;
subject to c124:
	0 <= 259.0*x1 + 259.0*x2 + 259.0*x3 + 259.0*x4 + 259.0*x5 + 259.0*x6 + 259.0*x7 
	+ 233.0*x8;
subject to c125:
	0 <= 382.0*x1 + 382.0*x2 + 380.0*x3 + 380.0*x4 + 382.0*x5 + 380.0*x6 + 380.0*x7 
	+ 380.0*x8;
subject to c126:
	0 <= 157.0*x1 + 162.0*x2 + 162.0*x3 + 162.0*x4 + 162.0*x5 + 160.0*x6 + 157.0*x7 
	+ 160.0*x8;
subject to c127:
	0 <= 137.0*x1 + 110.0*x2 + 120.0*x3 + 98.0*x4 + 147.0*x5 + 147.0*x6 + 120.0*x7 
	+ 155.0*x8;
subject to c128:
	0 <= 420.0*x1 + 420.0*x2 + 420.0*x3 + 420.0*x4 + 420.0*x5 + 420.0*x6 + 420.0*x7 
	+ 420.0*x8;
subject to c129:
	0 <= 140.0*x1 + 150.0*x2 + 140.0*x3 + 121.0*x4 + 140.0*x5 + 123.0*x6 + 140.0*x7 
	+ 123.0*x8;
subject to c130:
	0 <= 427.0*x1 + 427.0*x2 + 427.0*x3 + 427.0*x4 + 427.0*x5 + 427.0*x6 + 427.0*x7 
	+ 427.0*x8;
subject to c131:
	0 <= 514.0*x1 + 514.0*x2 + 514.0*x3 + 514.0*x4 + 514.0*x5 + 514.0*x6 + 514.0*x7 
	+ 514.0*x8;
subject to c132:
	0 <= 304.0*x1 + 309.0*x2 + 309.0*x3 + 309.0*x4 + 304.0*x5 + 301.0*x6 + 304.0*x7 
	+ 306.0*x8;
subject to c133:
	0 <= 223.0*x1 + 223.0*x2 + 223.0*x3 + 223.0*x4 + 223.0*x5 + 223.0*x6 + 223.0*x7 
	+ 205.0*x8;
subject to c134:
	0 <= 371.0*x1 + 371.0*x2 + 371.0*x3 + 371.0*x4 + 371.0*x5 + 371.0*x6 + 371.0*x7 
	+ 371.0*x8;
subject to c135:
	0 <= 227.0*x1 + 222.0*x2 + 222.0*x3 + 222.0*x4 + 222.0*x5 + 227.0*x6 + 206.0*x7 
	+ 227.0*x8;
subject to c136:
	0 <= 413.0*x1 + 440.0*x2 + 440.0*x3 + 440.0*x4 + 413.0*x5 + 430.0*x6 + 442.0*x7 
	+ 422.0*x8;
subject to c137:
	0 <= 24.0*x1 + 19.0*x2 + 19.0*x3 + 19.0*x4 + 24.0*x5 + 24.0*x6 + 43.0*x7 + 
	19.0*x8;
subject to c138:
	0 <= 485.0*x1 + 475.0*x2 + 475.0*x3 + 475.0*x4 + 475.0*x5 + 475.0*x6 + 475.0*x7 
	+ 493.0*x8;
subject to c139:
	0 <= 488.0*x1 + 488.0*x2 + 488.0*x3 + 488.0*x4 + 488.0*x5 + 488.0*x6 + 488.0*x7 
	+ 488.0*x8;
subject to c140:
	0 <= 531.0*x1 + 531.0*x2 + 531.0*x3 + 531.0*x4 + 531.0*x5 + 531.0*x6 + 531.0*x7 
	+ 531.0*x8;
subject to c141:
	0 <= 576.0*x1 + 576.0*x2 + 576.0*x3 + 576.0*x4 + 576.0*x5 + 576.0*x6 + 576.0*x7 
	+ 576.0*x8;
subject to c142:
	0 <= 214.0*x1 + 214.0*x2 + 214.0*x3 + 214.0*x4 + 214.0*x5 + 214.0*x6 + 214.0*x7 
	+ 214.0*x8;
subject to c143:
	0 <= 89.0*x1 + 89.0*x2 + 89.0*x3 + 89.0*x4 + 89.0*x5 + 89.0*x6 + 89.0*x7 + 
	89.0*x8;
subject to c144:
	0 <= 470.0*x1 + 470.0*x2 + 470.0*x3 + 470.0*x4 + 470.0*x5 + 470.0*x6 + 470.0*x7 
	+ 470.0*x8;
subject to c145:
	0 <= 402.0*x1 + 402.0*x2 + 402.0*x3 + 402.0*x4 + 402.0*x5 + 402.0*x6 + 402.0*x7 
	+ 402.0*x8;
subject to c146:
	0 <= 333.0*x1 + 333.0*x2 + 333.0*x3 + 333.0*x4 + 333.0*x5 + 333.0*x6 + 333.0*x7 
	+ 333.0*x8;
subject to c147:
	0 <= 54.0*x1 + 54.0*x2 + 85.0*x3 + 85.0*x4 + 54.0*x5 + 54.0*x6 + 15.0*x7 + 
	92.0*x8;
subject to c148:
	0 <= 223.0*x1 + 223.0*x2 + 223.0*x3 + 223.0*x4 + 223.0*x5 + 223.0*x6 + 223.0*x7 
	+ 223.0*x8;
subject to c149:
	0 <= 515.0*x1 + 515.0*x2 + 515.0*x3 + 515.0*x4 + 515.0*x5 + 515.0*x6 + 515.0*x7 
	+ 515.0*x8;
subject to c150:
	0 <= 491.0*x1 + 491.0*x2 + 491.0*x3 + 491.0*x4 + 491.0*x5 + 491.0*x6 + 491.0*x7 
	+ 491.0*x8;
subject to c151:
	0 <= 245.0*x1 + 272.0*x2 + 272.0*x3 + 284.0*x4 + 245.0*x5 + 245.0*x6 + 272.0*x7 
	+ 245.0*x8;
subject to c152:
	0 <= 172.0*x1 + 172.0*x2 + 172.0*x3 + 172.0*x4 + 172.0*x5 + 172.0*x6 + 172.0*x7 
	+ 172.0*x8;
subject to c153:
	0 <= 255.0*x1 + 255.0*x2 + 255.0*x3 + 255.0*x4 + 255.0*x5 + 255.0*x6 + 255.0*x7 
	+ 255.0*x8;
subject to c154:
	0 <= 185.0*x1 + 185.0*x2 + 185.0*x3 + 185.0*x4 + 185.0*x5 + 185.0*x6 + 185.0*x7 
	+ 185.0*x8;
subject to c155:
	0 <= 220.0*x1 + 220.0*x2 + 220.0*x3 + 276.0*x4 + 220.0*x5 + 216.0*x6 + 201.0*x7 
	+ 232.0*x8;
subject to c156:
	0 <= 169.0*x1 + 168.0*x2 + 168.0*x3 + 168.0*x4 + 180.0*x5 + 182.0*x6 + 169.0*x7 
	+ 177.0*x8;
subject to c157:
	0 <= 532.0*x1 + 532.0*x2 + 532.0*x3 + 532.0*x4 + 532.0*x5 + 532.0*x6 + 532.0*x7 
	+ 532.0*x8;
subject to c158:
	0 <= 181.0*x1 + 181.0*x2 + 181.0*x3 + 189.0*x4 + 181.0*x5 + 181.0*x6 + 181.0*x7 
	+ 201.0*x8;
subject to c159:
	0 <= 225.0*x1 + 225.0*x2 + 225.0*x3 + 225.0*x4 + 253.0*x5 + 225.0*x6 + 221.0*x7 
	+ 221.0*x8;
subject to c160:
	0 <= 294.0*x1 + 300.0*x2 + 311.0*x3 + 311.0*x4 + 272.0*x5 + 275.0*x6 + 294.0*x7 
	+ 303.0*x8;
subject to c161:
	0 <= 337.0*x1 + 337.0*x2 + 337.0*x3 + 337.0*x4 + 337.0*x5 + 337.0*x6 + 337.0*x7 
	+ 337.0*x8;
subject to c162:
	0 <= 157.0*x1 + 157.0*x2 + 157.0*x3 + 157.0*x4 + 157.0*x5 + 157.0*x6 + 157.0*x7 
	+ 157.0*x8;
subject to c163:
	0 <= 405.0*x1 + 405.0*x2 + 405.0*x3 + 405.0*x4 + 405.0*x5 + 405.0*x6 + 409.0*x7 
	+ 409.0*x8;
subject to c164:
	0 <= 327.0*x1 + 327.0*x2 + 358.0*x3 + 358.0*x4 + 296.0*x5 + 327.0*x6 + 288.0*x7 
	+ 358.0*x8;
subject to c165:
	0 <= 84.0*x1 + 84.0*x2 + 84.0*x3 + 28.0*x4 + 84.0*x5 + 103.0*x6 + 84.0*x7 + 
	110.0*x8;
subject to c166:
	0 <= 331.0*x1 + 331.0*x2 + 331.0*x3 + 331.0*x4 + 331.0*x5 + 331.0*x6 + 331.0*x7 
	+ 333.0*x8;
subject to c167:
	0 <= 188.0*x1 + 188.0*x2 + 188.0*x3 + 188.0*x4 + 188.0*x5 + 200.0*x6 + 188.0*x7 
	+ 200.0*x8;
subject to c168:
	0 <= 283.0*x1 + 283.0*x2 + 283.0*x3 + 283.0*x4 + 241.0*x5 + 283.0*x6 + 283.0*x7 
	+ 283.0*x8;
subject to c169:
	0 <= 107.0*x1 + 108.0*x2 + 120.0*x3 + 121.0*x4 + 53.0*x5 + 131.0*x6 + 101.0*x7 
	+ 146.0*x8;
subject to c170:
	0 <= 394.0*x1 + 394.0*x2 + 394.0*x3 + 394.0*x4 + 394.0*x5 + 394.0*x6 + 394.0*x7 
	+ 394.0*x8;
subject to c171:
	0 <= 258.0*x1 + 253.0*x2 + 253.0*x3 + 253.0*x4 + 253.0*x5 + 275.0*x6 + 258.0*x7 
	+ 263.0*x8;
subject to c172:
	0 <= 352.0*x1 + 352.0*x2 + 321.0*x3 + 321.0*x4 + 352.0*x5 + 352.0*x6 + 352.0*x7 
	+ 314.0*x8;
subject to c173:
	0 <= 111.0*x1 + 111.0*x2 + 111.0*x3 + 111.0*x4 + 111.0*x5 + 111.0*x6 + 111.0*x7 
	+ 111.0*x8;
subject to c174:
	0 <= 573.0*x1 + 573.0*x2 + 573.0*x3 + 573.0*x4 + 583.0*x5 + 573.0*x6 + 559.0*x7 
	+ 559.0*x8;
subject to c175:
	0 <= 84.0*x1 + 84.0*x2 + 84.0*x3 + 84.0*x4 + 84.0*x5 + 84.0*x6 + 78.0*x7 + 
	84.0*x8;
subject to c176:
	0 <= 561.0*x1 + 561.0*x2 + 561.0*x3 + 561.0*x4 + 561.0*x5 + 548.0*x6 + 561.0*x7 
	+ 548.0*x8;
subject to c177:
	0 <= 250.0*x1 + 240.0*x2 + 250.0*x3 + 240.0*x4 + 250.0*x5 + 267.0*x6 + 250.0*x7 
	+ 267.0*x8;
subject to c178:
	0 <= 204.0*x1 + 253.0*x2 + 193.0*x3 + 229.0*x4 + 193.0*x5 + 176.0*x6 + 207.0*x7 
	+ 191.0*x8;
subject to c179:
	0 <= 217.0*x1 + 217.0*x2 + 217.0*x3 + 217.0*x4 + 217.0*x5 + 217.0*x6 + 219.0*x7 
	+ 217.0*x8;
subject to c180:
	0 <= 538.0*x1 + 538.0*x2 + 538.0*x3 + 538.0*x4 + 538.0*x5 + 538.0*x6 + 538.0*x7 
	+ 538.0*x8;
subject to c181:
	0 <= 100.0*x1 + 100.0*x2 + 100.0*x3 + 100.0*x4 + 100.0*x5 + 100.0*x6 + 100.0*x7 
	+ 100.0*x8;
subject to c182:
	0 <= 485.0*x1 + 485.0*x2 + 485.0*x3 + 486.0*x4 + 485.0*x5 + 485.0*x6 + 485.0*x7 
	+ 474.0*x8;
subject to c183:
	0 <= 558.0*x1 + 558.0*x2 + 558.0*x3 + 558.0*x4 + 558.0*x5 + 558.0*x6 + 558.0*x7 
	+ 558.0*x8;
subject to c184:
	0 <= 393.0*x1 + 393.0*x2 + 393.0*x3 + 383.0*x4 + 383.0*x5 + 393.0*x6 + 393.0*x7 
	+ 393.0*x8;
subject to c185:
	0 <= 144.0*x1 + 144.0*x2 + 156.0*x3 + 166.0*x4 + 136.0*x5 + 169.0*x6 + 156.0*x7 
	+ 169.0*x8;
subject to c186:
	0 <= 322.0*x1 + 322.0*x2 + 322.0*x3 + 322.0*x4 + 322.0*x5 + 322.0*x6 + 322.0*x7 
	+ 322.0*x8;
subject to c187:
	0 <= 465.0*x1 + 465.0*x2 + 465.0*x3 + 465.0*x4 + 465.0*x5 + 465.0*x6 + 465.0*x7 
	+ 465.0*x8;
subject to c188:
	0 <= 157.0*x1 + 157.0*x2 + 157.0*x3 + 157.0*x4 + 157.0*x5 + 157.0*x6 + 157.0*x7 
	+ 157.0*x8;
subject to c189:
	0 <= 373.0*x1 + 374.0*x2 + 374.0*x3 + 374.0*x4 + 374.0*x5 + 374.0*x6 + 425.0*x7 
	+ 373.0*x8;
subject to c190:
	0 <= 114.0*x1 + 114.0*x2 + 114.0*x3 + 114.0*x4 + 114.0*x5 + 114.0*x6 + 114.0*x7 
	+ 114.0*x8;
subject to c191:
	0 <= 104.0*x1 + 104.0*x2 + 104.0*x3 + 104.0*x4 + 104.0*x5 + 104.0*x6 + 104.0*x7 
	+ 104.0*x8;
subject to c192:
	0 <= 437.0*x1 + 443.0*x2 + 443.0*x3 + 453.0*x4 + 453.0*x5 + 437.0*x6 + 437.0*x7 
	+ 443.0*x8;
subject to c193:
	0 <= 213.0*x1 + 174.0*x2 + 174.0*x3 + 174.0*x4 + 174.0*x5 + 174.0*x6 + 174.0*x7 
	+ 174.0*x8;
subject to c194:
	0 <= 6.0*x1 + 6.0*x2 + 6.0*x3 + 6.0*x4 + 6.0*x5 + 6.0*x6 + 6.0*x7 + 6.0*x8;
subject to c195:
	0 <= 158.0*x1 + 197.0*x2 + 197.0*x3 + 197.0*x4 + 197.0*x5 + 197.0*x6 + 197.0*x7 
	+ 197.0*x8;
subject to c196:
	0 <= 515.0*x1 + 515.0*x2 + 515.0*x3 + 515.0*x4 + 515.0*x5 + 515.0*x6 + 515.0*x7 
	+ 515.0*x8;
subject to c197:
	0 <= 24.0*x1 + 24.0*x2 + 24.0*x3 + 24.0*x4 + 24.0*x5 + 24.0*x6 + 24.0*x7 + 
	22.0*x8;
subject to c198:
	0 <= 524.0*x1 + 524.0*x2 + 524.0*x3 + 524.0*x4 + 524.0*x5 + 524.0*x6 + 524.0*x7 
	+ 524.0*x8;
subject to c199:
	0 <= 56.0*x1 + 56.0*x2 + 67.0*x3 + 67.0*x4 + 56.0*x5 + 56.0*x6 + 56.0*x7 + 
	69.0*x8;
subject to c200:
	0 <= 395.0*x1 + 395.0*x2 + 395.0*x3 + 395.0*x4 + 395.0*x5 + 395.0*x6 + 434.0*x7 
	+ 406.0*x8;
subject to c201:
	0 <= 588.0*x1 + 588.0*x2 + 588.0*x3 + 569.0*x4 + 577.0*x5 + 588.0*x6 + 588.0*x7 
	+ 568.0*x8;
subject to c202:
	0 <= 429.0*x1 + 429.0*x2 + 429.0*x3 + 429.0*x4 + 429.0*x5 + 429.0*x6 + 429.0*x7 
	+ 408.0*x8;
subject to c203:
	0 <= 275.0*x1 + 267.0*x2 + 267.0*x3 + 267.0*x4 + 267.0*x5 + 267.0*x6 + 275.0*x7 
	+ 267.0*x8;
subject to c204:
	0 <= 431.0*x1 + 426.0*x2 + 431.0*x3 + 431.0*x4 + 431.0*x5 + 431.0*x6 + 431.0*x7 
	+ 437.0*x8;
subject to c205:
	0 <= 181.0*x1 + 179.0*x2 + 179.0*x3 + 189.0*x4 + 189.0*x5 + 179.0*x6 + 179.0*x7 
	+ 179.0*x8;
subject to c206:
	0 <= 477.0*x1 + 476.0*x2 + 476.0*x3 + 475.0*x4 + 476.0*x5 + 476.0*x6 + 476.0*x7 
	+ 477.0*x8;
subject to c207:
	0 <= 322.0*x1 + 324.0*x2 + 324.0*x3 + 324.0*x4 + 324.0*x5 + 324.0*x6 + 324.0*x7 
	+ 324.0*x8;
subject to c208:
	0 <= 272.0*x1 + 264.0*x2 + 264.0*x3 + 272.0*x4 + 264.0*x5 + 245.0*x6 + 245.0*x7 
	+ 271.0*x8;
subject to c209:
	0 <= 529.0*x1 + 529.0*x2 + 529.0*x3 + 529.0*x4 + 529.0*x5 + 529.0*x6 + 529.0*x7 
	+ 517.0*x8;
subject to c210:
	0 <= 217.0*x1 + 217.0*x2 + 217.0*x3 + 212.0*x4 + 207.0*x5 + 219.0*x6 + 217.0*x7 
	+ 219.0*x8;
subject to c211:
	0 <= 124.0*x1 + 132.0*x2 + 132.0*x3 + 132.0*x4 + 132.0*x5 + 132.0*x6 + 124.0*x7 
	+ 153.0*x8;
subject to c212:
	0 <= 565.0*x1 + 565.0*x2 + 565.0*x3 + 565.0*x4 + 565.0*x5 + 565.0*x6 + 565.0*x7 
	+ 565.0*x8;
subject to c213:
	0 <= 600.0*x1 + 600.0*x2 + 600.0*x3 + 600.0*x4 + 600.0*x5 + 600.0*x6 + 600.0*x7 
	+ 600.0*x8;
subject to c214:
	0 <= 343.0*x1 + 343.0*x2 + 343.0*x3 + 343.0*x4 + 343.0*x5 + 343.0*x6 + 343.0*x7 
	+ 343.0*x8;
subject to c215:
	0 <= 251.0*x1 + 251.0*x2 + 251.0*x3 + 246.0*x4 + 251.0*x5 + 251.0*x6 + 251.0*x7 
	+ 251.0*x8;
subject to c216:
	0 <= 381.0*x1 + 381.0*x2 + 381.0*x3 + 381.0*x4 + 381.0*x5 + 379.0*x6 + 381.0*x7 
	+ 391.0*x8;
subject to c217:
	0 <= 325.0*x1 + 333.0*x2 + 333.0*x3 + 336.0*x4 + 344.0*x5 + 352.0*x6 + 352.0*x7 
	+ 326.0*x8;
subject to c218:
	0 <= 33.0*x1 + 33.0*x2 + 33.0*x3 + 4.0*x4 + 33.0*x5 + 52.0*x6 + 72.0*x7 + 
	42.0*x8;
subject to c219:
	0 <= 145.0*x1 + 145.0*x2 + 145.0*x3 + 134.0*x4 + 134.0*x5 + 126.0*x6 + 126.0*x7 
	+ 145.0*x8;
subject to c220:
	0 <= 340.0*x1 + 309.0*x2 + 340.0*x3 + 278.0*x4 + 340.0*x5 + 340.0*x6 + 340.0*x7 
	+ 340.0*x8;
subject to c221:
	0 <= 127.0*x1 + 137.0*x2 + 137.0*x3 + 137.0*x4 + 168.0*x5 + 137.0*x6 + 70.0*x7 
	+ 137.0*x8;
subject to c222:
	0 <= 366.0*x1 + 366.0*x2 + 366.0*x3 + 366.0*x4 + 366.0*x5 + 366.0*x6 + 401.0*x7 
	+ 366.0*x8;
subject to c223:
	0 <= 406.0*x1 + 356.0*x2 + 406.0*x3 + 352.0*x4 + 406.0*x5 + 406.0*x6 + 406.0*x7 
	+ 406.0*x8;
subject to c224:
	0 <= 160.0*x1 + 132.0*x2 + 132.0*x3 + 102.0*x4 + 159.0*x5 + 159.0*x6 + 132.0*x7 
	+ 160.0*x8;
subject to c225:
	0 <= 365.0*x1 + 365.0*x2 + 365.0*x3 + 349.0*x4 + 376.0*x5 + 384.0*x6 + 384.0*x7 
	+ 365.0*x8;
subject to c226:
	0 <= 311.0*x1 + 311.0*x2 + 311.0*x3 + 311.0*x4 + 311.0*x5 + 311.0*x6 + 311.0*x7 
	+ 311.0*x8;
subject to c227:
	0 <= 246.0*x1 + 246.0*x2 + 246.0*x3 + 246.0*x4 + 246.0*x5 + 246.0*x6 + 246.0*x7 
	+ 246.0*x8;
subject to c228:
	0 <= 398.0*x1 + 398.0*x2 + 398.0*x3 + 398.0*x4 + 398.0*x5 + 398.0*x6 + 398.0*x7 
	+ 416.0*x8;
subject to c229:
	0 <= 421.0*x1 + 420.0*x2 + 420.0*x3 + 420.0*x4 + 432.0*x5 + 432.0*x6 + 421.0*x7 
	+ 416.0*x8;
subject to c230:
	0 <= 312.0*x1 + 340.0*x2 + 340.0*x3 + 335.0*x4 + 340.0*x5 + 340.0*x6 + 340.0*x7 
	+ 340.0*x8;
subject to c231:
	0 <= 381.0*x1 + 381.0*x2 + 381.0*x3 + 381.0*x4 + 381.0*x5 + 381.0*x6 + 381.0*x7 
	+ 381.0*x8;
subject to c232:
	0 <= 414.0*x1 + 414.0*x2 + 414.0*x3 + 414.0*x4 + 414.0*x5 + 414.0*x6 + 414.0*x7 
	+ 414.0*x8;
subject to c233:
	0 <= 432.0*x1 + 432.0*x2 + 432.0*x3 + 432.0*x4 + 432.0*x5 + 432.0*x6 + 432.0*x7 
	+ 432.0*x8;
subject to c234:
	0 <= 96.0*x1 + 96.0*x2 + 96.0*x3 + 96.0*x4 + 96.0*x5 + 96.0*x6 + 96.0*x7 + 
	96.0*x8;
subject to c235:
	0 <= 133.0*x1 + 171.0*x2 + 171.0*x3 + 171.0*x4 + 171.0*x5 + 171.0*x6 + 171.0*x7 
	+ 171.0*x8;
subject to c236:
	0 <= 239.0*x1 + 240.0*x2 + 240.0*x3 + 258.0*x4 + 240.0*x5 + 240.0*x6 + 240.0*x7 
	+ 231.0*x8;
subject to c237:
	0 <= 415.0*x1 + 414.0*x2 + 414.0*x3 + 413.0*x4 + 414.0*x5 + 414.0*x6 + 414.0*x7 
	+ 415.0*x8;
subject to c238:
	0 <= 221.0*x1 + 221.0*x2 + 221.0*x3 + 221.0*x4 + 221.0*x5 + 221.0*x6 + 221.0*x7 
	+ 221.0*x8;
subject to c239:
	0 <= 470.0*x1 + 503.0*x2 + 503.0*x3 + 495.0*x4 + 475.0*x5 + 507.0*x6 + 489.0*x7 
	+ 496.0*x8;
subject to c240:
	0 <= 464.0*x1 + 464.0*x2 + 464.0*x3 + 472.0*x4 + 464.0*x5 + 464.0*x6 + 464.0*x7 
	+ 484.0*x8;
subject to c241:
	0 <= 216.0*x1 + 255.0*x2 + 255.0*x3 + 255.0*x4 + 255.0*x5 + 255.0*x6 + 255.0*x7 
	+ 255.0*x8;
subject to c242:
	0 <= 239.0*x1 + 250.0*x2 + 250.0*x3 + 242.0*x4 + 250.0*x5 + 250.0*x6 + 239.0*x7 
	+ 230.0*x8;
subject to c243:
	0 <= 270.0*x1 + 270.0*x2 + 270.0*x3 + 253.0*x4 + 298.0*x5 + 270.0*x6 + 270.0*x7 
	+ 270.0*x8;
subject to c244:
	0 <= 238.0*x1 + 227.0*x2 + 227.0*x3 + 227.0*x4 + 227.0*x5 + 227.0*x6 + 238.0*x7 
	+ 235.0*x8;
subject to c245:
	0 <= 410.0*x1 + 410.0*x2 + 410.0*x3 + 410.0*x4 + 410.0*x5 + 410.0*x6 + 410.0*x7 
	+ 410.0*x8;
subject to c246:
	0 <= 292.0*x1 + 292.0*x2 + 294.0*x3 + 294.0*x4 + 292.0*x5 + 294.0*x6 + 294.0*x7 
	+ 294.0*x8;
subject to c247:
	0 <= 103.0*x1 + 103.0*x2 + 103.0*x3 + 103.0*x4 + 103.0*x5 + 103.0*x6 + 103.0*x7 
	+ 103.0*x8;
subject to c248:
	0 <= 150.0*x1 + 150.0*x2 + 150.0*x3 + 150.0*x4 + 150.0*x5 + 141.0*x6 + 150.0*x7 
	+ 141.0*x8;
subject to c249:
	0 <= 324.0*x1 + 324.0*x2 + 324.0*x3 + 324.0*x4 + 324.0*x5 + 324.0*x6 + 324.0*x7 
	+ 324.0*x8;
subject to c250:
	0 <= 220.0*x1 + 220.0*x2 + 220.0*x3 + 220.0*x4 + 220.0*x5 + 220.0*x6 + 220.0*x7 
	+ 220.0*x8;
subject to c251:
	0 <= 14.0*x1 + 14.0*x2 + 12.0*x3 + 12.0*x4 + 20.0*x5 + 12.0*x6 + 12.0*x7 + 
	12.0*x8;
subject to c252:
	0 <= 147.0*x1 + 147.0*x2 + 147.0*x3 + 147.0*x4 + 147.0*x5 + 147.0*x6 + 147.0*x7 
	+ 147.0*x8;
subject to c253:
	0 <= 360.0*x1 + 360.0*x2 + 360.0*x3 + 360.0*x4 + 360.0*x5 + 360.0*x6 + 360.0*x7 
	+ 360.0*x8;
subject to c254:
	0 <= 306.0*x1 + 306.0*x2 + 306.0*x3 + 306.0*x4 + 306.0*x5 + 306.0*x6 + 302.0*x7 
	+ 280.0*x8;
subject to c255:
	0 <= 466.0*x1 + 466.0*x2 + 466.0*x3 + 466.0*x4 + 466.0*x5 + 466.0*x6 + 466.0*x7 
	+ 466.0*x8;
subject to c256:
	0 <= 251.0*x1 + 251.0*x2 + 251.0*x3 + 251.0*x4 + 251.0*x5 + 251.0*x6 + 251.0*x7 
	+ 245.0*x8;
subject to c257:
	0 <= 264.0*x1 + 264.0*x2 + 264.0*x3 + 264.0*x4 + 264.0*x5 + 264.0*x6 + 264.0*x7 
	+ 264.0*x8;
subject to c258:
	0 <= 403.0*x1 + 403.0*x2 + 403.0*x3 + 403.0*x4 + 403.0*x5 + 403.0*x6 + 403.0*x7 
	+ 403.0*x8;
subject to c259:
	0 <= 347.0*x1 + 347.0*x2 + 347.0*x3 + 347.0*x4 + 341.0*x5 + 347.0*x6 + 347.0*x7 
	+ 347.0*x8;
subject to c260:
	0 <= 99.0*x1 + 99.0*x2 + 99.0*x3 + 99.0*x4 + 99.0*x5 + 99.0*x6 + 99.0*x7 + 
	99.0*x8;
subject to c261:
	0 <= 123.0*x1 + 123.0*x2 + 123.0*x3 + 123.0*x4 + 123.0*x5 + 123.0*x6 + 127.0*x7 
	+ 123.0*x8;
subject to c262:
	0 <= 173.0*x1 + 175.0*x2 + 175.0*x3 + 175.0*x4 + 175.0*x5 + 175.0*x6 + 175.0*x7 
	+ 175.0*x8;
subject to c263:
	0 <= 554.0*x1 + 552.0*x2 + 552.0*x3 + 552.0*x4 + 552.0*x5 + 552.0*x6 + 552.0*x7 
	+ 552.0*x8;
subject to c264:
	0 <= 222.0*x1 + 222.0*x2 + 222.0*x3 + 222.0*x4 + 204.0*x5 + 222.0*x6 + 222.0*x7 
	+ 222.0*x8;
subject to c265:
	0 <= 448.0*x1 + 448.0*x2 + 448.0*x3 + 448.0*x4 + 448.0*x5 + 448.0*x6 + 448.0*x7 
	+ 448.0*x8;
subject to c266:
	0 <= 279.0*x1 + 283.0*x2 + 271.0*x3 + 271.0*x4 + 314.0*x5 + 278.0*x6 + 267.0*x7 
	+ 278.0*x8;
subject to c267:
	0 <= 154.0*x1 + 150.0*x2 + 150.0*x3 + 158.0*x4 + 119.0*x5 + 144.0*x6 + 131.0*x7 
	+ 166.0*x8;
subject to c268:
	0 <= 451.0*x1 + 451.0*x2 + 451.0*x3 + 451.0*x4 + 451.0*x5 + 451.0*x6 + 451.0*x7 
	+ 451.0*x8;
subject to c269:
	0 <= 291.0*x1 + 291.0*x2 + 291.0*x3 + 291.0*x4 + 291.0*x5 + 291.0*x6 + 291.0*x7 
	+ 291.0*x8;
subject to c270:
	0 <= 83.0*x1 + 83.0*x2 + 83.0*x3 + 83.0*x4 + 83.0*x5 + 91.0*x6 + 91.0*x7 + 
	90.0*x8;
subject to c271:
	0 <= 171.0*x1 + 171.0*x2 + 171.0*x3 + 171.0*x4 + 171.0*x5 + 171.0*x6 + 171.0*x7 
	+ 171.0*x8;
subject to c272:
	0 <= 70.0*x1 + 65.0*x2 + 70.0*x3 + 62.0*x4 + 70.0*x5 + 70.0*x6 + 70.0*x7 + 
	56.0*x8;
subject to c273:
	0 <= 317.0*x1 + 312.0*x2 + 319.0*x3 + 319.0*x4 + 294.0*x5 + 318.0*x6 + 329.0*x7 
	+ 318.0*x8;
subject to c274:
	0 <= 540.0*x1 + 540.0*x2 + 540.0*x3 + 540.0*x4 + 540.0*x5 + 540.0*x6 + 540.0*x7 
	+ 540.0*x8;
subject to c275:
	0 <= 330.0*x1 + 330.0*x2 + 330.0*x3 + 330.0*x4 + 330.0*x5 + 330.0*x6 + 330.0*x7 
	+ 330.0*x8;
subject to c276:
	0 <= 288.0*x1 + 288.0*x2 + 288.0*x3 + 288.0*x4 + 288.0*x5 + 288.0*x6 + 288.0*x7 
	+ 281.0*x8;
subject to c277:
	0 <= 429.0*x1 + 429.0*x2 + 429.0*x3 + 428.0*x4 + 429.0*x5 + 429.0*x6 + 429.0*x7 
	+ 478.0*x8;
subject to c278:
	0 <= 225.0*x1 + 225.0*x2 + 225.0*x3 + 215.0*x4 + 246.0*x5 + 225.0*x6 + 228.0*x7 
	+ 225.0*x8;

solve;
	display x1;
	display x2;
	display x3;
	display x4;
	display x5;
	display x6;
	display x7;
	display x8;
display obj;
