import pycutest


pyc_prob = pycutest.import_problem("TWIRIBG1")
x0 = pyc_prob.x0
print(f"Size: {len(x0)}")
print(f"First 10: {x0[:10]}")
print(f"Around 2496: {x0[2490:2500]}")
print(f"Last 10: {x0[-10:]}")
