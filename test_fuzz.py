from rapidfuzz import fuzz

q1 = "hi"
t1 = "this is something"

q2 = "hell"
t2 = "hello world"

q3 = "hello world"
t3 = "bla helloworld bla"

q4 = "hello world"
t4 = "hello  world"

queries = [(q1, t1), (q2, t2), (q3, t3), (q4, t4)]

print(f"{'Query':<15} | {'Text':<20} | {'partial_ratio'} | {'token_set'} | {'ratio'} | {'sliding_window'}")
for q, t in queries:
    pr = fuzz.partial_ratio(q, t)
    ts = fuzz.token_set_ratio(q, t)
    r = fuzz.ratio(q, t)
    
    # sliding window
    qw = q.split()
    tw = t.split()
    nq = len(qw)
    sw = 0
    if len(tw) >= nq and nq > 0:
        for i in range(len(tw) - nq + 1):
            w = " ".join(tw[i:i+nq])
            sw = max(sw, fuzz.ratio(q, w))
    
    print(f"{q:<15} | {t:<20} | {pr:13.1f} | {ts:9.1f} | {r:5.1f} | {sw:14.1f}")

