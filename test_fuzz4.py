from rapidfuzz import fuzz

q1 = "dof"
t1 = "this is a very long text that happens to contain the word dog somewhere in the middle of it"

print("token_set_ratio:", fuzz.token_set_ratio(q1, t1))

q2 = "helloworld"
t2 = "hello world"
print("token_set_ratio2:", fuzz.token_set_ratio(q2, t2))

