from rapidfuzz import fuzz

q1 = "do"
t1 = "this is a very long text that happens to contain the word dog somewhere in the middle of it"

print("token_set_ratio:", fuzz.token_set_ratio(q1, t1))
print("token_sort_ratio:", fuzz.token_sort_ratio(q1, t1))
print("partial_ratio:", fuzz.partial_ratio(q1, t1))

