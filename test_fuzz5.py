from rapidfuzz import fuzz

q = "monika grozio salonas"
t = "o"

print("token_set_ratio:", fuzz.token_set_ratio(q, t))
print("token_sort_ratio:", fuzz.token_sort_ratio(q, t))
print("partial_ratio:", fuzz.partial_ratio(q, t))

