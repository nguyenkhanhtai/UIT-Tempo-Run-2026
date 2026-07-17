from rapidfuzz import fuzz

q = "monika grozio salonas"
t1 = "monika"
t2 = "monika grozio"
t3 = "monika grozio salonas ở đằng sau"

print("1 word:", fuzz.token_set_ratio(q, t1))
print("2 words:", fuzz.token_set_ratio(q, t2))
print("3 words:", fuzz.token_set_ratio(q, t3))

