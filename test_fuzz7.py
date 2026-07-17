from rapidfuzz import fuzz

def asymmetric_partial_ratio(query, text):
    if len(query) > len(text):
        return fuzz.ratio(query, text)
    else:
        return fuzz.partial_ratio(query, text)

q = "monika grozio salonas"
t1 = "o"
t2 = "monika"
t3 = "monika grozio salonas ở đằng sau"

print("text='o':", asymmetric_partial_ratio(q, t1))
print("text='monika':", asymmetric_partial_ratio(q, t2))
print("text='...salonas...':", asymmetric_partial_ratio(q, t3))

# what if query is short and text is long?
q2 = "o"
t4 = "monika grozio salonas"
print("query='o', text long:", asymmetric_partial_ratio(q2, t4))

