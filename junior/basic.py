data = [
    {"name": "Alice", "score": "90"},
    {"name": "Bob", "score": "NaN"},
    {"name": "Charlie", "score": "85"},
    {"name": None, "score": "100"},
]

'''
Задача:

удалить элементы с None в name

отфильтровать score == "NaN"

привести score к int

вывести список имён, у кого score >= 90

'''

cleaned = []
for entry in data:
    if entry["name"] is not None and entry["score"] != "NaN":
        score = int(entry["score"])
        if score >= 90:
            cleaned.append(entry["name"])


#functional
cleaned = list(
    map(
        lambda x: x["name"],
        filter(
            lambda z: z["name"] is not None and z["score"] != "NaN" and int(z["score"]) >= 90,
            data
        )
    )
)

print(cleaned)