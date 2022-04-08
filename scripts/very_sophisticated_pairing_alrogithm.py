import random

team_members = {
    "Dmitry Mittov": True,
    "Amruth Anand": True,
    "Bruno Lansac Nieto": True,
    "Jack Agass": True,
    "Jurij Wollert": True,
    "Navid Ghayazi": True,
    "Stanislav Kozlov": True,
    "Zhenyu Wang": True,
    "Andrei Vishniakov": True,
}

valid_members = [k for k, v in team_members.items() if v]

if len(valid_members) % 2 != 0:
    valid_members.append(None)

random.shuffle(valid_members)

final_pairing = list(
    zip(
        valid_members[0::2],
        valid_members[1::2]
    )
)

[print(i) for i in final_pairing]
