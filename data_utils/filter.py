from collections import defaultdict

specific_characters = ['X', 'Y']
labels = ["\n", " ", '|', '&', '}', 'b', 't', 'j', 'G', 'x', '*', 'z', '$', 'D', 'Z', 'g', '_', 'q', 'l', 'n', 'w', 'y', 'N', 'a', 'i', 'o', '`', 'P', 'J', "'", '>', '<', 'V', 'A', 'p', 'v', 'H', 'd', 'r', 's', 'S', 'T', 'E', 'f', 'k', 'm', 'h', 'Y', 'F', 'K', 'u', '~', '{']

counter_dict = defaultdict(lambda: 0)

with open("company_names_bk.txt", "r") as in_file, open("company_names_filtered.txt","a") as out_file:
  while True:
    c = in_file.read(1)
    if not c:
      break
    if c in labels:
      out_file.write(c)

