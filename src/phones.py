import re

def load_symbols(path):
    lookup = {}
    with open(path, "r") as infile:
        for line in infile.readlines():
            lookup[line.strip()] = len(lookup)
    return lookup

def combine_related_phones(symbol_ids):
  symbol_ids_copy = {}

  last_idx = 0
  for phone in symbol_ids:
      if "[" in phone:
          continue
      if "%" in phone:
          symbol_ids_copy[phone] = symbol_ids_copy["sil"]    
          continue
          
      normed = re.sub("_[0-9]+", "", phone)
      if normed not in symbol_ids_copy:
          symbol_ids_copy[normed] = last_idx
          last_idx += 1
      symbol_ids_copy[phone] = symbol_ids_copy[normed]

  return symbol_ids_copy, last_idx