from collections import deque

def get_index(q, to_insert):
  for i, elem in enumerate(q):
    if elem < to_insert:
      return i

  return len(q)

test = deque([17,10,6,5,3,2])

ind = get_index(test, 1)
test.insert(ind, 1)
print(test)