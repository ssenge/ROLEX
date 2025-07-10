\ LP format example
\ Simple maximization problem

MAXIMIZE
  x1 + x2

SUBJECT TO
  x1 + x2 <= 1
  x1 <= 0.8
  x2 <= 0.7

BOUNDS
  x1 >= 0
  x2 >= 0

END 