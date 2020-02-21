from openValGenerator import openValGenerator

column = "pickup_latitude"
table = "rides"
db = "gda_taxi"

y = openValGenerator(db, table, column)
print(f"Do {column}")
print(y.istrained())
if not y.istrained():
    y.train()
print(y.column_type)
print(y.istrained())
# Get values in a given range
print(y.getVal(40.75, 40.755))
print(y.getVal(-73.955, -73.950))

print("-------------------------------")

column = "firstname"
table = "accounts"
db = 'gda_banking'
x = openValGenerator(db, table, column)
print(f"Do {column}")
print(x.istrained())
if not x.istrained():
    x.train()
print(x.column_type)
print(x.istrained())
# Get any text value
for _ in range(10):
    print(x.getVal())

print("-------------------------------")

column = "frequency"
table = "accounts"
db = 'gda_banking'

x = openValGenerator(db, table, column)
print(f"Do {column}")
print(x.istrained())
if not x.istrained():
    x.train()
print(x.column_type)
print(x.istrained())
for _ in range(10):
    print(x.getVal())

print("-------------------------------")

column = "email"
table = "accounts"
db = 'gda_banking'

x = openValGenerator(db, table, column)
print(f"Do {column}")
print(x.istrained())
if not x.istrained():
    x.train()
print(x.column_type)
print(x.istrained())
print(x.is_email())
for _ in range(10):
    print(x.getVal())

print("-------------------------------")

column = "trip_distance"
table = "rides"
db = "gda_taxi"

y = openValGenerator(db, table, column)
print(f"Do {column}")
print(y.istrained())
if not y.istrained():
    y.train()
print(y.column_type)
print(y.istrained())
print(y.getVal(1, 5))
print(y.getVal(1, 5))

print("-------------------------------")

column = "pickup_longitude"
table = "rides"
db = "gda_taxi"

y = openValGenerator(db, table, column)
print(f"Do {column}")
print(y.istrained())
if not y.istrained():
    y.train()
print(y.column_type)
print(y.istrained())
print(y.getVal(-73.96, -73.95))
print(y.getVal(-73.955, -73.950))
