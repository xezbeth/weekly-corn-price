import time
from datetime import datetime

t = datetime.now()
t1 = t.timetuple()

print time.mktime(t1)
