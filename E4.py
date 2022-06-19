class gc(object):
    def __init__(self,year,mon,day,hour=0,min=0,sec=0):
        self.day=day
        self.month=mon
        self.year=year
        self.hour=hour
        self.second=sec
        self.minute=min

def cal_jd(gc):
    jd = gc.day - 32075 + (1461 * (gc.year + 4800 + ((gc.month-14) / 12))/4) +(367 * (gc.month - 2 - ((gc.month-14) / 12) * 12)/12) - (3 * (((gc.year +4900 + ((gc.month-14) / 12)) / 100))/4);
    print('儒略日:',jd)
    jd = jd - 0.5 + gc.hour / 24.0 + gc.minute / 1440.0 +gc.second / 86400.0;
    print('修正儒略日:',jd)
    return jd
def runnian(year):
    if (year % 400 == 0) or ((year % 4 == 0) and (year % 100 != 0)):
        return 1
    else:
        return 0
def bigsmallmonth(year,mon):
    if mon in [1,3,5,7,8,10,12]:
        return 31
    elif mon in [4,6,9,11]:
        return 30
    elif mon==2:
        return 28+runnian(year)
def selfmadejd(gc):
    yearsday=0;monthsday=0
    for i in range(-4713,gc.year):
        if i != 0:
            yearsday+=365+runnian(i)
    for j in range(1,gc.month):
        monthsday+=bigsmallmonth(gc.year,j)
    jd=yearsday+monthsday+gc.day+(gc.hour-12)/24+gc.minute/24/60+gc.second/24/60/60-10+48
    return jd
gc1=gc(1996,10,26,14,20,0)
gc2=gc(2019,8,8,14,33,59)
print(selfmadejd(gc2))
print(selfmadejd(gc1))