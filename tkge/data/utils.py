import datetime
import arrow
from datetime import datetime

from typing import List


def is_leap_year(years: int):
    """
    if year is a leap year
    """

    assert isinstance(years, int), "Integer required."

    if ((years % 4 == 0 and years % 100 != 0) or (years % 400 == 0)):
        days_sum = 366
        return days_sum
    else:
        days_sum = 365
        return days_sum


def get_all_days_of_year(years: int, format: str = "YYYY-MM-DD") -> List[str]:
    """
    get all days of the year in string format
    """

    start_date = '%s-1-1' % years
    a = 0
    all_date_list = []
    days_sum = is_leap_year(int(years))
    while a < days_sum:
        b = arrow.get(start_date).shift(days=a).format(format)
        a += 1
        all_date_list.append(b)

    return all_date_list

def get_all_days_between(start_date, end_date, format="YYYY-MM-DD"):
    """
    get all days between starting date and ending date
    """
    start_date = arrow.get(start_date)
    end_date = arrow.get(end_date)

    res = []

    for d in range(0, (end_date-start_date).days + 1):
        new_day = start_date.shift(days=d).format(format)
        res.append(new_day)

    return res


if __name__ == '__main__':
    print(get_all_days_of_year(2011))