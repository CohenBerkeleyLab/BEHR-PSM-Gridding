#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the OMI Python package.
#
# The OMI Python package is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
#
# The OMI Python Package is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with the OMI Python Package. If not, see
# <http://www.gnu.org/licenses/>.


from __future__ import division
import datetime as dt
import numpy as np
import numpy.ma as ma



def rect2sphere(vector, degree=True):
    """\
    Convert vector (x,y,z) from rect to sphere coordinates. If degree is 
    ``True`` the unit will be in degree.

    Examples
    --------
    >>> convert.rect2sphere([1,1,1], degree=False)
    array([ 0.78539816,  0.61547971,  1.73205081])

    >>> convert.rect2sphere(numpy.array([[1,2],[1,0],[1,3]]), degree=True) 
    array([[ 45.        ,   0.        ],
           [ 35.26438968,  56.30993247],
           [  1.73205081,   3.60555128]])

    """
    x, y, z = vector

    r = np.sqrt(x**2 + y**2 + z**2)
    lon = np.arctan2(y,x)
    lat = np.arcsin(z/r)

    if degree:
        lon = np.rad2deg(lon)
        lat = np.rad2deg(lat)

    return ma.concatenate([ lon[np.newaxis], lat[np.newaxis], r[np.newaxis] ])


def sphere2rect(vector, degree=True):
    """\
    Convert vector (lon, lat, r) from sphere to rect coordinates. If degree
    is True, the unit of vector has to be in degree.

    Examples
    --------
    >>> convert.sphere2rect([120, 30, 1], degree=True)
    array([-0.4330127,  0.75     ,  0.5      ])
    """
    lon, lat, r = vector

    if degree:
        lon = np.deg2rad(lon)
        lat = np.deg2rad(lat)

    return ma.concatenate([
        (r * np.cos(lat) * np.cos(lon))[np.newaxis],
        (r * np.cos(lat) * np.sin(lon))[np.newaxis],
        (r * np.sin(lat))[np.newaxis]
    ])

def tai93toDatetime(tai93):
    """
    Converts and OMI TAI93 time code to a Python datetime object.
    :param tai93: a TAI93 time code
    :return: a Python datetime.datetime object representing the TAI93 time code

    TAI93 is a time specification that gives time as the number of seconds since midnight, Jan 1st, 1993. Leap seconds
    were added on:
        1 Jul 1993
        1 Jul 1994
        1 Jan 1996
        1 Jul 1997
        1 Jan 1999
        1 Jan 2006
        1 Jan 2009
        1 Jul 2012
        1 Jul 2015
    Since Python's datetime module does not account for these, we will account for them here.
    """

    base_dt = dt.datetime(1993, 1, 1, 0, 0, 0)
    leap_sec = [dt.datetime(2015, 7, 1) - base_dt,
                dt.datetime(2012, 7, 1) - base_dt,
                dt.datetime(2009, 1, 1) - base_dt,
                dt.datetime(2006, 1, 1) - base_dt,
                dt.datetime(1999, 1, 1) - base_dt,
                dt.datetime(1997, 7, 1) - base_dt,
                dt.datetime(1996, 1, 1) - base_dt,
                dt.datetime(1994, 7, 1) - base_dt,
                dt.datetime(1993, 7, 1) - base_dt]

    for ld in leap_sec:
        if tai93 > ld.total_seconds():
            tai93 -= 1

    return base_dt + dt.timedelta(seconds=tai93)




