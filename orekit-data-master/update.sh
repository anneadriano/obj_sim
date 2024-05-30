#!/bin/sh

# Copyright 2002-2024 CS GROUP
# Licensed to CS GROUP (CS) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# CS licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# this script updates the following files in the orekit-data directory
#  UTC-TAI.history
#  Earth-Orientation-Parameters/IAU-1980/finals.all
#  Earth-Orientation-Parameters/IAU-2000/finals2000A.all
#  MSAFE/mmm####f10{-|_}prd.txt (where mmm is a month abbreviation and #### a year)
#  CSSI-Space-Weather-Data/SpaceWeather-All-v1.2.txt

# base URLS
usno_ser7_url=https://maia.usno.navy.mil/ser7
iers_rapid_url=https://datacenter.iers.org/data
msafe_url_uploads=https://www.nasa.gov/wp-content/uploads
msafe_url_atom=https://www3.nasa.gov/sites/default/files/atoms/files
cssi_url=ftp://ftp.agi.com/pub/DynamicEarthData

last_fetched=""

# fetch a file from an URL
fetch_URL()
{ echo "fetching file $1" 1>&2
  local name=$(echo "$1" | sed 's,.*/,,')
  if [ -f "$name" ] ; then
    mv "$name" "$name.old"
  fi

  # convert either DOS or MAC file to Unix line endings
  if curl "$1" | sed 's,\r$,,' | tr '\015' '\012' > "$name" && test -s "$name" ; then
    if [ -f "$name.old" ] ; then
        # remove old file
        rm "$name.old"
    fi
    echo "$1 fetched" 1>&2
    last_fetched=$name
    return 0
  else
    if [ -f "$name" ] ; then
        # remove empty file
        rm "$name"
    fi
    if [ -f "$name.old" ] ; then
        # recover old file
        mv "$name.old" "$name"
    fi
    echo "$1 not fetched!" 1>&2
    return 1
  fi

}

# fetch an MSAFE file
fetch_MSAFE()
{
  # for an unknown reason, some files have been published with an URL refering to april 2019…
  for base_url in $msafe_url_atom $(date_based_MSAFE_url $1) $msafe_url_uploads/2019/04 ; do
    for suffix in "_prd.txt" "-prd.txt" ; do
      if fetch_URL ${base_url}/${1}${suffix} ; then
        case $(head -1 $last_fetched | tr ' ' '_') in
            __TABLE_3_*) echo 1>&2 "${1}${suffix} is complete" ; return 0 ;;
            *)           echo 1>&2 "${1}${suffix} removed (fetched only a 404 error page)"; rm $last_fetched ;;
        esac
      fi
    done
  done
  return 1
}

# find the MSAFE month
MSAFE_month()
{
    echo  "$1" | sed "s,\([a-z][a-z][a-z]\)\([0-9][0-9][0-9][0-9]\)f10,\1,"
}

# find the MSAFE year
MSAFE_year()
{
    echo  "$1" | sed "s,\([a-z][a-z][a-z]\)\([0-9][0-9][0-9][0-9]\)f10,\2,"
}

# find the MSAFE base name (month/year) following an existing base name
next_MSAFE()
{
    local month=$(MSAFE_month $1)
    local year=$(MSAFE_year $1)
    case $month in
      'jan') month='feb' ;;
      'feb') month='mar' ;;
      'mar') month='apr' ;;
      'apr') month='may' ;;
      'may') month='jun' ;;
      'jun') month='jul' ;;
      'jul') month='aug' ;;
      'aug') month='sep' ;;
      'sep') month='oct' ;;
      'oct') month='nov' ;;
      'nov') month='dec' ;;
      'dec') month='jan' ; year=$(($year + 1));;
      *) echo "wrong pattern $1" 1>&2 ; exit 1;;
    esac
    echo ${month}${year}f10
}

# find the MSAFE date-based URL
date_based_MSAFE_url()
{
    local month=$(MSAFE_month $1)
    local year=$(MSAFE_year $1)
    case $month in
      'jan') m='01' ;;
      'feb') m='02' ;;
      'mar') m='03' ;;
      'apr') m='04' ;;
      'may') m='05' ;;
      'jun') m='06' ;;
      'jul') m='07' ;;
      'aug') m='08' ;;
      'sep') m='09' ;;
      'oct') m='10' ;;
      'nov') m='11' ;;
      'dec') m='12' ;;
      *) echo "wrong pattern $1" 1>&2 ; exit 1;;
    esac
    echo ${msafe_url_uploads}/$year/$m
}

# find the first MSAFE file that is missing in current directory
first_missing_MSAFE()
{
  local msafe=dec2023f10
  while test -f "$msafe".txt || test -f "$msafe"_prd.txt  || test -f "$msafe"-prd.txt ; do
      msafe=$(next_MSAFE $msafe)
  done
  echo $msafe
}

# update (overwriting) leap seconds file
fetch_URL $usno_ser7_url/tai-utc.dat
 
# update (overwriting) Earth Orientation Parameters
(cd Earth-Orientation-Parameters/IAU-2000 && fetch_URL $iers_rapid_url/9/finals2000A.all)
(cd Earth-Orientation-Parameters/IAU-1980 && fetch_URL $iers_rapid_url/7/finals.all)

# update (adding files) Marshall Solar Activity Future Estimation
msafe_base=$(cd MSAFE ; first_missing_MSAFE)
while [ ! -z "$msafe_base" ] ; do
    if $(cd MSAFE ; fetch_MSAFE $msafe_base) ; then
        msafe_base=$(next_MSAFE $msafe_base)
    else
        msafe_base=""
    fi
done

# update (overwriting) CSSI space weather data
(cd CSSI-Space-Weather-Data && fetch_URL $cssi_url/SpaceWeather-All-v1.2.txt)
