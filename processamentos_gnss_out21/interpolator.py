import os, pickle
import datetime
import julian
# from math import ceil, floor

# def based_ceil(x, base=5):
#     return base * ceil(x/base)

# def based_floor(x, base=5):
#     return base * floor(x/base)

# vmf_coefs_path = '/home/lape04/Dropbox/laix2/dados VMF1'

# interpol_pickles_dir = "/home/lape04/data"

interpolated_data = "/home/lape04/data/stations_interps"

# constants
with open("/home/lape04/Dropbox/laix2/processamentos_gnss_out21/curr_constants.txt") as c_reader:
    station,rtklibdate  = c_reader.readline().split(',')


# lat = float(lat)
# lgt = float(lgt) + 360 # longitude 0 - 360
rtklibdate = int(rtklibdate)

conv_datetime = datetime.datetime.utcfromtimestamp(rtklibdate)

year,month,day,hour = conv_datetime.year,conv_datetime.month,conv_datetime.day,conv_datetime.hour


modifiedjuliandate = julian.to_jd(conv_datetime, fmt='mjd')

curr_filename = f'{hour}_{day:02d}{month:02d}_{station}.txt'

# ah_filename = f'{hour}_{day:02d}{month:02d}_ah.pickle'
# aw_filename = f'{hour}_{day:02d}{month:02d}_aw.pickle'



curr_filepath = os.path.join(interpolated_data,curr_filename)

if not os.path.exists(curr_filepath):
    while not os.path.exists(curr_filepath):

        hour -= 1
        curr_filename = f'{hour}_{day:02d}{month:02d}_{station}.txt'
        curr_filepath = os.path.join(interpolated_data,curr_filename)




# ah_filepath = os.path.join(interpol_pickles_dir,ah_filename)
# aw_filepath = os.path.join(interpol_pickles_dir,aw_filename)

# with open(ah_filepath,'rb') as ah_unpickler:
#     ah_interpolator = pickle.load(ah_unpickler)

# with open(aw_filepath,'rb') as aw_unpickler:
#     aw_interpolator = pickle.load(aw_unpickler)

# ah = ah_interpolator(lgt,lat)[0]
# aw = aw_interpolator(lgt,lat)[0]


with open(curr_filepath) as reader:
    ah,aw  = reader.readline().split(',')




# with open('/home/lape04/Dropbox/laix2/processamentos_gnss_out21/curr_interps.txt','w+') as results:
#     results.write(f'{ah},{aw},{modifiedjuliandate}')
    # results.write(f'ah,aw,{modifiedjuliandate}')

with open('/home/lape04/Dropbox/laix2/processamentos_gnss_out21/curr_ah.txt','w+') as results:
    results.write(f'{ah}')

with open('/home/lape04/Dropbox/laix2/processamentos_gnss_out21/curr_aw.txt','w+') as results:
    results.write(f'{aw}')

with open('/home/lape04/Dropbox/laix2/processamentos_gnss_out21/curr_jd.txt','w+') as results:
    results.write(f'{modifiedjuliandate}')