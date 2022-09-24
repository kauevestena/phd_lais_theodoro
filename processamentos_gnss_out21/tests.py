# # import datetime, os
# # from gnsscal import date2gpswd

# # rinex_sample_path = '/media/lais/SAMSUNG/INVERNO/AMCO/RINEX/amco1831corr2.20o'

# # def date_from_rnxdateline(inputrnxline):
# #     as_list = inputrnxline.split()

# #     year,month,day = as_list[:3]

# #     return datetime.date(year=int(year),month=int(month),day=int(day))

# # def get_rinex_date(inputpath,key='TIME OF FIRST OBS',return_gpsweek=False):
# #     with open(inputpath) as inputfile:
# #         for line in inputfile:
# #             if key in line:
# #                 return date_from_rnxdateline(line)

# # test = get_rinex_date(rinex_sample_path)

# # print(date2gpswd(test)[1])

# import numpy as np
# from scipy import interpolate

# import pickle

# # x = [100.0, 75.0, 50.0, 0.0, 100.0, 75.0, 50.0, 0.0, 100.0, 75.0, 50.0, 0.0, 100.0, 75.0, 50.0, 0.0, 100.0, 75.0, 50.0, 0.0, 100.0, 75.0, 50.0, 0.0, 100.0, 75.0, 50.0, 0.0, 100.0, 75.0, 50.0, 0.0, 100.0, 75.0, 50.0, 0.0, 100.0, 75.0, 50.0, 0.0]
# # y = [300.0, 300.0, 300.0, 300.0, 500.0, 500.0, 500.0, 500.0, 700.0, 700.0, 700.0, 700.0, 1000.0, 1000.0, 1000.0, 1000.0, 1500.0, 1500.0, 1500.0, 1500.0, 2000.0, 2000.0, 2000.0, 2000.0, 3000.0, 3000.0, 3000.0, 3000.0, 5000.0, 5000.0, 5000.0, 5000.0, 7500.0, 7500.0, 7500.0, 75000.0, 10000.0, 10000.0, 10000.0, 10000.0]
# # z = [100.0, 95.0, 87.5, 77.5, 60.0, 57.0, 52.5, 46.5, 40.0, 38.0, 35.0, 31.0, 30.0, 28.5, 26.25, 23.25, 23.0, 21.85, 20.125, 17.825, 17.0, 16.15, 14.875, 13.175, 13.0, 12.35, 11.375, 10.075, 10.0, 9.5, 8.75, 7.75, 7.0, 6.65, 6.125, 5.425, 5.0, 4.75, 4.375, 3.875]

# # f = interpolate.interp2d(x, y, z)

# # with open('test.pickle','wb') as pickler:
# #     pickle.dump(f,pickler)

# with open('test.pickle','rb') as unpickler:
#     new_interpolator = pickle.load(unpickler)

# print(new_interpolator(60,1100))

# import time

# timestr = str(time.time())

# with open('/home/lape04/Dropbox/laix2/processamentos_gnss_out21/test_out.txt','w+') as tester:
#     tester.write(timestr)


test_str = '012345'

print(test_str[0:4])