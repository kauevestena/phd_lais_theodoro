import os,pickle
import numpy as np
from scipy import interpolate

# stations:


AMCO=(-4.871989319444450,-65.333978925000000+360)
NAUS=(-3.022919663888890,-60.055016644444400+360)
SAGA=(-0.143854466666667,-67.057781041666700+360)
UFPR=(-25.44836859722220,-49.230954769444400+360)

stations = {
    "ufpr" : UFPR,
    'naus' : NAUS,
    'saga' : SAGA,
    'amco' : AMCO,
}




print(UFPR,NAUS,SAGA,AMCO)

vmf_coefs_path = '/home/lape04/Dropbox/laix2/dados VMF1'

outdir = '/home/lape04/data/stations_interps'

lat_min = -40
lat_max = 10

min_lgt = -80
max_lgt = -30

with open('grids_report.csv','w+') as deb_report:
    deb_report.write('filename,station,ah,aw\n')
    for filename in os.listdir(vmf_coefs_path):

        filepath = os.path.join(vmf_coefs_path,filename)
        print('file ',filepath)

        lgts = []
        lats = []
        ah_list = []
        aw_list = [] 

        with open(filepath) as filereader:
            for i,line in enumerate(filereader):
                if not '!' in line:
                    # print(i)
                    # if i % 200 == 0:
                    #     print('line ',i)

                    lat,lgt,ah,aw,_,_ = list(map(float,line.strip('\n').split()))

                    # if i % 10000 == 0:
                    #     print(lat,lgt)

                    if lat > lat_min and lat < lat_max:
                        if lgt > 280 and lgt < 330:

                            lgts.append(lgt)
                            lats.append(lat)
                            ah_list.append(ah)
                            aw_list.append(aw)


        as_mat = np.array([lats,lgts,ah_list,aw_list]).T

        as_mat = as_mat[as_mat[:, 0].argsort()]

        # print(as_mat[1:8,:])
        # print(as_mat[-8:-1,:])

        # print(as_mat[:,0])

        x = as_mat[:,0] #lats
        y = as_mat[:,1] #lgts


        z_ah = as_mat[:,2] #ah_list
        z_aw = as_mat[:,3] #aw_list




        # print(len(x),len(y),len(z_ah),len(z_aw))
        # print(min(x),min(y),min(z_ah),min(z_aw))
        # print(max(x),max(y),max(z_ah),max(z_aw))


        print('generating interpolators')

        interpolator_ah = interpolate.interp2d(x,y,z_ah)
        interpolator_aw = interpolate.interp2d(x,y,z_aw)

        print('interpolators are created!')


        basename = filename.replace('.txt','')

        print(basename)

        for station in stations:

            ah = interpolator_ah(*stations[station])[0]
            aw = interpolator_aw(*stations[station])[0]

            outpath = os.path.join(outdir,f"{basename}_{station}.txt")

            deb_report.write(f'{filename},{station},{ah},{aw}\n')
            

            with open(outpath,'w+') as writer:
                writer.write(f'{ah},{aw}')

        # filename_ah = filename.replace(".txt","_ah.pickle")
        # filename_aw = filename.replace(".txt","_aw.pickle")

        # filepath_ah = os.path.join(outdir,filename_ah)
        # filepath_aw = os.path.join(outdir,filename_aw)

        # # pickling:

        # with open(filepath_ah,'wb') as ah_pickler:
        #     pickle.dump(interpolator_ah,ah_pickler)

        # with open(filepath_aw,'wb') as aw_pickler:
        #     pickle.dump(interpolator_aw,aw_pickler)



