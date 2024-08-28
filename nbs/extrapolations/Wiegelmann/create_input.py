import numpy as np

def main():
    f = np.load('bx_by_bz.npz')

    bx = f['bx']
    by = f['by']
    bz = f['bz']

    xgridsize, ygridsize = bx.shape

    # input.dat
    file = open('input.dat',"w")

    for iy in range(ygridsize):
        for ix in range(xgridsize):
            
            file.write( '%.7f'%(bx[ ix, iy ]) + '\n' )
            file.write( '%.7f'%(by[ ix, iy ]) + '\n' )
            file.write( '%.7f'%(bz[ ix, iy ]) + '\n' )
    
    file.close()


    # grid.ini
    inifile = open('grid.ini',"w")
    
    inifile.write("nx")
    inifile.write(str(" "))
    inifile.write(str(int(xgridsize)))
    inifile.write(str(" "))
    
    inifile.write("ny")
    inifile.write(str(" "))
    inifile.write(str(int(ygridsize)))
    inifile.write(str(" "))
    
    inifile.write("nz")
    inifile.write(str(" "))
    inifile.write(str(int(ygridsize)))
    inifile.write(str(" "))
    
    inifile.write("mu")
    inifile.write(str(" "))
    inifile.write(str(0.0001))
    inifile.write(str(" "))
    
    inifile.write("nd")
    inifile.write(str(" "))
    # inifile.write(str(int(1+ygridsize/20)))
    inifile.write(str(int(0)))
    inifile.write(str(" "))

    inifile.close()

if __name__ == '__main__':
    main()