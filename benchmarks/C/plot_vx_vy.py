import pylab as P
import glob
import cPickle
xfiles = glob.glob('x-velocity_xspace*')
yfiles = glob.glob('y-velocity_xspace*')
xfiles.sort()
yfiles.sort()
i = 0
xmin = []
xmax = []
ymin = []
ymax = []
for x,y in zip(xfiles,yfiles):
    print "x = %s" % x
    print "y = %s" % y
    vx = (P.loadtxt(x)).reshape(100,100)
    xmin.append(vx.min())
    xmax.append(vx.max())
    vy = (P.loadtxt(y)).reshape(100,100)
    ymin.append(vy.min())
    ymax.append(vy.max())
    # P.subplot(121)
    # P.imshow(vx,vmin=-1,vmax=1)
    # P.title(r'$v_x$')
    # P.colorbar()    
    # P.subplot(122)
    # P.imshow(vy,vmin=-1,vmax=1)
    # P.title(r'$v_y$')
    # P.colorbar()    

    # P.savefig('frames/velocity_%04i.png' %i)
    # P.close()
    # i += 1

outfi = open('minmax.dat','w')
cPickle.dump(xmin,outfi)
cPickle.dump(xmax,outfi)
cPickle.dump(ymin,outfi)
cPickle.dump(ymax,outfi)
outfi.close()

P.subplot(211)
P.plot(xmin)
P.plot(xmax)
P.ylim(-10.,10.)
P.subplot(212)
P.plot(ymin)
P.plot(ymax)
P.ylim(-10.,10.)
P.savefig('velocity_minmax.png')
P.close()

    
