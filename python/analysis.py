
def snapshot(data, it):
    import pylab as P

    P.subplot(121)
    P.imshow(data['ux']['xspace'].real)
    P.subplot(122)
    P.imshow(data['ux']['xspace'].real)
    outfile = "snap_%04i.png" % it
    P.savefig(outfile)
    
