import os
import numpy as np
from astropy.io import fits

def write_pds(pds, filename='pds.fits', mjd=None, clobber=False):
    # Handle clobber
    if os.path.isfile(filename):
        if clobber:
            os.remove(filename)
        else:
            raise IOError("cannot write spectrum: file exists")

    # Extract the mjd
    mjd = getattr(pds, 'mjd', None)
    if mjd is None:
        mjd = 0

    # Reconstruct the counts
    counts = 0.5 * pds[:,0]
    # Extract tseg
    tseg = 1.0 / pds.df
    # Fake MJDs
    mjdvec = pds.mjd + np.arange(len(counts)) * (tseg/86400.0)
    
    # Construct the primary HDU
    prihdr = fits.Header()
    prihdr['OBSERVER'] = 'nicerlab'
    prihdu = fits.PrimaryHDU(header=prihdr)
    
    # Construct the table HDU
    tbhdr = fits.Header()
    tbhdr['TYPE']   = 'powerspectrum'
    tbhdr['FRES']   = pds.df
    tbhdr['SIZE']   = pds.shape[1]
    tbhdr['STACK']  = pds.shape[0]
    tbhdr['CSTACK'] = 1

    tbhdr['DURATION'] = pds.tstop - pds.tstart
    tbhdr['EXPOSURE'] = 1
    
    tbhdr['DETCOUNT']   = 52
    tbhdr['INSTRUME']  = 'XTI'
    tbhdr['TELESCOP'] = 'NICER'
    
    tbhdu = fits.BinTableHDU.from_columns(
        [fits.Column(name='MJD', format='D', array=mjdvec),
         fits.Column(name='PDS', format='{:d}D'.format(pds.shape[1]), array=pds),
         fits.Column(name='COUNTS', format='D', array=counts)],
        header=tbhdr
    )
   
    # Write to file
    thdulist = fits.HDUList([prihdu, tbhdu])
    thdulist.writeto(filename)


def write_spectrum(spectrum, filename='spec.fits', clobber=False):
    if os.path.isfile(filename):
        if clobber:
            os.remove(filename)
        else:
            raise IOError("cannot write spectrum: file exists")

    # Construct the primary HDU
    prihdr = fits.Header()
    prihdr['CREATOR' ] = 'nicerlab'
    prihdr['DATE'    ] = '2017-01-01'
    prihdr['ORIGIN'  ] = 'NASA/GSFC'
    prihdr['INSTRUME'] = 'XTI'
    prihdr['TELESCOP'] = 'NICER'
    prihdr['DATAMODE'] = 'PHOTON'
    prihdr['OBJECT'  ] = 'Crab'
    prihdr['TELAPSE' ] = 1
    prihdr['LIVETIME' ] = 1
    prihdr['TSTART'] = 0
    prihdr['TSTOP'] = 1
    
    prihdu = fits.PrimaryHDU(header=prihdr)
    
    # Construct the table HDU
    tbhdr = fits.Header()
    tbhdr['HDUCLASS'] = 'OGIP'
    tbhdr['HDUVERS1'] = '1.2.0'
    tbhdr['HDUVERS' ] = '1.2.0'
    tbhdr['HDUCLAS3'] = 'COUNT'
    tbhdr['TLMIN1'  ] = spectrum.cstart
    tbhdr['TLMAX1'  ] = spectrum.cstop-1
    tbhdr['TELESCOP'] = 'NICER'
    tbhdr['INSTRUME'] = 'XTI'
    tbhdr['FILTER'  ] = 'NONE'
    tbhdr['AREASCAL'] = 1.0
    tbhdr['BACKFILE'] = 'NONE'
    tbhdr['BACKSCAL'] = 1.0
    tbhdr['CORRFILE'] = 'NONE'
    tbhdr['CORRSCAL'] = 1.0
    tbhdr['RESPFILE'] = 'NONE'
    tbhdr['ANCRFILE'] = 'NONE'
    tbhdr['PHAVERSN'] = '1992a'
    tbhdr['DETCHANS'] = spectrum.cstop - spectrum.cstart
    tbhdr['CHANTYPE'] = 'PI'
    tbhdr['POISSERR'] = 'T'
    tbhdr['STAT_ERR'] = 0
    tbhdr['SYS_ERR' ] = 0
    tbhdr['GROUPING'] = 0
    tbhdr['QUALITY' ] = 0
    tbhdr['HDUCLAS1'] = 'SPECTRUM'
    tbhdr['DATAMODE'] = 'PHOTON'
    tbhdr['DATE'] = '2017-02-01T13:14:15'
    tbhdr['EXPOSURE'] = 100.0
    tbhdr['ONTIME'] = 100.0
    tbhdr['TIMEPIXR'] = 0.0
    tbhdr['TIMEDEL'] = 40e-9
    tbhdr['TIMEZERO'] = 0.0
    tbhdr['TARG_ID'] = 999999999
    tbhdr['OBS_ID'] = '0000000000'
    tbhdr['ORIGIN'] = 'NASA/GSFC'
    tbhdr['CREATOR'] = 'nicerlab'
    tbhdr['OBJECT'] = 'Crab'
    tbhdr['MJDREFI'] = 56658
    tbhdr['MJDREFF'] = 7.775925925925930E-04
    tbhdr['TIMEREF'] = 'LOCAL'
    tbhdr['TASSIGN'] = 'SATELLITE'
    tbhdr['TIMEUNIT'] = 's'
    tbhdr['TIERRELL'] = 1.0e-8
    tbhdr['TIERABSO'] = 1.0
    tbhdr['TSTART'] = 100.0
    tbhdr['TSTOP'] = 200.0
    tbhdr['CLOCKAPP'] = 'F'
    tbhdr['DEADAPP'] = 'F'
    tbhdr['LEAPINIT'] = 0.0
    tbhdr['TELAPSE'] = 100.0
    tbhdr['LIVETIME'] = 100.0
    tbhdr['DETNAME'] = 'NONE'
    tbhdr['MJD-OBS'] = 5.793705341648148E+04 
    tbhdr['USER'] = 'pbult'
    tbhdr['HDUCLAS2'] = 'TOTAL'
    tbhdr['TOTCTS'] = 0
    tbhdr['SPECDELT'] = 1
    tbhdr['SPECPIX' ] = spectrum.cstart
    tbhdr['SPECVAL' ] = spectrum.cstart
    tbhdr['GCOUNT'  ] = 1

    # I : 16-bit int (signed)
    # J : 32-bit int (signed)
    print(len(spectrum.channelspace()))
    print(len(spectrum))
    tbhdu = fits.BinTableHDU.from_columns(
        [fits.Column(name='CHANNEL', format='J', array=spectrum.channelspace()),
         fits.Column(name='COUNTS',  format='J', array=spectrum, unit='count')],
        header=tbhdr,
        name='SPECTRUM'
    )
   
    # Write to file
    thdulist = fits.HDUList([prihdu, tbhdu])
    thdulist.writeto(filename)

