
import os
from astropy.io.fits import getheader
from nicerlab.ftools import ftcopy, ftdelete, ftselect, ftappend

def ftsplit(inputfile, outputfile, clobber=False, verbose=False):
    """
    Split a fits event file on its GTI table values.

    Parameters
    ----------
    inputfile: string
        Fits file containg an EVENT list and a GTI table.

    outputfile: string
        Basename for the output files. Each file is formatted
        as 
            [outputfile][xx].fits
        where 'xx' is the GTI row number.
    """

    # Get the number of GTIs
    numrows = getheader(inputfile, extname='GTI')['NAXIS2']

    # Collect the output filenames
    files = []

    # Expand the GTI
    for n in range(numrows):
        if verbose:
            print("> Filtering row {}".format(n+1))

        # Construct the output filename
        outfile = "{}_{:02d}.fits".format(outputfile, n)

        # Store the filtered file name
        files.append(outfile)

        # Test if file exists
        if os.path.isfile(outfile) and clobber==False:
            continue

        # Copy the gti table to a temporary file
        gtiname = "gti{:02}.fits".format(n)
        ftcopy(inputfile, gtiname, ext=2, clobber=True)

        # Truncate the temporary GTI
        rows = [i for i in range(1,numrows+1) if i != n+1]
        ftdelete(gtiname, gtiname, rows, ext=1, clobber=True)

        # Select event rows bases on the trucated GTI
        ftselect(inputfile, outfile, 
                expression='gtifilter("{}")'.format(gtiname),
                copyall=False, clobber=True)

        # Append the GTI table to the filtered event list
        ftappend(gtiname, outfile, ext='GTI')

        # Remove the temorary file
        os.remove(gtiname)


    return files
