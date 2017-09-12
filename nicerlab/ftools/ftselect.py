
import os
import subprocess
from nicerlab.ftools.ftutils import *
 
def ftselect( inputfile, outputfile, expression, ext=1, copyall=True, clobber=False):
    """
    Calls ftselect in a subprocess. Returns the outputfile name on success and
    raises an error on failure.

    Parameters
    ----------
    inputfile: string
        Filename of the input fits table

    outputfile: string
        Filename of the output fits table

    expression: string
        Ftools style rowfilter expression. See
        https://heasarc.gsfc.nasa.gov/lheasoft/ftools/headas/rowfilter.html
        for detailed instructions.

    clobber: bool
        Overwrite flag. Default is False, meaning existing output will **not**
        be overwritten.

    ext: object
        Fits extension to filter. Takes the HDU name or index. Default is 1.

    """

    if (os.path.exists(outputfile) and clobber==False):
        return outputfile
    
    # Copy the use environment
    my_env = os.environ.copy()
    my_env["HEADASPROMPT"] = "/dev/null"
    
    # Construct the command
    cmd = "ftselect '{0}[{1}]' {2} '{3}' copyall={4} clobber={5}".format(
        inputfile, ext, outputfile, expression, 
        clobber_str(copyall), clobber_str(clobber)   
    )

    # Execute and get output
    p = subprocess.Popen(cmd, shell=True, env=my_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if (p.wait() == 0):
        return outputfile
    else:
        # Read from the stdout 
        message = p.stderr.read()
        raise FtoolsError(p.wait(), cmd, message)

