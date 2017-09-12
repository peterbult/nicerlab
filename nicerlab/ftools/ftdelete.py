
import os
import subprocess
from nicerlab.ftools.ftutils import *

def ftdelete( inputfile, outputfile, rows, ext=1, clobber=False):
    """
    Calls ftdelete in a subprocess. Returns the outputfile name on success and
    raises an error on failure.

    Parameters
    ----------
    inputfile: string
        Filename of the input fits table

    outputfile: string
        Filename of the output fits table

    rows: iterable
        list of row numbers that shoudl be deleted from the table

    ext: object
        Fits extension to filter. Takes the HDU name or index. Default is 1.

    clobber: bool
        Overwrite flag. Default is False, meaning existing output will **not**
        be overwritten.

    """

    if (os.path.exists(outputfile) and clobber==False):
        return outputfile
    
    # Copy the use environment
    my_env = os.environ.copy()
    my_env["HEADASPROMPT"] = "/dev/null"

    rowstr = ','.join(str(i) for i in rows)
    
    # Construct the command
    cmd = "ftdelrow '{0}[{1}]' {2} {3} clobber={4} confirm=no".format(
        inputfile, ext, outputfile, rowstr, 
        clobber_str(clobber)   
    )

    # Execute and get output
    p = subprocess.Popen(cmd, shell=True, env=my_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if (p.wait() == 0):
        return outputfile
    else:
        # Read from the stdout 
        message = p.stderr.read()
        raise FtoolsError(p.wait(), cmd, message)

