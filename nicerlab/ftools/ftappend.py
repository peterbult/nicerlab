
import os
import subprocess
from nicerlab.ftools.ftutils import *

def ftappend( inputfile, outputfile, ext=1):
    """
    Calls ftappend in a subprocess. Returns the outputfile name on success and
    raises an error on failure.

    Parameters
    ----------
    inputfile: string
        Filename of the input fits table

    outputfile: string
        Filename of the output fits table

    ext: object
        Fits extension to filter. Takes the HDU name or index. Default is 1.
    """

    # Copy the use environment
    my_env = os.environ.copy()
    my_env["HEADASPROMPT"] = "/dev/null"
    
    # Construct the command
    cmd = "ftappend '{0}[{1}]' {2}".format(
        inputfile, ext, outputfile
    )

    # Execute and get output
    p = subprocess.Popen(cmd, shell=True, env=my_env, 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if (p.wait() == 0):
        return outputfile
    else:
        # Read from the stdout 
        message = p.stderr.read()
        raise FtoolsError(p.wait(), cmd, message)

