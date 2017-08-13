
import os
import subprocess

class FtoolsError(Exception):
    """Error in FTOOLS execution"""

    def __init__(self, code, expression, message):
        self.code = code
        self.expression = expression
        self.message = message
        
def clobber_str(clobber):
    if (clobber == True):
        return 'yes'
    else:
        return 'no'

def ftcopy( inputfile, outputfile, ext=1, copyall=False, clobber=False):
    """
    Calls ftcopy in a subprocess. Returns the outputfile name on success and
    raises an error on failure.

    Parameters
    ----------
    inputfile: string
        Filename of the input fits table

    outputfile: string
        Filename of the output fits table

    ext: object
        Fits extension to filter. Takes the HDU name or index. Default is 1.

    copyall: bool
        Copy flag. Default is False, meaning only the selected HDU is
        copied.

    clobber: bool
        Overwrite flag. Default is False, meaning existing output will **not**
        be overwritten.

    """

    if (os.path.exists(outputfile) and clobber==False):
        return outputfile
    
    # Copy the use environment
    my_env = os.environ.copy()
    my_env["HEADASPROMPT"] = "/dev/null"
    
    # Construct the command
    cmd = "ftcopy '{0}[{1}]' {2} copyall='{3}' clobber={4}".format(
        inputfile, ext, outputfile, 
        clobber_str(copyall), clobber_str(clobber)   
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

