
import os
import subprocess
from nicerlab.ftools.ftutils import *

def xtefilter(appid, obsid, stem, dt, clobber=False):
    """
    Calls xtefilt in a subprocess. The command will create a xte_filter.xfl file
    in the stem directory. Returns the name of the output file on success and
    raises an error on failure. 

    Parameters
    ----------
    appid: string
        Path to the local 'appidlist.txt' file.

    obsid: string
        ObsID of the file to filter.

    stem: string
        Path of the ObsID folder.

    dt: float
        Time resolution of the filter file.

    clobber: bool [default=False]
        Overwrite flag. By default existing files will not be overwritten.
    """

    # Set output file
    outputfile = os.path.join(stem, 'xte_filter')

    # Process clobber flag
    if os.path.exists(outputfile+".xfl"):
        if clobber:
            os.remove(outputfile+".xfl")
        else:
            return outputfile+".xfl"
    
    # Copy the use environment
    my_env = os.environ.copy()
    my_env["HEADASPROMPT"] = "/dev/null"
    
    # Construct the command
    cmd = "xtefilt -a {} -o {} -p {} -t {} -f {} -c".format(
        appid, obsid, stem, dt, outputfile
    )

    # Execute and get output
    p = subprocess.Popen(cmd, shell=True, env=my_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if (p.wait() == 0):
        return outputfile+".xfl"
    else:
        # Read from the stdout 
        message = p.stderr.read()
        raise FtoolsError(p.wait(), cmd, message)

    
def maketime(filtername, gtiname, expression, clobber=False):
    """
    Calls maketime in a subprocess. This process will create a gti table at the
    file path specified by gtiname.

    Parameters
    ----------
    filtername: string
        Filename of the input xte filter file.

    gtiname: string
        Filename of the output gti table.

    expression: string
        Filter expression.

    clobber: bool (Default=False)
        Overwrite flag. By default existing files will not be overwritten.

    """
    
    # Process the clobber flag
    if os.path.exists(gtiname):
        if clobber:
            os.remove(gtiname)
        else:
            return gtiname

    # Copy the use environment
    my_env = os.environ.copy()
    my_env["HEADASPROMPT"] = "/dev/null"
    
    # Construct the command
    cmd = "maketime {} {} {} ".format(
        filtername, gtiname, expression
    )
    cmd += "name='NAME' time='TIME' value='VALUE' "
    cmd += "compact='no' prefr=0.5 postfr=0.5"

    # Execute and get output
    p = subprocess.Popen(cmd, shell=True, env=my_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if (p.wait() == 0):
        return gtiname
    else:
        # Read from the stdout 
        message = p.stderr.read()
        raise FtoolsError(p.wait(), cmd, message)

