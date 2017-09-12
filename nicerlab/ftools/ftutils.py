
__all__ = ['FtoolsError', 'clobber_str']

class FtoolsError(Exception):
    """Error in FTOOLS execution"""

    def __init__(self, code, expression, message):
        self.code = code
        self.expression = expression
        self.message = message

def clobber_str(clobber):
    """clobber bool formatter"""

    if (clobber == True):
        return 'yes'
    else:
        return 'no'

