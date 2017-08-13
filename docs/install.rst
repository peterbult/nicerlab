===============
Getting Started
===============

Obtaining nicerlab
------------------

The nicerlab package can be obtained through github. To obtain the source
code use::
    
    git clone https://github.com/peterbult/nicerlab.git

Installing nicerlab
-------------------

The recommended way to install nicerlab is to use pip. Navigate to the project
folder, and from the command line::
    
    make install

This will use pip to install the package, as well as all required dependencies.
Alternatively, one can also use pip directly to install as::

    pip install -r requirements.txt
    pip install .

Testing
-------

Most nicerlab modules have associated test functions. These tests can be run
using::

    make test

or alternatively, using pytest directly::

    pytest test/

License
-------

The nicerlab package is licensed using the `MIT license <http://opensource.org/licenses/MIT>`_.

    Copyright (c) 2017 Peter Bult <https://github.com/peterbult>

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.


Contributing
============

Contributions through github are welcome.

