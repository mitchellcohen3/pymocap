# pymocap
Collection of tools for working with mocap data

## Dependencies

- ``navlie`` (optional)
- ``pymlg`` (installed by pip)

## Installation

`cd` into the folder and as usual:

    pip install -e .

## Getting started
Check out the examples in the `./examples/` folder.

## Idea for improvement
This appears to work, it is essentially the same as `decar_mocap_tools`. However,
to minimize dead-reckoning error, we still need require velocity information 
from the mocap spline, which has error.

A solution to this could be to start the dead reckoning at static moments at 
in the data, and replace with zero velocity instead of the spline-derived 
velocity. 
