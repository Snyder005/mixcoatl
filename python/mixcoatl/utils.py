"""Utility functions and classes for MixCOATL.
"""

AMP2SEG = {1 : 'C17', 2 : 'C16', 3 : 'C15', 4 : 'C14', 5 : 'C13', 6 : 'C12', 7 : 'C11', 8 : 'C10',
           9 : 'C00', 10 : 'C01', 11 : 'C02', 12 : 'C03', 13 : 'C04', 14 : 'C05', 15 : 'C06', 16 : 'C07'}
"""dict: Dictionary mapping from CCD amplifier number to segment names."""

SEG2AMP = {'C00' : 9, 'C01' : 10, 'C02' : 11, 'C03' : 12, 'C04' : 13, 'C05' : 14, 'C06' : 15, 'C07' : 16,
           'C10' : 8, 'C11' : 7, 'C12' : 6, 'C13' : 5, 'C14' : 4, 'C15' : 3, 'C16' : 2, 'C17' : 1}
"""dict: Dictionary mapping from CCD segment names to amplifier number."""
