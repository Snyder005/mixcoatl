# This file is part of mixcoatl.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

AMP2SEG = {1 : 'C10', 2 : 'C11', 3 : 'C12', 4 : 'C13', 5 : 'C14', 6 : 'C15', 7 : 'C16', 8 : 'C17',
           9 : 'C07', 10 : 'C06', 11 : 'C05', 12 : 'C04', 13 : 'C03', 14 : 'C02', 15 : 'C01', 16 : 'C00'}
"""dict: Dictionary mapping from CCD amplifier number to segment names."""

SEG2AMP = {'C00' : 16, 'C01' : 15, 'C02' : 14, 'C03' : 13, 'C04' : 12, 'C05' : 11, 'C06' : 10, 'C07' : 9,
           'C10' : 1, 'C11' : 2, 'C12' : 3, 'C13' : 4, 'C14' : 5, 'C15' : 6, 'C16' : 7, 'C17' : 8}
"""dict: Dictionary mapping from CCD segment names to amplifier number."""
