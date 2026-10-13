"""Test package marker.

Present so `python -m unittest discover -s tests -t .` can import the tests directory as a
package; without it unittest refuses to discover from that start directory.
"""
