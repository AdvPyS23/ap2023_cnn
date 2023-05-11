### Pylint output (preprocessing.py)
preprocessing.py:27:0: C0301: Line too long (116/100) (line-too-long) *
preprocessing.py:44:0: C0301: Line too long (104/100) (line-too-long) *
preprocessing.py:48:0: C0301: Line too long (103/100) (line-too-long) #
preprocessing.py:73:0: C0305: Trailing newlines (trailing-newlines) *
preprocessing.py:9:0: R0402: Use 'from torchvision import transforms' instead (consider-using-from-import) *
preprocessing.py:10:0: E0401: Unable to import 'scripts.visualization' (import-error) #

Your code has been rated at 2.86/10

\*Points that have been corrected   
\#Points that are ignored (using pylint disable)

After corrections:
scripts/preprocessing.py:72:0: C0304: Final newline missing (missing-final-newline)
Your code has been rated at 9.29/10 (previous run: 2.86/10, +6.43)

Last change:
Your code has been rated at 10.00/10 (previous run: 9.29/10, +0.71)

### Pylint output (visualization.py)
Module visualization
scripts/visualization.py:8:0: E0401: Unable to import 'torchvision' (import-error) # sounds like a you problem
scripts/visualization.py:29:12: W0612: Unused variable 'labels' (unused-variable) *

-----------------------------------
Your code has been rated at 5.00/10

After correction:

Your code has been rated at 10.00/10 (previous run: 5.00/10, +5.00)


Updates will come as the project progresses






