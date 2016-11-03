# IRdiode_pyqt
Small python program to analyze IR photodiode dark current using NumPy, Matplotlib, PyQt, and guidata.

This program was developed for a specific use at JPL.  This program reads data from csv and Excel and generated
I-V plots for photodiodes under test.

# Status
This program was developed for Python 2.7 in 2014 and used PyQt4 and guidata for PyQt4.
There were some breaking changes from PyQt4 --> PyQt5 and also in guidata, I suspect.
As such, the program is NOT currently fully functional.  
I updated many of the calls to PyQt to get the application to launch without but I will need to rework
major portions of the UI code to return full functionality.

# ToDo
Update GUI to use PyQt5 natively and remove dependence upon guidata.
