def stop_notebook(query = "Type 'yes' to continue running notebook"):
    #This is a function to be able to stop a notebook when you run all cells
    keep_running = input(query)
    if keep_running.casefold() != 'yes':
        import sys
        import warnings
        warnings.filterwarnings("ignore")
        sys.exit('Pausing notebook execution')