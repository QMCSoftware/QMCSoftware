def stop_notebook(query = "Type 'yes' to continue long code"):
    #This is a function to be able to stop a notebook when you run all cells
    keep_running = input("Type 'yes' to continue long code")
    if keep_running.casefold() != 'yes':
        import sys
        import warnings
        warnings.filterwarnings("ignore")
        sys.exit('Stopping execution')