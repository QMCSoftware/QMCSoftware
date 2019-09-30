def summary_qmc(stopObj,measureObj,funObj,distribObj,dataObj):
    h1 = '%s (%s)\n'
    item_i = '%25s: %d\n'
    item_f = '%25s: %-15.4f\n'
    item_s = '%25s: %-15s\n'
    s = 'Solution: %-15.4f\n%s\n'%(dataObj.solution,'~'*50)
    
    s += h1%(type(funObj).__name__,'Function Object')

    s += h1%(type(measureObj).__name__,'Measure Type')
    s += item_s%('dimension',str(measureObj.dimension))

    s += h1%(type(distribObj).__name__,'Distribution Object')
    s += item_s%('trueD.measureName',type(distribObj.trueD).__name__)

    s += h1%(type(stopObj).__name__,'StoppingCriterion Object')
    s += item_f%('absTol',stopObj.absTol)
    s += item_f%('relTol',stopObj.relTol)
    s += item_i%('nMax',stopObj.nMax)
    s += item_f%('inflate',stopObj.inflate)
    s += item_f%('alpha',stopObj.alpha)
    
    s += h1%(type(dataObj).__name__,'Data Object')
    s += item_s%('nSamplesUsed',str(dataObj.nSamplesUsed))
    s += item_f%('timeUsed',dataObj.timeUsed)
    s += item_s%('confidInt',str(dataObj.confidInt))
    
    print(s)
    
    '''
    print()
    print(stopObj)
    print(measureObj)
    print(funObj)
    print(distribObj)
    print(dataObj)
    '''
    return 