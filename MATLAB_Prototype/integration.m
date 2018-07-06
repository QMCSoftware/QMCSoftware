function sol = integration(funObj, distribObj, stopCritObj)
%§\mcommentfont Specify and generate values $f(\vx)$ for $\vx \in \cx$§
% funObj = an object from class fun
% distribObj = an object from class discrete_distribution
% stopcritObj = an object from class stopping_criterion

stopYet(stopCritObj, funObj, distribObj);
while ~strcmp(stopCritObj.nextStep,"stop")
   updateData(stopCritObj, distribObj, fun_obj, decompType)
   newData = getData(stopCritObj, distribObj, funObj, decompType, oldData);
   [stop, oldData, sol] = stopYet(stopCritObj, distribObj, decompType, oldData, newData);
end

