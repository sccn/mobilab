% vector is a monotonically increasing sorted vector. 
function ind = binary_findClosest(vector,target)
% 
% N = length(vector);
% 
% if N <= 5
%     [~,ind] = min(abs(vector - target));
%     return;
% end
% 
% 
% ind = ceil(N/2);
% if vector(ind) > target
%     ind = binary_findClosest(vector(1:ind-1),target);
% else
%     if vector(ind) == target
%         return;
%     else
%         subind = binary_findClosest(vector(ind+1:end),target);
%         ind = ind + subind;
%         return;
%     end
% end


N = length(vector);
startInd = 1;
endInd = N;
while N > 5
    
    ind = ceil(N/2);
    if vector(startInd + ind - 1) > target
        %startInd = startInd;
        endInd = startInd+ ind - 2;
    else
        if vector(startInd + ind - 1) == target
            ind = startInd + ind - 1;
            return;
        else
            startInd = startInd + ind;
            %endInd = endInd;
            
            
        end
    end
    
    N = endInd - startInd + 1;
end

[~,ind] = min(abs(vector(startInd:endInd)-target));
ind = ind + startInd - 1;

