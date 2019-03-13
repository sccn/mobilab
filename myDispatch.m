function myDispatch(varargin)
try
    obj = varargin{3};
    if isa(obj,'dataSource')
        contObj = obj;
    else
        contObj = obj.container;
    end
    N = length(contObj.item);
    if ~isempty(strfind(varargin{4},'Browser'))
        eval(['obj.' varargin{4} ';']);
    elseif isnumeric(varargin{5})
        eval(['obj.' varargin{4} '(' num2str(varargin{5}) ');']);
    else
        eval(['obj.' varargin{4} '(' varargin{5} ');']);
    end
    if N ~= length(contObj.item)
        contObj.container.gui;
    end
catch ME
    errordlg(ME.message);
end